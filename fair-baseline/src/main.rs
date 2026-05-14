//! fair-baseline: replay an Anthropic workload through two enforcement
//! mechanisms with IDENTICAL cost arithmetic, differing only in WHEN
//! they check the cap.
//!
//! Mechanism 1 (TB, type-level discipline): pre-call check
//!   - Before each call: compute reservation = byte_estimate * input_rate
//!     + max_output_tokens * output_rate
//!   - Refuse if spent + reservation > cap
//!   - Otherwise issue call, add actual cost to spent
//!
//! Mechanism 2 (RuntimeBudgetCallback, post-call callback): post-call check
//!   - Before each call: check current spent; refuse if callback has tripped
//!   - Issue call, observe actual cost, add to spent
//!   - If spent > cap, mark callback as tripped; future calls refused
//!   - The call that crossed the cap was already issued and paid for
//!
//! The point: both mechanisms use the same estimator, same cost
//! arithmetic, same workload. The difference is purely the temporal
//! ordering of the check. Any overshoot the callback shows is therefore
//! a structural property of post-call enforcement, not an arithmetic
//! defect; the comparison isolates "type-level vs runtime" from
//! "designed for dollars vs not designed for dollars".
//!
//! Workload: the 30 tool-augmented Anthropic calls captured by a1-rerun
//! (real claude-haiku-4-5 input/output token counts from the API).
//!
//! Usage:
//!     cargo run --release -- \
//!         --input ../a1-rerun/a1_rerun_results.csv \
//!         --output fair_baseline_results.csv

use anyhow::{Context, Result};
use clap::Parser;
use csv::{Reader, Writer};
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------

#[derive(Parser)]
#[command(version, about = "Fair-baseline replay (TB vs runtime callback)")]
struct Cli {
    /// Input CSV (produced by the a1-rerun binary)
    #[arg(long, default_value = "a1_rerun_results.csv")]
    input: String,

    /// Output CSV (one row per (cap, mechanism))
    #[arg(long, default_value = "fair_baseline_results.csv")]
    output: String,

    /// Cap values to sweep in micro-cents (uc). Multiple --cap-uc flags allowed.
    #[arg(long = "cap-uc", default_values_t = vec![5000u64, 6500, 8000, 10000])]
    caps: Vec<u64>,

    /// Anthropic input rate (uc / token). Default: claude-haiku-4-5 at $1/Mtok.
    #[arg(long, default_value = "1.0")]
    input_rate: f64,

    /// Anthropic output rate (uc / token). Default: claude-haiku-4-5 at $5/Mtok.
    #[arg(long, default_value = "5.0")]
    output_rate: f64,

    /// max_output_tokens used for reservation (matches paper setup).
    #[arg(long, default_value = "200")]
    max_output_tokens: u32,

    /// Filter to a specific cell of the input CSV.
    #[arg(long, default_value = "cell_2_tools")]
    cell: String,
}

// -----------------------------------------------------------------------------
// CSV row types
// -----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct InputRow {
    cell: String,
    prompt_id: u32,
    prompt_class: String,
    #[serde(rename = "has_tools")]
    _has_tools: bool,
    request_body_bytes: u32,
    #[serde(rename = "tiktoken_estimator_tokens")]
    _tiktoken_estimator_tokens: u32,
    anthropic_input_tokens: u32,
    anthropic_output_tokens: u32,
    #[serde(rename = "bt_ratio")]
    _bt_ratio: f64,
    #[serde(rename = "k_byte")]
    _k_byte: f64,
    #[serde(rename = "k_tiktoken")]
    _k_tiktoken: f64,
}

#[derive(Debug, Serialize, Clone)]
struct OutputRow {
    cap_uc: u64,
    mechanism: String,
    n_calls: u32,
    n_admitted: u32,
    n_refused: u32,
    spent_uc: u64,
    overshoot_uc: u64,
    overshoot_pct_of_cap: f64,
    mean_reservation_per_call_uc: u64,
    mean_actual_per_call_uc: u64,
    cap_crossing_call: i32, // -1 if cap never crossed
}

// -----------------------------------------------------------------------------
// Per-call cost derivation
// -----------------------------------------------------------------------------

struct Call {
    call_id: u32,
    class: String,
    body_bytes: u32,
    in_tok: u32,
    out_tok: u32,
}

impl Call {
    fn reservation_uc(&self, input_rate: f64, output_rate: f64, max_out: u32) -> u64 {
        // Conservative reservation: treat each byte of the full request body
        // as one token at the input rate. Plus max_output_tokens at output rate.
        // This is the byte-length estimator as discussed in the paper §V-K0.
        let input_part = self.body_bytes as f64 * input_rate;
        let output_part = max_out as f64 * output_rate;
        (input_part + output_part).round() as u64
    }

    fn actual_cost_uc(&self, input_rate: f64, output_rate: f64) -> u64 {
        let input_part = self.in_tok as f64 * input_rate;
        let output_part = self.out_tok as f64 * output_rate;
        (input_part + output_part).round() as u64
    }
}

// -----------------------------------------------------------------------------
// Replay mechanisms
// -----------------------------------------------------------------------------

#[derive(Debug)]
struct ReplayResult {
    n_admitted: u32,
    n_refused: u32,
    spent_uc: u64,
    cap_crossing_call: i32,
}

/// Pre-call enforcement: compute would-be reservation, refuse if it would
/// push cumulative reservation over the cap.
fn replay_tb(calls: &[Call], cap_uc: u64, cli: &Cli) -> ReplayResult {
    let mut spent: u64 = 0;
    let mut admitted = 0;
    let mut refused = 0;
    let mut cap_crossing = -1i32;
    for c in calls {
        let r = c.reservation_uc(cli.input_rate, cli.output_rate, cli.max_output_tokens);
        if spent.saturating_add(r) > cap_uc {
            refused += 1;
            continue;
        }
        let actual = c.actual_cost_uc(cli.input_rate, cli.output_rate);
        let new_spent = spent.saturating_add(actual);
        if cap_crossing < 0 && new_spent > cap_uc {
            cap_crossing = c.call_id as i32;
        }
        spent = new_spent;
        admitted += 1;
    }
    ReplayResult { n_admitted: admitted, n_refused: refused, spent_uc: spent, cap_crossing_call: cap_crossing }
}

/// Post-call callback enforcement: check only the current spent before
/// issuing. The call goes through, the cost is recorded after, and the
/// callback trips only when cumulative actual cost exceeds the cap.
/// The call that crossed the cap has already been issued and paid for.
fn replay_callback(calls: &[Call], cap_uc: u64, cli: &Cli) -> ReplayResult {
    let mut spent: u64 = 0;
    let mut admitted = 0;
    let mut refused = 0;
    let mut tripped = false;
    let mut cap_crossing = -1i32;
    for c in calls {
        if tripped {
            refused += 1;
            continue;
        }
        let actual = c.actual_cost_uc(cli.input_rate, cli.output_rate);
        let new_spent = spent.saturating_add(actual);
        admitted += 1;
        if cap_crossing < 0 && new_spent > cap_uc {
            cap_crossing = c.call_id as i32;
        }
        spent = new_spent;
        if spent > cap_uc {
            tripped = true;
        }
    }
    ReplayResult { n_admitted: admitted, n_refused: refused, spent_uc: spent, cap_crossing_call: cap_crossing }
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load rows
    let mut rdr = Reader::from_path(&cli.input)
        .with_context(|| format!("opening input CSV: {}", cli.input))?;
    let mut calls: Vec<Call> = Vec::new();
    for result in rdr.deserialize::<InputRow>() {
        let row = result.context("CSV row parse failed")?;
        if row.cell == cli.cell {
            calls.push(Call {
                call_id: row.prompt_id,
                class: row.prompt_class,
                body_bytes: row.request_body_bytes,
                in_tok: row.anthropic_input_tokens,
                out_tok: row.anthropic_output_tokens,
            });
        }
    }
    eprintln!("Loaded {} calls from cell '{}'", calls.len(), cli.cell);
    if calls.is_empty() {
        anyhow::bail!("no rows matched cell='{}' in {}", cli.cell, cli.input);
    }

    // Per-call summary
    let mean_reservation: f64 = calls.iter()
        .map(|c| c.reservation_uc(cli.input_rate, cli.output_rate, cli.max_output_tokens) as f64)
        .sum::<f64>() / calls.len() as f64;
    let mean_actual: f64 = calls.iter()
        .map(|c| c.actual_cost_uc(cli.input_rate, cli.output_rate) as f64)
        .sum::<f64>() / calls.len() as f64;
    eprintln!("Per-call mean reservation: {:.1} uc", mean_reservation);
    eprintln!("Per-call mean actual:      {:.1} uc", mean_actual);
    eprintln!("Per-call over-reservation: {:.1} uc ({:.1}% of actual)",
        mean_reservation - mean_actual,
        100.0 * (mean_reservation - mean_actual) / mean_actual);

    // Replay each cap
    let mut writer = Writer::from_path(&cli.output)
        .with_context(|| format!("opening output CSV: {}", cli.output))?;
    let mut report: Vec<OutputRow> = Vec::new();

    eprintln!("\n{:-^80}", " REPLAY RESULTS ");
    for &cap in &cli.caps {
        let tb  = replay_tb      (&calls, cap, &cli);
        let cb  = replay_callback(&calls, cap, &cli);

        let tb_overshoot = tb.spent_uc.saturating_sub(cap);
        let cb_overshoot = cb.spent_uc.saturating_sub(cap);

        eprintln!("\nCap = {} uc", cap);
        eprintln!("  TB        : admitted={:>2}/{} refused={:>2}/{} spent={} uc OVERSHOOT={} uc",
            tb.n_admitted, calls.len(), tb.n_refused, calls.len(), tb.spent_uc, tb_overshoot);
        eprintln!("  Callback  : admitted={:>2}/{} refused={:>2}/{} spent={} uc OVERSHOOT={} uc",
            cb.n_admitted, calls.len(), cb.n_refused, calls.len(), cb.spent_uc, cb_overshoot);
        let delta = cb_overshoot.saturating_sub(tb_overshoot);
        eprintln!("  Delta (Callback - TB overshoot) = {} uc ({:.1}% of cap)",
            delta, 100.0 * delta as f64 / cap as f64);

        for (mech, r, overshoot) in [
            ("TB",       &tb, tb_overshoot),
            ("Callback", &cb, cb_overshoot),
        ] {
            report.push(OutputRow {
                cap_uc: cap,
                mechanism: mech.to_string(),
                n_calls: calls.len() as u32,
                n_admitted: r.n_admitted,
                n_refused: r.n_refused,
                spent_uc: r.spent_uc,
                overshoot_uc: overshoot,
                overshoot_pct_of_cap: 100.0 * overshoot as f64 / cap as f64,
                mean_reservation_per_call_uc: mean_reservation.round() as u64,
                mean_actual_per_call_uc: mean_actual.round() as u64,
                cap_crossing_call: r.cap_crossing_call,
            });
        }
    }

    for row in &report {
        writer.serialize(row)?;
    }
    writer.flush()?;
    eprintln!("\nWrote {} rows to {}", report.len(), cli.output);

    Ok(())
}
