//! Multi-turn conversation workload evaluation.
//!
//! For each of N_SESSIONS sessions, threads one Budget through ~12
//! conversational turns. Message history grows each turn; tests whether
//! the discipline holds under realistic agentic load where input tokens
//! grow super-linearly with turn count.
//!
//! Usage: ANTHROPIC_API_KEY=... cargo run --release --bin multi-turn-session
//! Output: multi_turn_sessions.csv + multi_turn_turns.csv

use anyhow::{Result};
use token_budgets::Budget;
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const SESSION_BUDGET_NC: u64 = 150_000_000;  // $0.15 per session, in nano-cents
const ANTHROPIC_IN_RATE_NC: u64 = 1000;
const ANTHROPIC_OUT_RATE_NC: u64 = 5000;
const MAX_TOKENS_PER_TURN: u32 = 512;

type SessionBudget = Budget<SESSION_BUDGET_NC>;

#[derive(Clone)]
struct Scenario {
    name: &'static str,
    turns: Vec<&'static str>,
}

fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario { name: "trip_planning", turns: vec![
            "Plan a 5-day trip to Tokyo. Include daily themes and one must-see per day.",
            "Add detail about food experiences each day.",
            "What's a reasonable daily budget for two adults?",
            "Any safety considerations for solo travel in Tokyo?",
            "How should I get around between districts?",
            "Best time of year to go and why?",
            "Essentials to pack that are easy to forget?",
            "Cultural etiquette tips for restaurants?",
            "Is a JR pass worth it for this itinerary?",
            "Suggest one off-the-beaten-path neighborhood.",
            "Summarize the full itinerary in 3 sentences.",
            "Top piece of advice from everything we covered.",
        ]},
        Scenario { name: "code_debug", turns: vec![
            "I have a Python function for longest palindrome substring but it returns wrong answers sometimes. def longest_palindrome(s): start=0; end=0; for i in range(len(s)): for j in range(i,len(s)): if s[i:j]==s[i:j][::-1] and j-i>end-start: start,end=i,j; return s[start:end]. What's wrong?",
            "Why does it not find palindromes correctly in 'babad'?",
            "Show me a corrected version.",
            "Walk through the corrected version with 'babad'.",
            "What's the time complexity?",
            "Is there a more efficient approach?",
            "Explain Manacher's algorithm briefly.",
            "Show pseudocode for Manacher's.",
            "What about edge cases like empty string?",
            "How would I test this in pytest?",
            "Write 3 representative test cases.",
            "Final summary of what we covered.",
        ]},
        Scenario { name: "rust_learning", turns: vec![
            "Teach me Rust ownership in 3 sentences.",
            "Show me a simple example of move semantics.",
            "Now show what happens when I try to use a moved value.",
            "Explain borrowing vs ownership.",
            "Show an example of mutable borrow.",
            "What's the borrow checker rule about aliasing?",
            "Why doesn't Rust allow two mutable borrows at once?",
            "Show me how lifetimes annotate borrows.",
            "What's the difference between &str and String?",
            "When should I use Arc vs Rc?",
            "Recap the 5 most important rules.",
            "What's the most common newbie mistake?",
        ]},
        Scenario { name: "podcast_brainstorm", turns: vec![
            "Brainstorm 10 podcast names for a tech-and-philosophy show.",
            "Make them more clever, less obvious.",
            "Now narrow to your top 3 with reasoning.",
            "Write a one-line tagline for each.",
            "Pick one and draft a 30-second intro script.",
            "Make it punchier.",
            "What's the first episode title?",
            "Suggest 5 guests for episode 1.",
            "What's the cover-art concept?",
            "Sponsor pitch in 2 sentences.",
            "Launch plan for week 1.",
            "Recap the final brand identity.",
        ]},
        Scenario { name: "essay_revision", turns: vec![
            "Write a 150-word intro paragraph about climate adaptation.",
            "Make it more formal in tone.",
            "Now shorten it to 80 words.",
            "Add one concrete statistic.",
            "Reframe to emphasize urgency.",
            "Now soften slightly so it doesn't feel alarmist.",
            "Improve the opening sentence.",
            "Rewrite as a question-first opening.",
            "Cut any redundancy.",
            "Add a single rhetorical hook.",
            "Final version, polished.",
            "One-sentence summary of the paragraph's argument.",
        ]},
        Scenario { name: "math_problem", turns: vec![
            "Solve: if x + y = 10 and x*y = 21, what are x and y?",
            "Walk me through the steps.",
            "What general method did you use?",
            "Apply the same method to: x+y=12, x*y=20.",
            "Show the quadratic equation form for the general case.",
            "What's the discriminant condition for real solutions?",
            "Give an example where the discriminant is negative.",
            "What does that mean geometrically?",
            "Connect this to Vieta's formulas.",
            "State Vieta's formulas for a cubic.",
            "Give one application of Vieta in real life.",
            "Summarize what we learned.",
        ]},
        Scenario { name: "tech_compare", turns: vec![
            "Compare Python and JavaScript for backend development.",
            "Focus specifically on async/concurrency models.",
            "Which is faster for I/O-heavy workloads?",
            "What about for CPU-bound tasks?",
            "Memory consumption comparison?",
            "Ecosystem maturity for database drivers?",
            "Which has better tooling for testing?",
            "Deployment complexity comparison?",
            "Cold-start latency in serverless?",
            "Best fit for a real-time chat app, and why?",
            "Best fit for a data-processing pipeline?",
            "Final recommendation framework: when to pick which.",
        ]},
        Scenario { name: "recipe_iter", turns: vec![
            "Suggest a weeknight dinner using chicken, rice, and lemons.",
            "How long does it take?",
            "Make it spicier.",
            "What if I don't have rice — what substitutes?",
            "Reduce salt by half — how?",
            "Make it vegetarian instead.",
            "Now make it vegan.",
            "What wine pairs with the original chicken version?",
            "Side-dish suggestion?",
            "Dessert under 200 calories?",
            "Full final menu summary.",
            "Cost estimate per serving in USD?",
        ]},
    ]
}

struct TurnRecord {
    session_idx: usize,
    turn_idx: usize,
    scenario: String,
    reservation_nc: u64,
    actual_nc: u64,
    refund_nc: u64,
    input_tokens: u64,
    output_tokens: u64,
    body_bytes: u64,
    margin_ratio: f64,
    violated: bool,
    latency_ms: u128,
}

struct SessionSummary {
    idx: usize,
    scenario: String,
    turns_completed: usize,
    turns_attempted: usize,
    total_reserved_nc: u64,
    total_actual_nc: u64,
    total_refund_nc: u64,
    final_budget_nc: u64,
    violations: usize,
    exhausted_early: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let n_sessions: usize = env::var("N_SESSIONS").unwrap_or_else(|_| "100".to_string()).parse()?;
    let api_key = env::var("ANTHROPIC_API_KEY")?;
    let scenarios = scenarios();
    let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build()?;

    let mut all_turns: Vec<TurnRecord> = Vec::new();
    let mut sessions: Vec<SessionSummary> = Vec::new();

    let run_start = Instant::now();

    for session_idx in 0..n_sessions {
        let scenario = &scenarios[session_idx % scenarios.len()];
        let mut budget: Option<SessionBudget> = Some(Budget::new(SESSION_BUDGET_NC)?);
        let mut messages: Vec<Value> = Vec::new();
        let mut total_r = 0u64;
        let mut total_a = 0u64;
        let mut total_f = 0u64;
        let mut violations = 0usize;
        let mut turns_completed = 0usize;
        let mut exhausted_early = false;
        let turns_attempted = scenario.turns.len();

        for (turn_idx, user_msg) in scenario.turns.iter().enumerate() {
            messages.push(json!({"role": "user", "content": user_msg}));

            let body = serde_json::to_string(&json!({
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": MAX_TOKENS_PER_TURN,
                "messages": &messages,
            }))?;
            let body_bytes = body.len() as u64;
            let reservation = body_bytes * ANTHROPIC_IN_RATE_NC
                + MAX_TOKENS_PER_TURN as u64 * ANTHROPIC_OUT_RATE_NC;

            let current = budget.take().expect("budget present");
            let (after_reserve, receipt) = match current.spend_with_receipt(reservation) {
                Ok(x) => x,
                Err(_) => {
                    println!("Session {} ({}) exhausted at turn {}", session_idx, scenario.name, turn_idx);
                    exhausted_early = true;
                    budget = Some(Budget::new(0).unwrap_or_else(|_| panic!()));
                    break;
                }
            };

            let start = Instant::now();
            let resp = client.post("https://api.anthropic.com/v1/messages")
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .body(body)
                .send().await;
            let latency_ms = start.elapsed().as_millis();

            let resp = match resp {
                Ok(r) if r.status().is_success() => r,
                Ok(r) => {
                    eprintln!("S{} T{} HTTP {}", session_idx, turn_idx, r.status());
                    receipt.forfeit();
                    budget = Some(after_reserve);
                    continue;
                }
                Err(e) => {
                    eprintln!("S{} T{} err: {}", session_idx, turn_idx, e);
                    receipt.forfeit();
                    budget = Some(after_reserve);
                    continue;
                }
            };

            let parsed: Value = resp.json().await?;
            let in_tok = parsed["usage"]["input_tokens"].as_u64().unwrap_or(0);
            let out_tok = parsed["usage"]["output_tokens"].as_u64().unwrap_or(0);
            let actual = in_tok * ANTHROPIC_IN_RATE_NC + out_tok * ANTHROPIC_OUT_RATE_NC;
            let violated = actual > reservation;

            if violated {
                violations += 1;
                println!("S{} T{} ⚠ VIOLATION reserve={} actual={}", session_idx, turn_idx, reservation, actual);
                receipt.forfeit();
                budget = Some(after_reserve);
                all_turns.push(TurnRecord {
                    session_idx, turn_idx, scenario: scenario.name.to_string(),
                    reservation_nc: reservation, actual_nc: actual, refund_nc: 0,
                    input_tokens: in_tok, output_tokens: out_tok, body_bytes,
                    margin_ratio: reservation as f64 / actual.max(1) as f64,
                    violated: true, latency_ms,
                });
                continue;
            }

            let refund = receipt.confirm(actual)?;
            let refund_amount = refund.amount();
            budget = Some(refund.apply_to(after_reserve)?);

            // Append assistant response to history
            let assistant_text = parsed["content"][0]["text"].as_str().unwrap_or("").to_string();
            messages.push(json!({"role": "assistant", "content": assistant_text}));

            total_r += reservation;
            total_a += actual;
            total_f += refund_amount;
            turns_completed = turn_idx + 1;

            all_turns.push(TurnRecord {
                session_idx, turn_idx, scenario: scenario.name.to_string(),
                reservation_nc: reservation, actual_nc: actual, refund_nc: refund_amount,
                input_tokens: in_tok, output_tokens: out_tok, body_bytes,
                margin_ratio: reservation as f64 / actual.max(1) as f64,
                violated: false, latency_ms,
            });
        }

        let final_budget_nc = budget.as_ref().map(|b| b.micro_cents()).unwrap_or(0);

        sessions.push(SessionSummary {
            idx: session_idx, scenario: scenario.name.to_string(),
            turns_completed, turns_attempted,
            total_reserved_nc: total_r, total_actual_nc: total_a, total_refund_nc: total_f,
            final_budget_nc, violations, exhausted_early,
        });

        if session_idx % 10 == 0 {
            println!("Session {}: scenario={} turns={}/{} actual=${:.4} viol={}{}",
                session_idx, scenario.name, turns_completed, turns_attempted,
                total_a as f64 / 1e9, violations,
                if exhausted_early { " EXHAUSTED" } else { "" });
        }
    }

    let elapsed = run_start.elapsed();
    let total_violations: usize = sessions.iter().map(|s| s.violations).sum();
    let total_turns: usize = sessions.iter().map(|s| s.turns_completed).sum();
    let total_actual_usd: f64 = sessions.iter().map(|s| s.total_actual_nc).sum::<u64>() as f64 / 1e9;
    let exhausted_count = sessions.iter().filter(|s| s.exhausted_early).count();

    println!();
    println!("=== Multi-turn session aggregate ===");
    println!("Wall time:           {:.1} min", elapsed.as_secs_f64() / 60.0);
    println!("Sessions:            {} (target {})", sessions.len(), n_sessions);
    println!("Turns completed:     {} (across all sessions)", total_turns);
    println!("Total A1 violations: {} ({:.3}%)", total_violations,
             100.0 * total_violations as f64 / total_turns.max(1) as f64);
    println!("Sessions exhausting budget early: {} ({:.1}%)",
             exhausted_count, 100.0 * exhausted_count as f64 / sessions.len() as f64);
    println!("Total spent:         ${:.4}", total_actual_usd);
    println!();

    // Per-scenario summary
    println!("Per-scenario stats:");
    let scenario_names: Vec<&str> = scenarios.iter().map(|s| s.name).collect();
    for sn in scenario_names {
        let subs: Vec<&SessionSummary> = sessions.iter().filter(|s| s.scenario == sn).collect();
        if subs.is_empty() { continue; }
        let avg_turns = subs.iter().map(|s| s.turns_completed).sum::<usize>() as f64 / subs.len() as f64;
        let avg_cost = subs.iter().map(|s| s.total_actual_nc).sum::<u64>() as f64 / subs.len() as f64 / 1e9;
        let exh = subs.iter().filter(|s| s.exhausted_early).count();
        println!("  {:20}  N={:>3}  avg_turns={:.1}  avg_cost=${:.4}  exhausted={}",
                 sn, subs.len(), avg_turns, avg_cost, exh);
    }

    // CSV: per-session summary
    let mut s_csv = File::create("multi_turn_sessions.csv")?;
    writeln!(s_csv, "idx,scenario,turns_completed,turns_attempted,total_reserved_nc,total_actual_nc,total_refund_nc,final_budget_nc,violations,exhausted_early")?;
    for s in &sessions {
        writeln!(s_csv, "{},{},{},{},{},{},{},{},{},{}",
                 s.idx, s.scenario, s.turns_completed, s.turns_attempted,
                 s.total_reserved_nc, s.total_actual_nc, s.total_refund_nc,
                 s.final_budget_nc, s.violations, s.exhausted_early)?;
    }

    // CSV: per-turn records
    let mut t_csv = File::create("multi_turn_turns.csv")?;
    writeln!(t_csv, "session_idx,turn_idx,scenario,reservation_nc,actual_nc,refund_nc,input_tokens,output_tokens,body_bytes,margin_ratio,violated,latency_ms")?;
    for t in &all_turns {
        writeln!(t_csv, "{},{},{},{},{},{},{},{},{},{:.6},{},{}",
                 t.session_idx, t.turn_idx, t.scenario, t.reservation_nc, t.actual_nc, t.refund_nc,
                 t.input_tokens, t.output_tokens, t.body_bytes, t.margin_ratio, t.violated, t.latency_ms)?;
    }

    println!("Wrote multi_turn_sessions.csv ({} sessions)", sessions.len());
    println!("Wrote multi_turn_turns.csv ({} turns)", all_turns.len());

    Ok(())
}
