//! Synchronous runtime demo for the affine Budget API.
//! Run: cargo run --bin demo_runtime

use budget_spike::{Budget, BudgetError, estimate_cost};

fn main() {
    println!("=== Budget Spike: Sync Runtime Demo ===\n");

    // ── [1] Linear spend across 5 sequential tool calls ──
    println!("[1] Linear spend across 5 sequential tool calls");
    let mut b = Budget::new(500_000);
    println!("  initial available: {} uc", b.available());
    for step in 0..5 {
        let cost = estimate_cost(800, 200, 3_000_000);
        let placeholder = Budget::new(0);
        let taken = std::mem::replace(&mut b, placeholder);
        match taken.spend(cost, || format!("tool_call_{}", step)) {
            Ok((remaining, label)) => {
                println!(
                    "  step {step}: spent {cost} uc ({label}), now {} uc",
                    remaining.available()
                );
                b = remaining;
            }
            Err(e) => {
                println!("  step {step}: spend failed: {e}");
                break;
            }
        }
    }
    println!("  final available: {} uc\n", b.available());

    // ── [2] Parent split into two children, each spends independently ──
    println!("[2] Parent split into two children, each spends independently");
    let parent = Budget::new(100_000);
    let (parent, sub_a) = parent.split(40_000).unwrap();
    let (parent, sub_b) = parent.split(30_000).unwrap();
    println!(
        "  parent kept: {} uc, sub_a: {} uc, sub_b: {} uc",
        parent.available(),
        sub_a.available(),
        sub_b.available()
    );

    let (sub_a, _) = sub_a.spend(15_000, || "sub_a step 1").unwrap();
    let (sub_a, _) = sub_a.spend(20_000, || "sub_a step 2").unwrap();
    println!(
        "  sub_a remaining after 2 spends: {} uc (expected 5_000)",
        sub_a.available()
    );

    let (sub_b, _) = sub_b.spend(10_000, || "sub_b step 1").unwrap();
    let (sub_b, _) = sub_b.spend(20_000, || "sub_b step 2").unwrap();
    println!("  sub_b remaining: {} uc (expected 0)", sub_b.available());

    let result = sub_b.spend(15_000, || "sub_b step 3");
    match result {
        Ok((sub_b, _)) => {
            println!(
                "  sub_b step 3 unexpectedly succeeded: {} uc remaining",
                sub_b.available()
            );
        }
        Err(BudgetError::Insufficient { requested, available }) => {
            println!(
                "  sub_b step 3 BLOCKED cleanly: requested {requested} uc, only {available} uc available"
            );
        }
        Err(other) => {
            println!("  sub_b step 3 unexpected error: {other}");
        }
    }
    println!();

    // ── [3] Insufficient budget aborts cleanly ──
    println!("[3] Insufficient budget aborts cleanly");
    let small = Budget::new(1_000);
    let err = small.spend(5_000, || ()).unwrap_err();
    println!("  small.spend(5_000) -> {err}");

    println!("\nAll runtime checks complete.");
}
