//! Refund flow + panic-safety integration tests.

use budget_spike::{Budget, BudgetError};

// ---------- Existing API tests (unchanged) ----------

#[test]
fn split_consumes_self() {
    let b = Budget::new(1000);
    let (parent, child) = b.split(300).unwrap();
    assert_eq!(parent.available(), 700);
    assert_eq!(child.available(), 300);
}

#[test]
fn spend_returns_successor_with_reduced_quota() {
    let b = Budget::new(1000);
    let (b2, _) = b.spend(300, || ()).unwrap();
    assert_eq!(b2.available(), 700);
}

#[test]
fn spend_then_spend_chains_correctly() {
    let b = Budget::new(1000);
    let (b2, _) = b.spend(200, || ()).unwrap();
    let (b3, _) = b2.spend(300, || ()).unwrap();
    assert_eq!(b3.available(), 500);
}

// ---------- Refund flow tests ----------

#[test]
fn refund_full_when_actual_charge_is_zero() {
    let b = Budget::new(1000);
    let (b2, r) = b.reserve(500).unwrap();
    assert_eq!(b2.available(), 500);
    assert_eq!(r.reserved(), 500);
    let refund = r.confirm(0).unwrap().expect("refund of 500");
    assert_eq!(refund.amount(), 500);
    let restored = b2.apply_refund(refund).unwrap();
    assert_eq!(restored.available(), 1000);
}

#[test]
fn refund_partial_when_actual_below_reservation() {
    let b = Budget::new(1000);
    let (b2, r) = b.reserve(500).unwrap();
    let refund = r.confirm(200).unwrap().expect("refund of 300");
    assert_eq!(refund.amount(), 300);
    let restored = b2.apply_refund(refund).unwrap();
    assert_eq!(restored.available(), 800); // 1000 - 500 + 300
}

#[test]
fn no_refund_when_actual_equals_reservation() {
    let b = Budget::new(1000);
    let (b2, r) = b.reserve(500).unwrap();
    let refund_opt = r.confirm(500).unwrap();
    assert!(refund_opt.is_none());
    assert_eq!(b2.available(), 500);
}

#[test]
fn estimator_violation_consumes_receipt_without_refund() {
    let b = Budget::new(1000);
    let (b2, r) = b.reserve(500).unwrap();
    match r.confirm(600) {
        Err(BudgetError::EstimatorViolation { reservation, actual }) => {
            assert_eq!(reservation, 500);
            assert_eq!(actual, 600);
        }
        other => panic!("expected EstimatorViolation, got {:?}", other),
    }
    // Successor budget still alive; only the receipt was consumed.
    assert_eq!(b2.available(), 500);
}

#[test]
fn forfeit_preserves_successor_budget() {
    let b = Budget::new(1000);
    let (b2, r) = b.reserve(500).unwrap();
    r.forfeit();
    assert_eq!(b2.available(), 500);
}

#[test]
fn retry_loop_with_partial_refunds_bounds_total_cost() {
    // Multi-step retry loop: each call costs less than its reservation.
    // Total provider-charged cost equals sum of actual charges,
    // bounded by the initial budget regardless of retry count.
    let initial = 10_000u64;
    let mut budget = Budget::new(initial);
    let calls: Vec<(u64, u64)> = vec![
        (1000, 800),  // reserve 1000, charge 800, refund 200
        (1000, 850),
        (1000, 900),
        (1000, 750),
    ];
    let mut total_charged = 0u64;
    for (reservation, actual) in &calls {
        let (b2, r) = budget.reserve(*reservation).unwrap();
        budget = match r.confirm(*actual).unwrap() {
            Some(refund) => b2.apply_refund(refund).unwrap(),
            None => b2,
        };
        total_charged += actual;
    }
    assert_eq!(total_charged, 800 + 850 + 900 + 750);
    assert_eq!(budget.available(), initial - total_charged);
    assert!(budget.available() <= initial);
}

// ---------- Panic safety across spawn boundaries ----------

#[cfg(feature = "tokio-tests")]
mod panic_safety {
    use super::*;

    #[tokio::test]
    async fn panic_in_spawn_does_not_corrupt_parent() {
        let b = Budget::new(10_000);
        let (parent, child) = b.split(3000).unwrap();
        assert_eq!(parent.available(), 7000);

        let handle = tokio::spawn(async move {
            let _moved = child;
            panic!("simulated worker panic");
        });

        let join_result = handle.await;
        assert!(join_result.is_err(), "expected JoinError from panicking task");

        let (parent2, _) = parent.spend(100, || ()).unwrap();
        assert_eq!(parent2.available(), 6900);
    }

    #[tokio::test]
    async fn worker_returns_unspent_budget_to_parent() {
        let b = Budget::new(10_000);
        let (parent, child) = b.split(3000).unwrap();

        let handle = tokio::spawn(async move {
            let (after_one, _) = child.spend(500, || ()).unwrap();
            after_one
        });

        let returned = handle.await.expect("worker did not panic");
        assert_eq!(returned.available(), 2500);

        let merged = parent.merge(returned);
        assert_eq!(merged.available(), 9500);
    }
}
