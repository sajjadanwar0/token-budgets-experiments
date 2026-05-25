use budget_spike::{Budget, BudgetError};
use proptest::prelude::*;

#[test]
fn spend_at_zero_returns_error() {
    let b = Budget::new(0);
    assert!(b.spend(1, || ()).is_err());
}

#[test]
fn spend_exact_amount_drains_budget() {
    let b = Budget::new(100);
    let (b2, _) = b.spend(100, || ()).unwrap();
    assert_eq!(b2.available(), 0);
}

#[test]
fn split_consumes_self_and_distributes() {
    let b = Budget::new(1000);
    let (parent, child) = b.split(300).unwrap();
    assert_eq!(parent.available(), 700);
    assert_eq!(child.available(), 300);
}

#[test]
fn split_merge_round_trips() {
    let b = Budget::new(1000);
    let (parent, child) = b.split(300).unwrap();
    let merged = parent.merge(child);
    assert_eq!(merged.available(), 1000);
}

#[test]
fn merge_checked_at_overflow_boundary_returns_error() {
    let b1 = Budget::new(u64::MAX / 2 + 1);
    let b2 = Budget::new(u64::MAX / 2 + 1);
    match b1.merge_checked(b2) {
        Err(BudgetError::Overflow) => {}
        other => panic!("expected Overflow, got {:?}", other),
    }
}

#[test]
fn merge_checked_just_below_overflow_succeeds() {
    let b1 = Budget::new(u64::MAX / 2);
    let b2 = Budget::new(u64::MAX / 2);
    let merged = b1.merge_checked(b2).unwrap();
    assert_eq!(merged.available(), u64::MAX - 1);
}

#[test]
fn merge_checked_at_u64_max_with_zero_succeeds() {
    let b1 = Budget::new(u64::MAX);
    let b2 = Budget::new(0);
    let merged = b1.merge_checked(b2).unwrap();
    assert_eq!(merged.available(), u64::MAX);
}

#[test]
fn merge_checked_within_a2_succeeds() {
    let b1 = Budget::new(1_000_000);
    let b2 = Budget::new(2_000_000);
    let merged = b1.merge_checked(b2).unwrap();
    assert_eq!(merged.available(), 3_000_000);
}

#[test]
fn apply_refund_overflow_returns_error() {
    let near_max = Budget::new(u64::MAX - 100);
    let b_small = Budget::new(200);
    let (_b_small_after, r) = b_small.reserve(200).unwrap();
    let refund = r.confirm(0).unwrap().expect("refund of 200");
    match near_max.apply_refund(refund) {
        Err(BudgetError::Overflow) => {}
        other => panic!("expected Overflow, got {:?}", other),
    }
}

proptest! {
    #[test]
    fn split_preserves_total(
        initial in 1000u64..1_000_000,
        split_at in 1u64..500,
    ) {
        let split_at = split_at.min(initial.saturating_sub(1));
        let b = Budget::new(initial);
        let (parent, child) = b.split(split_at).unwrap();
        prop_assert_eq!(parent.available() + child.available(), initial);
    }

    #[test]
    fn spend_decreases_by_amount(
        initial in 1000u64..1_000_000,
        amount in 1u64..500,
    ) {
        let amount = amount.min(initial);
        let b = Budget::new(initial);
        let (b2, _) = b.spend(amount, || ()).unwrap();
        prop_assert_eq!(b2.available(), initial - amount);
    }

    #[test]
    fn split_then_merge_preserves_total(
        initial in 1000u64..(u64::MAX / 4),
        split_at in 1u64..500,
    ) {
        let split_at = split_at.min(initial.saturating_sub(1));
        let b = Budget::new(initial);
        let (parent, child) = b.split(split_at).unwrap();
        let merged = parent.merge(child);
        prop_assert_eq!(merged.available(), initial);
    }

    #[test]
    fn merge_checked_succeeds_within_a2(
        a in 0u64..=(u64::MAX / 2),
        b in 0u64..=(u64::MAX / 2),
    ) {
        let b1 = Budget::new(a);
        let b2 = Budget::new(b);
        let merged = b1.merge_checked(b2).unwrap();
        prop_assert_eq!(merged.available(), a + b);
    }

    #[test]
    fn reserve_confirm_refund_round_trip_preserves_total(
        initial in 1000u64..1_000_000,
        reservation in 1u64..1000,
        actual in 0u64..1000,
    ) {
        let reservation = reservation.min(initial);
        let actual = actual.min(reservation);
        let b = Budget::new(initial);
        let (b2, r) = b.reserve(reservation).unwrap();
        let after = match r.confirm(actual).unwrap() {
            Some(refund) => b2.apply_refund(refund).unwrap(),
            None => b2,
        };
        prop_assert_eq!(after.available(), initial - actual);
    }
}
