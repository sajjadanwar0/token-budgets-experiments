//! Reservation receipts and refund tokens.
//!
//! These types extend the affine `Budget` discipline with refund
//! semantics for the case where the actual provider charge is below
//! the reserved amount. The integrity invariant of the discipline
//! is preserved: live_budget + outstanding_receipts +
//! outstanding_refunds is non-increasing under all operations
//! except `Budget::apply_refund`, which collapses a refund and a
//! budget into a single new budget of equal total.
//!
//! Like `Budget`, `ReservationReceipt` and `Refund` are affine: they
//! must be explicitly consumed (via `confirm`/`forfeit` for receipts,
//! `refund_to`/`discard` for refunds) or their `Drop` impl logs a
//! warning. The `consumed` flag tracks legitimate consumption so the
//! Drop check distinguishes accidental drops from explicit ones.

use crate::BudgetError;

/// A reservation receipt produced by `Budget::reserve`.
///
/// Represents an amount reserved-but-not-yet-confirmed against a
/// budget. The caller must `confirm(actual_charge)` it (returning
/// an optional `Refund`) or `forfeit()` it (consuming the
/// reservation without a refund) once the call completes.
#[derive(Debug)]
pub struct ReservationReceipt {
    micro_cents: u64,
    consumed: bool,
}

impl ReservationReceipt {
    pub(crate) fn new(amount: u64) -> Self {
        Self { micro_cents: amount, consumed: false }
    }

    /// The reserved amount in micro-cents.
    pub fn reserved(&self) -> u64 {
        self.micro_cents
    }

    /// Confirm the receipt with the actual provider charge.
    ///
    /// - If `actual <= reservation`, returns `Ok(Some(Refund))` with
    ///   the positive difference, or `Ok(None)` if they match exactly.
    /// - If `actual > reservation`, the conservative-estimator
    ///   condition (Assumption A1 in the paper) was violated; returns
    ///   `Err(BudgetError::EstimatorViolation)` and consumes the
    ///   receipt with no refund.
    pub fn confirm(mut self, actual: u64) -> Result<Option<Refund>, BudgetError> {
        self.consumed = true;
        if actual > self.micro_cents {
            return Err(BudgetError::EstimatorViolation {
                reservation: self.micro_cents,
                actual,
            });
        }
        let refund_amount = self.micro_cents - actual;
        if refund_amount == 0 {
            Ok(None)
        } else {
            Ok(Some(Refund::new(refund_amount)))
        }
    }

    /// Forfeit the receipt without producing a refund.
    ///
    /// Used when the LLM call failed with no provider charge
    /// information available (network error, timeout, etc.). The
    /// reserved amount is permanently consumed.
    pub fn forfeit(mut self) {
        self.consumed = true;
    }
}

impl Drop for ReservationReceipt {
    fn drop(&mut self) {
        if !self.consumed && self.micro_cents > 0 {
            eprintln!(
                "[budget] WARN: ReservationReceipt dropped without confirm/forfeit ({} uc unspent)",
                self.micro_cents
            );
        }
    }
}

/// A refund token produced by `ReservationReceipt::confirm` when the
/// actual provider charge was below the reservation.
///
/// Must be applied to a budget via `Budget::apply_refund` or
/// explicitly discarded via `discard`.
#[derive(Debug)]
pub struct Refund {
    micro_cents: u64,
    consumed: bool,
}

impl Refund {
    pub(crate) fn new(amount: u64) -> Self {
        Self { micro_cents: amount, consumed: false }
    }

    /// The refund amount in micro-cents.
    pub fn amount(&self) -> u64 {
        self.micro_cents
    }

    /// Internal consumer used by `Budget::apply_refund` to extract the
    /// amount and prevent the Drop warning from firing.
    pub(crate) fn into_amount(mut self) -> u64 {
        self.consumed = true;
        self.micro_cents
    }

    /// Discard the refund without applying it to a budget.
    ///
    /// Used in contexts where the parent budget is unreachable, e.g.
    /// across a panic boundary or when the budget has been moved into
    /// a different ownership domain.
    pub fn discard(mut self) {
        self.consumed = true;
    }
}

impl Drop for Refund {
    fn drop(&mut self) {
        if !self.consumed && self.micro_cents > 0 {
            eprintln!(
                "[budget] WARN: Refund dropped without refund_to/discard ({} uc lost)",
                self.micro_cents
            );
        }
    }
}
