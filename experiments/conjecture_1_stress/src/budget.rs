//! Minimal Budget API mirroring the paper's design, instrumented for
//! invariant monitoring under random concurrent schedules.
//!
//! Key differences from the production `token-budgets` crate:
//!   - Every operation reports the post-operation `available` to a
//!     monitor channel, so the test harness can compute Φ at any point.
//!   - `Drop` records the dropped capacity, so the monitor can
//!     distinguish "still live" from "permanently lost".
//!   - No estimator integration; this test isolates the affine
//!     ownership/arithmetic core that Lemma 2 is about.

use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc::UnboundedSender;

/// Globally-unique ID for tracking individual Budget values through their
/// lifetime (creation → spend/split/merge → drop).
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Events reported to the monitor for invariant checking.
#[derive(Debug, Clone, serde::Serialize)]
pub enum Event {
    Created { id: u64, capacity: u64, parent: Option<u64> },
    Spend   { id: u64, amount: u64, remaining: u64 },
    Split   { parent: u64, child_a: u64, child_b: u64, cap_a: u64, cap_b: u64 },
    Merge   { result: u64, capacity: u64, consumed_a: u64, consumed_b: u64 },
    Dropped { id: u64, available_at_drop: u64 },
}

/// The Budget value. Non-Clone, non-Copy: pure affine semantics.
#[derive(Debug)]
pub struct Budget {
    id: u64,
    available: u64,
    monitor: UnboundedSender<Event>,
    // Tracks whether this Budget has been consumed by spend/split/merge
    // (vs dropped naturally). Helps the monitor distinguish.
    consumed: bool,
}

impl Budget {
    /// Trusted constructor; mirrors paper's `Budget::new`.
    /// In the stress test, callers track how many times this is invoked
    /// so the monitor's expected B₀ matches the actual sum.
    pub fn new(capacity: u64, monitor: UnboundedSender<Event>) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let _ = monitor.send(Event::Created { id, capacity, parent: None });
        Self { id, available: capacity, monitor, consumed: false }
    }

    pub fn id(&self) -> u64 { self.id }
    pub fn available(&self) -> u64 { self.available }

    /// Spend (consume self by value, return Result with remainder).
    /// Mirrors paper Lemma 1 / Verus mechanisation.
    pub fn spend(mut self, amount: u64) -> Result<Self, BudgetError> {
        if amount > self.available {
            self.consumed = true;
            return Err(BudgetError::InsufficientFunds { 
                requested: amount, 
                available: self.available 
            });
        }
        let new_avail = self.available - amount;
        let _ = self.monitor.send(Event::Spend {
            id: self.id,
            amount,
            remaining: new_avail,
        });
        let id = self.id;
        let mon = self.monitor.clone();
        self.consumed = true;
        // Construct fresh Budget with same id (continuity) and new amount
        Ok(Self {
            id,
            available: new_avail,
            monitor: mon,
            consumed: false,
        })
    }

    /// Split into two budgets; sum equals original (conservation).
    pub fn split(mut self, amount_a: u64) -> Result<(Self, Self), BudgetError> {
        if amount_a > self.available {
            self.consumed = true;
            return Err(BudgetError::InsufficientFunds {
                requested: amount_a,
                available: self.available,
            });
        }
        let cap_a = amount_a;
        let cap_b = self.available - amount_a;
        let id_a = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let id_b = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let _ = self.monitor.send(Event::Split {
            parent: self.id,
            child_a: id_a,
            child_b: id_b,
            cap_a,
            cap_b,
        });
        let mon_a = self.monitor.clone();
        let mon_b = self.monitor.clone();
        self.consumed = true;
        let a = Self { id: id_a, available: cap_a, monitor: mon_a, consumed: false };
        let b = Self { id: id_b, available: cap_b, monitor: mon_b, consumed: false };
        Ok((a, b))
    }

    /// Merge two budgets; result equals sum (conservation).
    pub fn merge(mut self, mut other: Self) -> Self {
        let consumed_a = self.id;
        let consumed_b = other.id;
        let total = self.available + other.available;
        let id_new = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let _ = self.monitor.send(Event::Merge {
            result: id_new,
            capacity: total,
            consumed_a,
            consumed_b,
        });
        let mon = self.monitor.clone();
        self.consumed = true;
        other.consumed = true;
        Self { id: id_new, available: total, monitor: mon, consumed: false }
    }
}

impl Drop for Budget {
    fn drop(&mut self) {
        if !self.consumed {
            // Natural drop: this capacity is permanently lost. The monitor
            // counts dropped capacity towards Φ-decrease.
            let _ = self.monitor.send(Event::Dropped {
                id: self.id,
                available_at_drop: self.available,
            });
        }
    }
}

#[derive(Debug, Clone)]
pub enum BudgetError {
    InsufficientFunds { requested: u64, available: u64 },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::InsufficientFunds { requested, available } => {
                write!(f, "insufficient funds: requested {}, available {}", 
                       requested, available)
            }
        }
    }
}

impl std::error::Error for BudgetError {}
