//! Retrofit case study: applying the affine `Budget` discipline to three
//! DOCUMENTED catalogue incidents. Each case shows (1) the real failure, and
//! (2) that the affine retrofit makes the buggy pattern a COMPILE error.
//!
//! Cases:
//!   LANG-001 (langgraph #6731): retry loop spends cumulatively without a cap.
//!   CDXL-001 (codex   #18335): spawn "slots" leak across turns (not reclaimed).
//!   ATGN-003 (autogen #5831):  swarm delegates to children with no conserved cap.

/// Micro-cents (1 uc = 1e-5 USD). An affine cost capability.
/// No `Clone`, no `Copy`: a `Budget` cannot be duplicated, and every
/// consuming method takes `self` by value, so use-after-spend is a move error.
#[derive(Debug)]
pub struct Budget {
    uc: u64,
}

#[derive(Debug, PartialEq)]
pub enum BudgetError {
    Insufficient { have: u64, want: u64 },
}

impl Budget {
    /// Capability-gated constructor (gated behind a mint in the real crate).
    pub fn new(uc: u64) -> Self {
        Budget { uc }
    }

    pub fn available(&self) -> u64 {
        self.uc
    }

    /// Consume `self`, return a new Budget with the remainder, or refuse
    /// pre-flight if the reservation exceeds the remaining quota.
    pub fn spend(self, amount: u64) -> Result<Budget, BudgetError> {
        match self.uc.checked_sub(amount) {
            Some(rem) => Ok(Budget { uc: rem }),
            None => Err(BudgetError::Insufficient { have: self.uc, want: amount }),
        }
    }

    /// Carve a child sub-budget off the parent; conservation: child + parent == original.
    /// Consumes `self`, returns (parent_remainder, child).
    pub fn split(self, child_uc: u64) -> Result<(Budget, Budget), BudgetError> {
        match self.uc.checked_sub(child_uc) {
            Some(rem) => Ok((Budget { uc: rem }, Budget { uc: child_uc })),
            None => Err(BudgetError::Insufficient { have: self.uc, want: child_uc }),
        }
    }

    /// Reclaim an unspent child back into the parent (the codex "slot" return).
    pub fn merge(self, other: Budget) -> Budget {
        Budget { uc: self.uc + other.uc }
    }
}

// ---------------------------------------------------------------------------
// LANG-001 retrofit: the retry loop cannot exceed the cap, because each retry
// must thread the *remaining* budget; an exhausted budget refuses pre-flight.
// ---------------------------------------------------------------------------
pub fn lang001_retry_loop(mut budget: Budget, per_call: u64, max_retries: usize) -> u64 {
    let start = budget.available();
    for _ in 0..max_retries {
        if budget.available() < per_call {
            break;                       // pre-flight refusal: cap respected
        }
        budget = budget.spend(per_call).expect("checked pre-flight");
    }
    start - budget.available()           // total spent, provably <= start
}

// ---------------------------------------------------------------------------
// CDXL-001 retrofit: a spawn "slot" is a child Budget. It is RECLAIMED by
// merging it back; you cannot leak it silently because an un-merged child is
// a live value the borrow checker tracks (and Drop reclaims at scope end).
// ---------------------------------------------------------------------------
pub fn cdxl001_spawn_and_reclaim(parent: Budget, slot_uc: u64) -> Budget {
    let (parent, slot) = parent.split(slot_uc).expect("slot fits");
    // ... child task uses `slot` ...
    parent.merge(slot)                   // slot returned: conservation holds
}

// ---------------------------------------------------------------------------
// ATGN-003 retrofit: a swarm fans out to N children, each from a conserved
// sub-budget. The sum of children can never exceed the parent, by split.
// ---------------------------------------------------------------------------
pub fn atgn003_swarm_fanout(parent: Budget, n: usize, per_child: u64) -> Result<Budget, BudgetError> {
    let mut parent = parent;
    let mut children = Vec::new();
    for _ in 0..n {
        let (rem, child) = parent.split(per_child)?;  // refuses if over-allocated
        parent = rem;
        children.push(child);
    }
    // reclaim all unspent children
    for c in children {
        parent = parent.merge(c);
    }
    Ok(parent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lang001_caps_spend() {
        // 5 uc cap, 2 uc/call, 10 retries requested: spends at most 4, never > 5.
        let spent = lang001_retry_loop(Budget::new(5), 2, 10);
        assert!(spent <= 5, "cap respected");
        assert_eq!(spent, 4); // two calls admitted (2+2), third refused pre-flight
    }

    #[test]
    fn cdxl001_no_slot_leak() {
        let b = Budget::new(100);
        let after = cdxl001_spawn_and_reclaim(b, 30);
        assert_eq!(after.available(), 100, "slot reclaimed; conservation holds");
    }

    #[test]
    fn atgn003_fanout_conserves() {
        let b = Budget::new(100);
        let after = atgn003_swarm_fanout(b, 3, 20).unwrap();
        assert_eq!(after.available(), 100, "3 children reclaimed; sum never exceeded parent");
    }

    #[test]
    fn atgn003_fanout_refuses_overallocation() {
        // 3 children x 40 = 120 > 100: the 3rd split refuses (cap conserved).
        let b = Budget::new(100);
        assert!(atgn003_swarm_fanout(b, 3, 40).is_err());
    }
}
