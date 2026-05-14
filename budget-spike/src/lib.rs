//! Token Capabilities: affine resource types for LLM agent budgets.
//!
//! Unit convention: 1 micro-cent (uc) = 10^-6 USD. So $1.00 = 1,000,000 uc.

use std::fmt;

pub mod llm_client;
pub mod tokenizer;
pub mod receipt;

pub use receipt::{ReservationReceipt, Refund};

/// A finite quota of spendable resource, measured in micro-cents.
///
/// Affine: spend() and split() consume self by value and return a new
/// Budget carrying the remainder. No Clone, no Copy.
#[derive(Debug, PartialEq, Eq)]
pub struct Budget {
    micro_cents: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BudgetError {
    /// Reservation exceeds remaining quota; the call should not be issued.
    Insufficient { requested: u64, available: u64 },
    /// Arithmetic overflow in `merge_checked` or `apply_refund`.
    /// Should not occur under Assumption A2 (caps below u64::MAX/2);
    /// surfaced defensively.
    Overflow,
    /// Provider's reported charge exceeded the reservation, violating
    /// Assumption A1 (conservative estimator). The receipt is consumed
    /// and no refund is produced.
    EstimatorViolation { reservation: u64, actual: u64 },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BudgetError::Insufficient { requested, available } => write!(
                f,
                "insufficient budget: requested {} uc, available {} uc",
                requested, available
            ),
            BudgetError::Overflow => write!(
                f,
                "budget arithmetic overflow (cap exceeds u64::MAX/2; revise Assumption A2)"
            ),
            BudgetError::EstimatorViolation { reservation, actual } => write!(
                f,
                "estimator violation: reservation {} uc but provider charged {} uc",
                reservation, actual
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

impl Budget {
    /// Construct a new Budget with the given quota in micro-cents.
    pub fn new(micro_cents: u64) -> Self {
        Self { micro_cents }
    }

    /// Returns the remaining quota in micro-cents.
    pub fn available(&self) -> u64 {
        self.micro_cents
    }

    /// Spend `amount` micro-cents, returning a new Budget carrying the remainder
    /// plus the result of the audit-label closure.
    ///
    /// Affine: consumes self.
    pub fn spend<L, T>(self, amount: u64, label: L) -> Result<(Budget, T), BudgetError>
    where
        L: FnOnce() -> T,
    {
        if amount > self.micro_cents {
            return Err(BudgetError::Insufficient {
                requested: amount,
                available: self.micro_cents,
            });
        }
        let remaining = Budget { micro_cents: self.micro_cents - amount };
        let log = label();
        Ok((remaining, log))
    }

    /// Reserve `amount` micro-cents, returning a successor Budget plus a
    /// `ReservationReceipt` that must subsequently be `confirm`ed (with
    /// the actual provider charge) or `forfeit`ed.
    ///
    /// This is the refund-aware variant of `spend`. The integrity
    /// invariant is preserved: the reserved amount is held in the
    /// receipt, not destroyed.
    ///
    /// Affine: consumes self.
    pub fn reserve(self, amount: u64) -> Result<(Budget, ReservationReceipt), BudgetError> {
        if amount > self.micro_cents {
            return Err(BudgetError::Insufficient {
                requested: amount,
                available: self.micro_cents,
            });
        }
        let remaining = Budget { micro_cents: self.micro_cents - amount };
        let receipt = ReservationReceipt::new(amount);
        Ok((remaining, receipt))
    }

    /// Apply a refund to this Budget, producing a new Budget with the
    /// refund amount added back. Returns `BudgetError::Overflow` if
    /// the addition would exceed `u64::MAX`.
    ///
    /// Affine: consumes both self and the refund.
    pub fn apply_refund(self, refund: Refund) -> Result<Budget, BudgetError> {
        let amount = refund.into_amount();
        let total = self.micro_cents
            .checked_add(amount)
            .ok_or(BudgetError::Overflow)?;
        Ok(Budget { micro_cents: total })
    }

    /// Split this Budget into two: returns (remainder, child).
    /// Child carries `amount` micro-cents; remainder keeps the rest.
    /// Affine: consumes self.
    pub fn split(self, amount: u64) -> Result<(Budget, Budget), BudgetError> {
        if amount > self.micro_cents {
            return Err(BudgetError::Insufficient {
                requested: amount,
                available: self.micro_cents,
            });
        }
        let remainder = Budget { micro_cents: self.micro_cents - amount };
        let child = Budget { micro_cents: amount };
        Ok((remainder, child))
    }

    /// Merge another Budget into this one, consuming both and returning a new one.
    /// Uses `saturating_add`: capped at `u64::MAX`, never wraps.
    /// For checked overflow detection, use `merge_checked`.
    /// Affine: consumes both self and other.
    pub fn merge(self, other: Budget) -> Budget {
        Budget {
            micro_cents: self.micro_cents.saturating_add(other.micro_cents),
        }
    }

    /// Merge another Budget into this one with overflow detection.
    /// Returns `BudgetError::Overflow` if the sum exceeds `u64::MAX`.
    /// Use this variant when Assumption A2 cannot be enforced
    /// statically and overflow detection at the call site is required.
    /// Affine: consumes both self and other.
    pub fn merge_checked(self, other: Budget) -> Result<Budget, BudgetError> {
        let total = self.micro_cents
            .checked_add(other.micro_cents)
            .ok_or(BudgetError::Overflow)?;
        Ok(Budget { micro_cents: total })
    }

    /// Consume the Budget, returning the unspent micro-cents.
    pub fn consume(self) -> u64 {
        self.micro_cents
    }
}

/// Estimate cost of an LLM call given input/output tokens and combined price-per-Mtoken.
///
/// `price_per_mtok` is in micro-cents per million tokens.
pub fn estimate_cost(input_tokens: u64, max_output_tokens: u64, price_per_mtok: u64) -> u64 {
    let total = input_tokens.saturating_add(max_output_tokens);
    total.saturating_mul(price_per_mtok) / 1_000_000
}

/// Estimate cost with separate input/output prices (per-token, in micro-cents).
pub fn estimate_cost_split(
    input_tokens: u64,
    max_output_tokens: u64,
    input_price_per_token_mc: u64,
    output_price_per_token_mc: u64,
) -> u64 {
    input_tokens
        .saturating_mul(input_price_per_token_mc)
        .saturating_add(max_output_tokens.saturating_mul(output_price_per_token_mc))
}

// === ASYNC ADDITIONS =========================================================

/// Combined error type for budget-bounded LLM calls.
#[derive(Debug)]
pub enum CallError {
    Budget(BudgetError),
    Llm(llm_client::LLMError),
}

impl fmt::Display for CallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallError::Budget(e) => write!(f, "budget error: {}", e),
            CallError::Llm(e) => write!(f, "llm error: {}", e),
        }
    }
}

impl std::error::Error for CallError {}

impl From<BudgetError> for CallError {
    fn from(e: BudgetError) -> Self { CallError::Budget(e) }
}

impl From<llm_client::LLMError> for CallError {
    fn from(e: llm_client::LLMError) -> Self { CallError::Llm(e) }
}

/// Make an LLM call bounded by a Budget.
///
/// Conservative semantics: worst-case cost is reserved upfront. If reservation
/// fails, no API call is made. If reservation succeeds but the API call fails,
/// the budget is NOT refunded.
pub async fn call_with_budget<C: llm_client::LLMClient + ?Sized>(
    client: &C,
    budget: Budget,
    prompt: &str,
    max_output_tokens: u64,
) -> Result<(Budget, llm_client::CompletionResponse), CallError> {
    let input_tokens = client.estimate_input_tokens(prompt);
    let estimated = estimate_cost_split(
        input_tokens,
        max_output_tokens,
        client.input_price_per_token_mc(),
        client.output_price_per_token_mc(),
    );
    let (remaining, _) = budget.spend(estimated, || ())?;
    let resp = client.complete(prompt, max_output_tokens).await?;
    Ok((remaining, resp))
}

/// Refund-aware variant of `call_with_budget`.
///
/// Reserves worst-case cost upfront. If the call succeeds and the
/// provider reports an actual cost below the reservation, the refund
/// is applied back to the successor budget. If the call fails, the
/// reservation is forfeited (no refund possible without a
/// provider-reported charge).
pub async fn call_with_budget_refund<C: llm_client::LLMClient + ?Sized>(
    client: &C,
    budget: Budget,
    prompt: &str,
    max_output_tokens: u64,
) -> Result<(Budget, llm_client::CompletionResponse), CallError> {
    let input_tokens = client.estimate_input_tokens(prompt);
    let estimated = estimate_cost_split(
        input_tokens,
        max_output_tokens,
        client.input_price_per_token_mc(),
        client.output_price_per_token_mc(),
    );
    let (mut remaining, receipt) = budget.reserve(estimated)?;
    let resp = match client.complete(prompt, max_output_tokens).await {
        Ok(r) => r,
        Err(e) => {
            receipt.forfeit();
            return Err(e.into());
        }
    };
    // Compute the actual charge from the response. For now we use
    // the response's reported usage, falling back to the reservation
    // if the response doesn't report it. Providers that don't report
    // per-call cost cannot benefit from refunds.
    let actual = resp.actual_cost_micro_cents;
    match receipt.confirm(actual) {
        Ok(None) => {} // exact match
        Ok(Some(refund)) => {
            remaining = remaining.apply_refund(refund)?;
        }
        Err(e) => {
            return Err(CallError::Budget(e));
        }
    }
    Ok((remaining, resp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_spend_5_calls_clean_accounting() {
        let mut b = Budget::new(500_000);
        for _ in 0..5 {
            let (next, _) = b.spend(3_000, || ()).expect("spend ok");
            b = next;
        }
        assert_eq!(b.available(), 485_000);
    }

    #[test]
    fn split_spend_merge_returns_unspent() {
        let parent = Budget::new(100_000);
        let (parent, child) = parent.split(40_000).expect("split ok");
        assert_eq!(parent.available(), 60_000);
        assert_eq!(child.available(), 40_000);

        let (child, _) = child.spend(30_000, || ()).expect("child spend");
        let (parent, _) = parent.spend(50_000, || ()).expect("parent spend");

        let merged = parent.merge(child);
        assert_eq!(merged.available(), 20_000);
    }

    #[test]
    fn insufficient_aborts_atomically() {
        let b = Budget::new(100);
        let result = b.spend(101, || ());
        assert!(matches!(
            result,
            Err(BudgetError::Insufficient { requested: 101, available: 100 })
        ));
    }

    #[test]
    fn estimate_cost_basic() {
        let cost = estimate_cost(800, 200, 3_000_000);
        assert_eq!(cost, 3_000);
    }

    #[test]
    fn reserve_then_confirm_full_refund() {
        let b = Budget::new(1000);
        let (b2, r) = b.reserve(500).expect("reserve");
        let refund = r.confirm(0).expect("confirm").expect("refund");
        let restored = b2.apply_refund(refund).expect("apply");
        assert_eq!(restored.available(), 1000);
    }

    #[test]
    fn merge_checked_overflow() {
        let b1 = Budget::new(u64::MAX - 100);
        let b2 = Budget::new(200);
        assert!(matches!(b1.merge_checked(b2), Err(BudgetError::Overflow)));
    }
}
