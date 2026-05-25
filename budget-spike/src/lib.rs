use std::fmt;

pub mod llm_client;
pub mod tokenizer;
pub mod receipt;

pub use receipt::{ReservationReceipt, Refund};

#[derive(Debug, PartialEq, Eq)]
pub struct Budget {
    micro_cents: u64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum BudgetError {
    Insufficient { requested: u64, available: u64 },
    Overflow,
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
    pub fn new(micro_cents: u64) -> Self {
        Self { micro_cents }
    }

    pub fn available(&self) -> u64 {
        self.micro_cents
    }

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

    pub fn apply_refund(self, refund: Refund) -> Result<Budget, BudgetError> {
        let amount = refund.into_amount();
        let total = self.micro_cents
            .checked_add(amount)
            .ok_or(BudgetError::Overflow)?;
        Ok(Budget { micro_cents: total })
    }

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

    pub fn merge(self, other: Budget) -> Budget {
        Budget {
            micro_cents: self.micro_cents.saturating_add(other.micro_cents),
        }
    }

    pub fn merge_checked(self, other: Budget) -> Result<Budget, BudgetError> {
        let total = self.micro_cents
            .checked_add(other.micro_cents)
            .ok_or(BudgetError::Overflow)?;
        Ok(Budget { micro_cents: total })
    }

    pub fn consume(self) -> u64 {
        self.micro_cents
    }
}

pub fn estimate_cost(input_tokens: u64, max_output_tokens: u64, price_per_mtok: u64) -> u64 {
    let total = input_tokens.saturating_add(max_output_tokens);
    total.saturating_mul(price_per_mtok) / 1_000_000
}

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