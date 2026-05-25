use crate::BudgetError;

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

#[derive(Debug)]
pub struct Refund {
    micro_cents: u64,
    consumed: bool,
}

impl Refund {
    pub(crate) fn new(amount: u64) -> Self {
        Self { micro_cents: amount, consumed: false }
    }

    pub fn amount(&self) -> u64 {
        self.micro_cents
    }
    
    pub(crate) fn into_amount(mut self) -> u64 {
        self.consumed = true;
        self.micro_cents
    }
    
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