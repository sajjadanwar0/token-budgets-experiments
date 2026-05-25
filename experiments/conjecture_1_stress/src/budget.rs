use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc::UnboundedSender;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, serde::Serialize)]
pub enum Event {
    Created { id: u64, capacity: u64, parent: Option<u64> },
    Spend   { id: u64, amount: u64, remaining: u64 },
    Split   { parent: u64, child_a: u64, child_b: u64, cap_a: u64, cap_b: u64 },
    Merge   { result: u64, capacity: u64, consumed_a: u64, consumed_b: u64 },
    Dropped { id: u64, available_at_drop: u64 },
}

#[derive(Debug)]
pub struct Budget {
    id: u64,
    available: u64,
    monitor: UnboundedSender<Event>,
    consumed: bool,
}

impl Budget {
    pub fn new(capacity: u64, monitor: UnboundedSender<Event>) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
        let _ = monitor.send(Event::Created { id, capacity, parent: None });
        Self { id, available: capacity, monitor, consumed: false }
    }

    pub fn id(&self) -> u64 { self.id }
    pub fn available(&self) -> u64 { self.available }

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

        Ok(Self {
            id,
            available: new_avail,
            monitor: mon,
            consumed: false,
        })
    }

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