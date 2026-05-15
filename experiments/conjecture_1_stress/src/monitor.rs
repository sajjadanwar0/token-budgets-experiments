//! Invariant monitor: receives events from concurrent Budget operations
//! and checks Φ(s) ≤ B₀ at every step.
//!
//! Φ(s) is the sum of `.available` across all live Budgets. Tracked
//! incrementally as Created/Split/Merge add to Φ and Spend/Dropped
//! reduce it.

use crate::budget::Event;
use std::collections::HashMap;
use tokio::sync::mpsc::UnboundedReceiver;

#[derive(Debug, Clone, serde::Serialize)]
pub struct MonitorReport {
    pub initial_capacity: u64,
    pub max_phi_observed: u64,
    pub final_phi: u64,
    pub total_spent: u64,
    pub total_dropped: u64,
    pub violations: Vec<Violation>,
    pub events_processed: u64,
    pub seed: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Violation {
    pub event_index: u64,
    pub event: String,
    pub phi: u64,
    pub b0: u64,
    pub explanation: String,
}

/// Φ-monitor: per-budget-id liveness ledger.
pub struct Monitor {
    b0: u64,
    seed: u64,
    /// budget id -> current .available (live budgets only)
    live: HashMap<u64, u64>,
    total_spent: u64,
    total_dropped: u64,
    events_processed: u64,
    max_phi_observed: u64,
    violations: Vec<Violation>,
}

impl Monitor {
    pub fn new(b0: u64, seed: u64) -> Self {
        Self {
            b0,
            seed,
            live: HashMap::new(),
            total_spent: 0,
            total_dropped: 0,
            events_processed: 0,
            max_phi_observed: 0,
            violations: Vec::new(),
        }
    }

    fn phi(&self) -> u64 {
        self.live.values().sum::<u64>()
    }

    fn check_and_record(&mut self, event_repr: &str) {
        let phi = self.phi();
        if phi > self.b0 {
            self.violations.push(Violation {
                event_index: self.events_processed,
                event: event_repr.to_string(),
                phi,
                b0: self.b0,
                explanation: format!(
                    "Φ(s) = {} > B₀ = {}; conservation violated", phi, self.b0
                ),
            });
        }
        // Also check the invariant on total spent + Φ + dropped = B₀
        // (since every micro-cent must be either still live, spent, or lost-on-drop)
        let accounted = phi + self.total_spent + self.total_dropped;
        if accounted > self.b0 {
            self.violations.push(Violation {
                event_index: self.events_processed,
                event: event_repr.to_string(),
                phi: accounted,
                b0: self.b0,
                explanation: format!(
                    "accounted (live={} + spent={} + dropped={} = {}) > B₀ = {}",
                    phi, self.total_spent, self.total_dropped, accounted, self.b0
                ),
            });
        }
        self.max_phi_observed = self.max_phi_observed.max(phi);
    }

    pub async fn run(mut self, mut rx: UnboundedReceiver<Event>) -> MonitorReport {
        while let Some(event) = rx.recv().await {
            self.events_processed += 1;
            let repr = format!("{:?}", event);
            match event {
                Event::Created { id, capacity, parent: _ } => {
                    self.live.insert(id, capacity);
                }
                Event::Spend { id, amount, remaining } => {
                    // The id is reused (same Budget conceptually); update its
                    // ledger entry to remaining.
                    self.live.insert(id, remaining);
                    self.total_spent += amount;
                }
                Event::Split { parent, child_a, child_b, cap_a, cap_b } => {
                    // Parent dies, children take its capacity
                    self.live.remove(&parent);
                    self.live.insert(child_a, cap_a);
                    self.live.insert(child_b, cap_b);
                }
                Event::Merge { result, capacity, consumed_a, consumed_b } => {
                    self.live.remove(&consumed_a);
                    self.live.remove(&consumed_b);
                    self.live.insert(result, capacity);
                }
                Event::Dropped { id, available_at_drop } => {
                    self.live.remove(&id);
                    self.total_dropped += available_at_drop;
                }
            }
            self.check_and_record(&repr);
        }

        MonitorReport {
            initial_capacity: self.b0,
            max_phi_observed: self.max_phi_observed,
            final_phi: self.phi(),
            total_spent: self.total_spent,
            total_dropped: self.total_dropped,
            violations: self.violations,
            events_processed: self.events_processed,
            seed: self.seed,
        }
    }
}
