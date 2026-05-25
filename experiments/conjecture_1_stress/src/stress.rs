use crate::budget::Budget;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use tokio::sync::mpsc::UnboundedSender;
use crate::budget::Event;

#[derive(Clone, Copy)]
pub struct StressConfig {
    pub tasks: usize,
    pub ops_per_task: usize,
    pub initial_capacity: u64,
    pub panic_probability: f64,
    pub max_split_depth: usize,
    pub seed: u64,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            tasks: 32,
            ops_per_task: 1000,
            initial_capacity: 1_000_000,
            panic_probability: 0.0,
            max_split_depth: 4,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Spend,
    SpendThenSplit,
    SplitAndSpawn,
    DropNaturally,
}

fn pick_op(rng: &mut ChaCha8Rng, depth_remaining: usize) -> Op {
    let r = rng.gen::<f64>();
    if depth_remaining > 0 {
        match r {
            x if x < 0.50 => Op::Spend,
            x if x < 0.70 => Op::SpendThenSplit,
            x if x < 0.90 => Op::SplitAndSpawn,
            _             => Op::DropNaturally,
        }
    } else {
        match r {
            x if x < 0.80 => Op::Spend,
            _             => Op::DropNaturally,
        }
    }
}

pub fn task_loop(
    budget: Budget,
    config: StressConfig,
    depth: usize,
    seed: u64,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> {
    let depth_remaining = config.max_split_depth.saturating_sub(depth);
    Box::pin(async move {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut budget_opt: Option<Budget> = Some(budget);

        for op_idx in 0..config.ops_per_task {
            let cur = match budget_opt.take() {
                Some(b) if b.available() > 0 => b,
                _ => break,
            };

            match pick_op(&mut rng, depth_remaining) {
                Op::Spend => {
                    let amount = rng.gen_range(1..=(cur.available() / 10).max(1));
                    match cur.spend(amount) {
                        Ok(b) => budget_opt = Some(b),
                        Err(_) => break,
                    }
                }
                Op::SpendThenSplit => {
                    let amount = rng.gen_range(1..=(cur.available() / 20).max(1));
                    let after_spend = match cur.spend(amount) {
                        Ok(b) => b,
                        Err(_) => break,
                    };
                    if after_spend.available() >= 2 {
                        let split_amount = rng.gen_range(1..after_spend.available());
                        match after_spend.split(split_amount) {
                            Ok((a, b)) => {
                                budget_opt = Some(a);
                                drop(b);
                            }
                            Err(_) => break,
                        }
                    } else {
                        budget_opt = Some(after_spend);
                    }
                }
                Op::SplitAndSpawn => {
                    if cur.available() >= 2 {
                        let split_amount = rng.gen_range(1..cur.available());
                        match cur.split(split_amount) {
                            Ok((a, b)) => {
                                let child_seed: u64 = rng.gen();
                                let child_cfg = StressConfig {
                                    ops_per_task: config.ops_per_task / 4,
                                    ..config
                                };
                                let child_depth = depth + 1;
                                let _handle = tokio::spawn(async move {
                                    task_loop(b, child_cfg, child_depth, child_seed).await;
                                });
                                budget_opt = Some(a);
                            }
                            Err(_) => break,
                        }
                    } else {
                        budget_opt = Some(cur);
                    }
                }
                Op::DropNaturally => {
                    drop(cur);
                    return;
                }
            }

            if op_idx % 17 == 0 {
                tokio::task::yield_now().await;
            }
        }
    })
}

pub async fn run_iteration(
    monitor_tx: UnboundedSender<Event>,
    config: StressConfig,
) {
    let root = Budget::new(config.initial_capacity, monitor_tx.clone());
    let per_task = (config.initial_capacity / config.tasks as u64).max(1);

    let mut sub_budgets: Vec<Budget> = vec![];
    let mut remaining = Some(root);
    for i in 0..config.tasks {
        let cur = remaining.take().expect("remaining present");
        if i == config.tasks - 1 || cur.available() <= per_task {
            sub_budgets.push(cur);
            break;
        }
        match cur.split(per_task) {
            Ok((allocated, rest)) => {
                sub_budgets.push(allocated);
                remaining = Some(rest);
            }
            Err(_) => break,
        }
    }

    let mut handles = vec![];
    let mut task_rng = ChaCha8Rng::seed_from_u64(config.seed);
    for (i, b) in sub_budgets.into_iter().enumerate() {
        let task_seed: u64 = task_rng.gen();
        let cfg = StressConfig { seed: config.seed.wrapping_add(i as u64), ..config };
        handles.push(tokio::spawn(async move {
            task_loop(b, cfg, 0, task_seed).await;
        }));
    }

    for h in handles {
        let _ = h.await;
    }
}
