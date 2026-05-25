mod budget;
mod monitor;
mod stress;

use clap::Parser;
use monitor::{Monitor, MonitorReport};
use std::path::PathBuf;
use stress::{run_iteration, StressConfig};
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(name = "stress")]
#[command(about = "Conjecture 1 stress test: validate Lemma 2 invariant Φ ≤ B₀")]
struct Args {
    #[arg(long, default_value_t = 10_000)]
    iterations: usize,

    #[arg(long, default_value_t = 32)]
    tasks: usize,

    #[arg(long, default_value_t = 1_000)]
    ops_per_task: usize,

    #[arg(long, default_value_t = 1_000_000)]
    initial_capacity: u64,

    #[arg(long, default_value_t = 0.0)]
    panic_probability: f64,

    #[arg(long, default_value_t = 4)]
    max_split_depth: usize,

    #[arg(long, default_value_t = 42)]
    base_seed: u64,

    #[arg(long, default_value = "results")]
    output: PathBuf,

    #[arg(long, default_value_t = 100)]
    progress_every: usize,
}

#[tokio::main(flavor = "multi_thread", worker_threads = 16)]
async fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.output).expect("create output dir");

    println!("Conjecture 1 Stress Test");
    println!("Iterations:         {}", args.iterations);
    println!("Tasks per iter:     {}", args.tasks);
    println!("Ops per task:       {}", args.ops_per_task);
    println!("Initial capacity:   {}", args.initial_capacity);
    println!("Panic injection:    {:.2}%", args.panic_probability * 100.0);
    println!("Max split depth:    {}", args.max_split_depth);
    println!("Base seed:          {}", args.base_seed);
    println!();

    let start = std::time::Instant::now();
    let mut total_violations = 0u64;
    let mut total_events = 0u64;
    let mut violating_seeds: Vec<u64> = Vec::new();

    let mut csv_path = args.output.clone();
    csv_path.push("iterations.csv");
    let mut writer = csv::Writer::from_path(&csv_path).expect("open csv");
    writer.write_record(&[
        "iter", "seed", "events", "violations", "max_phi", 
        "final_phi", "total_spent", "total_dropped",
    ]).expect("write header");

    for iter_idx in 0..args.iterations {
        let seed = args.base_seed.wrapping_add(iter_idx as u64);
        let config = StressConfig {
            tasks: args.tasks,
            ops_per_task: args.ops_per_task,
            initial_capacity: args.initial_capacity,
            panic_probability: args.panic_probability,
            max_split_depth: args.max_split_depth,
            seed,
        };

        let (tx, rx) = mpsc::unbounded_channel();
        let monitor = Monitor::new(args.initial_capacity, seed);

        let monitor_handle = tokio::spawn(monitor.run(rx));
        run_iteration(tx, config).await;
        let report = monitor_handle.await.expect("monitor join");

        total_events += report.events_processed;
        let iter_violations = report.violations.len() as u64;
        total_violations += iter_violations;
        if iter_violations > 0 {
            violating_seeds.push(seed);
            let mut viol_path = args.output.clone();
            viol_path.push(format!("violation_seed_{}.json", seed));
            std::fs::write(
                &viol_path,
                serde_json::to_string_pretty(&report).expect("serialize"),
            )
            .expect("write violation");
            eprintln!("⚠️  VIOLATION at iter={} seed={}: details → {:?}",
                      iter_idx, seed, viol_path);
        }

        writer.write_record(&[
            iter_idx.to_string(),
            seed.to_string(),
            report.events_processed.to_string(),
            iter_violations.to_string(),
            report.max_phi_observed.to_string(),
            report.final_phi.to_string(),
            report.total_spent.to_string(),
            report.total_dropped.to_string(),
        ]).expect("write row");
        writer.flush().expect("flush");

        if (iter_idx + 1) % args.progress_every == 0 || iter_idx + 1 == args.iterations {
            let elapsed = start.elapsed().as_secs();
            let rate = (iter_idx + 1) as f64 / elapsed.max(1) as f64;
            println!(
                "  [{}/{}] elapsed: {}s, rate: {:.1} iter/s, violations: {}, events: {}M",
                iter_idx + 1, args.iterations, elapsed, rate, total_violations,
                total_events / 1_000_000,
            );
        }
    }

    let elapsed = start.elapsed();
    println!();
    println!("RESULTS");
    println!("Total iterations:    {}", args.iterations);
    println!("Total events:        {}", total_events);
    println!("Total violations:    {}", total_violations);
    println!("Violating seeds:     {:?}", &violating_seeds);
    println!("Wall-clock:          {:.2} min", elapsed.as_secs_f64() / 60.0);
    println!();
    if total_violations == 0 {
        println!("✅ NO VIOLATIONS observed in {} iterations.", args.iterations);
        println!("   Strong empirical support for Lemma 2 (safety preservation).");
        println!("   Conjecture 1 remains formally open (Iris mechanization required for proof).");
    } else {
        println!("❌ {} violations observed across {} iterations.", 
                 total_violations, args.iterations);
        println!("   See {:?} for details.", args.output);
        println!("   This is a CRITICAL FINDING — Lemma 2's monotonicity argument");
        println!("   may be incorrect, OR there is an implementation bug.");
        println!("   Required action: investigate the failing seed(s).");
    }
    
    let summary = serde_json::json!({
        "iterations": args.iterations,
        "tasks_per_iter": args.tasks,
        "ops_per_task": args.ops_per_task,
        "total_events": total_events,
        "total_violations": total_violations,
        "violating_seeds": violating_seeds,
        "wall_clock_secs": elapsed.as_secs(),
        "no_violations": total_violations == 0,
    });
    let mut summary_path = args.output.clone();
    summary_path.push("summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary).unwrap())
        .expect("write summary");
    println!("\nSummary: {:?}", summary_path);
}
