use anyhow::{Result};
use token_budgets::Budget;
use serde_json::{json, Value};
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

const TASK_BUDGET_NC: u64 = 50_000_000;
type TaskBudget = Budget<TASK_BUDGET_NC>;

const IN_RATE_NC: u64 = 1000;
const OUT_RATE_NC: u64 = 5000;
const MAX_ITERATIONS: usize = 6;
const MAX_TOKENS_PER_ITER: u32 = 512;

struct TaskRecord {
    task_idx: usize,
    task_name: String,
    iterations_completed: usize,
    tool_calls: usize,
    total_reserved_nc: u64,
    total_actual_nc: u64,
    completed: bool,
    budget_exhausted: bool,
    violations: usize,
    final_budget_nc: u64,
    tightest_margin: f64,
}

fn tasks() -> Vec<&'static str> {
    vec![
        "What's 17 * 23, and is the result prime?",
        "Compute the area of a triangle with base 14 and height 9.",
        "What year was the Eiffel Tower completed, and how tall is it?",
        "Find the GCD of 144 and 96.",
        "If a car travels at 65mph for 3.5 hours, how far does it go?",
        "Convert 100 degrees Fahrenheit to Celsius. Show formula.",
        "What's the molecular weight of water?",
        "Is 89 prime? Show your reasoning.",
        "Sum the first 10 squares: 1 + 4 + 9 + 16 + ... + 100.",
        "What's 2^16 - 1?",
        "Compute the hypotenuse of a right triangle with legs 7 and 24.",
        "Find the average of 12, 45, 67, 89, 23, 56, 78.",
        "What's 7 factorial?",
        "Roughly, how far is the Sun from Earth in km?",
        "What's the speed of light in m/s?",
        "Convert 5 km to miles.",
        "If you save $50 a week, how much in 2 years?",
        "Compute the volume of a sphere with radius 4.",
        "How many seconds in a day?",
        "What's 0.1 + 0.2 in floating point, and why?",
        "Convert 90 degrees to radians.",
        "What's the population of Tokyo, roughly?",
        "How many milligrams in a gram?",
        "What's e^2?",
        "Roughly, the boiling point of water at sea level?",
        "How many bits in a megabyte?",
        "What's the formula for the area of a circle?",
        "What's 15% of 240?",
        "If x + 3 = 8, what is x?",
        "What's the perimeter of a square with side 12?",
        "Sum: 1 + 2 + 3 + ... + 100",
        "What's the largest 3-digit prime?",
        "How many days in February 2024?",
        "Compute 50 choose 2.",
        "What's the chemical formula for table salt?",
        "Distance from London to Paris, roughly?",
        "How many minutes in a week?",
        "What's pi to 4 decimal places?",
        "If a recipe needs 3 cups for 4 people, how much for 6?",
        "What's the difference of squares: 10^2 - 9^2?",
        "Convert 72 inches to feet.",
        "What's the cube root of 27?",
        "Solve: 2x + 5 = 17.",
        "How many countries in the EU?",
        "What's the chemical symbol for gold?",
        "Compute 7 * 8 * 9.",
        "What's the median of 4, 7, 2, 9, 5?",
        "What year did WWII end?",
        "If the angle is 30 degrees, what's the sine?",
        "How many ounces in a pound?",
    ]
}

async fn run_react_task(
    client: &reqwest::Client,
    api_key: &str,
    task: &str,
    task_idx: usize,
) -> Result<TaskRecord> {
    let mut budget: Option<TaskBudget> = Some(Budget::new(TASK_BUDGET_NC)?);
    let mut messages: Vec<Value> = vec![
        json!({
            "role": "user",
            "content": format!(
                "You are a ReAct agent. Use this loop:\n\
                 THINK: <reasoning>\n\
                 ACT: <tool call as: tool_name(args)>\n\
                 OBSERVE: <wait for tool result>\n\
                 (repeat up to 4 times)\n\
                 ANSWER: <final answer>\n\n\
                 Available tools: calc(expression), lookup(topic).\n\
                 Task: {}", task)
        })
    ];

    let mut iterations = 0;
    let mut tool_calls = 0;
    let mut total_reserved = 0u64;
    let mut total_actual = 0u64;
    let mut violations = 0;
    let mut tightest = f64::INFINITY;
    let mut completed = false;
    let mut exhausted = false;

    for iter in 0..MAX_ITERATIONS {
        let body = serde_json::to_string(&json!({
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": MAX_TOKENS_PER_ITER,
            "messages": &messages,
        }))?;
        let reservation = body.len() as u64 * IN_RATE_NC
            + (MAX_TOKENS_PER_ITER as u64) * OUT_RATE_NC;

        let current = budget.take().expect("budget present");
        let (after_reserve, receipt) = match current.spend_with_receipt(reservation) {
            Ok(x) => x,
            Err(_) => {
                exhausted = true;
                budget = Some(Budget::new(0)?);
                break;
            }
        };

        let resp = client.post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body)
            .send().await;
        let resp = match resp {
            Ok(r) if r.status().is_success() => r,
            _ => {
                receipt.forfeit();
                budget = Some(after_reserve);
                continue;
            }
        };

        let parsed: Value = resp.json().await?;
        let in_tok = parsed["usage"]["input_tokens"].as_u64().unwrap_or(0);
        let out_tok = parsed["usage"]["output_tokens"].as_u64().unwrap_or(0);
        let actual = in_tok * IN_RATE_NC + out_tok * OUT_RATE_NC;

        if actual > reservation {
            violations += 1;
            receipt.forfeit();
            budget = Some(after_reserve);
            continue;
        }

        let margin = reservation as f64 / actual.max(1) as f64;
        if margin < tightest { tightest = margin; }

        let refund = receipt.confirm(actual)?;
        budget = Some(refund.apply_to(after_reserve)?);

        total_reserved += reservation;
        total_actual += actual;

        let assistant_text = parsed["content"][0]["text"].as_str().unwrap_or("").to_string();
        iterations = iter + 1;

        if assistant_text.contains("calc(") || assistant_text.contains("lookup(") {
            tool_calls += 1;
            messages.push(json!({"role": "assistant", "content": assistant_text}));
            messages.push(json!({"role": "user", "content": "Tool result: <mock result for demo>"}));
        } else if assistant_text.contains("ANSWER:") {
            completed = true;
            break;
        } else {
            messages.push(json!({"role": "assistant", "content": assistant_text}));
            messages.push(json!({"role": "user", "content": "Continue."}));
        }
    }

    let final_budget = budget.as_ref().map(|b| b.micro_cents()).unwrap_or(0);

    Ok(TaskRecord {
        task_idx,
        task_name: task.chars().take(40).collect(),
        iterations_completed: iterations,
        tool_calls,
        total_reserved_nc: total_reserved,
        total_actual_nc: total_actual,
        completed,
        budget_exhausted: exhausted,
        violations,
        final_budget_nc: final_budget,
        tightest_margin: if tightest.is_finite() { tightest } else { f64::NAN },
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("ANTHROPIC_API_KEY")?;
    let client = reqwest::Client::builder().timeout(Duration::from_secs(60)).build()?;
    let tasks = tasks();
    let n_tasks: usize = env::var("N_TASKS").ok().and_then(|s| s.parse().ok()).unwrap_or(tasks.len());

    let mut records: Vec<TaskRecord> = Vec::new();
    let start = Instant::now();

    for (i, task) in tasks.iter().take(n_tasks).enumerate() {
        let rec = run_react_task(&client, &api_key, task, i).await?;
        println!("Task {:>3} '{:.50}': iter={} tool={} cost=${:.4} margin_tight={:.3}x{}{}",
                 rec.task_idx, &rec.task_name, rec.iterations_completed, rec.tool_calls,
                 rec.total_actual_nc as f64 / 1e9, rec.tightest_margin,
                 if rec.completed { " ✓" } else { " ⊘" },
                 if rec.budget_exhausted { " EXHAUSTED" } else { "" });
        records.push(rec);
    }

    let elapsed = start.elapsed();
    let total_violations: usize = records.iter().map(|r| r.violations).sum();
    let total_iterations: usize = records.iter().map(|r| r.iterations_completed).sum();
    let completed_count = records.iter().filter(|r| r.completed).count();
    let exhausted_count = records.iter().filter(|r| r.budget_exhausted).count();
    let total_cost: u64 = records.iter().map(|r| r.total_actual_nc).sum();
    let tight_margins: Vec<f64> = records.iter()
        .filter(|r| r.tightest_margin.is_finite()).map(|r| r.tightest_margin).collect();

    println!();
    println!("=== ReAct agent eval summary ===");
    println!("Tasks:               {}", records.len());
    println!("Completed:           {} ({:.0}%)", completed_count, 100.0 * completed_count as f64 / records.len() as f64);
    println!("Budget-exhausted:    {} ({:.0}%)", exhausted_count, 100.0 * exhausted_count as f64 / records.len() as f64);
    println!("Total iterations:    {} (avg {:.1} per task)", total_iterations, total_iterations as f64 / records.len() as f64);
    println!("Total tool calls:    {}", records.iter().map(|r| r.tool_calls).sum::<usize>());
    println!("Total A1 violations: {}", total_violations);
    println!("Total spend:         ${:.4}", total_cost as f64 / 1e9);
    println!("Wall time:           {:.1} min", elapsed.as_secs_f64() / 60.0);
    if !tight_margins.is_empty() {
        let tight_min = tight_margins.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let tight_mean = tight_margins.iter().sum::<f64>() / tight_margins.len() as f64;
        println!("Tightest margin across tasks: min={:.3}x mean={:.3}x", tight_min, tight_mean);
    }

    let mut csv = File::create("react_agent_results.csv")?;
    writeln!(csv, "task_idx,task_name,iterations,tool_calls,total_reserved_nc,total_actual_nc,completed,exhausted,violations,final_budget_nc,tightest_margin")?;
    for r in &records {
        writeln!(csv, "{},\"{}\",{},{},{},{},{},{},{},{},{:.6}",
                 r.task_idx, r.task_name, r.iterations_completed, r.tool_calls,
                 r.total_reserved_nc, r.total_actual_nc, r.completed,
                 r.budget_exhausted, r.violations, r.final_budget_nc, r.tightest_margin)?;
    }
    println!("Wrote {} rows to react_agent_results.csv", records.len());
    Ok(())
}