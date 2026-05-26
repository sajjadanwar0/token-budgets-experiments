use anyhow::{Context, Result};
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Duration;

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const MODEL: &str = "claude-haiku-4-5-20251001";
const MAX_OUT: u32 = 200;
const PER_IN_UC: u64 = 1;
const PER_OUT_UC: u64 = 5;
const N_PER_CLASS: usize = 100;

#[derive(Deserialize)]
struct Resp { usage: Usage }
#[derive(Deserialize)]
struct Usage { input_tokens: u64, output_tokens: u64 }

fn build_prompt(class: &str, idx: usize) -> String {
    match class {
        "cjk_dense" => {
            const BASE: &[&str] = &[
                "请用中文回答以下问题：什么是机器学习？",
                "深度学习与传统机器学习有什么区别？",
                "卷积神经网络的工作原理是什么？",
                "循环神经网络如何处理序列数据？",
                "什么是注意力机制？请详细说明。",
                "Transformer模型为什么如此重要？",
                "什么是预训练和微调？",
                "强化学习与监督学习的区别是什么？",
                "贝叶斯方法在机器学习中的应用？",
                "请解释什么是迁移学习。",
            ];
            let phrases_per_prompt = 1 + (idx % 5);
            let mut out = String::new();
            for k in 0..phrases_per_prompt {
                out.push_str(BASE[(idx + k) % BASE.len()]);
            }
            out
        }
        "emoji_dense" => {
            const EMOJI_POOL: &[&str] = &[
                "🚀", "🌟", "🎯", "🔥", "💫", "🌈", "🎨", "🎭", "🎪", "🎢",
                "🎡", "🎠", "🎮", "🎲", "🎴", "🎳", "🎰", "🎱", "🃏", "🀄",
                "🎺", "🎸", "🎼", "🎹", "🎤", "🎧", "🥁", "🎷", "🪗", "🎻",
            ];
            let count = 10 + (idx % 20);
            let start = idx % EMOJI_POOL.len();
            (0..count).map(|k| EMOJI_POOL[(start + k) % EMOJI_POOL.len()]).collect()
        }
        "repeated_rare" => {
            const RARE: &[&str] = &[
                "supercalifragilisticexpialidocious",
                "pneumonoultramicroscopicsilicovolcanoconiosis",
                "antidisestablishmentarianism",
                "floccinaucinihilipilification",
                "thyroparathyroidectomized",
                "psychoneuroendocrinological",
                "incomprehensibility",
                "uncharacteristically",
                "counterproductiveness",
                "interdisciplinarity",
            ];
            let word = RARE[idx % RARE.len()];
            let reps = 10 + (idx % 20);
            (0..reps).map(|_| format!("{} ", word)).collect()
        }
        "long_output" => {
            const TOPICS: &[&str] = &[
                "the history of computing", "the discovery of penicillin",
                "the invention of the telephone", "the fall of the Roman Empire",
                "the Industrial Revolution", "the discovery of DNA structure",
                "the development of relativity", "the invention of the printing press",
                "the rise of the Mongol Empire", "the discovery of antibiotics",
            ];
            format!("Write a 500-word essay about {}.", TOPICS[idx % TOPICS.len()])
        }
        "mixed_scripts" => {
            const FRAGMENTS: &[&str] = &[
                "Hello world.", "مرحبا بالعالم.", "שלום עולם.", "नमस्ते दुनिया.", "你好世界",
                "Bonjour monde.", "こんにちは世界", "안녕 세상", "Здравствуй мир", "Γειά σου κόσμε",
            ];
            let count = 3 + (idx % 5);
            let start = idx % FRAGMENTS.len();
            let mut out = String::from("Translate or interpret: ");
            for k in 0..count {
                out.push_str(FRAGMENTS[(start + k) % FRAGMENTS.len()]);
                out.push(' ');
            }
            out
        }
        "json_dense" => {
            let depth = 5 + (idx % 10);
            let mut s = String::from("Parse this JSON: ");
            for _ in 0..depth { s.push_str(r#"{"a":"#); }
            s.push_str(&idx.to_string());
            for _ in 0..depth { s.push('}'); }
            s
        }
        "whitespace_pad" => {
            let pad = 100 + (idx % 200);
            format!("{}word{}", " ".repeat(pad), idx)
        }
        "code_heavy" => {
            let n = 10 + (idx % 50);
            format!(
                "def f_{i}():\n{indent}for i in range({n}):\n{indent}{indent}for j in range({n}):\n{indent}{indent}{indent}print(i*j)",
                i = idx, indent = "    ", n = n
            )
        }
        "long_prompt" => {
            let reps = 100 + (idx % 100);
            "Summarise the following text: ".to_string()
                + &"This is a sample sentence that will be repeated many times. ".repeat(reps)
        }
        "tool_format" => {
            format!(
                r#"<function_calls><invoke name="search_{}"><parameter name="q">query {}</parameter></invoke></function_calls>"#,
                idx, idx
            )
        }
        _ => format!("Unknown class: {}", class),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_key = env::var("ANTHROPIC_API_KEY")
        .context("ANTHROPIC_API_KEY must be set")?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    const CLASSES: &[&str] = &[
        "cjk_dense", "emoji_dense", "repeated_rare", "long_output",
        "mixed_scripts", "json_dense", "whitespace_pad", "code_heavy",
        "long_prompt", "tool_format",
    ];

    let mut per_class_violations: std::collections::HashMap<&str, usize> = Default::default();
    let mut per_class_min_margin: std::collections::HashMap<&str, f64> = Default::default();
    let mut all_records: Vec<(String, usize, u64, u64, f64, bool)> = Vec::new();

    for &class in CLASSES {
        per_class_violations.insert(class, 0);
        per_class_min_margin.insert(class, f64::INFINITY);

        println!("\n=== Class: {} (N={}) ===", class, N_PER_CLASS);
        for i in 0..N_PER_CLASS {
            let prompt = build_prompt(class, i);
            let req = serde_json::json!({
                "model": MODEL,
                "max_tokens": MAX_OUT,
                "messages": [{"role": "user", "content": prompt}],
            });
            let body = serde_json::to_string(&req)?;
            let reservation = body.len() as u64 * PER_IN_UC + MAX_OUT as u64 * PER_OUT_UC;

            let resp = client.post(ANTHROPIC_URL)
                .header("x-api-key", &api_key)
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .body(body)
                .send().await;

            let resp = match resp {
                Ok(r) if r.status().is_success() => r,
                Ok(r) => {
                    eprintln!("  [{}] class={} status={}", i, class, r.status());
                    continue;
                }
                Err(e) => {
                    eprintln!("  [{}] class={} err={}", i, class, e);
                    continue;
                }
            };
            let parsed: Resp = match resp.json().await {
                Ok(p) => p,
                Err(e) => { eprintln!("  [{}] parse err: {}", i, e); continue; }
            };

            let actual = parsed.usage.input_tokens * PER_IN_UC
                + parsed.usage.output_tokens * PER_OUT_UC;
            let margin = reservation as f64 / actual.max(1) as f64;
            let violated = actual > reservation;

            if violated {
                *per_class_violations.get_mut(class).unwrap() += 1;
                println!("  [{}] ⚠ VIOLATION reserve={} actual={}", i, reservation, actual);
            }
            let cur_min = per_class_min_margin.get_mut(class).unwrap();
            if margin < *cur_min { *cur_min = margin; }

            all_records.push((class.to_string(), i, reservation, actual, margin, violated));

            if i % 20 == 0 || i == N_PER_CLASS - 1 {
                println!("  [{:>3}] reserve={:>6} actual={:>6} margin={:.3}x{}",
                         i, reservation, actual, margin,
                         if violated { " ⚠" } else { "" });
            }
        }

        let v = per_class_violations[class];
        let m = per_class_min_margin[class];
        println!("  {}: {}/{} violations ({:.1}%), min margin {:.3}x",
                 class, v, N_PER_CLASS, 100.0 * v as f64 / N_PER_CLASS as f64, m);
    }

    println!("\n=== Aggregate ===");
    let total = all_records.len();
    let total_v: usize = per_class_violations.values().sum();
    let global_min = per_class_min_margin.values().fold(f64::INFINITY, |a, b| a.min(*b));
    println!("Total calls:      {} (target {})", total, CLASSES.len() * N_PER_CLASS);
    println!("Total violations: {} ({:.3}%)", total_v, 100.0 * total_v as f64 / total as f64);
    println!("Global min margin: {:.3}x", global_min);
    println!();
    println!("Per-class breakdown:");
    println!("{:>20} {:>10} {:>10} {:>12}", "class", "violations", "min_margin", "rate");
    for &class in CLASSES {
        let v = per_class_violations[class];
        let m = per_class_min_margin[class];
        println!("{:>20} {:>10} {:>10.3} {:>11.2}%", class, v, m,
                 100.0 * v as f64 / N_PER_CLASS as f64);
    }

    println!();
    println!("Statistical interpretation (Clopper-Pearson):");
    println!("  N={}, observed violations k, 95% CI upper bound on true rate:", total);
    if total_v == 0 {
        // For k=0: upper = 1 - 0.05^(1/n)
        let upper = 1.0 - 0.05_f64.powf(1.0 / total as f64);
        println!("  k=0 → upper bound = {:.4}% (rule of 3 / exact)", upper * 100.0);
    } else {
        println!("  k={} observed", total_v);
    }

    let mut csv = File::create("a1_adversarial_n100_results.csv")?;
    writeln!(csv, "class,idx,reservation_uc,actual_uc,margin_ratio,violated")?;
    for (cls, i, res, act, mar, vio) in &all_records {
        writeln!(csv, "{},{},{},{},{:.6},{}", cls, i, res, act, mar, vio)?;
    }
    println!("\nWrote {} rows to a1_adversarial_n100_results.csv", all_records.len());

    Ok(())
}
