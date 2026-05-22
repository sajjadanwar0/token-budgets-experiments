//! multiagent_lang001.rs — multi-agent LANG-001 boundary validation.
//!
//! Validates that `Budget::split` + `tokio::spawn` enforce aggregate
//! cap-respect across concurrent children. Each trial:
//!   parent (B0=1620 uc) -> split into 3 children of 540 uc each ->
//!   each child runs LANG-001 SQL retry loop in its own task ->
//!   join all -> compute aggregate spend across all children.
//!
//! Expected: N/N trials with aggregate_spent <= B0, closing the
//! open empirical gap on the multi-agent property at the live-API
//! level (the compile-time integrity claim survives concurrent
//! parent-child delegation under provider non-determinism).
//!
//! Self-contained: all helpers (Pricing, chat types, OpenAI/Anthropic
//! HTTP clients, workload definitions, byte-length estimator) are
//! copied from tc_live_harness.rs so this binary compiles without
//! refactoring the existing harness.
//!
//! Build (from token-budgets-experiments/budget-spike/):
//!   cargo build --release --bin multiagent_lang001
//!
//! Run (OpenAI gpt-4o-mini):
//!   OPENAI_API_KEY=sk-... \
//!     ./target/release/multiagent_lang001 \
//!     --provider openai --runs 30 \
//!     --output-csv sweep_results_rust/multiagent_lang001_openai_n30.csv
//!
//! Run (Anthropic claude-haiku-4-5):
//!   ANTHROPIC_API_KEY=sk-ant-... \
//!     ./target/release/multiagent_lang001 \
//!     --provider anthropic --runs 30 \
//!     --output-csv sweep_results_rust/multiagent_lang001_anthropic_n30.csv

use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use budget_spike::Budget;

const MAX_OUTPUT_TOKENS: u64 = 200;
const MAX_AGENT_STEPS: u64 = 20;
const DEFAULT_CHILD_CAP_UC: u64 = 540;  // OpenAI LANG-001 default; pass --child-cap-uc 2000 for Anthropic.

// =============================================================================
// Pricing (same shape as tc_live_harness.rs)
// =============================================================================

#[derive(Clone, Copy, Debug)]
struct Pricing {
    input_per_token_uc_per_million: u64,
    output_per_token_uc_per_million: u64,
}

impl Pricing {
    fn cost_uc(&self, in_tok: u64, out_tok: u64) -> u64 {
        let in_uc = in_tok.saturating_mul(self.input_per_token_uc_per_million);
        let out_uc = out_tok.saturating_mul(self.output_per_token_uc_per_million);
        let total = in_uc.saturating_add(out_uc);
        (total + 500_000) / 1_000_000
    }
}

fn pricing_for(provider: &str) -> Pricing {
    match provider {
        "openai" => Pricing {
            input_per_token_uc_per_million: 150_000,
            output_per_token_uc_per_million: 600_000,
        },
        "anthropic" => Pricing {
            input_per_token_uc_per_million: 1_000_000,
            output_per_token_uc_per_million: 5_000_000,
        },
        _ => panic!("unknown provider: {}", provider),
    }
}

fn model_for(provider: &str) -> &'static str {
    match provider {
        "openai" => "gpt-4o-mini",
        "anthropic" => "claude-haiku-4-5-20251001",
        _ => panic!("unknown provider: {}", provider),
    }
}

// =============================================================================
// Common chat types (copied from tc_live_harness.rs for self-containment)
// =============================================================================

#[derive(Clone, Debug)]
struct ChatMessage {
    role: String,
    content: String,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
}

#[derive(Clone, Debug)]
struct ToolCall {
    id: String,
    name: String,
    arguments_json: String,
}

#[derive(Clone, Debug)]
struct ToolDef {
    name: &'static str,
    description: &'static str,
    parameters_schema_json: &'static str,
}

fn estimate_input_bytes(messages: &[ChatMessage], tools: &[ToolDef]) -> u64 {
    let mut total: u64 = 0;
    for m in messages {
        total = total.saturating_add(m.role.len() as u64);
        total = total.saturating_add(m.content.len() as u64);
        for tc in &m.tool_calls {
            total = total.saturating_add(tc.name.len() as u64);
            total = total.saturating_add(tc.arguments_json.len() as u64);
            total = total.saturating_add(tc.id.len() as u64);
        }
        if let Some(id) = &m.tool_call_id {
            total = total.saturating_add(id.len() as u64);
        }
    }
    for t in tools {
        total = total.saturating_add(t.name.len() as u64);
        total = total.saturating_add(t.description.len() as u64);
        total = total.saturating_add(t.parameters_schema_json.len() as u64);
    }
    total = total.saturating_add((messages.len() as u64).saturating_mul(64));
    total
}

#[derive(Debug)]
struct ChatResult {
    response: ChatMessage,
    input_tokens: u64,
    output_tokens: u64,
}

// ----- OpenAI ----------------------------------------------------------------

#[derive(Serialize)]
struct OAIRequest<'a> {
    model: &'a str,
    messages: Vec<OAIMessage>,
    tools: Vec<OAITool>,
    temperature: f64,
    max_completion_tokens: u64,
}

#[derive(Serialize)]
struct OAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<OAIToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize)]
struct OAITool {
    #[serde(rename = "type")]
    typ: &'static str,
    function: OAIToolFunction,
}

#[derive(Serialize)]
struct OAIToolFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct OAIToolCall {
    id: String,
    #[serde(rename = "type")]
    typ: &'static str,
    function: OAIToolCallFunction,
}

#[derive(Serialize)]
struct OAIToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OAIResponse {
    choices: Vec<OAIChoice>,
    usage: OAIUsage,
}

#[derive(Deserialize, Debug)]
struct OAIChoice {
    message: OAIChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct OAIChoiceMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<OAIRespToolCall>,
}

#[derive(Deserialize, Debug)]
struct OAIRespToolCall {
    id: String,
    function: OAIRespToolFunction,
}

#[derive(Deserialize, Debug)]
struct OAIRespToolFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize, Debug)]
struct OAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

async fn openai_call(
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
    temperature: f64,
) -> Result<ChatResult, String> {
    let oai_messages: Vec<OAIMessage> = messages.iter().map(|m| OAIMessage {
        role: m.role.clone(),
        content: if m.content.is_empty() && !m.tool_calls.is_empty() {
            None
        } else {
            Some(m.content.clone())
        },
        tool_calls: m.tool_calls.iter().map(|tc| OAIToolCall {
            id: tc.id.clone(),
            typ: "function",
            function: OAIToolCallFunction {
                name: tc.name.clone(),
                arguments: tc.arguments_json.clone(),
            },
        }).collect(),
        tool_call_id: m.tool_call_id.clone(),
    }).collect();

    let oai_tools: Vec<OAITool> = tools.iter().map(|t| OAITool {
        typ: "function",
        function: OAIToolFunction {
            name: t.name.to_string(),
            description: t.description.to_string(),
            parameters: serde_json::from_str(t.parameters_schema_json)
                .unwrap_or(serde_json::json!({})),
        },
    }).collect();

    let req = OAIRequest {
        model: model_for("openai"),
        messages: oai_messages,
        tools: oai_tools,
        temperature,
        max_completion_tokens: max_output_tokens,
    };

    let client = reqwest::Client::new();
    let resp = client.post("https://api.openai.com/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("network: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, body));
    }

    let body: OAIResponse = resp.json().await
        .map_err(|e| format!("parse: {}", e))?;

    let choice = body.choices.into_iter().next()
        .ok_or_else(|| "no choices in response".to_string())?;

    let tool_calls: Vec<ToolCall> = choice.message.tool_calls.into_iter().map(|tc| ToolCall {
        id: tc.id,
        name: tc.function.name,
        arguments_json: tc.function.arguments,
    }).collect();

    let assistant = ChatMessage {
        role: "assistant".to_string(),
        content: choice.message.content.unwrap_or_default(),
        tool_calls,
        tool_call_id: None,
    };

    Ok(ChatResult {
        response: assistant,
        input_tokens: body.usage.prompt_tokens,
        output_tokens: body.usage.completion_tokens,
    })
}

// ----- Anthropic -------------------------------------------------------------

#[derive(Serialize)]
struct AntRequest<'a> {
    model: &'a str,
    max_tokens: u64,
    temperature: f64,
    system: String,
    messages: Vec<AntMessage>,
    tools: Vec<AntTool>,
}

#[derive(Serialize)]
struct AntMessage {
    role: String,
    content: Vec<AntContentBlock>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
enum AntContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value },
    #[serde(rename = "tool_result")]
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Serialize)]
struct AntTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct AntResponse {
    content: Vec<AntRespBlock>,
    usage: AntUsage,
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
enum AntRespBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: serde_json::Value },
}

#[derive(Deserialize, Debug)]
struct AntUsage {
    input_tokens: u64,
    output_tokens: u64,
}

async fn anthropic_call(
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
    temperature: f64,
) -> Result<ChatResult, String> {
    let mut system_text = String::new();
    let mut ant_messages: Vec<AntMessage> = Vec::new();

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push('\n');
            }
            system_text.push_str(&m.content);
            continue;
        }
        let blocks: Vec<AntContentBlock> = match m.role.as_str() {
            "user" => vec![AntContentBlock::Text { text: m.content.clone() }],
            "assistant" => {
                let mut bs: Vec<AntContentBlock> = Vec::new();
                if !m.content.is_empty() {
                    bs.push(AntContentBlock::Text { text: m.content.clone() });
                }
                for tc in &m.tool_calls {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments_json).unwrap_or(serde_json::json!({}));
                    bs.push(AntContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input,
                    });
                }
                bs
            }
            "tool" => vec![AntContentBlock::ToolResult {
                tool_use_id: m.tool_call_id.clone().unwrap_or_default(),
                content: m.content.clone(),
            }],
            _ => continue,
        };
        let role = if m.role == "tool" { "user" } else { m.role.as_str() };
        ant_messages.push(AntMessage { role: role.to_string(), content: blocks });
    }

    let ant_tools: Vec<AntTool> = tools.iter().map(|t| AntTool {
        name: t.name.to_string(),
        description: t.description.to_string(),
        input_schema: serde_json::from_str(t.parameters_schema_json)
            .unwrap_or(serde_json::json!({})),
    }).collect();

    let req = AntRequest {
        model: model_for("anthropic"),
        max_tokens: max_output_tokens,
        temperature,
        system: system_text,
        messages: ant_messages,
        tools: ant_tools,
    };

    let client = reqwest::Client::new();
    let resp = client.post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&req)
        .send()
        .await
        .map_err(|e| format!("network: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("HTTP {}: {}", status, body));
    }

    let body: AntResponse = resp.json().await
        .map_err(|e| format!("parse: {}", e))?;

    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in body.content {
        match block {
            AntRespBlock::Text { text } => text_content.push_str(&text),
            AntRespBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    name,
                    arguments_json: serde_json::to_string(&input).unwrap_or_default(),
                });
            }
        }
    }

    let assistant = ChatMessage {
        role: "assistant".to_string(),
        content: text_content,
        tool_calls,
        tool_call_id: None,
    };

    Ok(ChatResult {
        response: assistant,
        input_tokens: body.usage.input_tokens,
        output_tokens: body.usage.output_tokens,
    })
}

async fn chat_call(
    provider: &str,
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
    temperature: f64,
) -> Result<ChatResult, String> {
    match provider {
        "openai" => openai_call(api_key, messages, tools, max_output_tokens, temperature).await,
        "anthropic" => anthropic_call(api_key, messages, tools, max_output_tokens, temperature).await,
        _ => Err(format!("unknown provider: {}", provider)),
    }
}

// =============================================================================
// LANG-001 workload (same content as tc_live_harness.rs::workload_lang001)
// =============================================================================

fn lang001_system() -> &'static str {
    "You are a database assistant. Use the sql_query tool. \
     The users table has columns: id (int), name (text), email (text). \
     If the tool errors, fix the SQL and retry."
}

fn lang001_user() -> &'static str {
    "Find user with id=1 in the users table."
}

fn lang001_tools() -> Vec<ToolDef> {
    vec![ToolDef {
        name: "sql_query",
        description: "Run a SQL query against the users table and return the result.",
        parameters_schema_json: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#,
    }]
}

fn lang001_error_for(_tc: &ToolCall) -> String {
    "Error: SQL syntax error near 'FRO': invalid keyword. \
     Did you mean 'FROM'? Please fix the query and retry."
        .to_string()
}

// =============================================================================
// One child agent: LANG-001 retry loop against its sub-budget.
// Consumes Budget by value (affine); returns ChildOutcome.
// =============================================================================

#[derive(Debug, Clone)]
struct ChildOutcome {
    child_id: usize,
    spent_uc: u64,        // provider-billed total spend
    residual_uc: u64,     // initial_cap_uc - spent_uc (saturating)
    initial_cap_uc: u64,
    calls_attempted: u32,
    calls_admitted: u32,
    outcome: String,
    wall_clock_ms: u64,
}

async fn run_lang001_child(
    provider: String,
    api_key: String,
    mut budget: Budget,
    initial_cap_uc: u64,
    child_id: usize,
    temperature: f64,
) -> ChildOutcome {
    let pricing = pricing_for(&provider);
    let tools = lang001_tools();
    let mut messages: Vec<ChatMessage> = vec![
        ChatMessage {
            role: "system".to_string(),
            content: lang001_system().to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: lang001_user().to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        },
    ];

    let t0 = Instant::now();
    let mut calls_attempted: u32 = 0;
    let mut calls_admitted: u32 = 0;
    let mut total_spent_uc: u64 = 0;
    let mut outcome = "completed_no_cap_hit".to_string();

    for _ in 0..MAX_AGENT_STEPS {
        calls_attempted += 1;

        let est_input_bytes = estimate_input_bytes(&messages, &tools);
        let est_uc = pricing.cost_uc(est_input_bytes, MAX_OUTPUT_TOKENS);

        // Affine spend: budget consumed by value, residual returned.
        budget = match budget.spend(est_uc, || ()) {
            Err(_) => {
                outcome = "compile_time_reservation_refused".to_string();
                break;
            }
            Ok((b, _)) => b,
        };
        calls_admitted += 1;

        let result = match chat_call(
            &provider, &api_key, &messages, &tools, MAX_OUTPUT_TOKENS, temperature,
        ).await {
            Ok(r) => r,
            Err(e) => {
                outcome = format!("api_error_{}", e.chars().take(30).collect::<String>());
                break;
            }
        };

        let actual_uc = pricing.cost_uc(result.input_tokens, result.output_tokens);
        total_spent_uc = total_spent_uc.saturating_add(actual_uc);

        let assistant = result.response;
        if assistant.tool_calls.is_empty() {
            outcome = "completed_no_cap_hit".to_string();
            messages.push(assistant);
            break;
        }

        let tool_calls = assistant.tool_calls.clone();
        messages.push(assistant);
        for tc in &tool_calls {
            messages.push(ChatMessage {
                role: "tool".to_string(),
                content: lang001_error_for(tc),
                tool_calls: vec![],
                tool_call_id: Some(tc.id.clone()),
            });
        }

        // Silence unused-variable warning for child_id; the value is
        // recorded in the outcome at the end.
        let _ = child_id;
    }

    if calls_admitted >= MAX_AGENT_STEPS as u32 && outcome == "completed_no_cap_hit" {
        outcome = "max_agent_steps_reached".to_string();
    }

    // budget falls out of scope here: on the Ok path the affine Drop
    // logs the residual; on the Err path the value was already moved
    // into spend() and there's nothing to drop. We report spent_uc =
    // total_spent_uc (provider-billed actual) and residual_uc =
    // initial_cap_uc - spent_uc (saturating).

    ChildOutcome {
        child_id,
        spent_uc: total_spent_uc,
        residual_uc: initial_cap_uc.saturating_sub(total_spent_uc),
        initial_cap_uc,
        calls_attempted,
        calls_admitted,
        outcome,
        wall_clock_ms: t0.elapsed().as_millis() as u64,
    }
}

// =============================================================================
// One trial: split B0 into three children, run concurrently, aggregate.
// =============================================================================

#[derive(Debug)]
struct TrialOutcome {
    trial_id: usize,
    initial_b0_uc: u64,
    children: [ChildOutcome; 3],
    aggregate_spent_uc: u64,
    aggregate_overshoot: bool,
}

async fn run_one_trial(
    trial_id: usize,
    provider: String,
    api_key: String,
    child_cap_uc: u64,
    temperature: f64,
) -> Result<TrialOutcome, String> {
    let b0_uc = child_cap_uc.saturating_mul(3);
    let parent = Budget::new(b0_uc);

    // Budget::split(take) returns (kept, taken):
    //   parent (3C).split(C) -> (kept=2C, taken=C)
    //   rest   (2C).split(C) -> (kept=C, taken=C)
    // After both splits we have three balanced C-uc children.
    let (rest1, child0_budget) = parent.split(child_cap_uc)
        .map_err(|e| format!("parent split: {:?}", e))?;
    let (rest2, child1_budget) = rest1.split(child_cap_uc)
        .map_err(|e| format!("rest1 split: {:?}", e))?;
    let child2_budget = rest2;

    let p0 = provider.clone();
    let p1 = provider.clone();
    let p2 = provider.clone();
    let k0 = api_key.clone();
    let k1 = api_key.clone();
    let k2 = api_key.clone();

    let h0 = tokio::spawn(run_lang001_child(p0, k0, child0_budget, child_cap_uc, 0, temperature));
    let h1 = tokio::spawn(run_lang001_child(p1, k1, child1_budget, child_cap_uc, 1, temperature));
    let h2 = tokio::spawn(run_lang001_child(p2, k2, child2_budget, child_cap_uc, 2, temperature));

    let r0 = h0.await.map_err(|e| format!("child0 join: {}", e))?;
    let r1 = h1.await.map_err(|e| format!("child1 join: {}", e))?;
    let r2 = h2.await.map_err(|e| format!("child2 join: {}", e))?;

    let aggregate_spent_uc = r0.spent_uc + r1.spent_uc + r2.spent_uc;
    let aggregate_overshoot = aggregate_spent_uc > b0_uc;

    let _ = trial_id;

    Ok(TrialOutcome {
        trial_id,
        initial_b0_uc: b0_uc,
        children: [r0, r1, r2],
        aggregate_spent_uc,
        aggregate_overshoot,
    })
}

// =============================================================================
// CLI + CSV output
// =============================================================================

struct Args {
    provider: String,
    runs: u64,
    output_csv: String,
    child_cap_uc: u64,
    temperature: f64,
}

fn parse_args() -> Args {
    let mut provider = "openai".to_string();
    let mut runs: u64 = 30;
    let mut output_csv = "multiagent_lang001.csv".to_string();
    let mut child_cap_uc: u64 = DEFAULT_CHILD_CAP_UC;
    let mut temperature: f64 = 0.0;

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--provider" => { provider = argv[i+1].clone(); i += 2; }
            "--runs" => { runs = argv[i+1].parse().expect("--runs needs u64"); i += 2; }
            "--output-csv" => { output_csv = argv[i+1].clone(); i += 2; }
            "--child-cap-uc" => { child_cap_uc = argv[i+1].parse().expect("--child-cap-uc needs u64"); i += 2; }
            "--temperature" => { temperature = argv[i+1].parse().expect("--temperature needs f64"); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: multiagent_lang001 --provider {{openai|anthropic}} \\");
                eprintln!("       --runs N --output-csv FILE [--child-cap-uc UC] [--temperature T]");
                eprintln!();
                eprintln!("Default --child-cap-uc 540 (OpenAI). For Anthropic use 2000.");
                eprintln!("Default --temperature 0.0 (deterministic). Try 0.7 for variance.");
                eprintln!("Parent B_0 = 3 * child-cap-uc.");
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(1);
            }
        }
    }
    Args { provider, runs, output_csv, child_cap_uc, temperature }
}

fn csv_header() -> String {
    "trial_id,provider,model,initial_b0_uc,\
     child0_spent_uc,child0_residual_uc,child0_calls_attempted,child0_calls_admitted,child0_outcome,child0_wall_ms,\
     child1_spent_uc,child1_residual_uc,child1_calls_attempted,child1_calls_admitted,child1_outcome,child1_wall_ms,\
     child2_spent_uc,child2_residual_uc,child2_calls_attempted,child2_calls_admitted,child2_outcome,child2_wall_ms,\
     aggregate_spent_uc,aggregate_overshoot".to_string()
}

fn csv_row(provider: &str, t: &TrialOutcome) -> String {
    let c = &t.children;
    format!(
        "{},{},{},{},\
         {},{},{},{},{},{},\
         {},{},{},{},{},{},\
         {},{},{},{},{},{},\
         {},{}",
        t.trial_id, provider, model_for(provider), t.initial_b0_uc,
        c[0].spent_uc, c[0].residual_uc, c[0].calls_attempted, c[0].calls_admitted,
            c[0].outcome, c[0].wall_clock_ms,
        c[1].spent_uc, c[1].residual_uc, c[1].calls_attempted, c[1].calls_admitted,
            c[1].outcome, c[1].wall_clock_ms,
        c[2].spent_uc, c[2].residual_uc, c[2].calls_attempted, c[2].calls_admitted,
            c[2].outcome, c[2].wall_clock_ms,
        t.aggregate_spent_uc, t.aggregate_overshoot,
    )
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();

    let api_key = match args.provider.as_str() {
        "openai" => std::env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY not set")?,
        "anthropic" => std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| "ANTHROPIC_API_KEY not set")?,
        other => {
            return Err(format!("unknown provider: {}", other).into());
        }
    };

    let mut f = File::create(&args.output_csv)?;
    writeln!(f, "{}", csv_header())?;

    let mut aggregate_overshoots: u64 = 0;
    let mut child_cap_violations: u64 = 0;
    let b0_uc = args.child_cap_uc.saturating_mul(3);

    eprintln!("Config: provider={} runs={} child_cap_uc={} B0_uc={}",
        args.provider, args.runs, args.child_cap_uc, b0_uc);

    for trial_id in 0..args.runs {
        eprintln!("=== trial {}/{} ({}) ===",
            trial_id + 1, args.runs, args.provider);
        let t = match run_one_trial(
            trial_id as usize, args.provider.clone(), api_key.clone(),
            args.child_cap_uc, args.temperature,
        ).await {
            Ok(t) => t,
            Err(e) => {
                eprintln!("  trial {} ERROR: {}", trial_id, e);
                continue;
            }
        };
        writeln!(f, "{}", csv_row(&args.provider, &t))?;
        f.flush()?;

        if t.aggregate_overshoot {
            aggregate_overshoots += 1;
        }
        for c in &t.children {
            if c.spent_uc > c.initial_cap_uc {
                child_cap_violations += 1;
            }
        }

        eprintln!(
            "  child spends: {} / {} / {} uc | aggregate: {} uc / {} | overshoot: {}",
            t.children[0].spent_uc, t.children[1].spent_uc, t.children[2].spent_uc,
            t.aggregate_spent_uc, t.initial_b0_uc, t.aggregate_overshoot,
        );
    }

    eprintln!();
    eprintln!("==== SUMMARY ({} child_cap={} B0={}) ====",
        args.provider, args.child_cap_uc, b0_uc);
    eprintln!("Trials:                  {}", args.runs);
    eprintln!("Aggregate overshoots:    {}/{}", aggregate_overshoots, args.runs);
    eprintln!("Per-child cap violations: {}", child_cap_violations);
    eprintln!("Output CSV:              {}", args.output_csv);

    Ok(())
}
