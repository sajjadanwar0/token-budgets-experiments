use std::env;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use budget_spike::Budget;

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

        "anthropic-sonnet" => Pricing {
            input_per_token_uc_per_million: 3_000_000,
            output_per_token_uc_per_million: 15_000_000,
        },
        "groq" => Pricing {
            input_per_token_uc_per_million: 590_000,
            output_per_token_uc_per_million: 790_000,
        },
        "ollama" => Pricing {
            input_per_token_uc_per_million: 50_000,
            output_per_token_uc_per_million: 100_000,
        },
        _ => panic!("unknown provider: {}", provider),
    }
}

fn model_for(provider: &str) -> &'static str {
    match provider {
        "openai" => "gpt-4o-mini",
        "anthropic" => "claude-haiku-4-5-20251001",
        "anthropic-sonnet" => "claude-sonnet-4-5-20250929",
        "groq" => "llama-3.3-70b-versatile",
        "ollama" => "llama3.2:latest",
        _ => panic!("unknown provider: {}", provider),
    }
}

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

async fn chat_call(
    provider: &str,
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
) -> Result<ChatResult, String> {
    match provider {
        "openai" | "groq" | "ollama" => {
            openai_compat_call(provider, api_key, messages, tools, max_output_tokens).await
        }
        "anthropic" => {
            anthropic_call_with_model(
                model_for("anthropic"),
                api_key, messages, tools, max_output_tokens
            ).await
        }
        "anthropic-sonnet" => {
            anthropic_call_with_model(
                model_for("anthropic-sonnet"),
                api_key, messages, tools, max_output_tokens
            ).await
        }
        _ => Err(format!("unknown provider: {}", provider)),
    }
}

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

async fn openai_compat_call(
    provider: &str,
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
) -> Result<ChatResult, String> {
    let url = match provider {
        "openai" => "https://api.openai.com/v1/chat/completions",
        "groq" => "https://api.groq.com/openai/v1/chat/completions",
        "ollama" => "http://localhost:11434/v1/chat/completions",
        _ => unreachable!(),
    };

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
        model: model_for(provider),
        messages: oai_messages,
        tools: oai_tools,
        temperature: 0.0,
        max_completion_tokens: max_output_tokens,
    };

    let client = reqwest::Client::new();
    let resp = client.post(url)
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
) -> Result<ChatResult, String> {
    anthropic_call_with_model(
        model_for("anthropic"),
        api_key, messages, tools, max_output_tokens
    ).await
}

async fn anthropic_call_with_model(
    model: &str,
    api_key: &str,
    messages: &[ChatMessage],
    tools: &[ToolDef],
    max_output_tokens: u64,
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
        model,
        max_tokens: max_output_tokens,
        temperature: 0.0,
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

struct Workload {
    name: &'static str,
    system: &'static str,
    user: &'static str,
    tools: Vec<ToolDef>,
    error_for: fn(&ToolCall) -> String,
}

fn workload_lang001() -> Workload {
    Workload {
        name: "lang001",
        system: "You are a database assistant. Use the sql_query tool. \
                 The users table has columns: id (int), name (text), email (text). \
                 If the tool errors, fix the SQL and retry.",
        user: "Find user with id=1 in the users table.",
        tools: vec![ToolDef {
            name: "sql_query",
            description: "Run a SQL query against the users table and return the result.",
            parameters_schema_json: r#"{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}"#,
        }],
        error_for: |_tc: &ToolCall| {
            "Error: SQL syntax error near 'FRO': invalid keyword. \
             Did you mean 'FROM'? Please fix the query and retry."
                .to_string()
        },
    }
}

fn workload_clarification_loop() -> Workload {
    Workload {
        name: "clarification",
        system: "You are a file management assistant. Use the delete_record tool \
                 to remove records. Ensure you delete the correct record.",
        user: "Please delete the report record.",
        tools: vec![ToolDef {
            name: "delete_record",
            description: "Delete a record. Accepts either an id or a name.",
            parameters_schema_json: r#"{"type":"object","properties":{"id":{"type":"string"},"name":{"type":"string"}}}"#,
        }],
        error_for: |_tc: &ToolCall| {
            "Error: ambiguous record. Multiple records match. \
             Please be more specific about which record to delete."
                .to_string()
        },
    }
}

fn workload_arg_hallucination() -> Workload {
    Workload {
        name: "arg_hallucination",
        system: "You are a customer support assistant. Use lookup_customer to find \
                 customer accounts and answer questions about them.",
        user: "Look up the account for the customer named 'Alice'.",
        tools: vec![ToolDef {
            name: "lookup_customer",
            description: "Look up a customer account by name.",
            parameters_schema_json: r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#,
        }],
        error_for: |_tc: &ToolCall| {
            "Error: customer 'Alice' not found. \
             Tip: confirm the spelling matches an active account, or try lookup_customer again with the same name."
                .to_string()
        },
    }
}

fn workload_for(name: &str) -> Workload {
    match name {
        "lang001" => workload_lang001(),
        "clarification" => workload_clarification_loop(),
        "arg_hallucination" => workload_arg_hallucination(),
        _ => panic!("unknown workload: {}", name),
    }
}

#[derive(Debug)]
struct RunRecord {
    outcome: String,
    agent_steps: u64,
    total_spent_uc: u64,
    cap_uc: u64,
    overshoot_uc: u64,
    wall_seconds: f64,
    sum_input_tokens: u64,
    sum_output_tokens: u64,
    sum_byte_length_estimate: u64,
    sum_reservation_uc: u64,
    sum_actual_cost_uc: u64,
}

const MAX_OUTPUT_TOKENS: u64 = 200;
const MAX_AGENT_STEPS: u64 = 20;

async fn run_tc_once(
    provider: &str,
    api_key: &str,
    workload: &Workload,
    cap_uc: u64,
) -> RunRecord {
    let pricing = pricing_for(provider);
    let mut budget = Budget::new(cap_uc);
    let mut messages: Vec<ChatMessage> = vec![
        ChatMessage {
            role: "system".to_string(),
            content: workload.system.to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        },
        ChatMessage {
            role: "user".to_string(),
            content: workload.user.to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        },
    ];

    let mut total_spent_uc: u64 = 0;
    let mut steps: u64 = 0;
    let mut outcome = "completed_no_cap_hit".to_string();
    // v32 detail accumulators
    let mut sum_input_tokens: u64 = 0;
    let mut sum_output_tokens: u64 = 0;
    let mut sum_byte_length_estimate: u64 = 0;
    let mut sum_reservation_uc: u64 = 0;
    let mut sum_actual_cost_uc: u64 = 0;
    let t0 = Instant::now();

    for _ in 0..MAX_AGENT_STEPS {
        let est_input_bytes = estimate_input_bytes(&messages, &workload.tools);
        let est_uc = pricing.cost_uc(est_input_bytes, MAX_OUTPUT_TOKENS);

        budget = match budget.spend(est_uc, || ()) {
            Err(_) => {
                outcome = "compile_time_reservation_refused".to_string();
                break;
            }
            Ok((b, _)) => b,
        };

        let result = match chat_call(provider, api_key, &messages, &workload.tools, MAX_OUTPUT_TOKENS).await {
            Ok(r) => r,
            Err(e) => {
                outcome = format!("api_error_{}", e.chars().take(30).collect::<String>());
                break;
            }
        };

        sum_byte_length_estimate = sum_byte_length_estimate.saturating_add(est_input_bytes);
        sum_reservation_uc = sum_reservation_uc.saturating_add(est_uc);
        sum_input_tokens = sum_input_tokens.saturating_add(result.input_tokens);
        sum_output_tokens = sum_output_tokens.saturating_add(result.output_tokens);

        let actual_uc = pricing.cost_uc(result.input_tokens, result.output_tokens);
        total_spent_uc = total_spent_uc.saturating_add(actual_uc);
        sum_actual_cost_uc = sum_actual_cost_uc.saturating_add(actual_uc);
        steps += 1;

        let assistant = result.response;

        if assistant.tool_calls.is_empty() {
            outcome = "completed_no_cap_hit".to_string();
            messages.push(assistant);
            break;
        }

        let tool_calls = assistant.tool_calls.clone();
        messages.push(assistant);
        for tc in &tool_calls {
            let err_text = (workload.error_for)(tc);
            messages.push(ChatMessage {
                role: "tool".to_string(),
                content: err_text,
                tool_calls: vec![],
                tool_call_id: Some(tc.id.clone()),
            });
        }
    }

    if steps >= MAX_AGENT_STEPS && outcome == "completed_no_cap_hit" {
        outcome = "max_agent_steps_reached".to_string();
    }

    let wall_seconds = t0.elapsed().as_secs_f64();
    let overshoot_uc = total_spent_uc.saturating_sub(cap_uc);

    RunRecord {
        outcome,
        agent_steps: steps,
        total_spent_uc,
        cap_uc,
        overshoot_uc,
        wall_seconds,
        sum_input_tokens,
        sum_output_tokens,
        sum_byte_length_estimate,
        sum_reservation_uc,
        sum_actual_cost_uc,
    }
}

struct Args {
    provider: String,
    workload: String,
    runs: u64,
    cap_uc: u64,
    output_csv: String,
}

fn parse_args() -> Args {
    let mut provider = "openai".to_string();
    let mut workload = "lang001".to_string();
    let mut runs: u64 = 10;
    let mut cap_uc: u64 = 540;
    let mut output_csv = "tc_rust.csv".to_string();

    let argv: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--provider" => { provider = argv[i+1].clone(); i += 2; }
            "--workload" => { workload = argv[i+1].clone(); i += 2; }
            "--runs" => { runs = argv[i+1].parse().expect("--runs needs u64"); i += 2; }
            "--cap-uc" => { cap_uc = argv[i+1].parse().expect("--cap-uc needs u64"); i += 2; }
            "--output-csv" => { output_csv = argv[i+1].clone(); i += 2; }
            "--help" | "-h" => {
                eprintln!("Usage: tc_live_harness --provider {{openai|anthropic|anthropic-sonnet|groq|ollama}} \
                          --workload {{lang001|clarification|arg_hallucination}} \
                          --runs N --cap-uc UC --output-csv FILE");
                std::process::exit(0);
            }
            other => { eprintln!("unknown arg: {}", other); std::process::exit(1); }
        }
    }
    Args { provider, workload, runs, cap_uc, output_csv }
}

fn api_key_for(provider: &str) -> Result<String, String> {
    if provider == "ollama" {
        return Ok("ollama".to_string());
    }
    let var = match provider {
        "openai" => "OPENAI_API_KEY",
        "anthropic" => "ANTHROPIC_API_KEY",
        "anthropic-sonnet" => "ANTHROPIC_API_KEY",
        "groq" => "GROQ_API_KEY",
        _ => return Err(format!("unknown provider: {}", provider)),
    };
    env::var(var).map_err(|_| format!("env var {} not set", var))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let api_key = api_key_for(&args.provider)?;
    let workload = workload_for(&args.workload);

    println!(
        "tc_live_harness | provider={} | workload={} | cap={}uc | runs={}",
        args.provider, workload.name, args.cap_uc, args.runs
    );

    let mut rows: Vec<(u64, RunRecord)> = Vec::new();
    for run_id in 1..=args.runs {
        let rec = run_tc_once(&args.provider, &api_key, &workload, args.cap_uc).await;
        let bt_ratio = if rec.sum_input_tokens > 0 {
            rec.sum_byte_length_estimate as f64 / rec.sum_input_tokens as f64
        } else {
            0.0
        };
        println!(
            "  run {}: outcome={}, steps={}, spend={}uc ({:.1}% of cap), in_tok={}, out_tok={}, bl={}, B/T={:.2}x, wall={:.2}s",
            run_id,
            rec.outcome,
            rec.agent_steps,
            rec.total_spent_uc,
            (rec.total_spent_uc as f64 / args.cap_uc as f64) * 100.0,
            rec.sum_input_tokens,
            rec.sum_output_tokens,
            rec.sum_byte_length_estimate,
            bt_ratio,
            rec.wall_seconds
        );
        rows.push((run_id, rec));
    }

    let mut f = File::create(&args.output_csv)?;
    writeln!(
        f,
        "runtime,run_id,provider,outcome,agent_steps,cap_uc,total_spent_uc,\
         pct_of_cap,overshoot_uc,structural_undershoot_uc,wasted_call_cost_uc,\
         wall_seconds,workload,actual_input_tokens,actual_output_tokens,\
         byte_length_estimate,reservation_uc,actual_cost_uc"
    )?;
    for (run_id, rec) in &rows {
        let pct = (rec.total_spent_uc as f64 / args.cap_uc as f64) * 100.0;
        writeln!(
            f,
            "token_capabilities_rust,{},{},{},{},{},{},{:.2},{},0,0,{:.3},\
             {},{},{},{},{},{}",
            run_id,
            args.provider,
            rec.outcome,
            rec.agent_steps,
            rec.cap_uc,
            rec.total_spent_uc,
            pct,
            rec.overshoot_uc,
            rec.wall_seconds,
            args.workload,
            rec.sum_input_tokens,
            rec.sum_output_tokens,
            rec.sum_byte_length_estimate,
            rec.sum_reservation_uc,
            rec.sum_actual_cost_uc,
        )?;
    }
    println!("wrote {} rows to {}", rows.len(), args.output_csv);
    Ok(())
}