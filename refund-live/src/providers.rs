//! Provider adapters: Gemini, vLLM, and existing Anthropic/OpenAI.
//! Closes the "three-provider only" critique.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
pub enum Provider {
    Anthropic { model: &'static str },
    OpenAI { model: &'static str },
    Gemini { model: &'static str },
    VLLM { endpoint: &'static str, model: &'static str },
}

/// Per-token rates in micro-cents (1 uc = $0.000001).
#[derive(Debug, Clone, Copy)]
pub struct Rates {
    pub per_in_token_uc: u64,
    pub per_out_token_uc: u64,
}

impl Provider {
    pub fn rates(&self) -> Rates {
        match self {
            Provider::Anthropic { model } => match *model {
                "claude-haiku-4-5-20251001" => Rates { per_in_token_uc: 1, per_out_token_uc: 5 },
                "claude-sonnet-4-6" => Rates { per_in_token_uc: 3, per_out_token_uc: 15 },
                _ => Rates { per_in_token_uc: 3, per_out_token_uc: 15 },
            },
            Provider::OpenAI { model } => match *model {
                "gpt-4o-mini" => Rates { per_in_token_uc: 0, per_out_token_uc: 1 }, // ~$0.15/M, ~$0.60/M
                "gpt-4o" => Rates { per_in_token_uc: 2, per_out_token_uc: 10 },
                _ => Rates { per_in_token_uc: 1, per_out_token_uc: 5 },
            },
            Provider::Gemini { model } => match *model {
                "gemini-2.0-flash" => Rates { per_in_token_uc: 0, per_out_token_uc: 0 }, // approx
                "gemini-2.5-pro" => Rates { per_in_token_uc: 1, per_out_token_uc: 5 },
                _ => Rates { per_in_token_uc: 1, per_out_token_uc: 5 },
            },
            Provider::VLLM { .. } => Rates { per_in_token_uc: 0, per_out_token_uc: 0 }, // self-hosted
        }
    }

    pub fn endpoint(&self) -> String {
        match self {
            Provider::Anthropic { .. } => "https://api.anthropic.com/v1/messages".to_string(),
            Provider::OpenAI { .. } => "https://api.openai.com/v1/chat/completions".to_string(),
            Provider::Gemini { model } => format!(
                "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent",
                model
            ),
            Provider::VLLM { endpoint, .. } => format!("{}/v1/chat/completions", endpoint),
        }
    }

    pub fn auth_header(&self) -> &'static str {
        match self {
            Provider::Anthropic { .. } => "x-api-key",
            Provider::OpenAI { .. } => "Authorization",
            Provider::Gemini { .. } => "x-goog-api-key",
            Provider::VLLM { .. } => "Authorization",
        }
    }

    pub fn build_request(&self, prompt: &str, max_tokens: u32) -> serde_json::Value {
        match self {
            Provider::Anthropic { model } => serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }),
            Provider::OpenAI { model } | Provider::VLLM { model, .. } => serde_json::json!({
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }),
            Provider::Gemini { .. } => serde_json::json!({
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }),
        }
    }

    /// Parse provider response and extract (input_tokens, output_tokens).
    pub fn parse_usage(&self, response: &serde_json::Value) -> Option<(u64, u64)> {
        match self {
            Provider::Anthropic { .. } => {
                let usage = response.get("usage")?;
                Some((
                    usage.get("input_tokens")?.as_u64()?,
                    usage.get("output_tokens")?.as_u64()?,
                ))
            }
            Provider::OpenAI { .. } | Provider::VLLM { .. } => {
                let usage = response.get("usage")?;
                Some((
                    usage.get("prompt_tokens")?.as_u64()?,
                    usage.get("completion_tokens")?.as_u64()?,
                ))
            }
            Provider::Gemini { .. } => {
                let meta = response.get("usageMetadata")?;
                Some((
                    meta.get("promptTokenCount")?.as_u64()?,
                    meta.get("candidatesTokenCount")?.as_u64()?,
                ))
            }
        }
    }
}
