//! Provider-agnostic LLM client abstraction.
//!
//! ## Pricing-resolution note
//!
//! The legacy trait methods `input_price_per_token_mc` and
//! `output_price_per_token_mc` return integer micro-cents (uc) per token,
//! where 1 uc = 10⁻⁶ USD. Some real provider prices are sub-uc per token
//! (e.g. gpt-4o-mini at $0.15/Mtok = 0.15 uc/token), and u64 integer
//! arithmetic cannot represent fractional uc.
//!
//! To preserve soundness under the legacy accessors, the per-token methods
//! return the **ceiling** of the real per-token rate. For gpt-4o-mini this
//! means 1 uc/token for input (a 6.67× over-estimate of the real
//! 0.15 uc/token rate) and 1 uc/token for output (1.67× over). This
//! over-estimation is conservative for the cap-respecting claim
//! (reservation ≥ real cost) but inflates the smoke-test dollar figures
//! relative to real provider bills.
//!
//! For accurate cost accounting on `actual_cost_micro_cents`, the
//! per-Mtok `Pricing` struct (returned by `pricing()`) holds the real
//! per-million-token integer rates and computes actual cost at full
//! precision via integer math scaled by 1,000,000. Clients use `Pricing`
//! to compute the `actual_cost_micro_cents` field of the response, so
//! that downstream observers see real-dollar numbers rather than the
//! ceiling-rounded values.
//!
//! `tc_live_harness.rs` uses the same `Pricing` representation, so the
//! harness and the smoke test now agree on cost arithmetic.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// =============================================================================
// Errors and response type
// =============================================================================

#[derive(Debug)]
pub enum LLMError {
    Network(String),
    Api(String),
    Parse(String),
    MissingApiKey(&'static str),
}

impl std::fmt::Display for LLMError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LLMError::Network(m) => write!(f, "network error: {}", m),
            LLMError::Api(m) => write!(f, "API error: {}", m),
            LLMError::Parse(m) => write!(f, "parse error: {}", m),
            LLMError::MissingApiKey(name) => write!(f, "missing env var: {}", name),
        }
    }
}

impl std::error::Error for LLMError {}

#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub content: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Provider-billed cost computed via `Pricing::cost_uc`, in
    /// micro-cents (uc, where 1 uc = 10⁻⁶ USD). Accurate to the
    /// nearest uc; sub-uc fractions are rounded half-up.
    pub actual_cost_micro_cents: u64,
}

// =============================================================================
// Per-Mtok integer pricing (single source of truth)
// =============================================================================

/// Provider pricing expressed as micro-cents per million tokens.
/// 1 uc = 10⁻⁶ USD, so 1 uc/Mtok = 10⁻⁶ USD per million tokens =
/// $0.000001 / 1,000,000 tokens. A real price like
/// gpt-4o-mini input = $0.15/Mtok = 150,000 uc/Mtok.
///
/// Stored as u64 per million tokens so fractional uc/token rates
/// (e.g. 0.15 uc/token for gpt-4o-mini) become integer 150,000 uc/Mtok.
/// `cost_uc()` divides by 1,000,000 at the end (half-up rounding) to
/// give an integer uc cost; intermediate computation uses saturating
/// arithmetic to ensure overflow is impossible under Assumption A2.
#[derive(Clone, Copy, Debug)]
pub struct Pricing {
    pub input_per_mtok_uc: u64,
    pub output_per_mtok_uc: u64,
}

impl Pricing {
    /// Compute the cost of a call in micro-cents (uc).
    ///
    /// Uses saturating arithmetic so overflow returns `u64::MAX`
    /// rather than wrapping. Half-up rounding on the final division
    /// by 1,000,000 matches the Python harness's `round()` behaviour
    /// and the `tc_live_harness.rs` Rust implementation.
    pub fn cost_uc(&self, input_tokens: u64, output_tokens: u64) -> u64 {
        let in_part = input_tokens.saturating_mul(self.input_per_mtok_uc);
        let out_part = output_tokens.saturating_mul(self.output_per_mtok_uc);
        let total = in_part.saturating_add(out_part);
        // half-up: (x + 500_000) / 1_000_000 for positive x.
        (total.saturating_add(500_000)) / 1_000_000
    }

    /// Ceiling of the per-token input rate in uc/token. Used by the
    /// legacy `input_price_per_token_mc` accessor to provide a
    /// conservative-upper-bound estimator for reservation arithmetic.
    /// Always at least 1 for non-zero `input_per_mtok_uc`.
    pub fn input_per_token_uc_ceil(&self) -> u64 {
        if self.input_per_mtok_uc == 0 {
            0
        } else {
            // ceil_div(input_per_mtok_uc, 1_000_000) — at least 1.
            (self.input_per_mtok_uc + 999_999) / 1_000_000
        }
    }

    /// Ceiling of the per-token output rate in uc/token. Used by the
    /// legacy `output_price_per_token_mc` accessor.
    pub fn output_per_token_uc_ceil(&self) -> u64 {
        if self.output_per_mtok_uc == 0 {
            0
        } else {
            (self.output_per_mtok_uc + 999_999) / 1_000_000
        }
    }
}

// =============================================================================
// LLMClient trait
// =============================================================================

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(
        &self,
        prompt: &str,
        max_output_tokens: u64,
    ) -> Result<CompletionResponse, LLMError>;

    /// Real per-Mtok pricing. Concrete implementations should override
    /// with the provider's published per-million-token rates. The
    /// default falls back to synthesising a `Pricing` from the legacy
    /// per-token rates (which loses sub-uc precision but stays
    /// conservative).
    fn pricing(&self) -> Pricing {
        Pricing {
            input_per_mtok_uc: self
                .input_price_per_token_mc()
                .saturating_mul(1_000_000),
            output_per_mtok_uc: self
                .output_price_per_token_mc()
                .saturating_mul(1_000_000),
        }
    }

    /// Legacy: per-token input rate in uc (ceiling-rounded for soundness).
    /// Reservations computed via `estimate_cost_split` use this accessor.
    /// New implementations should override `pricing()` instead; this
    /// method has a default that derives from `pricing()`.
    fn input_price_per_token_mc(&self) -> u64 {
        self.pricing().input_per_token_uc_ceil()
    }

    /// Legacy: per-token output rate in uc (ceiling-rounded for soundness).
    fn output_price_per_token_mc(&self) -> u64 {
        self.pricing().output_per_token_uc_ceil()
    }

    /// Conservative-upper-bound input-token estimator: returns the
    /// UTF-8 byte length of the prompt. Sound for every BPE-family
    /// tokenizer surveyed in the paper's §V-J Construct Validity
    /// section; see `crate::tokenizer::TiktokenEstimator` for a tighter
    /// provider-specific bound.
    fn estimate_input_tokens(&self, prompt: &str) -> u64 {
        prompt.len() as u64
    }
}

// =============================================================================
// Mock client (used by demo_async_mock and unit tests)
// =============================================================================

pub struct MockClient {
    pub fixed_input_tokens: u64,
    pub fixed_output_tokens: u64,
    pub pricing: Pricing,
    pub simulated_latency_ms: u64,
}

impl MockClient {
    /// Claude-Sonnet-like pricing: $3/Mtok input, $15/Mtok output.
    pub fn sonnet_like() -> Self {
        Self {
            fixed_input_tokens: 50,
            fixed_output_tokens: 200,
            pricing: Pricing {
                input_per_mtok_uc: 3_000_000,
                output_per_mtok_uc: 15_000_000,
            },
            simulated_latency_ms: 50,
        }
    }
}

#[async_trait]
impl LLMClient for MockClient {
    async fn complete(
        &self,
        prompt: &str,
        _max_output_tokens: u64,
    ) -> Result<CompletionResponse, LLMError> {
        tokio::time::sleep(std::time::Duration::from_millis(self.simulated_latency_ms))
            .await;
        let actual = self
            .pricing
            .cost_uc(self.fixed_input_tokens, self.fixed_output_tokens);
        Ok(CompletionResponse {
            content: format!(
                "[mock response to: {}]",
                prompt.chars().take(40).collect::<String>()
            ),
            input_tokens: self.fixed_input_tokens,
            output_tokens: self.fixed_output_tokens,
            actual_cost_micro_cents: actual,
        })
    }

    fn pricing(&self) -> Pricing {
        self.pricing
    }
}

// =============================================================================
// Anthropic client
// =============================================================================

pub struct AnthropicClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u64,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
    usage: AnthropicUsage,
}

#[derive(Deserialize, Debug)]
struct AnthropicContent {
    #[serde(rename = "type")]
    #[allow(dead_code)]
    content_type: String,
    text: Option<String>,
}

#[derive(Deserialize, Debug)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
}

impl AnthropicClient {
    pub fn from_env(model: &str) -> Result<Self, LLMError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| LLMError::MissingApiKey("ANTHROPIC_API_KEY"))?;
        Ok(Self {
            api_key,
            model: model.to_string(),
            http: reqwest::Client::new(),
        })
    }

    /// Real per-Mtok pricing for the model in use, as published by
    /// Anthropic. Family-level matching on the model string; production
    /// deployments should pin exact snapshots.
    fn published_pricing(&self) -> Pricing {
        let m = self.model.as_str();
        if m.contains("opus") {
            // $15/Mtok input, $75/Mtok output (Claude Opus 4.x).
            Pricing { input_per_mtok_uc: 15_000_000, output_per_mtok_uc: 75_000_000 }
        } else if m.contains("haiku") {
            // $1/Mtok input, $5/Mtok output (Claude Haiku 4.5).
            Pricing { input_per_mtok_uc: 1_000_000, output_per_mtok_uc: 5_000_000 }
        } else {
            // Default Sonnet: $3/Mtok input, $15/Mtok output.
            Pricing { input_per_mtok_uc: 3_000_000, output_per_mtok_uc: 15_000_000 }
        }
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn complete(
        &self,
        prompt: &str,
        max_output_tokens: u64,
    ) -> Result<CompletionResponse, LLMError> {
        let req = AnthropicRequest {
            model: &self.model,
            max_tokens: max_output_tokens,
            messages: vec![AnthropicMessage { role: "user", content: prompt }],
        };
        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&req)
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::Api(format!("HTTP {}: {}", status, body)));
        }

        let body: AnthropicResponse = resp
            .json()
            .await
            .map_err(|e| LLMError::Parse(e.to_string()))?;

        let content = body
            .content
            .into_iter()
            .filter_map(|c| c.text)
            .collect::<Vec<_>>()
            .join("\n");

        // Use real per-Mtok pricing so actual_cost_micro_cents matches
        // the real provider bill (modulo half-up rounding to the nearest uc).
        let actual_cost = self
            .published_pricing()
            .cost_uc(body.usage.input_tokens, body.usage.output_tokens);

        Ok(CompletionResponse {
            content,
            input_tokens: body.usage.input_tokens,
            output_tokens: body.usage.output_tokens,
            actual_cost_micro_cents: actual_cost,
        })
    }

    fn pricing(&self) -> Pricing {
        self.published_pricing()
    }
}

// =============================================================================
// OpenAI client
// =============================================================================

pub struct OpenAIClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

#[derive(Serialize)]
struct OpenAIRequest<'a> {
    model: &'a str,
    max_completion_tokens: u64,
    messages: Vec<OpenAIMessage<'a>>,
}

#[derive(Serialize)]
struct OpenAIMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Deserialize, Debug)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoice {
    message: OpenAIChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct OpenAIChoiceMessage {
    content: Option<String>,
}

#[derive(Deserialize, Debug)]
struct OpenAIUsage {
    prompt_tokens: u64,
    completion_tokens: u64,
}

impl OpenAIClient {
    pub fn from_env(model: &str) -> Result<Self, LLMError> {
        let api_key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| LLMError::MissingApiKey("OPENAI_API_KEY"))?;
        Ok(Self {
            api_key,
            model: model.to_string(),
            http: reqwest::Client::new(),
        })
    }

    /// Real per-Mtok pricing as published by OpenAI. Family-level
    /// matching on the model string. The legacy per-token rates that
    /// the `LLMClient` default implementation derives from these are
    /// ceiling-rounded, so reservations computed via `estimate_cost_split`
    /// will over-reserve relative to the real per-Mtok rate. Use
    /// `pricing().cost_uc()` for accurate cost arithmetic.
    fn published_pricing(&self) -> Pricing {
        let m = self.model.as_str();
        if m.starts_with("gpt-5") {
            // gpt-5 family pricing (placeholder until officially announced).
            Pricing { input_per_mtok_uc: 5_000_000, output_per_mtok_uc: 15_000_000 }
        } else if m.starts_with("gpt-4o-mini") {
            // gpt-4o-mini: $0.15/Mtok input, $0.60/Mtok output (corrected).
            // Legacy accessors return 1 uc/token (ceiling) for both, giving
            // a 6.67× over-reservation on input and 1.67× on output. The
            // per-Mtok rate here is the one used to compute actual_cost.
            Pricing { input_per_mtok_uc: 150_000, output_per_mtok_uc: 600_000 }
        } else if m.starts_with("gpt-4o") {
            // gpt-4o (non-mini): $2.50/Mtok input, $10/Mtok output.
            Pricing { input_per_mtok_uc: 2_500_000, output_per_mtok_uc: 10_000_000 }
        } else if m.starts_with("gpt-4-turbo") {
            // gpt-4-turbo: $10/Mtok input, $30/Mtok output.
            Pricing { input_per_mtok_uc: 10_000_000, output_per_mtok_uc: 30_000_000 }
        } else if m.starts_with("gpt-4") {
            // gpt-4 (8k context): $30/Mtok input, $60/Mtok output.
            Pricing { input_per_mtok_uc: 30_000_000, output_per_mtok_uc: 60_000_000 }
        } else {
            // gpt-3.5-turbo fallback: $0.50/Mtok input, $1.50/Mtok output.
            Pricing { input_per_mtok_uc: 500_000, output_per_mtok_uc: 1_500_000 }
        }
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn complete(
        &self,
        prompt: &str,
        max_output_tokens: u64,
    ) -> Result<CompletionResponse, LLMError> {
        let req = OpenAIRequest {
            model: &self.model,
            max_completion_tokens: max_output_tokens,
            messages: vec![OpenAIMessage { role: "user", content: prompt }],
        };
        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .map_err(|e| LLMError::Network(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(LLMError::Api(format!("HTTP {}: {}", status, body)));
        }

        let body: OpenAIResponse = resp
            .json()
            .await
            .map_err(|e| LLMError::Parse(e.to_string()))?;

        let content = body
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .unwrap_or_default();

        // Use real per-Mtok pricing so actual_cost_micro_cents matches
        // the real OpenAI bill (modulo half-up rounding to the nearest uc).
        let actual_cost = self
            .published_pricing()
            .cost_uc(body.usage.prompt_tokens, body.usage.completion_tokens);

        Ok(CompletionResponse {
            content,
            input_tokens: body.usage.prompt_tokens,
            output_tokens: body.usage.completion_tokens,
            actual_cost_micro_cents: actual_cost,
        })
    }

    fn pricing(&self) -> Pricing {
        self.published_pricing()
    }
}

// =============================================================================
// Tests: pricing arithmetic and soundness of the ceiling-rounded fallback
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_cost_uc_gpt_4o_mini_typical_call() {
        // 1,000 input tokens + 200 output tokens at gpt-4o-mini rates.
        // Real cost: 1000 * 0.15/Mtok + 200 * 0.60/Mtok
        //          = 0.00015 + 0.00012
        //          = $0.00027 = 270 uc.
        let p = Pricing { input_per_mtok_uc: 150_000, output_per_mtok_uc: 600_000 };
        assert_eq!(p.cost_uc(1000, 200), 270);
    }

    #[test]
    fn pricing_cost_uc_anthropic_haiku() {
        // 1,000 input + 200 output at $1/$5 per Mtok.
        // Real cost: 1000 + 1000 = 2000 uc.
        let p = Pricing { input_per_mtok_uc: 1_000_000, output_per_mtok_uc: 5_000_000 };
        assert_eq!(p.cost_uc(1000, 200), 2000);
    }

    #[test]
    fn pricing_per_token_ceiling_is_conservative() {
        // gpt-4o-mini: 0.15 uc/token real → ceiling = 1 uc/token.
        // Reservation arithmetic over 1000 tokens via the legacy accessor:
        //   1000 * 1 = 1000 uc.
        // Real cost over 1000 tokens:
        //   1000 * 150,000 / 1,000,000 = 150 uc.
        // Reservation ≥ real cost: 1000 ≥ 150. Sound.
        let p = Pricing { input_per_mtok_uc: 150_000, output_per_mtok_uc: 600_000 };
        let ceil = p.input_per_token_uc_ceil();
        let n = 1000u64;
        assert!(n.saturating_mul(ceil) >= p.cost_uc(n, 0));
    }

    #[test]
    fn pricing_per_token_ceiling_at_least_one_for_nonzero_rate() {
        let p = Pricing { input_per_mtok_uc: 1, output_per_mtok_uc: 1 };
        assert_eq!(p.input_per_token_uc_ceil(), 1);
        assert_eq!(p.output_per_token_uc_ceil(), 1);
    }

    #[test]
    fn pricing_per_token_ceiling_zero_when_rate_zero() {
        let p = Pricing { input_per_mtok_uc: 0, output_per_mtok_uc: 0 };
        assert_eq!(p.input_per_token_uc_ceil(), 0);
        assert_eq!(p.output_per_token_uc_ceil(), 0);
    }

    #[test]
    fn cost_uc_saturating_under_overflow() {
        // Verify saturating arithmetic does not panic on huge inputs.
        let p = Pricing { input_per_mtok_uc: u64::MAX, output_per_mtok_uc: u64::MAX };
        let _ = p.cost_uc(u64::MAX, u64::MAX); // should not panic
    }

    #[test]
    fn mock_client_sonnet_pricing_matches_documented_rates() {
        // Sonnet $3/$15 per Mtok. 50 input + 200 output tokens.
        // Real cost: 50 * 3 + 200 * 15 = 150 + 3000 = 3150 uc.
        let p = MockClient::sonnet_like().pricing;
        assert_eq!(p.cost_uc(50, 200), 3150);
    }
}
