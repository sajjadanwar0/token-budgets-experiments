use async_trait::async_trait;
use serde::{Deserialize, Serialize};

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
    pub actual_cost_micro_cents: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct Pricing {
    pub input_per_mtok_uc: u64,
    pub output_per_mtok_uc: u64,
}

impl Pricing {
    pub fn cost_uc(&self, input_tokens: u64, output_tokens: u64) -> u64 {
        let in_part = input_tokens.saturating_mul(self.input_per_mtok_uc);
        let out_part = output_tokens.saturating_mul(self.output_per_mtok_uc);
        let total = in_part.saturating_add(out_part);
        // half-up: (x + 500_000) / 1_000_000 for positive x.
        (total.saturating_add(500_000)) / 1_000_000
    }

    pub fn input_per_token_uc_ceil(&self) -> u64 {
        if self.input_per_mtok_uc == 0 {
            0
        } else {
            (self.input_per_mtok_uc + 999_999) / 1_000_000
        }
    }

    pub fn output_per_token_uc_ceil(&self) -> u64 {
        if self.output_per_mtok_uc == 0 {
            0
        } else {
            (self.output_per_mtok_uc + 999_999) / 1_000_000
        }
    }
}

#[async_trait]
pub trait LLMClient: Send + Sync {
    async fn complete(
        &self,
        prompt: &str,
        max_output_tokens: u64,
    ) -> Result<CompletionResponse, LLMError>;
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

    fn input_price_per_token_mc(&self) -> u64 {
        self.pricing().input_per_token_uc_ceil()
    }

    fn output_price_per_token_mc(&self) -> u64 {
        self.pricing().output_per_token_uc_ceil()
    }

    fn estimate_input_tokens(&self, prompt: &str) -> u64 {
        prompt.len() as u64
    }
}

pub struct MockClient {
    pub fixed_input_tokens: u64,
    pub fixed_output_tokens: u64,
    pub pricing: Pricing,
    pub simulated_latency_ms: u64,
}

impl MockClient {
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

    fn published_pricing(&self) -> Pricing {
        let m = self.model.as_str();
        if m.contains("opus") {
            Pricing { input_per_mtok_uc: 15_000_000, output_per_mtok_uc: 75_000_000 }
        } else if m.contains("haiku") {
            Pricing { input_per_mtok_uc: 1_000_000, output_per_mtok_uc: 5_000_000 }
        } else {
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

    fn published_pricing(&self) -> Pricing {
        let m = self.model.as_str();
        if m.starts_with("gpt-5") {
            Pricing { input_per_mtok_uc: 5_000_000, output_per_mtok_uc: 15_000_000 }
        } else if m.starts_with("gpt-4o-mini") {
            Pricing { input_per_mtok_uc: 150_000, output_per_mtok_uc: 600_000 }
        } else if m.starts_with("gpt-4o") {
            Pricing { input_per_mtok_uc: 2_500_000, output_per_mtok_uc: 10_000_000 }
        } else if m.starts_with("gpt-4-turbo") {
            Pricing { input_per_mtok_uc: 10_000_000, output_per_mtok_uc: 30_000_000 }
        } else if m.starts_with("gpt-4") {
            Pricing { input_per_mtok_uc: 30_000_000, output_per_mtok_uc: 60_000_000 }
        } else {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pricing_cost_uc_gpt_4o_mini_typical_call() {
        let p = Pricing { input_per_mtok_uc: 150_000, output_per_mtok_uc: 600_000 };
        assert_eq!(p.cost_uc(1000, 200), 270);
    }

    #[test]
    fn pricing_cost_uc_anthropic_haiku() {
        let p = Pricing { input_per_mtok_uc: 1_000_000, output_per_mtok_uc: 5_000_000 };
        assert_eq!(p.cost_uc(1000, 200), 2000);
    }

    #[test]
    fn pricing_per_token_ceiling_is_conservative() {
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
        let p = Pricing { input_per_mtok_uc: u64::MAX, output_per_mtok_uc: u64::MAX };
        let _ = p.cost_uc(u64::MAX, u64::MAX); // should not panic
    }

    #[test]
    fn mock_client_sonnet_pricing_matches_documented_rates() {
        let p = MockClient::sonnet_like().pricing;
        assert_eq!(p.cost_uc(50, 200), 3150);
    }
}