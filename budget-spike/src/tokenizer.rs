#[cfg(feature = "tiktoken")]
use std::sync::Arc;

pub trait TokenEstimator: Send + Sync {
    fn estimate_input_tokens(&self, prompt: &str) -> u64;
    fn name(&self) -> &'static str;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ByteLengthEstimator;

impl TokenEstimator for ByteLengthEstimator {
    fn estimate_input_tokens(&self, prompt: &str) -> u64 {
        prompt.len() as u64
    }

    fn name(&self) -> &'static str {
        "byte-length"
    }
}

#[cfg(feature = "tiktoken")]
#[derive(Clone)]
pub struct TiktokenEstimator {
    bpe: Arc<tiktoken_rs::CoreBPE>,
    name: &'static str,
}

#[cfg(feature = "tiktoken")]
impl TiktokenEstimator {
    pub fn for_gpt4o_mini() -> Result<Self, String> {
        let bpe = tiktoken_rs::o200k_base()
            .map_err(|e| format!("failed to load o200k_base: {e}"))?;
        Ok(Self {
            bpe: Arc::new(bpe),
            name: "tiktoken-o200k-base",
        })
    }

    pub fn for_cl100k() -> Result<Self, String> {
        let bpe = tiktoken_rs::cl100k_base()
            .map_err(|e| format!("failed to load cl100k_base: {e}"))?;
        Ok(Self {
            bpe: Arc::new(bpe),
            name: "tiktoken-cl100k-base",
        })
    }
}

#[cfg(feature = "tiktoken")]
impl TokenEstimator for TiktokenEstimator {
    fn estimate_input_tokens(&self, prompt: &str) -> u64 {
        self.bpe.encode_with_special_tokens(prompt).len() as u64
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    #[test]
    fn byte_length_is_at_least_one_for_nonempty_ascii() {
        let e = ByteLengthEstimator;
        assert!(e.estimate_input_tokens("a") >= 1);
        assert_eq!(e.estimate_input_tokens(""), 0);
    }

    #[test]
    fn byte_length_handles_multibyte_utf8() {
        let e = ByteLengthEstimator;
        assert_eq!(e.estimate_input_tokens("你好"), 6);
    }

    #[test]
    fn byte_length_handles_digit_dense() {
        let e = ByteLengthEstimator;
        let s = "1234567890" .repeat(10);
        assert_eq!(e.estimate_input_tokens(&s), 100);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn tiktoken_o200k_loads() {
        let e = TiktokenEstimator::for_gpt4o_mini().expect("load failed");
        assert!(e.estimate_input_tokens("hello world") > 0);
    }

    #[cfg(feature = "tiktoken")]
    #[test]
    fn byte_length_dominates_tiktoken_on_english_prose() {
        let e_byte = ByteLengthEstimator;
        let e_tik = TiktokenEstimator::for_gpt4o_mini().expect("load failed");
        let prompts = [
            "Hello, world.",
            "The quick brown fox jumps over the lazy dog.",
            "SELECT * FROM users WHERE id = 1",
            "Tell me about resource-aware functional programming.",
        ];
        
        for p in prompts {
            let b = e_byte.estimate_input_tokens(p);
            let t = e_tik.estimate_input_tokens(p);
            assert!(
                b >= t,
                "byte length {} < tiktoken {} on prompt {:?} (would violate soundness)",
                b, t, p
            );
        }
    }
}