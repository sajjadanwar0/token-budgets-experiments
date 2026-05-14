//! Token estimation: pluggable conservative-upper-bound estimators.
//!
//! The Token Capabilities cap-respecting claim (Section 4.2 of the paper)
//! depends on a conservative input-token estimator: for every successful
//! call `i`, the estimator-reported reservation `r_i` must satisfy
//! `c_i <= r_i` where `c_i` is the provider-reported actual cost. Two
//! estimators are provided here, both sound; they trade efficiency for
//! reservation tightness.
//!
//! `ByteLengthEstimator` returns `prompt.len() as u64` (UTF-8 byte length).
//! Sound for every BPE-family tokenizer we are aware of: tiktoken
//! (OpenAI), the Anthropic tokenizer, SentencePiece. Every emitted token
//! spans at least one byte of the input string, so byte length upper-bounds
//! token count. Loose by ~4x on typical English text; tightness is an
//! efficiency property, not a soundness property.
//!
//! `TiktokenEstimator` returns the exact tiktoken-rs encode count for
//! OpenAI models. Tight upper bound for OpenAI (the bound is exact at the
//! tokenizer level; provider-side reporting may add small protocol
//! overhead which the conservative-reservation pattern absorbs by
//! reserving `max_output_tokens` worth of output budget upfront).
//!
//! Soundness preservation. Swapping `ByteLengthEstimator` for
//! `TiktokenEstimator` does not weaken the cap-respecting claim, because
//! tiktoken's count for OpenAI inputs is at least as large as the actual
//! emitted token count under the same encoding (it IS the same encoding).
//! The existing `LLMClient::estimate_input_tokens` default falls back to
//! byte length when a provider-specific estimator is not configured.

#[cfg(feature = "tiktoken")]
use std::sync::Arc;

/// A conservative-upper-bound estimator. Implementations must guarantee
/// that the returned value is greater than or equal to the actual number
/// of input tokens the provider will charge for the same prompt.
pub trait TokenEstimator: Send + Sync {
    fn estimate_input_tokens(&self, prompt: &str) -> u64;

    /// A short identifier used in audit logs and benchmarks. Production
    /// dashboards may want to know whether a budget was sized against
    /// the byte-length bound or a tighter tokenizer-accurate one.
    fn name(&self) -> &'static str;
}

/// Sound for every BPE-family tokenizer we surveyed (tiktoken, Anthropic,
/// SentencePiece). Loose by ~4x on typical English text. Use this when a
/// provider-specific tokenizer is not available, when the deployment must
/// avoid a per-provider tokenizer table, or when the artifact's
/// reproducibility constraint forbids network-fetched encodings.
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

/// Tiktoken-accurate estimator for OpenAI-family models.
///
/// Wraps `tiktoken_rs::CoreBPE` and exposes an estimator handle that can
/// be cloned cheaply across budget-bearing tasks. The encoder is loaded
/// lazily on first use and shared via `Arc`, so dropping a clone does not
/// reload the tokenizer.
///
/// Soundness: for any prompt, the returned count is the exact tiktoken
/// encoding count. For OpenAI providers using the same encoding (which
/// is the case for all current production OpenAI models the harness
/// targets), this is an exact bound; the conservative-reservation
/// pattern absorbs any small protocol overhead via the upfront
/// `max_output_tokens` reservation on the output side.
#[cfg(feature = "tiktoken")]
#[derive(Clone)]
pub struct TiktokenEstimator {
    bpe: Arc<tiktoken_rs::CoreBPE>,
    name: &'static str,
}

#[cfg(feature = "tiktoken")]
impl TiktokenEstimator {
    /// Estimator for `gpt-4`, `gpt-4-turbo`, `gpt-4o`, and `gpt-4o-mini`
    /// (cl100k_base / o200k_base depending on model family).
    pub fn for_gpt4o_mini() -> Result<Self, String> {
        let bpe = tiktoken_rs::o200k_base()
            .map_err(|e| format!("failed to load o200k_base: {e}"))?;
        Ok(Self {
            bpe: Arc::new(bpe),
            name: "tiktoken-o200k-base",
        })
    }

    /// Estimator for older `gpt-3.5-turbo` and `gpt-4` (cl100k_base).
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

// === Property tests ==========================================================

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
        // "你好" is 6 bytes in UTF-8 but 2 chars; the byte-length bound
        // returns 6, which is conservative because tokenizers can emit
        // up to 6 tokens for these characters depending on the encoding.
        let e = ByteLengthEstimator;
        assert_eq!(e.estimate_input_tokens("你好"), 6);
    }

    #[test]
    fn byte_length_handles_digit_dense() {
        // Adversarial digit strings: tiktoken often produces one token
        // per digit, so byte length (one byte per digit) is a tight
        // bound here. The byte-length bound is still safe.
        let e = ByteLengthEstimator;
        let s = "1234567890" .repeat(10); // 100 ASCII digits
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
        // Soundness check: byte length should always be >= tiktoken count
        // for English text under any BPE encoding.
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