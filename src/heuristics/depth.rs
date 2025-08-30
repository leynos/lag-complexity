//! Heuristic depth estimator.
//!
//! Counts linguistic signals that typically indicate reasoning depth. The
//! heuristic favours clarity over exhaustive linguistic analysis.

use crate::{heuristics::text::normalize_tokens, providers::TextProcessor};
use regex::Regex;
use std::sync::LazyLock;
use thiserror::Error;

/// Errors returned by [`DepthHeuristic`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DepthHeuristicError {
    /// Input was empty or whitespace only.
    #[error("input cannot be empty")]
    Empty,
}

/// Fast, lightweight estimator for reasoning depth.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::DepthHeuristic;
/// use lag_complexity::TextProcessor;
///
/// let estimator = DepthHeuristic;
/// let score = estimator.process("If A and B then C").unwrap();
/// assert!(score > 0.0);
/// ```
#[derive(Default)]
pub struct DepthHeuristic;

impl TextProcessor for DepthHeuristic {
    type Output = f32;
    type Error = DepthHeuristicError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(DepthHeuristicError::Empty);
        }
        let tokens = normalize_tokens(trimmed);
        let lower = trimmed.to_lowercase();
        let mut score = tokens.iter().fold(0u32, |acc, tok| {
            let mut s = acc;
            for (set, w) in TOKEN_WEIGHTS {
                if set.contains(&tok.as_str()) {
                    s += w;
                }
            }
            s
        });
        #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
        {
            score += PHRASE_WEIGHTS_RAW
                .iter()
                .map(|(pat, w)| lower.matches(pat).count() as u32 * w)
                .sum::<u32>();
        }
        score += boundary_score(&lower);
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        Ok(score as f32)
    }
}

const COORD_CONJ: &[&str] = &["and", "or", "but"];
const SUBORD_CONJ: &[&str] = &["if", "because", "while", "since", "although"];
const REL_PRON: &[&str] = &["who", "which", "that", "whose"];
const TOKEN_WEIGHTS: &[(&[&str], u32)] = &[(COORD_CONJ, 1), (SUBORD_CONJ, 2), (REL_PRON, 1)];
const PHRASE_WEIGHTS_RAW: &[(&str, u32)] = &[(", and", 1), (", or", 1)];
const PHRASE_WEIGHTS_BOUNDARY: &[(&str, u32)] = &[
    ("compared to", 2),
    ("versus", 2),
    ("as opposed to", 2),
    ("more than", 2),
    ("less than", 2),
];

static BOUNDARY_PATTERNS: LazyLock<Vec<(Regex, u32)>> = LazyLock::new(|| {
    PHRASE_WEIGHTS_BOUNDARY
        .iter()
        .map(|(p, w)| {
            #[expect(clippy::expect_used, reason = "escaped pattern cannot fail")]
            (
                Regex::new(&format!(r"\b{}\b", regex::escape(p))).expect("valid regex"),
                *w,
            )
        })
        .collect()
});

fn boundary_score(lower: &str) -> u32 {
    #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
    {
        BOUNDARY_PATTERNS
            .iter()
            .map(|(re, w)| re.find_iter(lower).count() as u32 * *w)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("If Alice and Bob play chess, who wins?", 4.0)]
    #[case("What are the differences between A, B, and C?", 2.0)]
    #[case("Is the sky more than blue?", 2.0)]
    fn scores_expected_values(#[case] query: &str, #[case] expected: f32) {
        let h = DepthHeuristic;
        assert_eq!(h.process(query), Ok(expected));
    }

    #[test]
    fn rejects_empty_input() {
        let h = DepthHeuristic;
        assert_eq!(h.process(""), Err(DepthHeuristicError::Empty));
        assert_eq!(h.process("   \t"), Err(DepthHeuristicError::Empty));
    }
}
