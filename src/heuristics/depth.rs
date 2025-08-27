//! Heuristic depth estimator.
//!
//! Counts linguistic signals that typically indicate reasoning depth. The
//! heuristic favours clarity over exhaustive linguistic analysis.

use crate::providers::TextProcessor;
use thiserror::Error;

/// Errors returned by [`DepthHeuristic`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DepthHeuristicError {
    /// Input was empty or whitespace only.
    #[error("input cannot be empty")]
    Empty,
}

/// Fast, dependency-free estimator for reasoning depth.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::DepthHeuristic;
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
        if input.trim().is_empty() {
            return Err(DepthHeuristicError::Empty);
        }
        let lower = input.to_lowercase();
        let mut score: u32 = 0;
        let words = lower.split_whitespace();
        for word in words {
            let token = word.trim_matches(|c: char| !c.is_alphabetic());
            if COORD_CONJ.contains(&token) {
                score += 1;
            }
            if SUBORD_CONJ.contains(&token) {
                score += 2;
            }
            if REL_PRON.contains(&token) {
                score += 1;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
        {
            score += lower.matches(", and").count() as u32;
        }
        #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
        {
            score += lower.matches(", or").count() as u32;
        }
        score += count_phrases(&lower, COMPARATIVES) * 2;
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        let score_f32 = score as f32;
        Ok(score_f32)
    }
}

const COORD_CONJ: &[&str] = &["and", "or", "but"];
const SUBORD_CONJ: &[&str] = &["if", "because", "while", "since", "although"];
const REL_PRON: &[&str] = &["who", "which", "that", "whose"];
const COMPARATIVES: &[&str] = &[
    "compared to",
    "versus",
    "as opposed to",
    "more than",
    "less than",
];

fn count_phrases(text: &str, phrases: &[&str]) -> u32 {
    phrases
        .iter()
        .map(|p| {
            #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
            {
                text.matches(p).count() as u32
            }
        })
        .sum()
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
    }
}
