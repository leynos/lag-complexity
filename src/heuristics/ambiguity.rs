//! Heuristic ambiguity estimator.
//!
//! Provides a lightweight ambiguity signal by counting pronouns, ambiguous
//! terms, and vague references. Uses Laplace smoothing to avoid zero scores.

use crate::providers::TextProcessor;
use thiserror::Error;

/// Errors returned by [`AmbiguityHeuristic`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum AmbiguityHeuristicError {
    /// Input was empty or whitespace only.
    #[error("input cannot be empty")]
    Empty,
}

/// Fast estimator for semantic ambiguity.
///
/// # Examples
///
/// ```
/// use lag_complexity::heuristics::AmbiguityHeuristic;
///
/// let estimator = AmbiguityHeuristic;
/// let score = estimator.process("It references Mercury").unwrap();
/// assert!(score >= 1.0);
/// ```
#[derive(Default)]
pub struct AmbiguityHeuristic;

impl TextProcessor for AmbiguityHeuristic {
    type Output = f32;
    type Error = AmbiguityHeuristicError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        if input.trim().is_empty() {
            return Err(AmbiguityHeuristicError::Empty);
        }
        let lower = input.to_lowercase();
        let mut score: u32 = 0;
        for word in lower.split_whitespace() {
            let token = word.trim_matches(|c: char| !c.is_alphabetic());
            if PRONOUNS.contains(&token) {
                score += 1;
            }
            let singular = token.strip_suffix('s').unwrap_or(token);
            if AMBIGUOUS_ENTITIES.contains(&singular) {
                score += 2;
            }
            if VAGUE_WORDS.contains(&token) {
                score += 1;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "match count fits in u32")]
        {
            score += lower.matches("a few").count() as u32;
        }
        let total = score + 1;
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        let score_f32 = total as f32;
        Ok(score_f32)
    }
}

const PRONOUNS: &[&str] = &["it", "he", "she", "they", "this", "that"];
const AMBIGUOUS_ENTITIES: &[&str] = &["mercury", "apple", "jaguar", "python"];
const VAGUE_WORDS: &[&str] = &["some", "several", "here", "there", "then"];

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("It is about Mercury and some others.", 5.0)]
    #[case("A few jaguars here", 5.0)]
    #[case("Plain question", 1.0)]
    fn scores_expected_values(#[case] query: &str, #[case] expected: f32) {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(query), Ok(expected));
    }

    #[test]
    fn rejects_empty_input() {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(""), Err(AmbiguityHeuristicError::Empty));
    }
}
