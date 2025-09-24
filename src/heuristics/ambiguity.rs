//! Heuristic ambiguity estimator.
//!
//! Provides a lightweight ambiguity signal by counting pronouns, ambiguous
//! terms, and vague references. Uses Laplace smoothing to avoid zero scores.

use crate::{
    heuristics::text::{normalize_tokens, singularise, substring_count_regex, weighted_count},
    providers::TextProcessor,
};
use regex::Regex;
use std::sync::LazyLock;
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
/// use lag_complexity::TextProcessor;
///
/// let estimator = AmbiguityHeuristic;
/// let score = estimator.process("It references Mercury").unwrap();
/// assert!(score >= 1.0);
/// ```
#[derive(Default, Debug, Clone)]
pub struct AmbiguityHeuristic;

impl TextProcessor for AmbiguityHeuristic {
    type Output = f32;
    type Error = AmbiguityHeuristicError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(AmbiguityHeuristicError::Empty);
        }
        let pronouns = score_pronouns(trimmed);
        let tokens = normalize_tokens(trimmed);
        let ambiguous =
            weighted_count(tokens.iter().map(|t| singularise(t)), AMBIGUOUS_ENTITIES, 2);
        let vague = weighted_count(tokens.iter().map(String::as_str), VAGUE_WORDS, 1);
        let extras = substring_count_regex(trimmed, &A_FEW_RE);
        let total = pronouns + ambiguous + vague + extras + 1;
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        Ok(total as f32)
    }
}

const PRONOUNS: &[&str] = &["it", "he", "she", "they", "this", "that"];
const PRONOUN_BASE_WEIGHT: u32 = 1;
const UNRESOLVED_PRONOUN_BONUS: u32 = 1;
const DEFINITE_ARTICLES: &[&str] = &["the", "this", "that", "these", "those"];
const AMBIGUOUS_ENTITIES: &[&str] = &["mercury", "apple", "jaguar", "python"];
const VAGUE_WORDS: &[&str] = &["some", "several", "here", "there", "then"];

static A_FEW_RE: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used, reason = "pattern is constant and valid")]
    Regex::new(r"(?i)\ba few\b").expect("valid regex")
});

fn score_pronouns(input: &str) -> u32 {
    let sentences = split_sentences(input);
    if sentences.is_empty() {
        return 0;
    }
    let candidate_presence: Vec<bool> = sentences
        .iter()
        .map(|sentence| sentence_has_candidate(sentence))
        .collect();
    let mut score = 0;
    let mut previous_has_candidate = false;
    for (has_candidate, sentence) in candidate_presence.iter().copied().zip(sentences.iter()) {
        let tokens = normalize_tokens(sentence);
        let has_nearby_candidate = has_candidate || previous_has_candidate;
        for token in tokens {
            if is_pronoun(&token) {
                score += PRONOUN_BASE_WEIGHT;
                if !has_nearby_candidate {
                    score += UNRESOLVED_PRONOUN_BONUS;
                }
            }
        }
        previous_has_candidate = has_candidate;
    }
    score
}

fn split_sentences(input: &str) -> Vec<&str> {
    input
        .split(['.', '!', '?'])
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

fn sentence_has_candidate(sentence: &str) -> bool {
    let tokens: Vec<String> = sentence
        .split_whitespace()
        .filter_map(clean_token)
        .collect();
    if tokens.is_empty() {
        return false;
    }
    for token in &tokens {
        if is_capitalised(token) {
            let lower = token.to_lowercase();
            if !is_pronoun(lower.as_str()) {
                return true;
            }
        }
    }
    for window in tokens.windows(2) {
        if let [article, noun] = window {
            let article_lower = article.to_lowercase();
            if DEFINITE_ARTICLES.contains(&article_lower.as_str())
                && noun.chars().any(char::is_alphabetic)
            {
                return true;
            }
        }
    }
    false
}

fn clean_token(token: &str) -> Option<String> {
    let trimmed = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn is_capitalised(token: &str) -> bool {
    token.chars().next().is_some_and(char::is_uppercase)
}

fn is_pronoun(token: &str) -> bool {
    PRONOUNS.contains(&token)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("It is about Mercury and some others.", 5.0)]
    #[case("A few jaguars here", 5.0)]
    #[case("Plain question", 1.0)]
    #[case("It broke last night.", 3.0)]
    #[case("Alice fixed the radio. It works now?", 2.0)]
    fn scores_expected_values(#[case] query: &str, #[case] expected: f32) {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(query), Ok(expected));
    }

    #[test]
    fn rejects_empty_input() {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(""), Err(AmbiguityHeuristicError::Empty));
        assert_eq!(h.process("   \n\t"), Err(AmbiguityHeuristicError::Empty));
    }
}
