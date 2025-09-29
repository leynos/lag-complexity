//! Heuristic ambiguity estimator.
//!
//! Provides a lightweight ambiguity signal by counting pronouns, ambiguous
//! terms, and vague references. Uses Laplace smoothing to avoid zero scores.

use crate::{
    heuristics::text::{normalize_tokens, substring_count_regex, weighted_count},
    providers::TextProcessor,
};
use regex::Regex;
use std::sync::LazyLock;
use thiserror::Error;

mod pronouns;

pub use self::pronouns::score_pronouns;

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

impl AmbiguityHeuristic {
    fn process_input(input: &str) -> Result<f32, AmbiguityHeuristicError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(AmbiguityHeuristicError::Empty);
        }

        let pronouns = score_pronouns(trimmed);
        let tokens = normalize_tokens(trimmed);
        let ambiguous = ambiguous_entity_score(trimmed);
        let vague = weighted_count(tokens.iter().map(String::as_str), VAGUE_WORDS, 1);
        let extras = substring_count_regex(trimmed, &A_FEW_RE);
        let total = pronouns
            .saturating_add(ambiguous)
            .saturating_add(vague)
            .saturating_add(extras)
            .saturating_add(1);
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        Ok(total as f32)
    }
}

impl TextProcessor for AmbiguityHeuristic {
    type Output = f32;
    type Error = AmbiguityHeuristicError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        Self::process_input(input)
    }
}

fn ambiguous_entity_score(text: &str) -> u32 {
    AMBIGUOUS_ENTITY_REGEXES
        .iter()
        .fold(0u32, |acc, pattern| {
            acc.saturating_add(substring_count_regex(text, pattern))
        })
        .saturating_mul(AMBIGUOUS_ENTITY_WEIGHT)
}

const AMBIGUOUS_ENTITY_WEIGHT: u32 = 2;
/// Word-boundary regex patterns capturing the curated ambiguity lexicon
/// described in the design doc. Raw strings avoid extra escaping and keep
/// the patterns readable.
const AMBIGUOUS_ENTITY_PATTERNS: &[&str] = &[
    r"(?i)\bmercur(?:y|ies)(?:'s)?\b",
    r"(?i)\bapple(?:'s|s)?\b",
    r"(?i)\bjaguar(?:'s|s)?\b",
    r"(?i)\bpython(?:'s|s)?\b",
    r"(?i)\bnile(?:'s|s)?\b",
    r"(?i)\bamazon(?:'s|s)?\b",
    r"(?i)\bjordan(?:'s|s)?\b",
    r"(?i)\borion(?:'s|s)?\b",
    r"(?i)\bsaturn(?:'s|s)?\b",
    r"(?i)\bdelta(?:'s|s)?\b",
];
static AMBIGUOUS_ENTITY_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    AMBIGUOUS_ENTITY_PATTERNS
        .iter()
        .map(|pattern| {
            #[expect(clippy::expect_used, reason = "patterns validated during tests")]
            Regex::new(pattern).expect("valid ambiguous entity regex")
        })
        .collect()
});
const VAGUE_WORDS: &[&str] = &["some", "several", "here", "there", "then"];

static A_FEW_RE: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used, reason = "pattern is constant and valid")]
    Regex::new(r"(?i)\ba few\b").expect("valid regex")
});

pub(super) fn is_sentence_boundary(token: &str) -> bool {
    for c in token.chars().rev() {
        if matches!(c, '"' | '\u{27}' | ')' | ']' | '}') {
            continue;
        }

        return matches!(c, '.' | '!' | '?' | '…');
    }
    false
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
    #[case("Yesterday it broke.", 3.0)]
    #[case("Suddenly it broke.", 3.0)]
    #[case("His idea failed.", 3.0)]
    #[case("Its hinge snapped.", 3.0)]
    #[case("Their plan stalled.", 3.0)]
    #[case("Theirs was missing.", 3.0)]
    #[case("Alice fixed the radio. It works now?", 2.0)]
    #[case("He told her to go home.", 5.0)]
    #[case("Those results are final. It stands.", 2.0)]
    #[case("It… works!", 3.0)]
    fn scores_expected_values(#[case] query: &str, #[case] expected: f32) {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(query), Ok(expected));
    }

    #[rstest]
    #[case("Mercury.", 3.0)]
    #[case("Mercury-based alloys", 3.0)]
    #[case("Discuss Python's syntax.", 3.0)]
    #[case("Amazon, Nile, and Jordan", 7.0)]
    fn recognises_ambiguous_entities(#[case] query: &str, #[case] expected: f32) {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(query), Ok(expected));
    }

    #[test]
    fn rejects_empty_input() {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(""), Err(AmbiguityHeuristicError::Empty));
        assert_eq!(h.process("   \n\t"), Err(AmbiguityHeuristicError::Empty));
    }

    #[rstest]
    #[case("Word.", true)]
    #[case("Word?", true)]
    #[case("Word…", true)]
    #[case(r#""Who?""#, true)]
    #[case("Alert!]", true)]
    #[case("token", false)]
    #[case(r#"token,""#, false)]
    #[case(r#"token-""#, false)]
    fn detects_sentence_boundaries(#[case] token: &str, #[case] expected: bool) {
        assert_eq!(is_sentence_boundary(token), expected);
    }
}
