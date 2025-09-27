//! Heuristic ambiguity estimator.
//!
//! Provides a lightweight ambiguity signal by counting pronouns, ambiguous
//! terms, and vague references. Uses Laplace smoothing to avoid zero scores.

use crate::{
    heuristics::text::{normalize_tokens, singularise, substring_count_regex, weighted_count},
    providers::TextProcessor,
};
use regex::Regex;
use std::{mem, sync::LazyLock};
use thiserror::Error;

/// Represents input text for complexity analysis
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InputText(String);

impl InputText {
    pub fn new<T: Into<String>>(value: T) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn trim(&self) -> Self {
        Self(self.0.trim().to_owned())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[must_use]
    pub fn split_sentences(&self) -> Vec<Sentence> {
        let mut sentences = Vec::new();
        let mut current = Sentence::default();

        for raw in self.as_str().split_whitespace() {
            current.push_token(raw);

            if is_sentence_boundary(raw) {
                sentences.push(mem::take(&mut current));
            }
        }

        if !current.is_empty() {
            sentences.push(current);
        }

        sentences
    }
}

impl From<&str> for InputText {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for InputText {
    fn from(value: String) -> Self {
        Self(value)
    }
}

/// Represents a single sentence extracted from input
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sentence(String);

impl Sentence {
    pub fn new<T: Into<String>>(value: T) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn push_token(&mut self, token: &str) {
        if !self.0.is_empty() {
            self.0.push(' ');
        }
        self.0.push_str(token);
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Represents a raw token before processing
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RawToken(String);

impl RawToken {
    pub fn new<T: Into<String>>(value: T) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for RawToken {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for RawToken {
    fn from(value: String) -> Self {
        Self(value)
    }
}

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
    fn process_input(input: &InputText) -> Result<f32, AmbiguityHeuristicError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(AmbiguityHeuristicError::Empty);
        }

        let pronouns = score_pronouns(&trimmed);
        let tokens = normalize_tokens(trimmed.as_str());
        let ambiguous =
            weighted_count(tokens.iter().map(|t| singularise(t)), AMBIGUOUS_ENTITIES, 2);
        let vague = weighted_count(tokens.iter().map(String::as_str), VAGUE_WORDS, 1);
        let extras = substring_count_regex(trimmed.as_str(), &A_FEW_RE);
        let total = pronouns + ambiguous + vague + extras + 1;
        #[expect(clippy::cast_precision_loss, reason = "score within f32 range")]
        Ok(total as f32)
    }
}

impl TextProcessor for AmbiguityHeuristic {
    type Output = f32;
    type Error = AmbiguityHeuristicError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        let input = InputText::new(input);
        Self::process_input(&input)
    }
}

const PRONOUNS: &[&str] = &[
    "it", "he", "she", "they", "this", "that", "him", "her", "them", "his", "its", "their",
    "theirs",
];
const PRONOUN_BASE_WEIGHT: u32 = 1;
const UNRESOLVED_PRONOUN_BONUS: u32 = 1;
const FLAG_PRONOUN: u8 = 1 << 0;
const FLAG_CAPITALISED: u8 = 1 << 1;
const FLAG_ARTICLE: u8 = 1 << 2;
const FLAG_LIKELY_NOUN: u8 = 1 << 3;
const FLAG_FUNCTION_WORD: u8 = 1 << 4;
const FLAG_SENTENCE_ADVERB: u8 = 1 << 5;
const FUNCTION_WORDS: &[&str] = &[
    "however",
    "therefore",
    "meanwhile",
    "moreover",
    "yesterday",
    "today",
    "tomorrow",
    "suddenly",
    "finally",
    "initially",
    "eventually",
];
const LY_NOUN_EXCEPTIONS: &[&str] = &["assembly", "family", "italy", "july", "supply"];
const DEFINITE_ARTICLES: &[&str] = &["the", "this", "that", "these", "those"];
const AMBIGUOUS_ENTITIES: &[&str] = &["mercury", "apple", "jaguar", "python"];
const VAGUE_WORDS: &[&str] = &["some", "several", "here", "there", "then"];

/// Token metadata reused when scanning for antecedents.
/// Normalises case once so the heuristic avoids repeated lowercase allocations.
struct TokenCandidate {
    flags: u8,
}

impl TokenCandidate {
    fn from_raw(token: &RawToken) -> Option<Self> {
        let raw = token.as_str();
        let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-');
        if trimmed.is_empty() {
            return None;
        }
        let lower = trimmed.to_ascii_lowercase();
        let has_letters = trimmed.chars().any(char::is_alphabetic);
        let mut flags = 0;
        if PRONOUNS.contains(&lower.as_str()) {
            flags |= FLAG_PRONOUN;
        }
        if trimmed.chars().next().is_some_and(char::is_uppercase) {
            flags |= FLAG_CAPITALISED;
        }
        if DEFINITE_ARTICLES.contains(&lower.as_str()) {
            flags |= FLAG_ARTICLE;
        }
        if FUNCTION_WORDS.contains(&lower.as_str()) {
            flags |= FLAG_FUNCTION_WORD;
        }
        if has_letters && lower.ends_with("ly") && !LY_NOUN_EXCEPTIONS.contains(&lower.as_str()) {
            flags |= FLAG_SENTENCE_ADVERB;
        }
        if has_letters {
            flags |= FLAG_LIKELY_NOUN;
        }
        Some(Self { flags })
    }

    fn is_pronoun(&self) -> bool {
        self.flags & FLAG_PRONOUN != 0
    }

    fn is_capitalised(&self) -> bool {
        self.flags & FLAG_CAPITALISED != 0
    }

    fn is_article(&self) -> bool {
        self.flags & FLAG_ARTICLE != 0
    }

    fn is_function_word(&self) -> bool {
        self.flags & FLAG_FUNCTION_WORD != 0
    }

    fn is_sentence_adverb(&self) -> bool {
        self.flags & FLAG_SENTENCE_ADVERB != 0
    }

    fn is_likely_noun(&self) -> bool {
        self.flags & FLAG_LIKELY_NOUN != 0
    }

    /// Returns `true` when the token looks noun-like enough to anchor pronouns.
    ///
    /// Sentence-initial capitalised adverbs (for example "However" or "Suddenly")
    /// are ignored so discourse markers do not clear the unresolved-pronoun bonus
    /// without supporting noun context.
    fn is_candidate(&self, at_sentence_start: bool) -> bool {
        self.is_capitalised()
            && self.is_likely_noun()
            && !self.is_pronoun()
            && !self.is_function_word()
            && !(at_sentence_start && self.is_sentence_adverb())
    }
}

struct SentenceAnalysis {
    has_candidate: bool,
}

impl SentenceAnalysis {
    fn from_text(sentence: &Sentence) -> Self {
        let mut has_candidate = false;
        let mut pending_article = false;
        let mut at_sentence_start = true;

        for raw in sentence.as_str().split_whitespace() {
            if let Some(token) = TokenCandidate::from_raw(&RawToken::from(raw)) {
                if token.is_candidate(at_sentence_start)
                    || (pending_article && token.is_likely_noun())
                {
                    has_candidate = true;
                }
                pending_article = token.is_article();
            }
            at_sentence_start = false;
        }

        Self { has_candidate }
    }

    fn has_candidate(&self) -> bool {
        self.has_candidate
    }
}

static A_FEW_RE: LazyLock<Regex> = LazyLock::new(|| {
    #[expect(clippy::expect_used, reason = "pattern is constant and valid")]
    Regex::new(r"(?i)\ba few\b").expect("valid regex")
});

fn score_pronouns(input: &InputText) -> u32 {
    let mut score = 0;
    let mut previous_has_candidate = false;

    for sentence in input.split_sentences() {
        let analysis = SentenceAnalysis::from_text(&sentence);
        let has_nearby_candidate = previous_has_candidate || analysis.has_candidate();
        score += score_pronouns_in_sentence(&sentence, has_nearby_candidate);
        previous_has_candidate = analysis.has_candidate();
    }

    score
}

fn score_pronouns_in_sentence(sentence: &Sentence, has_nearby_candidate: bool) -> u32 {
    let mut score = 0;
    // `has_nearby_candidate` includes antecedents from the current sentence
    // and its immediate predecessor, so this helper only tallies pronouns.

    for raw in sentence.as_str().split_whitespace() {
        if let Some(token) = TokenCandidate::from_raw(&RawToken::from(raw)) {
            if token.is_pronoun() {
                score += PRONOUN_BASE_WEIGHT;
                if !has_nearby_candidate {
                    score += UNRESOLVED_PRONOUN_BONUS;
                }
            }
        }
    }

    score
}

fn is_sentence_boundary(token: &str) -> bool {
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

    #[test]
    fn rejects_empty_input() {
        let h = AmbiguityHeuristic;
        assert_eq!(h.process(""), Err(AmbiguityHeuristicError::Empty));
        assert_eq!(h.process("   \n\t"), Err(AmbiguityHeuristicError::Empty));
    }
}
