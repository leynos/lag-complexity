//! `rstest` tests for `AmbiguityHeuristic`.
//!
//! Replaces the previous BDD-style harness with concise, parameterised tests
//! using `rstest`. This keeps intent clear without macro-generated indirection.

use lag_complexity::TextProcessor;
use lag_complexity::heuristics::{AmbiguityHeuristic, AmbiguityHeuristicError};
use proptest::prelude::*;
use rstest::rstest;

mod support;
use support::approx_eq;

const EPSILON: f32 = 1e-6;
const PRONOUN_SAMPLES: &[&str] = &[
    "It", "He", "She", "They", "This", "That", "Him", "Her", "Them", "His", "Its", "Their",
    "Theirs", "it", "he", "she", "they", "this", "that", "him", "her", "them", "his", "its",
    "their", "theirs",
];
const CANDIDATE_NAMES: &[&str] = &["Alice", "Berlin", "Mercury", "Orion", "Sirius"];
const SENTENCE_ADVERBS: &[&str] = &[
    "However",
    "Suddenly",
    "Finally",
    "Moreover",
    "Meanwhile",
    "Today",
    "Yesterday",
];
const DEMONSTRATIVE_PRONOUNS: &[&str] = &["This", "That"];

/// Validate ambiguity scoring and error handling.
///
/// Examples:
/// - Input with pronoun + ambiguous entity + vague term yields 5.0.
/// - Pronoun without a nearby antecedent yields 3.0.
/// - Pronoun anchored by the previous sentence yields 2.0.
/// - Empty input yields `AmbiguityHeuristicError::Empty`.
#[rstest]
#[case("It is about Mercury and some others.", Ok(5.0))]
#[case("It broke last night.", Ok(3.0))]
#[case("Alice fixed the radio. It works now?", Ok(2.0))]
#[case("", Err(AmbiguityHeuristicError::Empty))]
fn evaluates_ambiguity_heuristic(
    #[case] input: &str,
    #[case] expected: Result<f32, AmbiguityHeuristicError>,
) {
    let h = AmbiguityHeuristic;
    let result = h.process(input);
    match (result, expected) {
        (Ok(actual), Ok(score)) => {
            assert!(
                approx_eq(actual, score, EPSILON),
                "expected {score}, got {actual}"
            );
        }
        (Err(err), Err(expected_err)) => assert_eq!(err, expected_err),
        (res, exp) => panic!("expected {exp:?}, got {res:?}"),
    }
}

fn to_sentence(words: &[String]) -> String {
    format!("{}.", words.join(" "))
}

#[rstest]
fn article_noun_pattern_anchors_pronoun() {
    let h = AmbiguityHeuristic;
    let score = match h.process("The engineer repaired it.") {
        Ok(score) => score,
        Err(err) => panic!("expected scoring to succeed: {err:?}"),
    };
    assert!(
        approx_eq(score, 2.0, EPSILON),
        "expected pronoun anchored score to be 2.0, got {score}"
    );
}

#[rstest]
fn sentence_initial_adverb_does_not_anchor_pronoun() {
    let h = AmbiguityHeuristic;
    let score = match h.process("However, it broke.") {
        Ok(score) => score,
        Err(err) => panic!("expected scoring to succeed: {err:?}"),
    };
    assert!(
        approx_eq(score, 3.0, EPSILON),
        "expected unresolved pronoun score to be 3.0, got {score}"
    );
}

#[rstest]
fn article_followed_by_function_word_does_not_anchor_pronoun() {
    let h = AmbiguityHeuristic;
    let score = match h.process("The very quickly failed. It broke.") {
        Ok(score) => score,
        Err(err) => panic!("expected scoring to succeed: {err:?}"),
    };
    assert!(
        approx_eq(score, 3.0, EPSILON),
        "expected unresolved pronoun score to be 3.0, got {score}"
    );
}

macro_rules! assert_non_decreasing {
    ($heuristic:expr, $base:expr, $augmented:expr, $context:expr) => {{
        let context = $context;
        let base_result = $heuristic.process($base);
        prop_assert!(
            base_result.is_ok(),
            "{} base sentence should score but returned {base_result:?}",
            context
        );
        let base_score = base_result.unwrap();
        let augmented_result = $heuristic.process($augmented);
        prop_assert!(
            augmented_result.is_ok(),
            "{} augmented text should score but returned {augmented_result:?}",
            context
        );
        let augmented_score = augmented_result.unwrap();
        prop_assert!(
            augmented_score >= base_score,
            "{} lowered score: base={base_score}, augmented={augmented_score}",
            context
        );
    }};
}

proptest! {
    #[test]
    fn injecting_unresolved_pronoun_is_monotonic(
        base_words in prop::collection::vec("[a-z]{1,10}", 1..6),
        pronoun in prop::sample::select(PRONOUN_SAMPLES),
    ) {
        let base_sentence = to_sentence(&base_words);
        let pronoun_sentence = format!("{pronoun}.");
        let augmented = format!("{pronoun_sentence} {base_sentence}");
        let h = AmbiguityHeuristic;
        assert_non_decreasing!(
            &h,
            &base_sentence,
            &augmented,
            "unresolved pronoun"
        );
    }

    #[test]
    fn injecting_anchored_pronoun_is_monotonic(
        base_words in prop::collection::vec("[a-z]{1,10}", 1..6),
        pronoun in prop::sample::select(PRONOUN_SAMPLES),
        name in prop::sample::select(CANDIDATE_NAMES),
        verb in "[a-z]{3,12}",
    ) {
        let base_sentence = to_sentence(&base_words);
        let candidate_sentence = format!("{name} {verb}.");
        let pronoun_sentence = format!("{pronoun}.");
        let augmented = format!(
            "{candidate_sentence} {pronoun_sentence} {base_sentence}"
        );
        let h = AmbiguityHeuristic;
        assert_non_decreasing!(
            &h,
            &base_sentence,
            &augmented,
            "anchored pronoun"
        );
    }

    #[test]
    fn capitalised_adverb_prefix_keeps_pronoun_unresolved(
        adverb in prop::sample::select(SENTENCE_ADVERBS),
        pronoun in prop::sample::select(PRONOUN_SAMPLES),
    ) {
        let sentence = format!("{adverb} {pronoun}.");
        let h = AmbiguityHeuristic;
        let score = h
            .process(&sentence)
            .unwrap_or_else(|err| panic!("expected scoring to succeed: {err:?}"));
        prop_assert!(
            score >= 3.0,
            "expected unresolved pronoun score >= 3.0, got {score}"
        );
    }

    #[test]
    fn demonstrative_pronoun_with_verb_is_anchored(
        pronoun in prop::sample::select(DEMONSTRATIVE_PRONOUNS),
    ) {
        let sentence = format!("{pronoun} failed.");
        let h = AmbiguityHeuristic;
        let score = h
            .process(&sentence)
            .unwrap_or_else(|err| panic!("expected scoring to succeed: {err:?}"));
        prop_assert!(
            approx_eq(score, 2.0, EPSILON),
            "expected demonstrative pronoun to be anchored by the verb, got {score}"
        );
    }
}
