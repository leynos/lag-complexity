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
    "It", "He", "She", "They", "This", "That", "Him", "Her", "Them",
];
const CANDIDATE_NAMES: &[&str] = &["Alice", "Berlin", "Mercury", "Orion", "Sirius"];

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
        let base_result = h.process(&base_sentence);
        prop_assert!(
            base_result.is_ok(),
            "base sentence should score but returned {base_result:?}"
        );
        let Ok(base_score) = base_result else {
            unreachable!("checked by prop_assert");
        };
        let augmented_result = h.process(&augmented);
        prop_assert!(
            augmented_result.is_ok(),
            "augmented text should score but returned {augmented_result:?}"
        );
        let Ok(unresolved_score) = augmented_result else {
            unreachable!("checked by prop_assert");
        };
        prop_assert!(
            unresolved_score >= base_score,
            "unresolved pronoun lowered score: base={base_score}, with_pronoun={unresolved_score}"
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
        let base_result = h.process(&base_sentence);
        prop_assert!(
            base_result.is_ok(),
            "base sentence should score but returned {base_result:?}"
        );
        let Ok(base_score) = base_result else {
            unreachable!("checked by prop_assert");
        };
        let anchored_result = h.process(&augmented);
        prop_assert!(
            anchored_result.is_ok(),
            "anchored text should score but returned {anchored_result:?}"
        );
        let Ok(anchored_score) = anchored_result else {
            unreachable!("checked by prop_assert");
        };
        prop_assert!(
            anchored_score >= base_score,
            "anchored pronoun lowered score: base={base_score}, with_pronoun={anchored_score}"
        );
    }
}
