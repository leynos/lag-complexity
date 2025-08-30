//! `rstest` tests for `AmbiguityHeuristic`.
//!
//! Replaces the previous BDD-style harness with concise, parameterised tests
//! using `rstest`. This keeps intent clear without macro-generated indirection.

use lag_complexity::TextProcessor;
use lag_complexity::heuristics::{AmbiguityHeuristic, AmbiguityHeuristicError};
use rstest::rstest;

mod support;
use support::approx_eq;

const EPSILON: f32 = 1e-6;

/// Validate ambiguity scoring and error handling.
///
/// Examples:
/// - Input with pronoun + ambiguous entity + vague term yields 5.0.
/// - Empty input yields `AmbiguityHeuristicError::Empty`.
#[rstest]
#[case("It is about Mercury and some others.", Ok(5.0))]
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
