use lag_complexity::TextProcessor;
use lag_complexity::heuristics::{DepthHeuristic, DepthHeuristicError};
use rstest::rstest;
mod support;
use support::approx_eq;

const EPSILON: f32 = 1e-6;

#[rstest]
#[case("If Alice and Bob play chess, who wins?", Ok(4.0))]
#[case("", Err(DepthHeuristicError::Empty))]
fn evaluates_depth_heuristic(
    #[case] input: &str,
    #[case] expected: Result<f32, DepthHeuristicError>,
) {
    let h = DepthHeuristic;
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
