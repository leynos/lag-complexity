//! BDD tests for `AmbiguityHeuristic`.

use lag_complexity::TextProcessor;
use lag_complexity::heuristics::{AmbiguityHeuristic, AmbiguityHeuristicError};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;

const EPSILON: f32 = 1e-6;

#[derive(Default)]
struct TestContext {
    heuristic: AmbiguityHeuristic,
    result: RefCell<Option<Result<f32, AmbiguityHeuristicError>>>,
}

#[fixture]
fn test_context() -> TestContext {
    TestContext::default()
}

#[given("an ambiguity heuristic")]
fn given_heuristic(#[from(test_context)] ctx: &TestContext) {
    let _ = ctx;
}

#[when("evaluating {query}")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn when_evaluating(query: String, #[from(test_context)] ctx: &TestContext) {
    *ctx.result.borrow_mut() = Some(ctx.heuristic.process(&query));
}

#[when("evaluating empty input")]
fn when_evaluating_empty(#[from(test_context)] ctx: &TestContext) {
    *ctx.result.borrow_mut() = Some(ctx.heuristic.process(""));
}

#[then("the ambiguity score is {expected:f32}")]
#[expect(clippy::float_arithmetic, reason = "tolerance comparison")]
fn then_score(#[from(test_context)] ctx: &TestContext, expected: f32) {
    let actual = match ctx.result.borrow().as_ref() {
        Some(Ok(val)) => *val,
        Some(Err(e)) => panic!("expected score, got error: {e:?}"),
        None => panic!("no result"),
    };
    assert!(
        (actual - expected).abs() < EPSILON,
        "expected {expected}, got {actual}"
    );
}

#[then("an ambiguity error is returned")]
fn then_error(#[from(test_context)] ctx: &TestContext) {
    assert_eq!(
        ctx.result.borrow().as_ref(),
        Some(&Err(AmbiguityHeuristicError::Empty))
    );
}

#[scenario(path = "tests/features/ambiguity_heuristic.feature", index = 0)]
fn ambiguity_complex(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/ambiguity_heuristic.feature", index = 1)]
fn ambiguity_empty(test_context: TestContext) {
    let _ = test_context;
}
