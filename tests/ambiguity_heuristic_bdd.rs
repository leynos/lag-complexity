//! BDD tests for `AmbiguityHeuristic`.

use lag_complexity::TextProcessor;
use lag_complexity::heuristics::{AmbiguityHeuristic, AmbiguityHeuristicError};
mod support;
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;
use support::approx_eq;

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
#[expect(clippy::expect_used, reason = "test asserts presence")]
fn then_score(#[from(test_context)] ctx: &TestContext, expected: f32) {
    let actual = *ctx
        .result
        .borrow()
        .as_ref()
        .expect("expected result")
        .as_ref()
        .expect("expected score");
    assert!(approx_eq(actual, expected, 1e-6));
}

#[then("an ambiguity error is returned")]
fn then_error(#[from(test_context)] ctx: &TestContext) {
    assert!(matches!(
        ctx.result.borrow().as_ref(),
        Some(Err(AmbiguityHeuristicError::Empty))
    ));
}

#[scenario(path = "tests/features/ambiguity_heuristic.feature", index = 0)]
fn ambiguity_complex(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/ambiguity_heuristic.feature", index = 1)]
fn ambiguity_empty(test_context: TestContext) {
    let _ = test_context;
}
