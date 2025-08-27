//! BDD tests for `DepthHeuristic`.

use lag_complexity::heuristics::{DepthHeuristic, DepthHeuristicError};
use lag_complexity::providers::TextProcessor;
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;

#[derive(Default)]
struct TestContext {
    heuristic: DepthHeuristic,
    result: RefCell<Option<Result<f32, DepthHeuristicError>>>,
}

#[fixture]
fn test_context() -> TestContext {
    TestContext::default()
}

#[given("a depth heuristic")]
fn given_heuristic(#[from(test_context)] ctx: &TestContext) {
    let _ = ctx;
}

#[when("processing {query}")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn when_processing(query: String, #[from(test_context)] ctx: &TestContext) {
    *ctx.result.borrow_mut() = Some(ctx.heuristic.process(&query));
}

#[when("processing empty input")]
fn when_processing_empty(#[from(test_context)] ctx: &TestContext) {
    *ctx.result.borrow_mut() = Some(ctx.heuristic.process(""));
}

#[then("the depth score is {expected:f32}")]
fn then_score(#[from(test_context)] ctx: &TestContext, expected: f32) {
    assert_eq!(ctx.result.borrow().as_ref(), Some(&Ok(expected)));
}

#[then("a depth error is returned")]
fn then_error(#[from(test_context)] ctx: &TestContext) {
    assert!(matches!(ctx.result.borrow().as_ref(), Some(Err(_))));
}

#[scenario(path = "tests/features/depth_heuristic.feature", index = 0)]
fn depth_complex(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/depth_heuristic.feature", index = 1)]
fn depth_empty(test_context: TestContext) {
    let _ = test_context;
}
