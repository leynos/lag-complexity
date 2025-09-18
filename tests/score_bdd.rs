use lag_complexity::{
    Complexity, ComplexityFn, HeuristicComplexity, HeuristicComplexityError, Trace,
    heuristics::DepthHeuristicError,
};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;

#[derive(Default)]
struct TestContext {
    scorer: HeuristicComplexity,
    score_result: RefCell<Option<Result<Complexity, HeuristicComplexityError>>>,
    trace_result: RefCell<Option<Result<Trace, HeuristicComplexityError>>>,
}

impl TestContext {
    fn scored_ok(&self) -> Complexity {
        match self.score_result.borrow().as_ref() {
            Some(Ok(comp)) => comp.clone(),
            Some(Err(err)) => panic!("expected score to succeed, got error: {err}"),
            None => panic!("score result missing"),
        }
    }

    fn traced_ok(&self) -> Trace {
        match self.trace_result.borrow().as_ref() {
            Some(Ok(trace)) => trace.clone(),
            Some(Err(err)) => panic!("expected trace to succeed, got error: {err}"),
            None => panic!("trace result missing"),
        }
    }
}

fn assert_close(actual: f32, expected: f32) {
    let tolerance = 1e-6;
    #[expect(
        clippy::float_arithmetic,
        reason = "tests compare floating point values"
    )]
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "expected {expected}, got {actual}, diff {diff} exceeds tolerance {tolerance}"
    );
}

#[fixture]
fn test_context() -> TestContext {
    TestContext::default()
}

#[given("a heuristic complexity scorer")]
fn given_scorer(#[from(test_context)] context: &TestContext) {
    let _ = context;
}

#[when("scoring \"{query}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn when_scoring(query: String, #[from(test_context)] context: &TestContext) {
    *context.score_result.borrow_mut() = Some(context.scorer.score(&query));
}

#[when("scoring empty query")]
fn when_scoring_empty(#[from(test_context)] context: &TestContext) {
    *context.score_result.borrow_mut() = Some(context.scorer.score(""));
}

#[when("tracing \"{query}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn when_tracing(query: String, #[from(test_context)] context: &TestContext) {
    *context.trace_result.borrow_mut() = Some(context.scorer.trace(&query));
}

#[when("tracing empty query")]
fn when_tracing_empty(#[from(test_context)] context: &TestContext) {
    *context.trace_result.borrow_mut() = Some(context.scorer.trace(""));
}

#[then("the scored complexity components are {scope:f32}, {depth:f32}, {ambiguity:f32}")]
fn then_scored_components(
    #[from(test_context)] context: &TestContext,
    scope: f32,
    depth: f32,
    ambiguity: f32,
) {
    let comp = context.scored_ok();
    assert_close(comp.scope(), scope);
    assert_close(comp.depth(), depth);
    assert_close(comp.ambiguity(), ambiguity);
}

#[then("the scored total is {expected:f32}")]
fn then_scored_total(#[from(test_context)] context: &TestContext, expected: f32) {
    let comp = context.scored_ok();
    assert_close(comp.total(), expected);
}

#[then("the traced complexity components are {scope:f32}, {depth:f32}, {ambiguity:f32}")]
fn then_traced_components(
    #[from(test_context)] context: &TestContext,
    scope: f32,
    depth: f32,
    ambiguity: f32,
) {
    let trace = context.traced_ok();
    assert_close(trace.complexity.scope(), scope);
    assert_close(trace.complexity.depth(), depth);
    assert_close(trace.complexity.ambiguity(), ambiguity);
}

#[then("the traced total is {expected:f32}")]
fn then_traced_total(#[from(test_context)] context: &TestContext, expected: f32) {
    let trace = context.traced_ok();
    assert_close(trace.complexity.total(), expected);
}

#[then("the trace query is \"{expected}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn then_trace_query(expected: String, #[from(test_context)] context: &TestContext) {
    let trace = context.traced_ok();
    assert_eq!(trace.query, expected);
}

#[then("a depth error is returned")]
fn then_depth_error(#[from(test_context)] context: &TestContext) {
    match context.score_result.borrow().as_ref() {
        Some(Err(HeuristicComplexityError::Depth(DepthHeuristicError::Empty))) => {}
        Some(Err(other)) => panic!("expected depth error, got {other}"),
        Some(Ok(_)) => panic!("expected error, got success"),
        None => panic!("score result missing"),
    }
}

#[then("a depth error is returned from trace")]
fn then_depth_error_trace(#[from(test_context)] context: &TestContext) {
    match context.trace_result.borrow().as_ref() {
        Some(Err(HeuristicComplexityError::Depth(DepthHeuristicError::Empty))) => {}
        Some(Err(other)) => panic!("expected depth error, got {other}"),
        Some(Ok(_)) => panic!("expected error, got success"),
        None => panic!("trace result missing"),
    }
}

#[scenario(path = "tests/features/score.feature", index = 0)]
fn score_multi_clause(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/score.feature", index = 1)]
fn score_ambiguous_question(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/score.feature", index = 2)]
fn score_empty_query(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/score.feature", index = 3)]
fn trace_simple_question(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/score.feature", index = 4)]
fn trace_empty_query(test_context: TestContext) {
    let _ = test_context;
}
