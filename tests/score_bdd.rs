//! Behavioural tests exercising heuristic scoring and tracing.
//! Validates `score()` and `trace()` flows, totals, components, and error propagation.

use lag_complexity::{
    Complexity, ComplexityFn, DepthHeuristicError, HeuristicComplexity, HeuristicComplexityError,
    Trace,
};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::{Ref, RefCell};

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

    fn traced_ok(&self) -> Ref<'_, Trace> {
        let borrow = self.trace_result.borrow();
        match borrow.as_ref() {
            Some(Ok(_)) => {}
            Some(Err(err)) => panic!("expected trace to succeed, got error: {err}"),
            None => panic!("trace result missing"),
        }
        Ref::map(borrow, |maybe| match maybe {
            Some(Ok(trace)) => trace,
            _ => unreachable!("validated above"),
        })
    }
}

fn assert_close(actual: f32, expected: f32) {
    const ABS_TOLERANCE: f32 = 1e-5;
    const REL_TOLERANCE: f32 = 1e-4;

    #[expect(
        clippy::float_arithmetic,
        reason = "tests compare floating point values"
    )]
    let abs = (actual - expected).abs();
    #[expect(
        clippy::float_arithmetic,
        reason = "tests compare floating point values"
    )]
    let rel = abs / expected.abs().max(1.0);
    assert!(
        abs <= ABS_TOLERANCE || rel <= REL_TOLERANCE,
        "expected ≈{expected}, got {actual} (abs={abs}, rel={rel})"
    );
}

fn assert_depth_empty_error<T>(
    cell: &RefCell<Option<Result<T, HeuristicComplexityError>>>,
    label: &str,
) {
    match cell.borrow().as_ref() {
        Some(Err(HeuristicComplexityError::Depth(DepthHeuristicError::Empty))) => {}
        Some(Err(other)) => panic!("expected depth error, got {other}"),
        Some(Ok(_)) => panic!("expected error, got success"),
        None => panic!("{label} result missing"),
    }
}

#[fixture]
fn test_context() -> TestContext {
    TestContext::default()
}

#[given("a heuristic complexity scorer")]
fn given_scorer() {}

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
    assert_depth_empty_error(&context.score_result, "score");
}

#[then("a depth error is returned from trace")]
fn then_depth_error_trace(#[from(test_context)] context: &TestContext) {
    assert_depth_empty_error(&context.trace_result, "trace");
}

#[scenario(path = "tests/features/heuristic_scoring/multi_clause.feature")]
fn score_multi_clause(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/heuristic_scoring/ambiguous.feature")]
fn score_ambiguous_question(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/heuristic_scoring/empty_query.feature")]
fn score_empty_query(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/heuristic_scoring/pronoun_context.feature")]
fn score_pronoun_with_context(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(
    path = "tests/features/heuristic_scoring/pronoun_context.feature",
    index = 1
)]
fn score_pronoun_without_antecedent(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(
    path = "tests/features/heuristic_scoring/pronoun_context.feature",
    index = 2
)]
fn score_multiple_pronouns(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(
    path = "tests/features/heuristic_scoring/pronoun_context.feature",
    index = 3
)]
fn score_ambiguous_antecedents(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(
    path = "tests/features/heuristic_scoring/pronoun_context.feature",
    index = 4
)]
fn score_idiomatic_pronoun_usage(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/heuristic_scoring/trace_simple.feature")]
fn trace_simple_question(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/heuristic_scoring/trace_empty.feature")]
fn trace_empty_query(test_context: TestContext) {
    let _ = test_context;
}
