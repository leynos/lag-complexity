//! BDD tests for scoring behaviour.

use lag_complexity::{Complexity, ComplexityFn, Trace};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;
use thiserror::Error;

#[derive(Default, Clone)]
struct Dummy;

#[derive(Debug, Error, PartialEq, Eq, Clone)]
enum DummyError {
    #[error("empty query")]
    Empty,
}

impl ComplexityFn for Dummy {
    type Error = DummyError;

    fn score(&self, query: &str) -> Result<Complexity, Self::Error> {
        if query.is_empty() {
            Err(DummyError::Empty)
        } else {
            Ok(Complexity::new(1.0, 1.0, 1.0))
        }
    }

    fn trace(&self, query: &str) -> Result<Trace, Self::Error> {
        self.score(query).map(|complexity| Trace {
            query: query.to_string(),
            complexity,
        })
    }
}

/// Fixtures shared across BDD steps.
///
/// Each scenario receives an isolated instance to avoid shared mutable state.
#[derive(Default)]
struct TestContext {
    dummy: Dummy,
    result: RefCell<Option<Result<Complexity, DummyError>>>,
}

#[fixture]
fn test_context() -> TestContext {
    TestContext::default()
}

#[given("a dummy complexity function")]
fn given_function(#[from(test_context)] context: &TestContext) {
    let _ = context;
}

#[when("scoring {query}")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn when_scoring(query: String, #[from(test_context)] context: &TestContext) {
    *context.result.borrow_mut() = Some(context.dummy.score(&query));
}

#[when("scoring empty query")]
fn when_scoring_empty(#[from(test_context)] context: &TestContext) {
    *context.result.borrow_mut() = Some(context.dummy.score(""));
}

#[then("the total score is {expected:f32}")]
fn then_total(#[from(test_context)] context: &TestContext, expected: f32) {
    if let Some(Ok(comp)) = context.result.borrow().as_ref() {
        #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
        let diff = comp.total() - expected;
        assert!(diff.abs() < f32::EPSILON);
    } else {
        panic!("result missing");
    }
}

#[then("an error is returned")]
fn then_error(#[from(test_context)] context: &TestContext) {
    assert!(matches!(context.result.borrow().as_ref(), Some(Err(_))));
}

#[scenario(path = "tests/features/score.feature", index = 0)]
fn score_valid(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/score.feature", index = 1)]
fn score_empty(test_context: TestContext) {
    let _ = test_context;
}

#[test]
fn dummy_trace_returns_trace_struct() {
    let dummy = Dummy;
    let query = "valid_query";
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    let trace = dummy.trace(query).expect("unexpected error");
    assert_eq!(trace.query, query);
    assert_eq!(trace.complexity, Complexity::new(1.0, 1.0, 1.0));
}

#[test]
fn dummy_trace_handles_empty_query_error() {
    let dummy = Dummy;
    assert_eq!(dummy.trace(""), Err(DummyError::Empty));
}
