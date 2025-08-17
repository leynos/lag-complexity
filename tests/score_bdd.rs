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

#[fixture]
fn dummy() -> Dummy {
    Dummy
}

#[fixture]
fn outcome() -> RefCell<Option<Result<Complexity, DummyError>>> {
    RefCell::new(None)
}

#[given("a dummy complexity function")]
fn given_function() {}

#[when("scoring {query}")]
#[expect(clippy::needless_pass_by_value, reason = "step takes ownership")]
fn when_scoring(
    query: String,
    #[from(dummy)] scorer: &Dummy,
    #[from(outcome)] result: &RefCell<Option<Result<Complexity, DummyError>>>,
) {
    result.replace(Some(scorer.score(&query)));
}

#[when("scoring empty query")]
fn when_scoring_empty(
    #[from(dummy)] scorer: &Dummy,
    #[from(outcome)] result: &RefCell<Option<Result<Complexity, DummyError>>>,
) {
    result.replace(Some(scorer.score("")));
}

#[then("the total score is {expected:f32}")]
fn then_total(
    #[from(outcome)] result: &RefCell<Option<Result<Complexity, DummyError>>>,
    expected: f32,
) {
    if let Some(Ok(comp)) = result.borrow().as_ref() {
        #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
        let diff = comp.total - expected;
        assert!(diff.abs() < f32::EPSILON);
    } else {
        panic!("result missing");
    }
}

#[then("an error is returned")]
fn then_error(#[from(outcome)] result: &RefCell<Option<Result<Complexity, DummyError>>>) {
    assert!(matches!(result.borrow().as_ref(), Some(Err(_))));
}

#[scenario(path = "tests/features/score.feature", index = 0)]
fn score_valid(dummy: Dummy, outcome: RefCell<Option<Result<Complexity, DummyError>>>) {
    let _ = (dummy, outcome);
}

#[scenario(path = "tests/features/score.feature", index = 1)]
fn score_empty(dummy: Dummy, outcome: RefCell<Option<Result<Complexity, DummyError>>>) {
    let _ = (dummy, outcome);
}

#[test]
fn dummy_trace_returns_trace_struct() {
    let dummy = Dummy;
    let query = "valid_query";
    let trace = dummy
        .trace(query)
        .unwrap_or_else(|e| panic!("unexpected error: {e}"));
    assert_eq!(trace.query, query);
    assert_eq!(trace.complexity, Complexity::new(1.0, 1.0, 1.0));
}

#[test]
fn dummy_trace_handles_empty_query_error() {
    let dummy = Dummy;
    assert_eq!(dummy.trace(""), Err(DummyError::Empty));
}
