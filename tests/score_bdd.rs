use lag_complexity::{Complexity, ComplexityFn, Trace};
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

thread_local! {
    static RESULT: RefCell<Option<Result<Complexity, DummyError>>> = const { RefCell::new(None) };
}

#[given("a dummy complexity function")]
fn given_function() {}

#[when("scoring {query}")]
#[expect(clippy::needless_pass_by_value, reason = "step takes ownership")]
fn when_scoring(query: String) {
    RESULT.with(|r| r.replace(Some(Dummy.score(&query))));
}

#[when("scoring empty query")]
fn when_scoring_empty() {
    RESULT.with(|r| r.replace(Some(Err(DummyError::Empty))));
}

#[then("the total score is {expected:f32}")]
fn then_total(expected: f32) {
    RESULT.with(|r| {
        if let Some(Ok(comp)) = r.borrow().as_ref() {
            #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
            let diff = comp.total - expected;
            assert!(diff.abs() < f32::EPSILON);
        } else {
            panic!("result missing");
        }
    });
}

#[then("an error is returned")]
fn then_error() {
    RESULT.with(|r| assert!(matches!(r.borrow().as_ref(), Some(Err(_)))));
}

#[scenario(path = "tests/features/score.feature", index = 0)]
fn score_valid() {}

#[scenario(path = "tests/features/score.feature", index = 1)]
fn score_empty() {}

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
