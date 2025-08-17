use lag_complexity::{Complexity, ComplexityFn, Trace};
use rstest_bdd_macros::{given, scenario, then, when};
use std::sync::{LazyLock, Mutex};
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

static RESULT: LazyLock<Mutex<Option<Result<Complexity, DummyError>>>> =
    LazyLock::new(|| Mutex::new(None));

#[given("a dummy complexity function")]
fn given_function() {}

#[when("scoring {query}")]
#[expect(clippy::needless_pass_by_value, reason = "step takes ownership")]
fn when_scoring(query: String) {
    let res = Dummy.score(&query);
    let mut guard = RESULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some(res);
}

#[when("scoring empty query")]
fn when_scoring_empty() {
    let mut guard = RESULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    *guard = Some(Err(DummyError::Empty));
}

#[then("the total score is {expected:f32}")]
fn then_total(expected: f32) {
    let result = RESULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(Ok(comp)) = result.as_ref() {
        #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
        let diff = comp.total - expected;
        assert!(diff.abs() < f32::EPSILON);
    } else {
        panic!("result missing");
    }
}

#[then("an error is returned")]
fn then_error() {
    let result = RESULT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert!(matches!(result.as_ref(), Some(Err(_))));
}

#[scenario(path = "tests/features/score.feature", index = 0)]
fn score_valid() {}

#[scenario(path = "tests/features/score.feature", index = 1)]
fn score_empty() {}
