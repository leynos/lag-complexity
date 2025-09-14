use lag_complexity::{Complexity, ComplexityFn, HeuristicComplexity, Trace};
use serde::Deserialize;
use serde_json::from_str;
use std::error::Error;

const TRACES_JSONL: &str = include_str!("golden/traces.jsonl");

#[derive(Deserialize)]
struct GoldenTrace {
    id: u32,
    #[serde(flatten)]
    trace: Trace,
}

#[expect(
    clippy::float_arithmetic,
    reason = "compute delta between expected and actual values"
)]
fn assert_complexity_close(actual: &Complexity, expected: &Complexity, id: u32, query: &str) {
    let close = |a: f32, b: f32| (a - b).abs() < f32::EPSILON;
    assert!(
        close(actual.total(), expected.total()),
        "total mismatch for id {id} (query = `{query}`)"
    );
    assert!(
        close(actual.scope(), expected.scope()),
        "scope mismatch for id {id} (query = `{query}`)"
    );
    assert!(
        close(actual.depth(), expected.depth()),
        "depth mismatch for id {id} (query = `{query}`)"
    );
    assert!(
        close(actual.ambiguity(), expected.ambiguity()),
        "ambiguity mismatch for id {id} (query = `{query}`)"
    );
}

#[test]
fn golden_traces() -> Result<(), Box<dyn Error>> {
    let hc = HeuristicComplexity::default();
    for line in TRACES_JSONL.lines() {
        let expected: GoldenTrace = from_str(line)?;
        let actual = hc.trace(&expected.trace.query)?;
        assert_eq!(
            actual.query, expected.trace.query,
            "query mismatch for id {}",
            expected.id,
        );
        assert_complexity_close(
            &actual.complexity,
            &expected.trace.complexity,
            expected.id,
            &expected.trace.query,
        );
    }
    Ok(())
}
