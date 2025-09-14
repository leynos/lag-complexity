use lag_complexity::{ComplexityFn, HeuristicComplexity, Trace};
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

#[test]
fn golden_traces() -> Result<(), Box<dyn Error>> {
    let hc = HeuristicComplexity::default();
    for line in TRACES_JSONL.lines() {
        let expected: GoldenTrace = from_str(line)?;
        let actual = hc.trace(&expected.trace.query)?;
        assert_eq!(
            actual, expected.trace,
            "mismatch for id {} (query = `{}`)",
            expected.id, expected.trace.query,
        );
    }
    Ok(())
}
