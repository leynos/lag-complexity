use lag_complexity::{ComplexityFn, HeuristicComplexity, Trace};
use rstest::rstest;
use std::fs::File;
use std::io::{BufRead, BufReader};

#[rstest]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn golden_traces() {
    let file = File::open("tests/golden/traces.jsonl").expect("missing golden file");
    let reader = BufReader::new(file);
    let hc = HeuristicComplexity::default();

    for line in reader.lines() {
        let line = line.expect("read line");
        let expected: Trace = serde_json::from_str(&line).expect("parse trace");
        let actual = hc.trace(&expected.query).expect("trace error");
        assert_eq!(actual, expected, "mismatch for query `{}`", expected.query);
    }
}
