use lag_complexity::{Complexity, ComplexityFn, HeuristicComplexity, Trace};
use serde::Deserialize;
use serde_json::from_str;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, ErrorKind, Write};
use tempfile::NamedTempFile;

const TRACES_JSONL: &str = include_str!("golden/traces.jsonl");

const TOL_ABS: f32 = 1e-5;
const TOL_REL: f32 = 1e-4;

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
    let close = |a: f32, b: f32| {
        let diff = (a - b).abs();
        diff <= TOL_ABS || diff <= TOL_REL * b.abs()
    };
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
        // Invariant: total equals sum of components
        let s =
            actual.complexity.scope() + actual.complexity.depth() + actual.complexity.ambiguity();
        assert!(
            {
                let d = (actual.complexity.total() - s).abs();
                d <= TOL_ABS || d <= TOL_REL * s.abs()
            },
            "total invariant violated for id {} (query = `{}`)",
            expected.id,
            expected.trace.query,
        );
    }
    Ok(())
}

#[test]
fn missing_golden_file() {
    match File::open("tests/golden/missing_file.jsonl") {
        Err(e) => assert_eq!(e.kind(), ErrorKind::NotFound),
        Ok(_) => panic!("unexpectedly opened missing file"),
    }
}

#[test]
fn empty_golden_file() -> Result<(), Box<dyn Error>> {
    let tmp = NamedTempFile::new()?;
    let reader = BufReader::new(File::open(tmp.path())?);
    let hc = HeuristicComplexity::default();
    for line in reader.lines() {
        let line = line?;
        let expected: GoldenTrace = from_str(&line)?;
        let actual = hc.trace(&expected.trace.query)?;
        assert_eq!(actual, expected.trace);
    }
    Ok(())
}

#[test]
fn malformed_golden_file() -> Result<(), Box<dyn Error>> {
    let mut tmp = NamedTempFile::new()?;
    writeln!(tmp, "{{ this is not valid json }}")?;
    let reader = BufReader::new(File::open(tmp.path())?);
    for line in reader.lines() {
        let line = line?;
        let result: Result<GoldenTrace, _> = from_str(&line);
        assert!(result.is_err(), "expected JSON parsing to fail");
    }
    Ok(())
}

#[test]
fn missing_trace_fields() -> Result<(), Box<dyn Error>> {
    let mut tmp = NamedTempFile::new()?;
    tmp.write_all(b"{\"id\":1,\"query\":\"Q\"}\n")?;
    tmp.write_all(b"{\"id\":2,\"complexity\":{\"total\":1.0,\"scope\":0.0,\"depth\":0.0,\"ambiguity\":1.0}}\n")?;
    // Missing a required field inside `complexity` should also fail with an error.
    tmp.write_all(b"{\"id\":3,\"query\":\"Q3\",\"complexity\":{\"scope\":0.0,\"depth\":0.0}}\n")?;
    let reader = BufReader::new(File::open(tmp.path())?);
    for line in reader.lines() {
        let line = line?;
        let result: Result<GoldenTrace, _> = from_str(&line);
        assert!(result.is_err(), "expected deserialisation to fail");
    }
    Ok(())
}
