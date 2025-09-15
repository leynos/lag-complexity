use lag_complexity::{ComplexityFn, HeuristicComplexity, Trace};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

#[derive(Deserialize, Serialize)]
struct GoldenTrace {
    id: u32,
    #[serde(flatten)]
    trace: Trace,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = PathBuf::from("tests/golden/traces.jsonl");
    let hc = HeuristicComplexity::default();
    let reader = BufReader::new(File::open(&path)?);
    let mut traces = Vec::new();
    for line in reader.lines() {
        let line = line?;
        let gt: GoldenTrace = serde_json::from_str(&line)?;
        let trace = hc.trace(&gt.trace.query)?;
        traces.push(GoldenTrace { id: gt.id, trace });
    }
    let mut writer = BufWriter::new(File::create(&path)?);
    for (i, t) in traces.iter().enumerate() {
        let line = serde_json::to_string(t)?;
        writer.write_all(line.as_bytes())?;
        if i + 1 < traces.len() {
            writer.write_all(b"\n")?;
        }
    }
    writer.flush()?;
    Ok(())
}
