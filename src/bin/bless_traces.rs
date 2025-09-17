use lag_complexity::{ComplexityFn, HeuristicComplexity, Trace};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::{env, fs::File};
use tempfile::NamedTempFile;

#[derive(Deserialize, Serialize)]
struct GoldenTrace {
    id: u32,
    #[serde(flatten)]
    trace: Trace,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("tests/golden/traces.jsonl"), PathBuf::from);
    let hc = HeuristicComplexity::default();
    let reader = BufReader::new(File::open(&path)?);
    let dir = path
        .parent()
        .ok_or_else(|| "snapshot directory missing".to_owned())?;
    let mut tmp = NamedTempFile::new_in(dir)?;
    {
        let mut writer = BufWriter::new(tmp.as_file_mut());
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let gt: GoldenTrace = serde_json::from_str(&line)?;
            let trace = hc.trace(&gt.trace.query)?;
            let out = GoldenTrace { id: gt.id, trace };
            serde_json::to_writer(&mut writer, &out)?;
            writeln!(&mut writer)?;
        }
        writer.flush()?;
    }
    tmp.as_file_mut().sync_all()?;
    #[cfg(target_family = "windows")]
    let _ = std::fs::remove_file(&path);
    tmp.persist(&path)?;
    Ok(())
}
