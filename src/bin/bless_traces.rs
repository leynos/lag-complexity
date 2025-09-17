//! Regenerate golden trace snapshots (JSONL) deterministically.
//!
//! Usage: `cargo run --bin bless_traces -- [PATH]` (default:
//! `tests/golden/traces.jsonl`). Recomputes each record with the
//! default heuristics, writes to a temporary file, then replaces the
//! snapshot (atomic on Unix; best-effort on Windows).
use lag_complexity::{ComplexityFn, HeuristicComplexity, Trace};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
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
    let mut reader = BufReader::new(File::open(&path)?);
    let dir = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::NotFound, "snapshot directory missing")
    })?;
    let mut tmp = NamedTempFile::new_in(dir)?;
    {
        let mut writer = BufWriter::new(tmp.as_file_mut());
        for (i, line) in reader.by_ref().lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let gt: GoldenTrace = serde_json::from_str(&line).map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("line {}: {e}", i + 1),
                )
            })?;
            let trace = hc.trace(&gt.trace.query)?;
            let out = GoldenTrace { id: gt.id, trace };
            let record = serde_json::to_vec(&out)?;
            if record.contains(&b'\n') {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("serialised trace {} spans multiple lines", gt.id),
                )
                .into());
            }
            writer.write_all(&record)?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
    }
    tmp.as_file_mut().sync_all()?;
    drop(reader);
    let temp_path = tmp.into_temp_path();
    #[cfg(target_family = "unix")]
    {
        temp_path.persist(&path).map_err(std::io::Error::from)?;
        // Ensure directory entry is durable (Unix only; opening directories on Windows is unsupported).
        std::fs::File::open(dir)?.sync_all()?;
    }
    #[cfg(target_family = "windows")]
    {
        let _ = std::fs::remove_file(&path);
        std::fs::rename(temp_path.as_ref(), &path)?;
    }
    #[cfg(not(any(target_family = "unix", target_family = "windows")))]
    {
        temp_path.persist(&path).map_err(std::io::Error::from)?;
    }
    Ok(())
}
