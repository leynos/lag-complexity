//! CLI entrypoint for `lagc`. Loads configuration from CLI flags,
//! environment variables (prefix `LAGC`), and optional config files via
//! `ortho_config`.

use eyre::Result;
use lag_complexity::cli::LagcArgs;

fn main() -> Result<()> {
    // Load configuration from CLI, environment, and config files. Future
    // subcommands will branch from these parsed arguments.
    let _cfg = <LagcArgs as ortho_config::OrthoConfig>::load().map_err(|e| {
        if e.to_string().contains("invalid value") {
            eyre::eyre!("invalid boolean: {e}")
        } else {
            e.into()
        }
    })?;
    Ok(())
}
