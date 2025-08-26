//! CLI entrypoint for `lagc`. Loads configuration from CLI flags,
//! environment variables (prefix `LAGC`), and optional config files via
//! `ortho_config`.

use eyre::Result;
use lag_complexity::cli::LagcArgs;

fn main() -> Result<()> {
    // Load configuration from CLI, environment, and config files. Future
    // subcommands will branch from these parsed arguments.
    let _cfg = <LagcArgs as ortho_config::OrthoConfig>::load().map_err(|e| {
        // Prefer typed matching over substring checks for robustness.
        if let ortho_config::OrthoError::CliParsing(clap_err) = &e
            && clap_err.kind() == clap::error::ErrorKind::InvalidValue
        {
            return eyre::eyre!("invalid boolean: {clap_err}");
        }
        e.into()
    })?;
    Ok(())
}
