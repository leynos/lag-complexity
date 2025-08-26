use lag_complexity::cli::LagcArgs;
use ortho_config::{OrthoConfig, OrthoError};

fn main() -> Result<(), OrthoError> {
    // Load configuration from CLI, environment, and config files. Future
    // subcommands will branch from these parsed arguments.
    let _cfg = LagcArgs::load()?;
    Ok(())
}
