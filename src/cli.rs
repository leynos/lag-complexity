//! CLI argument types and layered configuration for the `lagc` binary.
//! Loads from CLI args, environment (prefix `LAGC_`), and optional config
//! files.

use figment::{
    Figment,
    providers::{Env, Format, Toml},
};
use ortho_config::OrthoError;
use serde::Deserialize;
use std::path::PathBuf;

/// Command-line arguments for the `lagc` binary.
///
/// This structure is intentionally minimal; it will grow alongside the
/// application's capabilities. Configuration values are loaded from command
/// line arguments, environment variables (prefixed with `LAGC_`), and an
/// optional configuration file.
///
/// # Examples
///
/// Parse flags directly:
/// ```
/// use lag_complexity::cli::LagcArgs;
/// use ortho_config::OrthoConfig;
///
/// let args = LagcArgs::load_from_iter(["lagc", "--dry-run=true"])
///     .expect("load args from CLI iterator");
/// assert!(args.dry_run);
/// ```
///
/// Load from a configuration file:
/// ```
/// use lag_complexity::cli::LagcArgs;
/// use ortho_config::OrthoConfig;
/// use std::io::Write;
/// use tempfile::NamedTempFile;
///
/// let mut file = NamedTempFile::new().expect("create temp file");
/// writeln!(file, "dry_run = true").expect("write config");
/// let path = file.path().to_str().expect("path str");
/// let args = LagcArgs::load_from_iter(["lagc", "--config-path", path])
///     .expect("load args from config path");
/// assert!(args.dry_run);
/// ```
#[derive(Debug, Deserialize, ortho_config::OrthoConfig)]
#[ortho_config(prefix = "LAGC")]
pub struct LagcArgs {
    /// Run without performing any side effects.
    #[ortho_config(default = false)]
    #[serde(default)]
    pub dry_run: bool,

    /// Optional path to a configuration file.
    #[serde(skip)]
    pub config_path: Option<PathBuf>,
}

impl LagcArgs {
    /// Load configuration solely from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if any variable cannot be parsed.
    pub fn load_from_env() -> Result<Self, OrthoError> {
        Figment::new()
            .merge(Env::prefixed("LAGC_"))
            .extract()
            .map_err(Into::into)
    }

    /// Load configuration from a file path.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if the file cannot be read or parsed.
    pub fn load_from_config(path: &str) -> Result<Self, OrthoError> {
        Figment::new()
            .merge(Toml::file(path))
            .extract()
            .map_err(Into::into)
    }

    /// Load configuration from environment variables and a file path.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if either source contains invalid values.
    pub fn load_from_env_and_config(path: &str) -> Result<Self, OrthoError> {
        Figment::new()
            .merge(Toml::file(path))
            .merge(Env::prefixed("LAGC_"))
            .extract()
            .map_err(Into::into)
    }
}
