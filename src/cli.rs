//! CLI argument types and layered configuration for the `lagc` binary.
//! Loads from CLI args, environment (prefix `LAGC_`), and optional config
//! files.

use figment::{
    Figment,
    providers::{Env, Format, Json, Toml, Yaml},
};
use ortho_config::OrthoError;
use serde::Deserialize;
use std::path::{Path, PathBuf};

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
/// # Loading from a TOML file
///
/// ```no_run
/// use lag_complexity::cli::LagcArgs;
/// use ortho_config::OrthoConfig;
///
/// // With a config.toml file containing: dry_run = true
/// let args = LagcArgs::load_from_config("config.toml")
///     .expect("load args from TOML file");
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
    #[ortho_config(cli_long = "config-path")]
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

    /// Load configuration from a TOML file path.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if the file cannot be read or parsed as TOML.
    pub fn load_from_config(path: &str) -> Result<Self, OrthoError> {
        Figment::new()
            .merge(Toml::file(path))
            .extract()
            .map_err(Into::into)
    }

    /// Load configuration from a path, selecting a provider by extension.
    ///
    /// Supported formats: `.toml`, `.yaml`/`.yml`, `.json`. Unknown extensions
    /// default to TOML.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if the file cannot be read or parsed in the
    /// chosen format.
    pub fn load_from_path(path: &Path) -> Result<Self, OrthoError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        let figment = match ext.as_deref() {
            Some("yaml" | "yml") => Figment::new().merge(Yaml::file(path)),
            Some("json") => Figment::new().merge(Json::file(path)),
            _ => Figment::new().merge(Toml::file(path)),
        };
        figment.extract().map_err(Into::into)
    }

    /// Load configuration from environment variables and a TOML file path.
    ///
    /// Environment values take precedence over the file.
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
