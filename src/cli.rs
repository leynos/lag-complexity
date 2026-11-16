//! CLI argument types and layered configuration for the `lagc` binary.
//! Loads from CLI args, environment (prefix `LAGC_`), and optional config
//! files.

use figment::{
    Figment,
    providers::{Json, Toml, Yaml},
};
use ortho_config::{CsvEnv, OrthoError, ProcessEnv, SharedScanEnvSource};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// The prefix every `lagc` environment variable carries.
const ENV_PREFIX: &str = "LAGC_";

/// The environment layer, read from `source` rather than from the process.
fn env_layer(source: SharedScanEnvSource) -> CsvEnv {
    CsvEnv::prefixed(ENV_PREFIX).with_source(source)
}

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
        Self::load_from_env_source(Arc::new(ProcessEnv))
    }

    /// Load configuration solely from the variables `source` supplies.
    ///
    /// [`load_from_env`](Self::load_from_env) passes the process environment;
    /// a test passes a closed [`ortho_config::MapEnv`], so nothing reads or
    /// writes the process.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if any variable cannot be parsed.
    ///
    /// # Examples
    ///
    /// ```
    /// use lag_complexity::cli::LagcArgs;
    /// use ortho_config::MapEnv;
    /// use std::sync::Arc;
    ///
    /// let env = MapEnv::new().with_var("LAGC_DRY_RUN", "true");
    /// let args = LagcArgs::load_from_env_source(Arc::new(env))
    ///     .expect("load args from an injected environment");
    /// assert!(args.dry_run);
    /// ```
    pub fn load_from_env_source(source: SharedScanEnvSource) -> Result<Self, OrthoError> {
        Figment::new()
            .merge(env_layer(source))
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
            .merge(<Toml as figment::providers::Format>::file(path))
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
            Some("yaml" | "yml") => {
                Figment::new().merge(<Yaml as figment::providers::Format>::file(path))
            }
            Some("json") => Figment::new().merge(<Json as figment::providers::Format>::file(path)),
            _ => Figment::new().merge(<Toml as figment::providers::Format>::file(path)),
        };
        figment.extract().map_err(Into::into)
    }

    /// Load configuration from environment variables and a TOML file path.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if either source contains invalid values.
    /// Environment takes precedence over file values.
    pub fn load_from_env_and_config(path: &str) -> Result<Self, OrthoError> {
        Self::load_from_source_and_config(Arc::new(ProcessEnv), path)
    }

    /// Load configuration from the variables `source` supplies and a TOML file
    /// path.
    ///
    /// [`load_from_env_and_config`](Self::load_from_env_and_config) passes the
    /// process environment.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if either source contains invalid values.
    /// Environment takes precedence over file values.
    pub fn load_from_source_and_config(
        source: SharedScanEnvSource,
        path: &str,
    ) -> Result<Self, OrthoError> {
        Figment::new()
            .merge(<Toml as figment::providers::Format>::file(path))
            .merge(env_layer(source))
            .extract()
            .map_err(Into::into)
    }
}
