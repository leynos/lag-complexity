use figment::{
    Error as FigmentError, Figment,
    providers::{Format, Toml},
};
use ortho_config::{OrthoConfig, OrthoError};
use serde::Deserialize;
use std::env;

/// Command-line arguments for the `lagc` binary.
///
/// This structure is intentionally minimal; it will grow alongside the
/// application's capabilities. Configuration values are loaded from command
/// line arguments, environment variables (prefixed with `LAGC_`), and an
/// optional configuration file.
///
/// # Examples
///
/// ```
/// use lag_complexity::cli::LagcArgs;
/// use ortho_config::OrthoConfig;
///
/// let args = LagcArgs::load_from_iter(["lagc", "--dry-run"]).unwrap();
/// assert!(args.dry_run);
/// ```
#[derive(Debug, Deserialize, OrthoConfig)]
#[ortho_config(prefix = "LAGC")]
pub struct LagcArgs {
    /// Run without performing any side effects.
    #[ortho_config(default = false)]
    pub dry_run: bool,
}

impl LagcArgs {
    /// Load configuration solely from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an [`OrthoError`] if any variable cannot be parsed.
    pub fn load_from_env() -> Result<Self, OrthoError> {
        match env::var("LAGC_DRY_RUN") {
            Ok(v) => {
                let dry_run = v
                    .parse::<bool>()
                    .map_err(|e| OrthoError::gathering(FigmentError::from(e.to_string())))?;
                Ok(Self { dry_run })
            }
            Err(env::VarError::NotPresent) => Ok(Self { dry_run: false }),
            Err(e) => Err(OrthoError::gathering(FigmentError::from(e.to_string()))),
        }
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
        let mut cfg = Self::load_from_config(path)?;
        if let Ok(v) = env::var("LAGC_DRY_RUN") {
            cfg.dry_run = v
                .parse::<bool>()
                .map_err(|e| OrthoError::gathering(FigmentError::from(e.to_string())))?;
        }
        Ok(cfg)
    }
}
