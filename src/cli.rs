use ortho_config::OrthoConfig;
use serde::Deserialize;

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
