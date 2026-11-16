//! Unit tests for `LagcArgs` argument parsing and validation.
//!
//! Every case loads through an injected [`MapEnv`], so nothing reads or writes
//! the process environment or working directory and the cases run
//! concurrently.

use lag_complexity::cli::LagcArgs;
use ortho_config::{MapEnv, OrthoConfig};
use rstest::{fixture, rstest};
use std::io::Write;
use std::sync::Arc;
use tempfile::NamedTempFile;

/// A directory no test creates, named for every location discovery searches.
const NOWHERE: &str = "/nonexistent-lagc-test";

#[fixture]
fn temp_toml_file() -> std::io::Result<NamedTempFile> {
    NamedTempFile::new()
}

/// Writes the given TOML content into the supplied temporary file, returning
/// the file for subsequent path lookups.
fn written_config(
    temp_toml_file: std::io::Result<NamedTempFile>,
    content: &str,
) -> std::io::Result<NamedTempFile> {
    let mut file = temp_toml_file?;
    writeln!(file, "{content}")?;
    Ok(file)
}

fn get_config_path(file: &NamedTempFile) -> Option<&str> {
    file.path().to_str()
}

/// A closed environment that contributes nothing from the host.
///
/// The working directory is pinned, and `HOME` and the XDG variables name a
/// directory that does not exist, because an unset `HOME` falls back to the
/// host and an empty `XDG_CONFIG_DIRS` selects `/etc/xdg`.
fn isolated_env() -> MapEnv {
    MapEnv::new()
        .with_current_dir(NOWHERE)
        .with_var("HOME", NOWHERE)
        .with_var("XDG_CONFIG_HOME", NOWHERE)
        .with_var("XDG_CONFIG_DIRS", NOWHERE)
}

/// Loads from `argv` through `env`, flattening the error for comparison.
fn load_args(env: MapEnv, argv: &[&str]) -> Result<LagcArgs, String> {
    <LagcArgs as OrthoConfig>::load_from_iter_with_env(argv.iter().copied(), Arc::new(env))
        .map_err(|error| error.to_string())
}

#[rstest]
#[case(vec!["lagc"], false)]
#[case(vec!["lagc", "--dry-run=true"], true)]
fn load_parses_dry_run(
    #[case] argv: Vec<&str>,
    #[case] expected: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = load_args(isolated_env(), &argv)?;
    assert_eq!(cfg.dry_run, expected);
    Ok(())
}

#[rstest]
#[case("maybe")]
#[case("Maybe")]
#[case("MAYBE")]
#[case("1")]
#[case("0")]
#[case("yes")]
fn load_rejects_invalid_bool(#[case] value: &str) {
    let arg = format!("--dry-run={value}");
    let result = load_args(isolated_env(), &["lagc", arg.as_str()]);
    assert!(result.is_err());
}

#[rstest]
fn env_var_parsing_sets_dry_run() -> Result<(), Box<dyn std::error::Error>> {
    let env = MapEnv::new().with_var("LAGC_DRY_RUN", "true");
    let cfg = LagcArgs::load_from_env_source(Arc::new(env))?;
    assert!(cfg.dry_run);
    Ok(())
}

#[rstest]
fn env_var_parsing_invalid_bool() {
    let env = MapEnv::new().with_var("LAGC_DRY_RUN", "notabool");
    let result = LagcArgs::load_from_env_source(Arc::new(env));
    assert!(result.is_err());
}

/// Which loading entry point a configuration-file case exercises.
#[derive(Clone, Copy)]
enum LoadMode {
    /// Load the configuration file alone.
    ConfigFile,
    /// Load with a CLI override on top of the environment and file.
    CliOverride,
    /// Load from the environment merged with the file.
    EnvAndConfig,
}

/// Loads `path` through the entry point `mode` selects, with `env_value` as
/// `LAGC_DRY_RUN` when present.
fn load_in_mode(mode: LoadMode, path: &str, env_value: Option<&str>) -> Result<LagcArgs, String> {
    let env = env_value.map_or_else(isolated_env, |value| {
        isolated_env().with_var("LAGC_DRY_RUN", value)
    });
    match mode {
        LoadMode::ConfigFile => LagcArgs::load_from_config(path).map_err(|e| e.to_string()),
        LoadMode::CliOverride => load_args(env, &["lagc", "--dry-run=true", "--config-path", path]),
        LoadMode::EnvAndConfig => {
            LagcArgs::load_from_source_and_config(Arc::new(env), path).map_err(|e| e.to_string())
        }
    }
}

#[rstest]
#[case::config_sets_dry_run("dry_run = true", None, LoadMode::ConfigFile, Some(true))]
#[case::config_invalid_bool("dry_run = notabool", None, LoadMode::ConfigFile, None)]
#[case::cli_overrides_env_and_config(
    "dry_run = false",
    Some("false"),
    LoadMode::CliOverride,
    Some(true)
)]
#[case::env_overrides_config("dry_run = false", Some("true"), LoadMode::EnvAndConfig, Some(true))]
fn config_file_loading(
    temp_toml_file: std::io::Result<NamedTempFile>,
    #[case] content: &str,
    #[case] env_value: Option<&str>,
    #[case] mode: LoadMode,
    #[case] expected_dry_run: Option<bool>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = written_config(temp_toml_file, content)?;
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    let result = load_in_mode(mode, path, env_value);
    match expected_dry_run {
        Some(expected) => {
            let cfg = result?;
            assert_eq!(cfg.dry_run, expected);
        }
        None => assert!(result.is_err()),
    }
    Ok(())
}

/// Marks the re-run of this binary that reads its own live environment.
const CHILD_MARKER: &str = "LAGC_TEST_PROCESS_BACKED_CHILD";

/// The process-backed loaders read the live process environment on purpose, so
/// the parent re-runs this binary as a child that carries `LAGC_DRY_RUN` and
/// changes nothing in its own process.
///
/// The child inherits the rest of the parent's environment, which keeps
/// `LLVM_PROFILE_FILE` so a coverage run still sees what the child executes.
#[rstest]
fn process_backed_loaders_read_the_live_environment() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os(CHILD_MARKER).is_some() {
        return assert_process_backed_loaders_see_the_environment();
    }
    let status = std::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "process_backed_loaders_read_the_live_environment",
            "--nocapture",
        ])
        .env(CHILD_MARKER, "1")
        .env("LAGC_DRY_RUN", "true")
        .status()?;
    assert!(status.success(), "the child run failed: {status}");
    Ok(())
}

/// The child half: both process-backed loaders see `LAGC_DRY_RUN=true`, and the
/// environment still wins over a file that says `false`.
fn assert_process_backed_loaders_see_the_environment() -> Result<(), Box<dyn std::error::Error>> {
    assert!(LagcArgs::load_from_env()?.dry_run);
    let file = written_config(temp_toml_file(), "dry_run = false")?;
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    assert!(LagcArgs::load_from_env_and_config(path)?.dry_run);
    Ok(())
}
