//! Unit tests for `LagcArgs` argument parsing and validation.

use lag_complexity::cli::LagcArgs;
use rstest::{fixture, rstest};
use serial_test::serial;
use std::env;
use std::io::Write;
use std::sync::{LazyLock, Mutex, MutexGuard};
use tempfile::NamedTempFile;

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

static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct EnvVarGuard {
    key: String,
    _lock: MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    fn new(key: &str, val: &str) -> Self {
        // Recover from poisoning: the guarded state is (), so a panic in a
        // previous holder cannot leave it inconsistent.
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Safety: process-wide env mutation is synchronised by ENV_LOCK.
        unsafe { env::set_var(key, val) };
        Self {
            key: key.to_owned(),
            _lock: lock,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // Safety: process-wide env mutation is synchronised by ENV_LOCK.
        unsafe { env::remove_var(&self.key) };
    }
}

#[rstest]
#[case(vec!["lagc"], false)]
#[case(vec!["lagc", "--dry-run=true"], true)]
fn load_parses_dry_run(#[case] argv: Vec<&str>, #[case] expected: bool) {
    let cfg = <LagcArgs as ortho_config::OrthoConfig>::load_from_iter(argv)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert_eq!(cfg.dry_run, expected);
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
    let result = <LagcArgs as ortho_config::OrthoConfig>::load_from_iter(["lagc", arg.as_str()]);
    assert!(result.is_err());
}

#[rstest]
#[serial]
fn env_var_parsing_sets_dry_run() {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "true");
    let cfg =
        LagcArgs::load_from_env().unwrap_or_else(|e| panic!("unexpected env parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
#[serial]
fn env_var_parsing_invalid_bool() {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "notabool");
    let result = LagcArgs::load_from_env();
    assert!(result.is_err());
}

#[rstest]
fn config_file_parsing_sets_dry_run(temp_toml_file: std::io::Result<NamedTempFile>) {
    let file = written_config(temp_toml_file, "dry_run = true")
        .unwrap_or_else(|e| panic!("prepare config: {e}"));
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    let cfg = LagcArgs::load_from_config(path)
        .unwrap_or_else(|e| panic!("unexpected config parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
fn config_file_parsing_invalid_bool(temp_toml_file: std::io::Result<NamedTempFile>) {
    let file = written_config(temp_toml_file, "dry_run = notabool")
        .unwrap_or_else(|e| panic!("prepare config: {e}"));
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    let result = LagcArgs::load_from_config(path);
    assert!(result.is_err());
}

#[rstest]
#[serial]
fn precedence_cli_over_env_and_config(temp_toml_file: std::io::Result<NamedTempFile>) {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "false");
    let file = written_config(temp_toml_file, "dry_run = false")
        .unwrap_or_else(|e| panic!("prepare config: {e}"));
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    let argv = vec!["lagc", "--dry-run=true", "--config-path", path];
    let cfg = <LagcArgs as ortho_config::OrthoConfig>::load_from_iter(argv)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
#[serial]
fn precedence_env_over_config(temp_toml_file: std::io::Result<NamedTempFile>) {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "true");
    let file = written_config(temp_toml_file, "dry_run = false")
        .unwrap_or_else(|e| panic!("prepare config: {e}"));
    let Some(path) = get_config_path(&file) else {
        panic!("config path is not valid UTF-8")
    };
    let cfg = LagcArgs::load_from_env_and_config(path)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
}
