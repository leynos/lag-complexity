//! Unit tests for `LagcArgs` argument parsing and validation.

use lag_complexity::cli::LagcArgs;
use rstest::{fixture, rstest};
use serial_test::serial;
use std::env;
use std::io::Write;
use std::sync::{LazyLock, Mutex, MutexGuard};
use tempfile::NamedTempFile;

#[fixture]
fn temp_toml_file() -> NamedTempFile {
    NamedTempFile::new().unwrap_or_else(|e| panic!("create temp file: {e}"))
}

fn write_toml_content(file: &mut NamedTempFile, content: &str) {
    writeln!(file, "{content}").unwrap_or_else(|e| panic!("write config: {e}"));
}

fn get_config_path(file: &NamedTempFile) -> &str {
    file.path().to_str().unwrap_or_else(|| panic!("path str"))
}

static ENV_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

struct EnvVarGuard {
    key: String,
    _lock: MutexGuard<'static, ()>,
}

impl EnvVarGuard {
    fn new(key: &str, val: &str) -> Self {
        let lock = ENV_LOCK
            .lock()
            .unwrap_or_else(|e| panic!("env lock poisoned: {e}"));
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
fn config_file_parsing_sets_dry_run(mut temp_toml_file: NamedTempFile) {
    write_toml_content(&mut temp_toml_file, "dry_run = true");
    let path = get_config_path(&temp_toml_file);
    let cfg = LagcArgs::load_from_config(path)
        .unwrap_or_else(|e| panic!("unexpected config parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
fn config_file_parsing_invalid_bool(mut temp_toml_file: NamedTempFile) {
    write_toml_content(&mut temp_toml_file, "dry_run = notabool");
    let path = get_config_path(&temp_toml_file);
    let result = LagcArgs::load_from_config(path);
    assert!(result.is_err());
}

#[rstest]
#[serial]
fn precedence_cli_over_env_and_config(mut temp_toml_file: NamedTempFile) {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "false");
    write_toml_content(&mut temp_toml_file, "dry_run = false");
    let path = get_config_path(&temp_toml_file);
    let argv = vec!["lagc", "--dry-run=true", "--config-path", path];
    let cfg = <LagcArgs as ortho_config::OrthoConfig>::load_from_iter(argv)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
#[serial]
fn precedence_env_over_config(mut temp_toml_file: NamedTempFile) {
    let _guard = EnvVarGuard::new("LAGC_DRY_RUN", "true");
    write_toml_content(&mut temp_toml_file, "dry_run = false");
    let path = get_config_path(&temp_toml_file);
    let cfg = LagcArgs::load_from_env_and_config(path)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
}
