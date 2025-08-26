use lag_complexity::cli::LagcArgs;
use ortho_config::OrthoConfig;
use rstest::rstest;
use serial_test::serial;
use std::env;
use std::io::Write;
use tempfile::NamedTempFile;

fn set_var(key: &str, val: &str) {
    // Safety: tests manipulate process-wide environment serially.
    unsafe { env::set_var(key, val) };
}

fn remove_var(key: &str) {
    // Safety: tests manipulate process-wide environment serially.
    unsafe { env::remove_var(key) };
}

#[rstest]
#[case(vec!["lagc"], false)]
#[case(vec!["lagc", "--dry-run=true"], true)]
fn load_parses_dry_run(#[case] argv: Vec<&str>, #[case] expected: bool) {
    let cfg =
        LagcArgs::load_from_iter(argv).unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
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
    let result = LagcArgs::load_from_iter(["lagc", arg.as_str()]);
    assert!(result.is_err());
}

#[rstest]
#[serial]
fn env_var_parsing_sets_dry_run() {
    set_var("LAGC_DRY_RUN", "true");
    let cfg =
        LagcArgs::load_from_env().unwrap_or_else(|e| panic!("unexpected env parse error: {e}"));
    assert!(cfg.dry_run);
    remove_var("LAGC_DRY_RUN");
}

#[rstest]
#[serial]
fn env_var_parsing_invalid_bool() {
    set_var("LAGC_DRY_RUN", "notabool");
    let result = LagcArgs::load_from_env();
    assert!(result.is_err());
    remove_var("LAGC_DRY_RUN");
}

#[rstest]
fn config_file_parsing_sets_dry_run() {
    let mut file = NamedTempFile::new().unwrap_or_else(|e| panic!("create temp file: {e}"));
    writeln!(file, "dry_run = true").unwrap_or_else(|e| panic!("write config: {e}"));
    let path = file.path().to_str().unwrap_or_else(|| panic!("path str"));
    let cfg = LagcArgs::load_from_config(path)
        .unwrap_or_else(|e| panic!("unexpected config parse error: {e}"));
    assert!(cfg.dry_run);
}

#[rstest]
fn config_file_parsing_invalid_bool() {
    let mut file = NamedTempFile::new().unwrap_or_else(|e| panic!("create temp file: {e}"));
    writeln!(file, "dry_run = notabool").unwrap_or_else(|e| panic!("write config: {e}"));
    let path = file.path().to_str().unwrap_or_else(|| panic!("path str"));
    let result = LagcArgs::load_from_config(path);
    assert!(result.is_err());
}

#[rstest]
#[serial]
fn precedence_cli_over_env_and_config() {
    set_var("LAGC_DRY_RUN", "false");
    let mut file = NamedTempFile::new().unwrap_or_else(|e| panic!("create temp file: {e}"));
    writeln!(file, "dry_run = false").unwrap_or_else(|e| panic!("write config: {e}"));
    let path = file.path().to_str().unwrap_or_else(|| panic!("path str"));
    let argv = vec!["lagc", "--dry-run=true", "--config-path", path];
    let cfg =
        LagcArgs::load_from_iter(argv).unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
    remove_var("LAGC_DRY_RUN");
}

#[rstest]
#[serial]
fn precedence_env_over_config() {
    set_var("LAGC_DRY_RUN", "true");
    let mut file = NamedTempFile::new().unwrap_or_else(|e| panic!("create temp file: {e}"));
    writeln!(file, "dry_run = false").unwrap_or_else(|e| panic!("write config: {e}"));
    let path = file.path().to_str().unwrap_or_else(|| panic!("path str"));
    let cfg = LagcArgs::load_from_env_and_config(path)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert!(cfg.dry_run);
    remove_var("LAGC_DRY_RUN");
}
