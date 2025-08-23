use lag_complexity::cli::LagcArgs;
use ortho_config::OrthoConfig;
use rstest::rstest;

#[rstest]
#[case(vec!["lagc"], false)]
#[case(vec!["lagc", "--dry-run=true"], true)]
fn load_parses_dry_run(#[case] argv: Vec<&str>, #[case] expected: bool) {
    let cfg = LagcArgs::load_from_iter(argv)
        .unwrap_or_else(|e| panic!("unexpected parse error: {e}"));
    assert_eq!(cfg.dry_run, expected);
}

#[rstest]
fn load_rejects_invalid_bool() {
    let result = LagcArgs::load_from_iter(vec!["lagc", "--dry-run=maybe"]);
    assert!(result.is_err());
}
