//! Behaviour tests for the `lagc` CLI.

use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;
use assert_cmd::Command;
use std::process::Output;

#[derive(Default)]
struct CliContext {
    output: RefCell<Option<Output>>,
}

#[fixture]
fn cli_context() -> CliContext {
    CliContext::default()
}

#[given("the lagc binary")]
fn given_binary(#[from(cli_context)] ctx: &CliContext) { let _ = ctx; }

#[when("running with \"{args}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
fn when_running(args: String, #[from(cli_context)] ctx: &CliContext) {
    let output = Command::cargo_bin("lagc")
        .unwrap_or_else(|e| panic!("failed to locate lagc binary: {e}"))
        .args(args.split_whitespace())
        .output()
        .expect("failed to run lagc");
    *ctx.output.borrow_mut() = Some(output);
}

#[then("it exits successfully")]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
fn then_success(#[from(cli_context)] ctx: &CliContext) {
    let status = ctx.output.borrow().as_ref().expect("missing output").status;
    assert!(status.success());
}

#[then("it exits with an error")]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
fn then_error(#[from(cli_context)] ctx: &CliContext) {
    let status = ctx.output.borrow().as_ref().expect("missing output").status;
    assert!(!status.success());
}

#[scenario(path = "tests/features/lagc_cli.feature", index = 0)]
fn dry_run(cli_context: CliContext) { let _ = cli_context; }

#[scenario(path = "tests/features/lagc_cli.feature", index = 1)]
fn invalid_flag(cli_context: CliContext) { let _ = cli_context; }
