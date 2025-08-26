//! Behaviour tests for the `lagc` CLI.

use assert_cmd::Command;
use once_cell::unsync::OnceCell;
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::collections::HashMap;
use std::process::Output;

#[derive(Default)]
struct CliContext {
    env: OnceCell<HashMap<String, String>>,
    output: OnceCell<Output>,
}

#[fixture]
fn cli_context() -> CliContext {
    CliContext::default()
}

#[given("the lagc binary")]
fn given_binary(#[from(cli_context)] ctx: &CliContext) {
    let _ = ctx;
}

#[given("env \"{pairs}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
fn given_env(pairs: String, #[from(cli_context)] ctx: &CliContext) {
    let map = pairs
        .split(',')
        .filter_map(|kv| {
            let mut iter = kv.splitn(2, '=');
            match (iter.next(), iter.next()) {
                (Some(k), Some(v)) => Some((k.trim().to_owned(), v.trim().to_owned())),
                _ => None,
            }
        })
        .collect();
    ctx.env
        .set(map)
        .unwrap_or_else(|_| panic!("env already set"));
}

#[when("running with \"{args}\"")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
fn when_running(args: String, #[from(cli_context)] ctx: &CliContext) {
    let parsed_args = shlex::split(&args)
        .unwrap_or_else(|| args.split_whitespace().map(ToOwned::to_owned).collect());
    let mut cmd = Command::cargo_bin("lagc").expect("failed to locate lagc binary");
    if let Some(env) = ctx.env.get() {
        cmd.envs(env);
    }
    let output = cmd.args(&parsed_args).output().expect("failed to run lagc");
    ctx.output.set(output).expect("output already set");
}

#[when("running with no args")]
fn when_running_empty(#[from(cli_context)] ctx: &CliContext) {
    when_running(String::new(), ctx);
}

#[then("it exits successfully")]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
#[expect(clippy::print_stderr, reason = "diagnose test failures")]
fn then_success(#[from(cli_context)] ctx: &CliContext) {
    let out = ctx.output.get().expect("missing output");
    if !out.status.success() {
        eprintln!("stdout:\n{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    assert!(out.status.success());
}

#[then("it exits with an error")]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
#[expect(clippy::print_stderr, reason = "diagnose test failures")]
fn then_error(#[from(cli_context)] ctx: &CliContext) {
    let out = ctx.output.get().expect("missing output");
    if out.status.success() {
        eprintln!("stdout:\n{}", String::from_utf8_lossy(&out.stdout));
        eprintln!("stderr:\n{}", String::from_utf8_lossy(&out.stderr));
    }
    assert!(!out.status.success());
}

#[then("stderr contains \"{text}\"")]
#[expect(clippy::expect_used, reason = "tests should fail loudly")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
#[expect(clippy::print_stderr, reason = "diagnose test failures")]
fn then_stderr_contains(text: String, #[from(cli_context)] ctx: &CliContext) {
    let out = ctx.output.get().expect("missing output");
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !stderr.contains(&text) {
        eprintln!("stderr:\n{stderr}");
    }
    assert!(stderr.contains(&text));
}

#[scenario(path = "tests/features/lagc_cli.feature", index = 0)]
fn dry_run(cli_context: CliContext) {
    let _ = cli_context;
}

#[scenario(path = "tests/features/lagc_cli.feature", index = 1)]
fn dry_run_disabled(cli_context: CliContext) {
    let _ = cli_context;
}

#[scenario(path = "tests/features/lagc_cli.feature", index = 2)]
fn env_dry_run(cli_context: CliContext) {
    let _ = cli_context;
}

#[scenario(path = "tests/features/lagc_cli.feature", index = 3)]
fn invalid_flag(cli_context: CliContext) {
    let _ = cli_context;
}
