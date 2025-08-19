//! BDD tests for Sigma normalisation strategies.

use lag_complexity::{Sigma, tests::support::approx_eq};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;

#[derive(Default)]
struct SigmaContext {
    sigma: RefCell<Option<Sigma>>,
    result: RefCell<Option<f32>>,
}

#[fixture]
fn sigma_context() -> SigmaContext {
    SigmaContext::default()
}

#[given("a minmax sigma")]
fn given_minmax(#[from(sigma_context)] ctx: &SigmaContext) {
    ctx.sigma.replace(Some(Sigma::MinMax {
        p01: 0.0,
        p99: 10.0,
    }));
}

#[given("a zscore sigma with zero std")]
fn given_zscore_zero(#[from(sigma_context)] ctx: &SigmaContext) {
    ctx.sigma.replace(Some(Sigma::ZScore {
        mean: 0.0,
        std: 0.0,
    }));
}

#[given("a robust sigma")]
fn given_robust(#[from(sigma_context)] ctx: &SigmaContext) {
    ctx.sigma.replace(Some(Sigma::Robust {
        median: 0.0,
        mad: 1.0,
    }));
}

#[given("a robust sigma with zero mad")]
fn given_robust_zero(#[from(sigma_context)] ctx: &SigmaContext) {
    ctx.sigma.replace(Some(Sigma::Robust {
        median: 0.0,
        mad: 0.0,
    }));
}

#[when("normalising {value:f32}")]
fn when_normalising(value: f32, #[from(sigma_context)] ctx: &SigmaContext) {
    let binding = ctx.sigma.borrow();
    let sigma = binding
        .as_ref()
        .unwrap_or_else(|| panic!("sigma to be set"));
    ctx.result.replace(Some(sigma.apply(value)));
}

#[then("the result is {expected:f32}")]
fn then_result(expected: f32, #[from(sigma_context)] ctx: &SigmaContext) {
    let result = ctx
        .result
        .borrow()
        .as_ref()
        .copied()
        .unwrap_or_else(|| panic!("result to be set"));
    assert!(approx_eq(result, expected, 1e-4));
}

#[scenario(path = "tests/features/sigma.feature", index = 0)]
fn minmax_in_range(sigma_context: SigmaContext) {
    let _ = sigma_context;
}

#[scenario(path = "tests/features/sigma.feature", index = 1)]
fn minmax_clamped_below(sigma_context: SigmaContext) {
    let _ = sigma_context;
}

#[scenario(path = "tests/features/sigma.feature", index = 2)]
fn minmax_clamped_above(sigma_context: SigmaContext) {
    let _ = sigma_context;
}

#[scenario(path = "tests/features/sigma.feature", index = 3)]
fn zscore_zero_std(sigma_context: SigmaContext) {
    let _ = sigma_context;
}

#[scenario(path = "tests/features/sigma.feature", index = 4)]
fn robust_standard(sigma_context: SigmaContext) {
    let _ = sigma_context;
}

#[scenario(path = "tests/features/sigma.feature", index = 5)]
fn robust_zero_mad(sigma_context: SigmaContext) {
    let _ = sigma_context;
}
