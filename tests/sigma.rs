use lag_complexity::Sigma;
mod support;
use rstest::rstest;
use support::approx_eq;

const ROBUST_EXPECTED: f32 = 0.662_507_95;

#[rstest]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, 5.0_f32, Some(0.5_f32))]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, -5.0, Some(0.0))]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, 15.0, Some(1.0))]
#[case(Sigma::MinMax { p01: 0.0, p99: f32::INFINITY }, 1.0, None)]
#[case(Sigma::ZScore { mean: 0.0, std: 0.0 }, 1.23, None)]
#[case(Sigma::ZScore { mean: 0.0, std: -1.0 }, 1.23, None)]
#[case(Sigma::Robust { median: 0.0, mad: 1.0 }, 1.0, Some(ROBUST_EXPECTED))]
#[case(Sigma::Robust { median: 0.0, mad: 0.0 }, 2.5, None)]
#[case(Sigma::Robust { median: 0.0, mad: -1.0 }, 2.5, None)]
#[case(Sigma::MinMax { p01: f32::NEG_INFINITY, p99: 1.0 }, 0.0, None)]
#[case(Sigma::ZScore { mean: 0.0, std: f32::NAN }, 1.0, None)]
#[case(Sigma::Robust { median: 0.0, mad: f32::INFINITY }, 1.0, None)]
fn test_sigma_apply(#[case] sigma: Sigma, #[case] input: f32, #[case] expected: Option<f32>) {
    let out = sigma.apply(input);
    match expected {
        Some(exp) => {
            #[expect(clippy::expect_used, reason = "tests unwrap known Some")]
            let val = out.expect("expected Some");
            assert!(approx_eq(val, exp, 1e-4));
        }
        None => assert!(out.is_none()),
    }
}
