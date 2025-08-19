use lag_complexity::{Sigma, tests::support::approx_eq};
use rstest::rstest;

const ROBUST_EXPECTED: f32 = 0.662_507_95;

#[rstest]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, 5.0_f32, Some(0.5_f32))]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, -5.0, Some(0.0))]
#[case(Sigma::MinMax { p01: 0.0, p99: 10.0 }, 15.0, Some(1.0))]
#[case(Sigma::ZScore { mean: 0.0, std: 0.0 }, 1.23, None)]
#[case(Sigma::Robust { median: 0.0, mad: 1.0 }, 1.0, Some(ROBUST_EXPECTED))]
#[case(Sigma::Robust { median: 0.0, mad: 0.0 }, 2.5, None)]
fn test_sigma_apply(#[case] sigma: Sigma, #[case] input: f32, #[case] expected: Option<f32>) {
    let out = sigma.apply(input);
    match expected {
        Some(exp) => {
            let val = out.unwrap_or_else(|| panic!("expected Some"));
            assert!(approx_eq(val, exp, 1e-4));
        }
        None => assert!(out.is_none()),
    }
}
