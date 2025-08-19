//! Normalisation strategies for bringing raw scores onto a comparable scale.
//!
//! `Sigma` implements three strategies:
//! - [`MinMax`] performs linear scaling based on the 1st and 99th percentiles.
//! - [`ZScore`] standardises using the mean and standard deviation followed by a
//!   sigmoid to map to `[0, 1]`.
//! - [`Robust`] uses the median and Median Absolute Deviation (MAD) for
//!   outlier-resistant normalisation.

use serde::{Deserialize, Serialize};

/// Values with an absolute magnitude below this are treated as zero.
const NEAR_ZERO: f32 = 1e-6;

/// Scales the Median Absolute Deviation to approximate the standard deviation
/// of a normal distribution.
const MAD_SCALING_FACTOR: f32 = 1.4826;

/// Logistic sigmoid function mapping any real number to `[0, 1]`.
#[expect(clippy::float_arithmetic, reason = "sigmoid uses floats")]
#[inline]
#[must_use]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Strategy for normalising raw scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case", deny_unknown_fields)]
pub enum Sigma {
    /// Linear scaling using percentile estimates. Values outside the `[p01, p99]`
    /// range are clamped.
    ///
    /// # Panics
    ///
    /// Panics if `p99` is not greater than `p01`.
    MinMax { p01: f32, p99: f32 },
    /// Standard Z-score normalisation with mean and standard deviation. A
    /// sigmoid is applied to map the Z-score to `[0, 1]`.
    ZScore { mean: f32, std: f32 },
    /// Robust normalisation based on the median and MAD. The MAD is scaled by
    /// [`MAD_SCALING_FACTOR`] to approximate the standard deviation of a normal
    /// distribution.
    Robust { median: f32, mad: f32 },
}

impl Sigma {
    /// Apply the normalisation strategy to a raw value.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use lag_complexity::Sigma;
    /// let sigma = Sigma::ZScore { mean: 0.0, std: 1.0 };
    /// let y = sigma.apply(1.0);
    /// assert!(y > 0.5 && y < 1.0);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if a [`MinMax`] sigma has `p99` not greater than `p01`.
    #[must_use]
    pub fn apply(&self, value: f32) -> f32 {
        match *self {
            Self::MinMax { p01, p99 } => {
                #[expect(clippy::float_arithmetic, reason = "percentile difference check")]
                let range = p99 - p01;
                assert!(range >= NEAR_ZERO, "p99 must be greater than p01");
                #[expect(clippy::float_arithmetic, reason = "linear scaling")]
                {
                    ((value - p01) / range).clamp(0.0, 1.0)
                }
            }
            Self::ZScore { mean, std } => {
                if std.abs() < NEAR_ZERO {
                    0.5
                } else {
                    #[expect(clippy::float_arithmetic, reason = "z-score normalisation")]
                    {
                        sigmoid((value - mean) / std)
                    }
                }
            }
            Self::Robust { median, mad } => {
                if mad.abs() < NEAR_ZERO {
                    0.5
                } else {
                    #[expect(clippy::float_arithmetic, reason = "robust scaling")]
                    {
                        let scale = mad * MAD_SCALING_FACTOR;
                        sigmoid((value - median) / scale)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::support::approx_eq;
    use rstest::rstest;

    const ROBUST_EXPECTED: f32 = 0.662_507_95;

    #[rstest]
    #[case(0.0, 0.0)]
    #[case(5.0, 0.5)]
    #[case(10.0, 1.0)]
    #[case(-5.0, 0.0)]
    #[case(15.0, 1.0)]
    fn minmax_cases(#[case] input: f32, #[case] expected: f32) {
        let sigma = Sigma::MinMax {
            p01: 0.0,
            p99: 10.0,
        };
        let result = sigma.apply(input);
        assert!(approx_eq(result, expected, 1e-6));
    }

    #[rstest]
    fn zscore_standard() {
        let sigma = Sigma::ZScore {
            mean: 0.0,
            std: 1.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, sigmoid(1.0), 1e-6));
    }

    #[rstest]
    fn zscore_zero_std() {
        let sigma = Sigma::ZScore {
            mean: 0.0,
            std: 0.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, 0.5, 1e-6));
    }

    #[rstest]
    fn robust_standard() {
        let sigma = Sigma::Robust {
            median: 0.0,
            mad: 1.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, ROBUST_EXPECTED, 1e-6));
    }

    #[rstest]
    fn robust_zero_mad() {
        let sigma = Sigma::Robust {
            median: 0.0,
            mad: 0.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, 0.5, 1e-6));
    }

    #[rstest]
    #[should_panic(expected = "p99 must be greater than p01")]
    fn minmax_inverted_range_panics() {
        let sigma = Sigma::MinMax {
            p01: 10.0,
            p99: 0.0,
        };
        let _ = sigma.apply(5.0);
    }

    #[rstest]
    #[should_panic(expected = "p99 must be greater than p01")]
    #[expect(clippy::float_arithmetic, reason = "test inputs requiring floats")]
    fn minmax_near_zero_range_panics() {
        let sigma = Sigma::MinMax {
            p01: 0.0,
            p99: NEAR_ZERO / 10.0,
        };
        let _ = sigma.apply(5.0);
    }

    #[rstest]
    #[expect(clippy::float_arithmetic, reason = "test inputs requiring floats")]
    fn zscore_near_zero_std() {
        let sigma = Sigma::ZScore {
            mean: 0.0,
            std: NEAR_ZERO / 10.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, 0.5, 1e-6));
    }

    #[rstest]
    #[expect(clippy::float_arithmetic, reason = "test inputs requiring floats")]
    fn robust_near_zero_mad() {
        let sigma = Sigma::Robust {
            median: 0.0,
            mad: NEAR_ZERO / 10.0,
        };
        let result = sigma.apply(1.0);
        assert!(approx_eq(result, 0.5, 1e-6));
    }
}
