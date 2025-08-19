//! Variance calculation utilities.
//!
//! Implements a numerically stable online algorithm for computing the
//! population variance of a slice of `f32` values. The implementation uses
//! Welford's method to avoid the catastrophic cancellation that can occur with
//! the naive two-pass formula.

use thiserror::Error;

/// Errors that can occur during variance calculation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum VarianceError {
    /// The input slice is empty and no variance can be calculated.
    #[error("at least one value is required to calculate variance")]
    Empty,
}

/// Calculate the population variance of the provided values.
///
/// The function returns an error if the slice is empty. For a single value the
/// variance is defined as zero.
///
/// # Examples
///
/// ```
/// use lag_complexity::variance::variance;
///
/// #[expect(clippy::unwrap_used, reason = "test should fail loudly")]
/// let var = variance(&[1.0, 2.0, 3.0]).unwrap();
/// assert!((var - 2.0 / 3.0).abs() < 1e-6);
/// ```
///
/// # Errors
///
/// Returns [`VarianceError::Empty`] if `values` contains no elements.
#[expect(
    clippy::cast_possible_truncation,
    reason = "variance calculation accumulates in f64 then converts to f32"
)]
pub fn variance(values: &[f32]) -> Result<f32, VarianceError> {
    if values.is_empty() {
        return Err(VarianceError::Empty);
    }

    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    let mut count = 0f64;

    #[expect(clippy::float_arithmetic, reason = "variance accumulation")]
    for &x in values {
        count += 1.0;
        let delta = f64::from(x) - mean;
        mean += delta / count;
        let delta2 = f64::from(x) - mean;
        m2 += delta * delta2;
    }

    #[expect(clippy::float_arithmetic, reason = "final division")]
    Ok((m2 / count) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    #[expect(clippy::float_arithmetic, reason = "tolerance comparison")]
    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    const TWO_THIRDS: f32 = 0.666_666_7;

    #[rstest]
    fn variance_empty() {
        assert_eq!(variance(&[]), Err(VarianceError::Empty));
    }

    #[rstest]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn variance_single() {
        let result = variance(&[1.0]).expect("variance of single value");
        assert!(result.abs() < f32::EPSILON);
    }

    #[rstest]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn variance_three_values() {
        let result = variance(&[1.0, 2.0, 3.0]).expect("variance of three values");
        assert!(approx_eq(result, TWO_THIRDS));
    }
}
