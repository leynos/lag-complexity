//! Shared assertions and helpers for integration tests.

/// Return whether `a` and `b` differ by no more than `tol`.
///
/// # Parameters
/// - `a`: The first value.
/// - `b`: The second value.
/// - `tol`: The largest absolute difference that still counts as equal.
///
/// # Returns
/// `true` when the absolute difference of `a` and `b` is at most `tol`, and
/// `false` otherwise.
#[expect(clippy::float_arithmetic, reason = "tolerance comparison")]
#[must_use]
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol
}
