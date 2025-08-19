#[expect(clippy::float_arithmetic, reason = "tolerance comparison")]
#[must_use]
pub fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() < tol
}
