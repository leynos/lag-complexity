use serde::{Deserialize, Serialize};

/// Aggregated complexity scores for a query.
///
/// The `total` field is the sum of the component scores.
///
/// # Examples
///
/// ```
/// use lag_complexity::api::Complexity;
///
/// let c = Complexity::new(1.0, 2.0, 3.0);
/// assert_eq!(c.total, 6.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Complexity {
    pub total: f32,
    pub scope: f32,
    pub depth: f32,
    pub ambiguity: f32,
}

impl Complexity {
    /// Create a new `Complexity` with the given component scores.
    #[must_use]
    pub fn new(scope: f32, depth: f32, ambiguity: f32) -> Self {
        #[expect(clippy::float_arithmetic, reason = "summing component scores")]
        let total = scope + depth + ambiguity;
        Self {
            total,
            scope,
            depth,
            ambiguity,
        }
    }
}

/// Diagnostic trace for a complexity computation.
///
/// # Examples
///
/// ```
/// use lag_complexity::api::{Complexity, Trace};
///
/// let c = Complexity::new(0.0, 0.0, 0.0);
/// let trace = Trace { query: "hi".into(), complexity: c.clone() };
/// assert_eq!(trace.complexity.total, 0.0);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Trace {
    pub query: String,
    pub complexity: Complexity,
}

/// Trait for scoring the complexity of a query.
pub trait ComplexityFn {
    /// Error type returned when scoring fails.
    type Error: std::error::Error + Send + Sync;

    /// Score the complexity of a query.
    ///
    /// # Errors
    ///
    /// Returns an error if scoring fails.
    fn score(&self, query: &str) -> Result<Complexity, Self::Error>;

    /// Produce a diagnostic trace for a query.
    ///
    /// # Errors
    ///
    /// Returns an error if trace generation fails.
    fn trace(&self, query: &str) -> Result<Trace, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(0.0, 0.0, 0.0)]
    #[case(1.0, 2.0, 3.0)]
    fn totals(#[case] scope: f32, #[case] depth: f32, #[case] ambiguity: f32) {
        let c = Complexity::new(scope, depth, ambiguity);
        #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
        let expected = scope + depth + ambiguity;
        #[expect(clippy::float_arithmetic, reason = "test requires float arithmetic")]
        let diff = c.total - expected;
        assert!(diff.abs() < f32::EPSILON);
    }
}
