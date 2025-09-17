//! Heuristic-based complexity scorer.
//!
//! Combines the depth and ambiguity heuristics to provide an end-to-end
//! `ComplexityFn` implementation. The scope component is a constant baseline
//! controlled via [`HeuristicComplexity::with_scope_baseline`] (alias
//! [`HeuristicComplexity::with_scope_weight`], default `0.0`) until a
//! dedicated scope estimator is introduced. The struct exists primarily to
//! facilitate early integration tests and will evolve as additional signals
//! are added.

use crate::{
    Complexity, ComplexityFn, Trace,
    heuristics::{
        AmbiguityHeuristic, AmbiguityHeuristicError, DepthHeuristic, DepthHeuristicError,
    },
    providers::TextProcessor,
};
use thiserror::Error;

/// Errors returned by [`HeuristicComplexity`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum HeuristicComplexityError {
    #[error(transparent)]
    Depth(#[from] DepthHeuristicError),
    #[error(transparent)]
    Ambiguity(#[from] AmbiguityHeuristicError),
}

/// Basic `ComplexityFn` backed by lightweight heuristics.
///
/// The `scope` component is a configurable baseline (default `0.0`) in this baseline implementation.
///
#[derive(Default, Debug, Clone)]
pub struct HeuristicComplexity {
    depth: DepthHeuristic,
    ambiguity: AmbiguityHeuristic,
    /// Additive baseline for the `scope` component in the complexity score.
    /// This is not a multiplier.
    ///
    /// Values are clamped to remain non-negative. `0.0` disables the scope
    /// signal, while larger values increase its contribution. Values within
    /// `[0.0, 1.0]` are recommended for current heuristics. Higher weights are
    /// still valid and further emphasise the scope signal. Non-finite inputs
    /// are normalised to zero.
    scope_weight: f32,
}

impl HeuristicComplexity {
    /// Create a heuristic scorer with default heuristics.
    ///
    /// # Examples
    /// ```
    /// use lag_complexity::heuristics::HeuristicComplexity;
    /// let hc = HeuristicComplexity::new();
    /// ```
    #[must_use]
    pub fn new() -> Self { Self::default() }

    /// Set the additive baseline for the scope component (alias of [`HeuristicComplexity::with_scope_baseline`]).
    ///
    /// Prefer calling [`with_scope_baseline`] for clarity.
    ///
    /// The `weight` argument is clamped to `>= 0.0` so negative values are
    /// ignored. Values in `[0.0, 1.0]` match the current heuristics: `0.0`
    /// disables the scope signal, while larger numbers raise
    /// [`Complexity::scope`]. Weights above `1.0` remain valid and further
    /// emphasise the signal.
    ///
    /// Non-finite inputs (`NaN`, `±∞`) are normalised to `0.0`.
    ///
    /// # Examples
    /// ```
    /// use lag_complexity::ComplexityFn;
    /// use lag_complexity::heuristics::HeuristicComplexity;
    /// let hc = HeuristicComplexity::new().with_scope_baseline(0.5);
    /// // `.with_scope_weight(0.5)` provides the same result.
    /// let score = hc
    ///     .score("Plain question")
    ///     .expect("unexpected error");
    /// assert_eq!(score.scope(), 0.5);
    /// ```
    #[must_use]
    pub fn with_scope_weight(mut self, weight: f32) -> Self {
        self.scope_weight = if weight.is_finite() {
            weight.max(0.0)
        } else {
            0.0
        };
        self
    }

    /// Set the additive scope baseline (preferred; alias of [`with_scope_weight`]).
    ///
    /// Non-finite inputs (`NaN`, `±∞`) are normalised to `0.0`.
    #[must_use]
    pub fn with_scope_baseline(self, baseline: f32) -> Self {
        self.with_scope_weight(baseline)
    }

    /// Replace the depth heuristic.
    ///
    /// # Examples
    /// ```
    /// use lag_complexity::ComplexityFn;
    /// use lag_complexity::heuristics::{HeuristicComplexity, DepthHeuristic};
    /// let hc = HeuristicComplexity::new()
    ///     .with_depth(DepthHeuristic::default());
    /// assert!(hc.score("If A and B then C").is_ok());
    /// ```
    #[must_use]
    pub fn with_depth(mut self, depth: DepthHeuristic) -> Self {
        self.depth = depth;
        self
    }

    /// Replace the ambiguity heuristic.
    ///
    /// # Examples
    /// ```
    /// use lag_complexity::ComplexityFn;
    /// use lag_complexity::heuristics::{HeuristicComplexity, AmbiguityHeuristic};
    /// let hc = HeuristicComplexity::new()
    ///     .with_ambiguity(AmbiguityHeuristic::default());
    /// assert!(hc.score("Plain question").is_ok());
    /// ```
    #[must_use]
    pub fn with_ambiguity(mut self, ambiguity: AmbiguityHeuristic) -> Self {
        self.ambiguity = ambiguity;
        self
    }
}

impl ComplexityFn for HeuristicComplexity {
    type Error = HeuristicComplexityError;

    fn score(&self, query: &str) -> Result<Complexity, Self::Error> {
        let depth = self.depth.process(query)?;
        let ambiguity = self.ambiguity.process(query)?;
        Ok(Complexity::new(self.scope_weight, depth, ambiguity))
    }

    fn trace(&self, query: &str) -> Result<Trace, Self::Error> {
        self.score(query).map(|complexity| Trace {
            query: query.to_owned(),
            complexity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case(
        "If Alice and Bob play chess, who wins?",
        Complexity::new(0.0, 4.0, 1.0)
    )]
    #[case("Plain question", Complexity::new(0.0, 0.0, 1.0))]
    fn scores_expected(#[case] query: &str, #[case] expected: Complexity) {
        let hc = HeuristicComplexity::default();
        assert_eq!(hc.score(query), Ok(expected));
    }

    #[rstest]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn rejects_empty() {
        let hc = HeuristicComplexity::default();
        let err = hc.score("").expect_err("expected empty query to error");
        assert!(matches!(
            err,
            HeuristicComplexityError::Depth(DepthHeuristicError::Empty)
        ));
    }

    #[test]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn trace_echoes_query() {
        let hc = HeuristicComplexity::default();
        let query = "Plain question";
        let trace = hc.trace(query).expect("unexpected error");
        assert_eq!(trace.query, query);
        assert_eq!(trace.complexity, Complexity::new(0.0, 0.0, 1.0));
    }

    #[test]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn clamps_negative_scope_baseline() {
        let hc = HeuristicComplexity::new().with_scope_baseline(-0.5);
        let score = hc.score("Plain question").expect("unexpected error");
        assert_eq!(score, Complexity::new(0.0, 0.0, 1.0));
    }

    #[test]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    #[expect(clippy::float_cmp, reason = "clamped scope should equal zero")]
    fn scope_weight_non_finite_normalises_to_zero() {
        let hc = HeuristicComplexity::new()
            .with_scope_weight(f32::NAN)
            .with_scope_weight(f32::INFINITY)
            .with_scope_weight(f32::NEG_INFINITY);
        let score = hc.score("Plain question").expect("unexpected error");
        assert_eq!(score.scope(), 0.0);
    }
}
