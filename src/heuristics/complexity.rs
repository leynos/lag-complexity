//! Heuristic-based complexity scorer.
//!
//! Combines the depth and ambiguity heuristics to provide an end-to-end
//! `ComplexityFn` implementation. The scope component is currently a constant
//! baseline controlled by a configurable field (defaulting to zero) until a
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
/// The `scope` component is a configurable baseline (default `0.0`) in this
/// baseline implementation.
#[derive(Debug, Clone)]
pub struct HeuristicComplexity {
    depth: DepthHeuristic,
    ambiguity: AmbiguityHeuristic,
    scope_weight: f32,
}

impl Default for HeuristicComplexity {
    fn default() -> Self {
        Self::builder().build()
    }
}

impl HeuristicComplexity {
    #[must_use]
    pub fn builder() -> HeuristicComplexityBuilder {
        HeuristicComplexityBuilder::default()
    }
}

#[derive(Debug, Clone)]
pub struct HeuristicComplexityBuilder {
    depth: DepthHeuristic,
    ambiguity: AmbiguityHeuristic,
    scope_weight: f32,
}

impl Default for HeuristicComplexityBuilder {
    fn default() -> Self {
        Self {
            depth: DepthHeuristic,
            ambiguity: AmbiguityHeuristic,
            scope_weight: 0.0,
        }
    }
}

impl HeuristicComplexityBuilder {
    #[must_use]
    pub fn depth(mut self, depth: DepthHeuristic) -> Self {
        self.depth = depth;
        self
    }

    #[must_use]
    pub fn ambiguity(mut self, ambiguity: AmbiguityHeuristic) -> Self {
        self.ambiguity = ambiguity;
        self
    }

    #[must_use]
    pub fn scope_weight(mut self, weight: f32) -> Self {
        self.scope_weight = weight;
        self
    }

    #[must_use]
    pub fn build(self) -> HeuristicComplexity {
        HeuristicComplexity {
            depth: self.depth,
            ambiguity: self.ambiguity,
            scope_weight: self.scope_weight,
        }
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
    #[expect(clippy::unwrap_used, reason = "test should fail loudly")]
    fn rejects_empty() {
        let hc = HeuristicComplexity::default();
        let err = hc.score("").unwrap_err();
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
}
