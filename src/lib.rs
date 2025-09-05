//! Core library entry point.
//! Re-exports public types and traits.

pub mod api;
#[cfg(feature = "provider-api")]
pub mod api_embedding;
#[cfg(feature = "cli")]
pub mod cli;
pub mod config;
pub mod heuristics;
pub mod providers;
pub mod sigma;
pub mod variance;

pub use api::{Complexity, ComplexityFn, Trace};
#[cfg(feature = "provider-api")]
pub use api_embedding::{ApiEmbedding, ApiEmbeddingError};
#[cfg(feature = "cli")]
pub use cli::LagcArgs;
pub use config::{ScopingConfig, VarianceScopingConfig};
pub use heuristics::{
    AmbiguityHeuristic, AmbiguityHeuristicError, DepthHeuristic, DepthHeuristicError,
};
pub use providers::{AmbiguityEstimator, DepthEstimator, EmbeddingProvider, TextProcessor};
pub use sigma::Sigma;
pub use variance::VarianceError;

pub mod tests;

/// Returns a greeting for the library.
///
/// # Examples
///
/// ```
/// use lag_complexity::greet;
///
/// assert_eq!(greet(), "Hello from LAG Complexity Function!");
/// ```
#[must_use]
pub fn greet() -> &'static str {
    "Hello from LAG Complexity Function!"
}
