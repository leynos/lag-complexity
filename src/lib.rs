//! Core library entry point.
//! Re-exports public types and traits.

pub mod api;
pub mod config;
pub mod providers;
pub mod sigma;
pub mod variance;

pub use api::{Complexity, ComplexityFn, Trace};
pub use config::{ScopingConfig, VarianceScopingConfig};
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
