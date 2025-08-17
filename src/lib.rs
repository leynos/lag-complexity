//! Core library entry point.
//! Re-exports public types and traits.

pub mod api;
pub mod config;
pub mod providers;

pub use api::{Complexity, ComplexityFn, Trace};
pub use config::{ScopingConfig, VarianceScopingConfig};
pub use providers::{AmbiguityEstimator, DepthEstimator, EmbeddingProvider};

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
