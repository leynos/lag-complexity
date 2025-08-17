//! Provider interfaces for embeddings, depth estimation, and ambiguity
//! estimation.
//!
//! Defines the `TextProcessor` trait and the provider aliases used by the
//! scoring engine.

/// Processes text to produce a structured output.
///
/// Providers for embeddings, depth, and ambiguity all implement this trait via
/// type aliases. Centralising the interface reduces boilerplate and eases
/// documentation.
pub trait TextProcessor {
    /// Structured result returned by the processor.
    type Output: Send + Sync + 'static;
    /// Error type returned when processing fails.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Process the supplied text.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    fn process(&self, input: &str) -> Result<Self::Output, Self::Error>;
}

/// Provides vector embeddings.
pub type EmbeddingProvider<E> = dyn TextProcessor<Output = Box<[f32]>, Error = E> + Send + Sync;

/// Estimates reasoning depth.
pub type DepthEstimator<E> = dyn TextProcessor<Output = f32, Error = E> + Send + Sync;

/// Estimates ambiguity.
pub type AmbiguityEstimator<E> = dyn TextProcessor<Output = f32, Error = E> + Send + Sync;
