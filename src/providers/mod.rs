//! Provider interfaces for embeddings, depth estimation, and ambiguity
//! estimation.
//!
//! Defines the `TextProcessor` trait and the provider aliases used by the
//! scoring engine.

#[cfg(feature = "onnx")]
pub mod onnx;

/// Processes text to produce a structured, thread-safe output.
///
/// Providers for embeddings, depth, and ambiguity all implement this trait via
/// type aliases. Centralising the interface reduces boilerplate and ensures
/// a single, well-documented contract.
pub trait TextProcessor {
    /// Structured result returned by the processor.
    ///
    /// Outputs must be `Send + Sync + 'static` so they can be safely shared
    /// across threads and stored in trait objects without borrowing.
    type Output: Send + Sync + 'static;
    /// Error type returned when processing fails.
    ///
    /// Errors must implement `std::error::Error` and be `Send + Sync + 'static`
    /// to propagate cleanly across threads and outlive the processor.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Process the supplied text.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails.
    fn process(&self, input: &str) -> Result<Self::Output, Self::Error>;
}

/// Provides vector embeddings.
pub type EmbeddingProvider<E> =
    dyn TextProcessor<Output = Box<[f32]>, Error = E> + Send + Sync + 'static;

/// Estimates reasoning depth.
pub type DepthEstimator<E> = dyn TextProcessor<Output = f32, Error = E> + Send + Sync + 'static;

/// Estimates ambiguity.
pub type AmbiguityEstimator<E> = dyn TextProcessor<Output = f32, Error = E> + Send + Sync + 'static;
