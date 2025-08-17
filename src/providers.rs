/// Provides vector embeddings for text.
pub trait EmbeddingProvider {
    /// Error type returned when embedding fails.
    type Error;

    /// Produce an embedding for the given input text.
    ///
    /// # Errors
    ///
    /// Returns an error if embedding fails.
    fn embed(&self, input: &str) -> Result<Vec<f32>, Self::Error>;
}

/// Estimates the reasoning depth of a query.
pub trait DepthEstimator {
    /// Error type returned when estimation fails.
    type Error;

    /// Estimate the reasoning depth of the supplied text.
    ///
    /// # Errors
    ///
    /// Returns an error if estimation fails.
    fn estimate_depth(&self, query: &str) -> Result<f32, Self::Error>;
}

/// Estimates the ambiguity of a query.
pub trait AmbiguityEstimator {
    /// Error type returned when estimation fails.
    type Error;

    /// Estimate the ambiguity score of the supplied text.
    ///
    /// # Errors
    ///
    /// Returns an error if estimation fails.
    fn estimate_ambiguity(&self, query: &str) -> Result<f32, Self::Error>;
}
