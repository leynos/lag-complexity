mod aggregation;
mod artefact;
mod classifier;
mod config;
mod errors;

pub use aggregation::OrdinalAggregation;
pub use artefact::OnnxArtefact;
pub use classifier::OnnxTextClassifier;
pub use config::OnnxClassifierConfig;
pub use errors::{AmbiguityClassifierOnnxError, DepthClassifierOnnxError, OnnxClassifierError};

use crate::providers::TextProcessor;

#[derive(Debug)]
pub struct DepthClassifierOnnx {
    inner: OnnxTextClassifier,
}

impl DepthClassifierOnnx {
    /// Builds a depth classifier from disk artefacts.
    ///
    /// # Errors
    ///
    /// Returns [`DepthClassifierOnnxError`] when artefact verification or ONNX initialisation fails.
    pub fn new(config: OnnxClassifierConfig) -> Result<Self, DepthClassifierOnnxError> {
        OnnxTextClassifier::new(config)
            .map(|inner| Self { inner })
            .map_err(DepthClassifierOnnxError::Classifier)
    }
}

impl TextProcessor for DepthClassifierOnnx {
    type Output = f32;
    type Error = DepthClassifierOnnxError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        self.inner
            .predict(input)
            .map_err(DepthClassifierOnnxError::Classifier)
    }
}

#[derive(Debug)]
pub struct AmbiguityClassifierOnnx {
    inner: OnnxTextClassifier,
}

impl AmbiguityClassifierOnnx {
    /// Builds an ambiguity classifier from disk artefacts.
    ///
    /// # Errors
    ///
    /// Returns [`AmbiguityClassifierOnnxError`] when artefact verification or ONNX initialisation fails.
    pub fn new(config: OnnxClassifierConfig) -> Result<Self, AmbiguityClassifierOnnxError> {
        OnnxTextClassifier::new(config)
            .map(|inner| Self { inner })
            .map_err(AmbiguityClassifierOnnxError::Classifier)
    }
}

impl TextProcessor for AmbiguityClassifierOnnx {
    type Output = f32;
    type Error = AmbiguityClassifierOnnxError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        self.inner
            .predict(input)
            .map_err(AmbiguityClassifierOnnxError::Classifier)
    }
}
