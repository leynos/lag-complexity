//! ONNX-backed text classifiers for depth and ambiguity with artefact verification.
mod aggregation;
mod artefact;
mod classifier;
mod config;
mod errors;

pub use aggregation::OrdinalAggregation;
pub use artefact::OnnxArtefact;
pub use config::OnnxClassifierConfig;
pub use errors::{AmbiguityClassifierOnnxError, DepthClassifierOnnxError, OnnxClassifierError};

use crate::providers::TextProcessor;
use classifier::OnnxTextClassifier;

/// Depth classifier using an ONNX Transformer-Ordinal model.
///
/// Returns the expected reasoning steps as a raw depth score for Sigma normalisation.
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
    ///
    /// # Examples
    /// ```no_run
    /// use lag_complexity::providers::onnx::{DepthClassifierOnnx, DepthClassifierOnnxError, OnnxArtefact, OnnxClassifierConfig, OrdinalAggregation};
    ///
    /// # fn main() -> Result<(), DepthClassifierOnnxError> {
    /// let config = OnnxClassifierConfig {
    ///     model: OnnxArtefact {
    ///         path: std::path::PathBuf::from("/models/depth_transformer_ordinal.onnx"),
    ///         sha256: "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into(),
    ///     },
    ///     tokenizer: OnnxArtefact {
    ///         path: std::path::PathBuf::from("/models/depth_tokenizer.json"),
    ///         sha256: "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210".into(),
    ///     },
    ///     input_names: vec!["input_ids".into(), "attention_mask".into()],
    ///     output_names: vec!["logits_ord".into()],
    ///     max_sequence_length: 512,
    ///     pad_token: "[PAD]".into(),
    ///     pad_id: 0,
    ///     ordinal_head_count: 5,
    ///     aggregator: OrdinalAggregation::ExpectedValue { base: 0.0, scale: 1.0 },
    /// };
    /// let classifier = DepthClassifierOnnx::new(config)?;
    /// let score = classifier.process("Which university did the CEO of the company that developed the original iPhone attend?")?;
    /// # println!("{score}");
    /// # Ok(())
    /// # }
    /// ```
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

/// Ambiguity classifier using an ONNX Transformer-Ordinal model.
///
/// Returns a raw ambiguity score in probability units for Sigma normalisation.
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
    ///
    /// # Examples
    /// ```no_run
    /// use lag_complexity::providers::onnx::{AmbiguityClassifierOnnx, AmbiguityClassifierOnnxError, OnnxArtefact, OnnxClassifierConfig, OrdinalAggregation};
    ///
    /// # fn main() -> Result<(), AmbiguityClassifierOnnxError> {
    /// let config = OnnxClassifierConfig {
    ///     model: OnnxArtefact {
    ///         path: std::path::PathBuf::from("/models/ambiguity_transformer_ordinal.onnx"),
    ///         sha256: "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff".into(),
    ///     },
    ///     tokenizer: OnnxArtefact {
    ///         path: std::path::PathBuf::from("/models/ambiguity_tokenizer.json"),
    ///         sha256: "ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766554433221100".into(),
    ///     },
    ///     input_names: vec!["input_ids".into(), "attention_mask".into()],
    ///     output_names: vec!["logits_ord".into()],
    ///     max_sequence_length: 384,
    ///     pad_token: "[PAD]".into(),
    ///     pad_id: 0,
    ///     ordinal_head_count: 4,
    ///     aggregator: OrdinalAggregation::ExpectedValue { base: 0.0, scale: 1.0 },
    /// };
    /// let classifier = AmbiguityClassifierOnnx::new(config)?;
    /// let score = classifier.process("What is the primary source of the Nile?")?;
    /// # println!("{score}");
    /// # Ok(())
    /// # }
    /// ```
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
