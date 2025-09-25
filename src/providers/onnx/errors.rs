use std::path::PathBuf;

use thiserror::Error;

/// Errors produced by the shared ONNX classifier implementation.
#[derive(Debug, Error)]
pub enum OnnxClassifierError {
    #[error("failed to read artefact at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("artefact at {path} expected SHA-256 {expected} but found {actual}")]
    ChecksumMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    #[error("failed to load tokenizer from {path}: {source}")]
    LoadTokenizer {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },
    #[error("failed to configure tokenizer truncation: {0}")]
    ConfigureTruncation(#[source] tokenizers::Error),
    #[error("failed to construct ONNX session builder: {0}")]
    CreateSessionBuilder(#[source] ort::Error),
    #[error("failed to create ONNX session: {0}")]
    CreateSession(#[source] ort::Error),
    #[error("classifier requires at least one input name")]
    MissingInputNames,
    #[error("classifier requires at least {expected} input names but found {actual}")]
    InsufficientInputNames { expected: usize, actual: usize },
    #[error("classifier requires at least one output name")]
    MissingOutputNames,
    #[error("classifier requires max_sequence_length > 0")]
    ZeroSequenceLength,
    #[error("classifier requires ordinal_head_count > 0")]
    ZeroHeadCount,
    #[error("tokenizer produced sequence of length {actual} but expected {expected}")]
    SequenceLength { expected: usize, actual: usize },
    #[error("failed to encode text: {0}")]
    Encode(#[source] tokenizers::Error),
    #[error("failed to convert encoded text into tensor: {0}")]
    EncodeTensor(#[source] ort::Error),
    #[error("session mutex was poisoned by a previous panic")]
    SessionPoisoned,
    #[error("failed to run inference: {0}")]
    Inference(#[source] ort::Error),
    #[error("ONNX output \"{name}\" missing from session results")]
    OutputMissing { name: String },
    #[error("ONNX output \"{name}\" has {actual} logits but expected {expected}")]
    UnexpectedLogitCount {
        name: String,
        expected: usize,
        actual: usize,
    },
    #[error("mid-bin mapping requires {expected} bins but found {actual}")]
    InvalidBinCount { expected: usize, actual: usize },
}

/// Errors returned by [`DepthClassifierOnnx`].
#[derive(Debug, Error)]
pub enum DepthClassifierOnnxError {
    #[error(transparent)]
    Classifier(#[from] OnnxClassifierError),
}

/// Errors returned by [`AmbiguityClassifierOnnx`].
#[derive(Debug, Error)]
pub enum AmbiguityClassifierOnnxError {
    #[error(transparent)]
    Classifier(#[from] OnnxClassifierError),
}
