use super::{aggregation::OrdinalAggregation, artefact::OnnxArtefact};

/// Configuration for an ordinal ONNX text classifier.
#[derive(Debug, Clone)]
pub struct OnnxClassifierConfig {
    /// Model artefact (ONNX graph) to load.
    pub model: OnnxArtefact,
    /// Tokeniser artefact consumed by `tokenizers`.
    pub tokenizer: OnnxArtefact,
    /// Ordered input names as defined in the ONNX graph.
    pub input_names: Vec<String>,
    /// Ordered output names to query after inference.
    pub output_names: Vec<String>,
    /// Maximum token count accepted by the model. Inputs are padded and truncated to this size.
    pub max_sequence_length: usize,
    /// Token inserted when padding shorter sequences.
    pub pad_token: String,
    /// Identifier of the padding token.
    pub pad_id: u32,
    /// Number of ordinal heads exposed by the model output.
    pub ordinal_head_count: usize,
    /// Mapping from ordinal probabilities to the scalar score.
    pub aggregator: OrdinalAggregation,
}
