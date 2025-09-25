use std::sync::{Arc, Mutex};

use ort::{session::Session, value::TensorRef};
use tokenizers::{
    Encoding, PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};

use super::{
    aggregation::OrdinalAggregation, config::OnnxClassifierConfig, errors::OnnxClassifierError,
};

#[derive(Debug)]
pub struct OnnxTextClassifier {
    session: Arc<Mutex<Session>>,
    tokenizer: Arc<Tokenizer>,
    input_names: Arc<Vec<String>>,
    output_names: Arc<Vec<String>>,
    max_sequence_length: usize,
    head_count: usize,
    aggregator: OrdinalAggregation,
}

/// Tokenised request converted to ONNX tensor inputs.
#[derive(Debug)]
struct EncodedInput {
    ids: Vec<i64>,
    attention: Vec<i64>,
    type_ids: Vec<i64>,
}

impl EncodedInput {
    fn from_encoding(
        encoding: &Encoding,
        expected_len: usize,
    ) -> Result<Self, OnnxClassifierError> {
        let ids = encoding.get_ids();
        Self::ensure_length(ids.len(), expected_len)?;
        let attention = encoding.get_attention_mask();
        Self::ensure_length(attention.len(), expected_len)?;
        let type_ids = encoding.get_type_ids();
        Self::ensure_length(type_ids.len(), expected_len)?;
        Ok(Self {
            ids: ids.iter().map(|id| i64::from(*id)).collect(),
            attention: attention.iter().map(|id| i64::from(*id)).collect(),
            type_ids: type_ids.iter().map(|id| i64::from(*id)).collect(),
        })
    }

    fn ensure_length(length: usize, expected: usize) -> Result<(), OnnxClassifierError> {
        if length == expected {
            Ok(())
        } else {
            Err(OnnxClassifierError::SequenceLength {
                expected,
                actual: length,
            })
        }
    }
}

impl OnnxTextClassifier {
    /// Builds an ONNX classifier from the supplied configuration.
    ///
    /// # Errors
    ///
    /// Returns configuration and runtime errors when artefacts cannot be verified, tokeniser setup fails, or the ONNX session cannot be created.
    pub fn new(config: OnnxClassifierConfig) -> Result<Self, OnnxClassifierError> {
        if config.max_sequence_length == 0 {
            return Err(OnnxClassifierError::ZeroSequenceLength);
        }
        if config.input_names.len() < 2 {
            return Err(OnnxClassifierError::InsufficientInputNames {
                expected: 2,
                actual: config.input_names.len(),
            });
        }
        if config.output_names.is_empty() {
            return Err(OnnxClassifierError::MissingOutputNames);
        }
        if config.ordinal_head_count == 0 {
            return Err(OnnxClassifierError::ZeroHeadCount);
        }

        config.model.verify()?;
        config.tokenizer.verify()?;

        let mut tokenizer = Tokenizer::from_file(&config.tokenizer.path).map_err(|source| {
            OnnxClassifierError::LoadTokenizer {
                path: config.tokenizer.path.clone(),
                source,
            }
        })?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: config.max_sequence_length,
                strategy: TruncationStrategy::OnlyFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(OnnxClassifierError::ConfigureTruncation)?;

        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(config.max_sequence_length),
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: config.pad_id,
            pad_type_id: 0,
            pad_token: config.pad_token.clone(),
        }));

        config.aggregator.validate(config.ordinal_head_count)?;

        let session = Session::builder()
            .map_err(OnnxClassifierError::CreateSessionBuilder)?
            .commit_from_file(&config.model.path)
            .map_err(OnnxClassifierError::CreateSession)?;

        Ok(Self {
            session: Arc::new(Mutex::new(session)),
            tokenizer: Arc::new(tokenizer),
            input_names: Arc::new(config.input_names),
            output_names: Arc::new(config.output_names),
            max_sequence_length: config.max_sequence_length,
            head_count: config.ordinal_head_count,
            aggregator: config.aggregator,
        })
    }

    /// Runs inference for a single input string and returns the aggregated ordinal score.
    ///
    /// # Errors
    ///
    /// Returns encoding, inference, or aggregation errors when tokenisation fails, the ONNX runtime errors, or the logits do not match the configured head count.
    pub fn predict(&self, input: &str) -> Result<f32, OnnxClassifierError> {
        let encoding = self
            .tokenizer
            .encode(input, true)
            .map_err(OnnxClassifierError::Encode)?;
        let encoded = EncodedInput::from_encoding(&encoding, self.max_sequence_length)?;
        let type_ids_name = self.input_names.get(2).map(String::as_str);
        let (output_name, logits) = self.run_session(&encoded, type_ids_name)?;
        self.aggregate_logits(&output_name, &logits)
    }

    fn run_session(
        &self,
        encoded: &EncodedInput,
        type_ids_name: Option<&str>,
    ) -> Result<(String, Vec<f32>), OnnxClassifierError> {
        let ids_tensor = TensorRef::from_array_view((
            [1usize, self.max_sequence_length],
            encoded.ids.as_slice(),
        ))
        .map_err(OnnxClassifierError::EncodeTensor)?;
        let attention_tensor = TensorRef::from_array_view((
            [1usize, self.max_sequence_length],
            encoded.attention.as_slice(),
        ))
        .map_err(OnnxClassifierError::EncodeTensor)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| OnnxClassifierError::SessionPoisoned)?;

        let input_ids_name = self
            .input_names
            .first()
            .ok_or(OnnxClassifierError::MissingInputNames)?
            .as_str();
        let attention_mask_name = self
            .input_names
            .get(1)
            .ok_or(OnnxClassifierError::InsufficientInputNames {
                expected: 2,
                actual: self.input_names.len(),
            })?
            .as_str();

        let outputs = match type_ids_name {
            Some(type_ids_name) => {
                let type_ids_tensor = TensorRef::from_array_view((
                    [1usize, self.max_sequence_length],
                    encoded.type_ids.as_slice(),
                ))
                .map_err(OnnxClassifierError::EncodeTensor)?;
                session
                    .run(ort::inputs! {
                        input_ids_name => ids_tensor,
                        attention_mask_name => attention_tensor,
                        type_ids_name => type_ids_tensor,
                    })
                    .map_err(OnnxClassifierError::Inference)?
            }
            None => session
                .run(ort::inputs! {
                    input_ids_name => ids_tensor,
                    attention_mask_name => attention_tensor,
                })
                .map_err(OnnxClassifierError::Inference)?,
        };

        let output_name = self
            .output_names
            .first()
            .ok_or(OnnxClassifierError::MissingOutputNames)?
            .clone();
        let logits_value =
            outputs
                .get(&output_name)
                .ok_or_else(|| OnnxClassifierError::OutputMissing {
                    name: output_name.clone(),
                })?;
        let (_, logits) = logits_value
            .try_extract_tensor::<f32>()
            .map_err(OnnxClassifierError::Inference)?;
        Ok((output_name, logits.to_vec()))
    }

    fn aggregate_logits(
        &self,
        output_name: &str,
        logits: &[f32],
    ) -> Result<f32, OnnxClassifierError> {
        if logits.len() != self.head_count {
            return Err(OnnxClassifierError::UnexpectedLogitCount {
                name: output_name.to_string(),
                expected: self.head_count,
                actual: logits.len(),
            });
        }

        let probabilities: Vec<f32> = logits.iter().map(|value| logistic(*value)).collect();
        match &self.aggregator {
            OrdinalAggregation::ExpectedValue { base, scale } => {
                let sum = probabilities.iter().copied().sum::<f32>();
                #[expect(
                    clippy::float_arithmetic,
                    reason = "ordinal aggregation requires float scaling"
                )]
                let value = base + scale * sum;
                Ok(value)
            }
            OrdinalAggregation::MidBin {
                bin_midpoints,
                decision_threshold,
            } => {
                let positives = probabilities
                    .iter()
                    .filter(|probability| **probability >= *decision_threshold)
                    .count();
                bin_midpoints
                    .get(positives)
                    .copied()
                    .ok_or(OnnxClassifierError::InvalidBinCount {
                        expected: self.head_count + 1,
                        actual: bin_midpoints.len(),
                    })
            }
        }
    }
}

#[expect(clippy::float_arithmetic, reason = "sigmoid requires float operations")]
fn logistic(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::logistic;

    #[test]
    fn logistic_is_monotonic() {
        assert!(logistic(-5.0) < logistic(0.0));
        assert!(logistic(0.0) < logistic(5.0));
    }
}
