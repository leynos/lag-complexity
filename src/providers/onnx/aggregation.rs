use super::errors::OnnxClassifierError;

/// Aggregation strategy that converts ordinal head probabilities into a scalar score.
#[derive(Debug, Clone)]
pub enum OrdinalAggregation {
    /// Sum the head probabilities and apply an affine mapping.
    ///
    /// The sum yields a value in the range `[0, head_count]`. Scaling and offsetting map the
    /// count to a calibrated step domain.
    ExpectedValue {
        /// Additive offset applied after summing the probabilities.
        base: f32,
        /// Multiplicative factor applied to the summed probabilities.
        scale: f32,
    },
    /// Choose a representative mid-bin value by counting how many heads exceed a threshold.
    MidBin {
        /// Representative step value for each possible count of positive heads. Length must be
        /// `head_count + 1`.
        bin_midpoints: Vec<f32>,
        /// Probability threshold that marks a head as “positive”.
        decision_threshold: f32,
    },
}

impl OrdinalAggregation {
    /// Validates the ordinal aggregation configuration.
    ///
    /// # Errors
    ///
    /// Returns `InvalidBinCount` when the mid-bin mapping length does not match `head_count + 1`.
    pub fn validate(&self, head_count: usize) -> Result<(), OnnxClassifierError> {
        match self {
            Self::ExpectedValue { .. } => Ok(()),
            Self::MidBin { bin_midpoints, .. } => {
                let expected = head_count + 1;
                if bin_midpoints.len() == expected {
                    Ok(())
                } else {
                    Err(OnnxClassifierError::InvalidBinCount {
                        expected,
                        actual: bin_midpoints.len(),
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OrdinalAggregation;

    #[test]
    fn ordinal_aggregation_validates_mid_bin_length() {
        let aggregation = OrdinalAggregation::MidBin {
            bin_midpoints: vec![0.0, 1.0, 2.0],
            decision_threshold: 0.5,
        };
        assert!(aggregation.validate(2).is_ok());
        assert!(aggregation.validate(1).is_err());
    }
}
