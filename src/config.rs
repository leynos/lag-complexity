//! Scoring configuration types and serialisation for semantic scope.

use serde::{Deserialize, Serialize};

/// Configuration for the semantic scope component.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum ScopingConfig {
    /// Configure variance-based scoping.
    Variance(VarianceScopingConfig),
}

/// Configuration for variance-based scoping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VarianceScopingConfig {
    /// Sliding window size; must be greater than zero.
    pub window: usize,
}

impl VarianceScopingConfig {
    /// Ensure the configuration values are within acceptable bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `window` is zero.
    #[must_use = "Validation should not be ignored"]
    pub fn validate(self) -> Result<Self, String> {
        if self.window == 0 {
            Err("window must be greater than 0".into())
        } else {
            Ok(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn serialise_variance() {
        let cfg = ScopingConfig::Variance(VarianceScopingConfig { window: 5 });
        #[expect(clippy::expect_used, reason = "test should fail loudly")]
        let json = serde_json::to_string(&cfg).expect("serialise ScopingConfig to JSON");
        assert_eq!(json, r#"{"strategy":"variance","window":5}"#);
    }

    #[rstest]
    fn deserialise_variance() {
        let json = r#"{"strategy":"variance","window":5}"#;
        #[expect(clippy::expect_used, reason = "test should fail loudly")]
        let cfg: ScopingConfig = serde_json::from_str(json).expect("deserialise ScopingConfig");
        assert_eq!(
            cfg,
            ScopingConfig::Variance(VarianceScopingConfig { window: 5 })
        );
    }

    #[rstest]
    fn deserialise_invalid() {
        let json = r#"{"strategy":"unknown"}"#;
        let cfg: Result<ScopingConfig, _> = serde_json::from_str(json);
        assert!(cfg.is_err());
    }

    #[rstest]
    fn validate_window() {
        let cfg = VarianceScopingConfig { window: 0 };
        assert!(cfg.validate().is_err());
    }
}
