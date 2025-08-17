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
pub struct VarianceScopingConfig {
    pub window: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    #[rstest]
    fn serialise_variance() {
        let cfg = ScopingConfig::Variance(VarianceScopingConfig { window: 5 });
        let json = serde_json::to_string(&cfg).unwrap_or_else(|e| panic!("serialize: {e}"));
        assert_eq!(json, r#"{"strategy":"variance","window":5}"#);
    }

    #[rstest]
    fn deserialise_invalid() {
        let json = r#"{"strategy":"unknown"}"#;
        let cfg: Result<ScopingConfig, _> = serde_json::from_str(json);
        assert!(cfg.is_err());
    }
}
