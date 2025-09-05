//! API-based embedding provider.
//!
//! Sends text to an HTTP endpoint and expects a JSON response
//! containing an `embedding` array of `f32` values. An optional
//! API key can be supplied and will be sent as a bearer token.
//!
//! # Examples
//!
//! ```
//! use lag_complexity::{ApiEmbedding, TextProcessor};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = ApiEmbedding::new("http://localhost:8080/embed", None);
//! let embedding = provider.process("hello")?;
//! assert!(!embedding.is_empty());
//! # Ok(())
//! # }
//! ```

use reqwest::blocking::Client;
use serde::Deserialize;
use thiserror::Error;

use crate::providers::TextProcessor;

/// Error returned by [`ApiEmbedding`].
#[derive(Debug, Error)]
pub enum ApiEmbeddingError {
    /// Input was empty.
    #[error("empty input")]
    Empty,
    /// HTTP request failed.
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    /// Response did not contain a valid embedding.
    #[error("invalid response")]
    InvalidResponse,
}

impl PartialEq for ApiEmbeddingError {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (Self::Empty, Self::Empty)
                | (Self::InvalidResponse, Self::InvalidResponse)
                | (Self::Request(_), Self::Request(_))
        )
    }
}

impl Eq for ApiEmbeddingError {}

#[derive(Deserialize)]
struct ApiResponse {
    embedding: Vec<f32>,
}

/// Embedding provider backed by an HTTP API.
#[derive(Debug, Clone)]
pub struct ApiEmbedding {
    client: Client,
    url: String,
    api_key: Option<String>,
}

impl ApiEmbedding {
    /// Create a new provider for the given endpoint.
    #[must_use]
    pub fn new(url: impl Into<String>, api_key: Option<String>) -> Self {
        Self {
            client: Client::new(),
            url: url.into(),
            api_key,
        }
    }
}

impl TextProcessor for ApiEmbedding {
    type Output = Box<[f32]>;
    type Error = ApiEmbeddingError;

    fn process(&self, input: &str) -> Result<Self::Output, Self::Error> {
        if input.trim().is_empty() {
            return Err(ApiEmbeddingError::Empty);
        }
        let mut req = self
            .client
            .post(&self.url)
            .json(&serde_json::json!({ "input": input }));
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send()?.error_for_status()?;
        let api: ApiResponse = resp
            .json()
            .map_err(|_| ApiEmbeddingError::InvalidResponse)?;
        Ok(api.embedding.into_boxed_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use rstest::rstest;

    fn provider(server: &MockServer) -> ApiEmbedding {
        ApiEmbedding::new(format!("{}/embed", server.base_url()), None)
    }

    #[rstest]
    #[expect(clippy::expect_used, reason = "test should fail loudly")]
    fn returns_embedding() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/embed");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(serde_json::json!({ "embedding": [1.0, 2.0] }));
        });
        let p = provider(&server);
        let emb = p.process("hi").expect("embedding");
        assert_eq!(&*emb, &[1.0, 2.0]);
    }

    #[rstest]
    fn empty_error() {
        let server = MockServer::start();
        let p = provider(&server);
        assert_eq!(p.process(""), Err(ApiEmbeddingError::Empty));
    }
}
