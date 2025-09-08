#![cfg(feature = "provider-api")]
use httpmock::{Method::POST, MockServer};
use lag_complexity::{ApiEmbedding, ApiEmbeddingError, TextProcessor};

#[test]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn embedding_success() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "embedding": [0.1, 0.2] }));
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    let emb = provider.process("hello").expect("embedding");
    assert_eq!(&*emb, &[0.1, 0.2]);
}

#[test]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn sets_bearer_auth_header() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST)
            .path("/embed")
            .header("authorization", "Bearer secret")
            .json_body(serde_json::json!({ "input": "hi" }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "embedding": [1.0] }));
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, Some("secret".into()));
    let emb = provider.process("hi").expect("embedding");
    assert_eq!(&*emb, &[1.0]);
}

#[test]
fn rejects_empty_input() {
    let server = MockServer::start();
    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    assert_eq!(provider.process(""), Err(ApiEmbeddingError::Empty));
}

#[test]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn server_failure_returns_request_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(500);
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    let err = provider.process("oops").expect_err("error");
    assert!(matches!(err, ApiEmbeddingError::Request(_)));
}

#[test]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn empty_embedding_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "embedding": [] }));
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    let err = provider.process("text").expect_err("error");
    assert_eq!(err, ApiEmbeddingError::Empty);
}

#[test]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn invalid_embedding_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(200)
            .header("content-type", "application/json")
            .body(r#"{"embedding":["x"]}"#);
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    let err = provider.process("text").expect_err("error");
    assert_eq!(err, ApiEmbeddingError::InvalidResponse);
}

#[test]
fn invalid_json_yields_invalid_response() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        // Missing `embedding` field
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "oops": true }));
    });

    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    assert_eq!(
        provider.process("hi"),
        Err(ApiEmbeddingError::InvalidResponse),
    );
}
