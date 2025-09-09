#![cfg(feature = "provider-api")] // Gate tests on provider-api feature
use httpmock::{Method::POST, MockServer};
use lag_complexity::{ApiEmbedding, ApiEmbeddingError, TextProcessor};
use rstest::*;

#[fixture]
fn mock_server() -> MockServer {
    MockServer::start()
}

#[fixture]
fn api_provider(mock_server: MockServer) -> (ApiEmbedding, MockServer) {
    let url = format!("{}/embed", mock_server.base_url());
    (ApiEmbedding::new(url, None), mock_server)
}

#[fixture]
fn api_provider_with_auth(mock_server: MockServer) -> (ApiEmbedding, MockServer) {
    let url = format!("{}/embed", mock_server.base_url());
    (ApiEmbedding::new(url, Some("secret".into())), mock_server)
}

#[rstest]
#[case("hello", vec![0.1, 0.2], serde_json::json!({ "embedding": [0.1, 0.2] }))]
fn test_success_cases(
    #[case] input: &str,
    #[case] expected_embedding: Vec<f32>,
    #[case] mock_json: serde_json::Value,
    api_provider: (ApiEmbedding, MockServer),
) {
    let (provider, server) = api_provider;
    server.mock(|when, then| {
        when.method(POST)
            .path("/embed")
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "input": input }));
        then.status(200).json_body(mock_json);
    });
    let emb = match provider.process(input) {
        Ok(e) => e,
        Err(e) => panic!("embedding error: {e:?}"),
    };
    assert_eq!(&*emb, expected_embedding.as_slice());
}

#[rstest]
#[case("hi", vec![1.0], serde_json::json!({ "embedding": [1.0] }))]
fn test_bearer_auth_success(
    #[case] input: &str,
    #[case] expected_embedding: Vec<f32>,
    #[case] mock_json: serde_json::Value,
    api_provider_with_auth: (ApiEmbedding, MockServer),
) {
    let (provider, server) = api_provider_with_auth;
    server.mock(|when, then| {
        when.method(POST)
            .path("/embed")
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret")
            .json_body(serde_json::json!({ "input": input }));
        then.status(200).json_body(mock_json);
    });
    let emb = match provider.process(input) {
        Ok(e) => e,
        Err(e) => panic!("embedding error: {e:?}"),
    };
    assert_eq!(&*emb, expected_embedding.as_slice());
}

#[derive(Clone, Copy)]
enum ExpectedError {
    Request,
    Empty,
    InvalidResponse,
}

#[rstest]
#[case("oops", 500, None::<serde_json::Value>, Some(""), ExpectedError::Request)]
#[case(
    "text",
    200,
    Some(serde_json::json!({ "embedding": [] })),
    None,
    ExpectedError::Empty
)]
#[case(
    "text",
    200,
    None,
    Some(r#"{\"embedding\":[\"x\"]}"#),
    ExpectedError::InvalidResponse
)]
#[case(
    "hi",
    200,
    Some(serde_json::json!({ "oops": true })),
    None,
    ExpectedError::InvalidResponse
)]
fn test_error_cases(
    #[case] input: &str,
    #[case] http_status: u16,
    #[case] response_json: Option<serde_json::Value>,
    #[case] response_body: Option<&str>,
    #[case] expected_error: ExpectedError,
    api_provider: (ApiEmbedding, MockServer),
) {
    let (provider, server) = api_provider;
    server.mock(|when, then| {
        when.method(POST).path("/embed");
        match (response_json, response_body) {
            (Some(json), _) => {
                then.status(http_status)
                    .header("content-type", "application/json")
                    .json_body(json);
            }
            (None, Some(body)) => {
                then.status(http_status)
                    .header("content-type", "application/json")
                    .body(body);
            }
            (None, None) => {
                then.status(http_status);
            }
        }
    });
    let Err(err) = provider.process(input) else {
        panic!("expected error")
    };
    match expected_error {
        ExpectedError::Request => assert!(matches!(err, ApiEmbeddingError::Request(_))),
        ExpectedError::Empty => assert_eq!(err, ApiEmbeddingError::Empty),
        ExpectedError::InvalidResponse => assert_eq!(err, ApiEmbeddingError::InvalidResponse),
    }
}

#[test]
fn rejects_empty_input() {
    let server = MockServer::start();
    let url = format!("{}/embed", server.base_url());
    let provider = ApiEmbedding::new(url, None);
    assert_eq!(provider.process(""), Err(ApiEmbeddingError::Empty));
}
