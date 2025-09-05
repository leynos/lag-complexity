//! BDD tests for the API embedding provider.

use httpmock::{Method::POST, MockServer};
use lag_complexity::{ApiEmbedding, ApiEmbeddingError, TextProcessor};
use rstest::fixture;
use rstest_bdd_macros::{given, scenario, then, when};
use std::cell::RefCell;

struct TestContext {
    server: MockServer,
    provider: RefCell<Option<ApiEmbedding>>,
    result: RefCell<Option<Result<Box<[f32]>, ApiEmbeddingError>>>,
}

#[fixture]
fn test_context() -> TestContext {
    let server = MockServer::start();
    TestContext {
        server,
        provider: RefCell::new(None),
        result: RefCell::new(None),
    }
}

#[given("an API embedding provider")]
fn given_provider(#[from(test_context)] ctx: &TestContext) {
    let url = format!("{}/embed", ctx.server.base_url());
    *ctx.provider.borrow_mut() = Some(ApiEmbedding::new(url, None));
}

#[when("embedding {text}")]
#[expect(
    clippy::needless_pass_by_value,
    reason = "BDD macro injects owned value"
)]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn when_embedding(text: String, #[from(test_context)] ctx: &TestContext) {
    ctx.server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(serde_json::json!({ "embedding": [0.1, 0.2] }));
    });
    let provider = ctx.provider.borrow();
    let provider = provider.as_ref().expect("provider missing");
    *ctx.result.borrow_mut() = Some(provider.process(&text));
}

#[when("embedding fails")]
#[expect(clippy::expect_used, reason = "test should fail loudly")]
fn when_embedding_fails(#[from(test_context)] ctx: &TestContext) {
    ctx.server.mock(|when, then| {
        when.method(POST).path("/embed");
        then.status(500);
    });
    let provider = ctx.provider.borrow();
    let provider = provider.as_ref().expect("provider missing");
    *ctx.result.borrow_mut() = Some(provider.process("oops"));
}

#[then("an embedding of length {len}")]
fn then_len(#[from(test_context)] ctx: &TestContext, len: usize) {
    if let Some(Ok(emb)) = ctx.result.borrow().as_ref() {
        assert_eq!(emb.len(), len);
    } else {
        panic!("missing result");
    }
}

#[then("an embedding error is returned")]
fn then_error(#[from(test_context)] ctx: &TestContext) {
    assert!(matches!(ctx.result.borrow().as_ref(), Some(Err(_))));
}

#[scenario(path = "tests/features/api_embedding.feature", index = 0)]
fn api_success(test_context: TestContext) {
    let _ = test_context;
}

#[scenario(path = "tests/features/api_embedding.feature", index = 1)]
fn api_failure(test_context: TestContext) {
    let _ = test_context;
}
