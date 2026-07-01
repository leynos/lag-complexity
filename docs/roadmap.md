# Roadmap


## 1. Scaffolding & Core API (Duration: 1 week)

This foundational phase establishes the crate's architecture and defines the
primary public interfaces.


### 1.1. Scaffolding & Core API

- [ ] Initialise the Rust project via `cargo new`.
- [x] Define all public data structures (`Complexity`, `Trace`, `ScopingConfig`
    and its subtypes) and derive `serde` traits for configuration types.
- [x] Define all public traits (`ComplexityFn`, `EmbeddingProvider`,
    `DepthEstimator`, `AmbiguityEstimator`).
- [x] Implement the mathematical logic for variance calculation and all `Sigma`
  normalization strategies.
- [x] Create the stub for the `lagc` CLI binary using `ortho_config`
  (published as `ortho-config` on crates.io and imported as `ortho_config` in
  code; Cargo converts hyphens to underscores for import paths).
- [ ] **Acceptance Criteria**: The crate and all its core types compile
  successfully.
- [x] **Acceptance Criteria**: A comprehensive suite of unit tests for the
  mathematical and normalization logic passes.
- [x] **Acceptance Criteria**: The `lagc` CLI application can be built and run
  (verify with: `cargo run --bin lagc -- --help`), though it will have no
  functional commands yet.


## 2. Heuristic Baseline (Duration: 1–2 weeks)

This phase delivers the first end-to-end, functional version of the scorer,
relying on fast, lightweight heuristics.


### 2.1. Heuristic Baseline

- [x] Implement the `DepthHeuristic` and `AmbiguityHeuristic` providers.
- [x] Add antecedent-aware pronoun weighting to the ambiguity heuristic.
- [x] Add the case-insensitive, word-boundary regex pre-pass for the curated
  ambiguous entity lexicon (see `docs/lag-complexity-function-design.md` and PR
  [#41](https://github.com/leynos/lag-complexity/pull/41)).
- [x] Implement the `ApiEmbedding` provider, guarded by the `provider-api`
  feature flag.
- [x] Create the golden-file integration test suite with an initial set of
  approximately 50 curated queries and their expected trace outputs.
- [x] **Acceptance Criteria**: The `score()` and `trace()` methods are fully
  functional using the heuristic providers.
- [x] **Acceptance Criteria**: The golden-file integration tests pass,
  establishing a baseline for regression testing.


## 3. Model-Backed Providers & Performance (Duration: 2 weeks)

This phase focuses on enhancing accuracy with model-based providers and
optimizing for performance.


### 3.1. Model-Backed Providers & Performance

- [ ] 3.1.1. Deliver the depth classifier ONNX artefact as specified in the
  accepted ADR:
  - [ ] 3.1.1.1. Fine-tune the DistilBERT ordinal model on ordered depth bins,
    confirm the expected-value projection matches the Sigma calibration
    interface, and export the opset 17 graph with pinned tokenizer assets.
  - [ ] 3.1.1.2. Apply post-training static INT8 quantisation, benchmark CPU
    latency to p95 ≤ 10 ms, and record calibration coefficients for the
    provider configuration.
  - [ ] 3.1.1.3. Publish the model package with checksum manifest, versioned
    artefact paths, and documentation of fallback MLP expectations.
- [ ] 3.1.2. Produce the ambiguity classifier ONNX artefact aligned with the
  design document:
  - [ ] 3.1.2.1. Curate and label the ambiguity dataset (AmbigQA plus curated
    edge cases) into the Clear / Possibly Ambiguous / Highly Ambiguous classes
    defined in the design.
  - [ ] 3.1.2.2. Fine-tune the lightweight transformer, export the ONNX graph
    with the categorical-to-numeric mapping required by Sigma, and pin the
    tokenizer vocabulary.
  - [ ] 3.1.2.3. Quantise and benchmark the graph to meet latency and
    footprint targets, then publish artefacts and checksum metadata alongside
    calibration notes.
- [ ] 3.1.3. Implement the `DepthClassifierOnnx` provider behind the `onnx`
  feature flag:
  - [ ] 3.1.3.1. Load the opset 17 session through `ort`, enforce checksum
    validation, and wire the expected-value scalar projection into
    `TextProcessor<Output = f32>` per the ADR.
  - [ ] 3.1.3.2. Integrate tracing, metrics, and configuration plumbing (model
    paths, quantisation variant selection, calibration coefficients)
    consistent with the Complexity pipeline.
  - [ ] 3.1.3.3. Extend golden traces and unit tests to cover inference happy
    paths, error mapping, and deterministic outputs across feature
    combinations.
- [ ] 3.1.4. Implement the `AmbiguityClassifierOnnx` provider behind the
  `onnx` feature flag:
  - [ ] 3.1.4.1. Mirror the runtime scaffolding (session loading, checksum
    enforcement, feature gating) used by the depth provider while mapping
    class logits to the numeric ambiguity score expected by Sigma.
  - [ ] 3.1.4.2. Expose configuration toggles for model variants and
    calibration, add tracing and metrics instrumentation, and ensure
    integration with `DefaultComplexity`.
  - [ ] 3.1.4.3. Expand regression tests and golden traces to validate
    deterministic scoring and parity with the documented ambiguity thresholds.
- [ ] 3.1.5. Implement the `score_batch` method and integrate `rayon` for
  parallel execution.
- [ ] 3.1.6. Set up the `criterion` benchmarking suite and implement the
  initial set of micro and macro benchmarks.
- [ ] 3.1.7. **Acceptance Criteria**: The ONNX-based providers can be
  successfully composed into a `DefaultComplexity` engine and produce valid
  scores.
- [ ] 3.1.8. **Acceptance Criteria**: The `score_batch` method demonstrates a
  significant performance speedup when the `rayon` feature is enabled.
- [ ] 3.1.9. **Acceptance Criteria**: Initial performance metrics (latency,
  throughput) are recorded in `BENCHMARKS.md`.


## 4. Evaluation & Calibration (Duration: 1 week)

This phase is dedicated to empirically validating the scorer's effectiveness
and tuning its parameters.


### 4.1. Evaluation & Calibration

- [ ] Build the dataset evaluation harness binary.
- [ ] Integrate loaders for the target datasets (`HotpotQA`, `AmbigQA`, etc.).
- [ ] Implement the calculation of correlation (`Kendall-τ`, `Spearman-ρ`) and
  calibration (`ECE`) metrics.
- [ ] Run the evaluation harness and analyse the results to fine-tune the
  `Sigma` normalization parameters and the weights within the heuristic models.
- [ ] **Acceptance Criteria**: The evaluation harness successfully generates a
  report (`EVALUATION.md`).
- [ ] **Acceptance Criteria**: The report demonstrates a statistically
  significant positive correlation between the crate's component scores and the
  corresponding dataset labels.
- [ ] **Acceptance Criteria**: The calibrated parameters are finalized and
  committed as the default configuration.


## 5. Bindings & Demos (Duration: 2 weeks)

This phase focuses on making the crate accessible from other ecosystems and
creating compelling demonstrations.


### 5.1. Bindings & Demos

- [ ] Implement the Python bindings using `pyo3`.
- [ ] Implement the WebAssembly bindings using `wasm-bindgen`.
- [ ] Develop the interactive "Complexity Meter" web page using the WASM module.
- [ ] Create the Jupyter notebooks for the "Smart Assistant" and "Ambiguity
  Resolver" stakeholder demonstrations.
- [ ] **Acceptance Criteria**: The Python package can be built, installed via
  `pip`, and used to score queries.
- [ ] **Acceptance Criteria**: The WASM demo is fully functional, interactive,
  and hosted on a static page.
- [ ] **Acceptance Criteria**: The demonstration notebooks are complete and
  successfully showcase the crate's value.


## 6. Production Hardening (Duration: 1 week)

The final phase adds the remaining features required for robust, secure, and
observable production deployment.


### 6.1. Production Hardening

- [ ] Instrument the entire crate with `tracing` spans and `metrics` calls.
- [ ] Implement the `moka`-based `CachingEmbeddingProvider`.
- [ ] Implement the `with_redaction_hook` method for PII scrubbing.
- [ ] Write comprehensive `rustdoc` documentation for all public APIs,
  including detailed usage examples.
- [ ] Finalize the `README.md` to include installation instructions, usage
  examples, and links to benchmarks and evaluation reports.
- [ ] **Acceptance Criteria**: The crate is fully documented, with
  `cargo doc --open` producing a complete and navigable API reference.
- [ ] **Acceptance Criteria**: All production features (observability, caching,
  security hooks) are implemented and tested.
- [ ] **Acceptance Criteria**: The final project is ready for its first
  official release.
