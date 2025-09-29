# Roadmap

## Phase 0 — Scaffolding & Core API (Duration: 1 week)

This foundational phase establishes the crate's architecture and defines the
primary public interfaces.

- [ ] Initialise the Rust project via `cargo new`.
- [x] Define all public data structures (`Complexity`, `Trace`, `ScopingConfig`
    and its subtypes) and derive `serde` traits for configuration types.
- [x] Define all public traits (`ComplexityFn`, `EmbeddingProvider`,
    `DepthEstimator`, `AmbiguityEstimator`).
- [x] Implement the mathematical logic for variance calculation and all `Sigma`
  normalization strategies.
- [x] Create the stub for the `lagc` CLI binary using `ortho_config`
  (published as `ortho-config` on crates.io) [^hyphen-underscore].
- [ ] **Acceptance Criteria**: The crate and all its core types compile
  successfully.
- [x] **Acceptance Criteria**: A comprehensive suite of unit tests for the
  mathematical and normalization logic passes.
- [x] **Acceptance Criteria**: The `lagc` CLI application can be built and run
  (verify with: `cargo run --bin lagc -- --help`), though it will have no
  functional commands yet.

## Phase 1 — Heuristic Baseline (Duration: 1–2 weeks)

This phase delivers the first end-to-end, functional version of the scorer,
relying on fast, lightweight heuristics.

- [x] Implement the `DepthHeuristic` and `AmbiguityHeuristic` providers.
- [x] Add antecedent-aware pronoun weighting to the ambiguity heuristic.
- [x] Add the regex-based ambiguous entity pre-pass for the heuristic lexicon.
- [x] Implement the `ApiEmbedding` provider, guarded by the `provider-api`
  feature flag.
- [x] Create the golden-file integration test suite with an initial set of
  approximately 50 curated queries and their expected trace outputs.
- [x] **Acceptance Criteria**: The `score()` and `trace()` methods are fully
  functional using the heuristic providers.
- [x] **Acceptance Criteria**: The golden-file integration tests pass,
  establishing a baseline for regression testing.

## Phase 2 — Model-Backed Providers & Performance (Duration: 2 weeks)

This phase focuses on enhancing accuracy with model-based providers and
optimizing for performance.

- [ ] Train or adapt and export the initial ONNX models for depth and ambiguity
  classification.
- [ ] Implement the `DepthClassifierOnnx` and `AmbiguityClassifierOnnx`
  providers, gated by the `onnx` feature flag.
- [ ] Implement the `score_batch` method and integrate `rayon` for parallel
  execution.
- [ ] Set up the `criterion` benchmarking suite and implement the initial set
  of micro and macro benchmarks.
- [ ] **Acceptance Criteria**: The ONNX-based providers can be successfully
  composed into a `DefaultComplexity` engine and produce valid scores.
- [ ] **Acceptance Criteria**: The `score_batch` method demonstrates a
  significant performance speedup when the `rayon` feature is enabled.
- [ ] **Acceptance Criteria**: Initial performance metrics (latency,
  throughput) are recorded in `BENCHMARKS.md`.

## Phase 3 — Evaluation & Calibration (Duration: 1 week)

This phase is dedicated to empirically validating the scorer's effectiveness
and tuning its parameters.

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

## Phase 4 — Bindings & Demos (Duration: 2 weeks)

This phase focuses on making the crate accessible from other ecosystems and
creating compelling demonstrations.

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

## Phase 5 — Production Hardening (Duration: 1 week)

The final phase adds the remaining features required for robust, secure, and
observable production deployment.

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

[^hyphen-underscore]: Cargo converts hyphens to underscores for import paths.
                      The package is `ortho-config` on crates.io and is
                      imported as `ortho_config` in code.
