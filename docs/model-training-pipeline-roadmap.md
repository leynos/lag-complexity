# Model training pipeline implementation roadmap

This roadmap decomposes the design in `docs/model-training-pipeline-design.md`
into executable work items. Phases align to the major capability shifts in the
design; within each phase, steps outline coherent workstreams, and tasks define
measurable deliverables. Actionable tasks are prefixed with checkboxes for easy
progress tracking.

## Phase 1: Establish reproducible infrastructure baselines

### Step: Standardise compute procurement and cost controls

- [ ] Automate g4dn.xlarge, g6.xlarge, and p4d.24xlarge provisioning via
      Terraform with spot/on-demand toggles validated in sandboxes.
- [ ] Publish hourly cost benchmarks for each instance class with variance
      thresholds (±5%) and alerting when breached.
- [ ] Implement spot interruption handling tests that confirm checkpoint resume
      succeeds after simulated two-minute warnings.

### Step: Harden storage and artefact governance

- [ ] Roll out S3-compatible bucket layout matching the design’s directory
      schema with lifecycle rules for hot, warm, and archived tiers.
- [ ] Enforce object immutability and versioning policies, including bucket
      replication rehearsal between primary and disaster-recovery regions.
- [ ] Provide IAM policies granting least-privilege access by pipeline stage,
      validated with automated policy-as-code tests.

## Phase 2: Deliver the Python fine-tuning pipeline

### Step: Implement deterministic data ingestion and preparation

- [ ] Build ingestion scripts that hydrate `/datasets/raw/` from approved
      sources and emit checksums recorded in metadata manifests.
- [ ] Create preprocessing jobs that materialise `/datasets/processed/`
      artefacts with schema validation and rejection workflows for anomalies.
- [ ] Introduce calibration dataset generation producing 100–500 curated
      samples stored under `/datasets/processed/{dataset}/calibration/`.

### Step: Ship ordinal-aware training components

- [ ] Develop the `OrdinalRegressionTrainer` with cutpoint ordering guarantees
      and unit tests covering boundary cases (monotonicity, identical labels).
- [ ] Parameterise model head replacement logic to support both BERT-like and
      ViT architectures, validated on representative text and image datasets.
- [ ] Achieve ≥90% statement coverage on the training module using `pytest`
      plus fixtures from `rstest`.

### Step: Orchestrate resilient training execution

- [ ] Containerise the training entrypoint with dependency pinning and
      deterministic start-up seeded from environment variables.
- [ ] Implement checkpoint cadence control ensuring <10 minutes of lost work
      under forced termination in staging drills.
- [ ] Capture structured metrics (loss, cutpoints, throughput) and emit them to
      the observability stack with dashboards for experiment comparison.

## Phase 3: Operationalise export and optimisation

### Step: Build ONNX export automation

- [ ] Script `optimum-cli export onnx` runs with reproducible configuration
      files checked into version control alongside commit hashes.
- [ ] Add regression tests asserting exported graphs match expected opset,
      input shapes, and dynamic axes for target models.
- [ ] Store FP32 exports under `/models/onnx/{experiment}/fp32/` with metadata
      capturing opset, optimiser flags, and git references.

### Step: Enforce parity verification gates

- [ ] Implement parity comparison jobs using `onnxruntime` that evaluate ≥500
      representative samples with `atol`/`rtol` thresholds from the design.
- [ ] Fail the pipeline when parity deviations exceed tolerance, surfacing
      diagnostics (max delta tensors, sample inputs) in build artefacts.
- [ ] Schedule nightly parity spot-checks on retained models to detect drift in
      dependencies (CUDA, ONNX Runtime) before release branches freeze.

### Step: Deliver quantisation and benchmarking workflow

- [ ] Codify static INT8 quantisation scripts with calibration dataset reuse
      and artefact write-back to `/models/onnx/{experiment}/int8/`.
- [ ] Measure INT8 throughput versus FP32 on target CPU instances, documenting
      improvements and acceptable accuracy deltas (<1% MAE loss).
- [ ] Integrate quantised-model verification into CI, including latency smoke
      tests under realistic batch sizes.

## Phase 4: Integrate with the Rust inference surface

### Step: Package deployment artefacts

- [ ] Assemble deployment bundles comprising ONNX models, tokenizer assets, and
      metadata JSON with schema validation during publish.
- [ ] Provide changelog automation that records experiment IDs, cutpoints, and
      evaluation metrics for each released bundle.
- [ ] Implement retention policies ensuring at least three previous artefact
      versions remain retrievable for rollback drills.

### Step: Embed ONNX Runtime usage in Rust services

- [ ] Add `ort` crate integration with the `load-dynamic` feature, including
      integration tests that exercise the dynamic loader across Linux targets.
- [ ] Supply environment bootstrap scripts that install ONNX Runtime binaries
      and validate `ORT_DYLIB_PATH` prior to service start.
- [ ] Benchmark end-to-end inference latencies (p50, p95) under representative
      traffic, ensuring INT8 pipelines meet sub-50 ms targets on t4g.large.

### Step: Operationalise release governance

- [ ] Define promotion criteria linking training experiment success, parity
      checks, and Rust e2e tests before artefacts become GA releases.
- [ ] Implement audit logging for artefact promotions and consumption events so
      provenance is traceable during incident reviews.
- [ ] Schedule quarterly resilience tests covering checkpoint restore, model
      rollback, and dependency upgrade rehearsals with pass/fail reporting.
