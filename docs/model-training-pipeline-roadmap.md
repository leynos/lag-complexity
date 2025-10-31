# Model training pipeline implementation roadmap

This roadmap decomposes the design in `docs/model-training-pipeline-design.md`
into executable work items. Phases align to the major capability shifts in the
design; within each phase, steps outline coherent workstreams, and tasks define
measurable deliverables. Actionable tasks are prefixed with checkboxes for easy
progress tracking.

## Phase 1: Establish self-hosted orchestration baselines

### Step: Deploy Prefect Orion control plane

- [ ] Stand up a Prefect Orion server with Postgres backing, TLS termination,
      and daily backup restore drills completing inside 30 minutes.
  - [ ] Document Orion control plane topology, including Postgres sizing,
        networking, and certificate authority integration.
  - [ ] Automate provisioning for Postgres, object storage, and Orion services
        via infrastructure-as-code checked into version control.
  - [ ] Configure TLS termination with managed certificate renewal and run
        acceptance tests confirming HTTPS enforcement.
  - [ ] Schedule daily logical backups and store encrypted snapshots with
        retention meeting recovery point objectives.
  - [ ] Rehearse restore drills into a staging environment, recording elapsed
        time and remediation notes when runs exceed 30 minutes.
- [ ] Register dedicated Prefect agents for control and compute queues, proven
      by smoke flows covering provisioning, training, and teardown paths.
  - [ ] Define queue taxonomy (control versus compute) with resource limits,
        labels, and expected concurrency documented for operators.
  - [ ] Deploy long-lived Prefect agent processes (e.g., systemd units or K8s
        workloads) bound to their target queues with observability hooks.
  - [ ] Implement smoke flows that exercise provisioning, training, and
        teardown steps using representative infrastructure modules.
  - [ ] Run smoke flows on a schedule and capture artefacts that prove agents
        drain work without orphaned runs or leaked machines.
- [ ] Model secrets, artefact buckets, and Terraform variables with Prefect
      blocks, including automated rotation tests and alerting on failures.
  - [ ] Inventory required secrets, bucket endpoints, and Terraform variables,
        mapping ownership and renewal cadences for each item.
  - [ ] Define Prefect block schemas per environment and populate them from
        the approved secret management sources.
  - [ ] Implement automated rotation or refresh flows that validate block
        values and roll back when verification fails.
  - [ ] Wire alerting (Slack, PagerDuty, email) to Prefect block failures and
        add runbook links for operator response.

### Step: Standardise compute procurement and cost controls

- [ ] Automate OpenTofu modules for g4dn.xlarge, g6.xlarge, and c6i.xlarge
      instances with provider parameterisation and teardown under 10 minutes.
- [ ] Validate spot-to-on-demand fallbacks by injecting capacity failures and
      confirming retries succeed within three orchestration attempts.
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
- [ ] Generate metadata manifest templates (experiment ID, git SHA, checksums)
      and enforce schema validation before artefacts transition phases.

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

- [ ] Compose a Prefect flow wiring data ingestion, provisioning, training, and
      teardown tasks with typed parameters and run documentation.
- [ ] Implement a `provision_training_vm` Prefect task invoking OpenTofu GPU
      modules and confirm teardown runs on every flow completion path.
- [ ] Implement checkpoint cadence control ensuring <10 minutes of lost work
      under forced termination in staging drills.
- [ ] Capture structured metrics (loss, cutpoints, throughput) and emit them to
      Prefect Orion and the observability stack with comparison dashboards.

## Phase 3: Operationalise export and optimisation

### Step: Build ONNX export automation

- [ ] Script `optimum-cli export onnx` runs with reproducible configuration
      files checked into version control alongside commit hashes.
- [ ] Add regression tests asserting exported graphs match expected opset,
      input shapes, and dynamic axes for target models.
- [ ] Execute export and verification Prefect tasks on CPU OpenTofu modules,
      proving GPU instances are torn down before CPU provisioning.
- [ ] Store FP32 exports under `/models/onnx/{experiment}/fp32/` with metadata
      capturing opset, optimiser flags, and git references.

### Step: Enforce parity verification gates

- [ ] Implement parity comparison jobs using `onnxruntime` that evaluate ≥500
      representative samples with `atol=1e-5` and matching `rtol` thresholds.
- [ ] Fail the pipeline when parity deviations exceed tolerance, surfacing
      diagnostics (max delta tensors, sample inputs) in build artefacts.
- [ ] Schedule nightly parity spot-checks on retained models to detect drift in
      dependencies (CUDA, ONNX Runtime) before release branches freeze.

### Step: Deliver quantisation and benchmarking workflow

- [ ] Codify static INT8 quantisation scripts with calibration dataset reuse
      and artefact write-back to `/models/onnx/{experiment}/int8/`.
- [ ] Measure INT8 throughput versus FP32 on target CPU instances, documenting
      improvements and acceptable accuracy deltas (<1% MAE loss).
- [ ] Integrate quantised-model verification into CI with `atol=1e-2`
      tolerances, including latency smoke tests under realistic batch sizes.

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
