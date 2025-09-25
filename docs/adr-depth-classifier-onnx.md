# Architecture decision record: DepthClassifierOnnx — model selection and architecture for the Cognitive Clutch

**Status:** Accepted (Revised) \
**Date:** 24 September 2025 \
**Owner:** LAG Complexity / Cognitive Clutch \
**Related:** LAG Complexity design document (Complexity Function & Cognitive
Clutch)

______________________________________________________________________

## 1. Context

The Cognitive Clutch selects between fast heuristic inference and
logic-augmented generation (LAG) using a composite **Complexity** signal. The
LAG Complexity design document defines a provider-based architecture where
independent signals (e.g., depth, ambiguity) are computed, normalized via
**Sigma**, and aggregated with weights and schedules. The depth provider must
output a scalar score that correlates with the compositional reasoning load of
a textual question while meeting production constraints: low latency,
deterministic behaviour, no network dependencies, and clean integration with
the Rust codebase (tracing, metrics, golden tests).

## 2. Problem statement

Select a model family and Open Neural Network Exchange (ONNX) architecture for
a depth estimator that:

- Achieves single-digit millisecond latency on CPU and a small memory footprint.
- Integrates with existing provider traits; returns a scalar **raw depth**
  compatible with Sigma normalization and threshold schedules defined in the
  LAG Complexity design.
- Preserves determinism and reproducibility across platforms.
- Provides a path to improved robustness and interpretability over time.

## 3. Decision drivers

- **Latency & footprint:** p95 < 10 ms per query on CPU; model artefact compact;
  limited heap churn.
- **Determinism & portability:** pinned opset, frozen weights, embedded
  pre-processing constants; identical outputs across supported platforms via
  ONNX Runtime.
- **Integration fit:** implements `TextProcessor<Output = f32>`; clean wiring
  into Complexity → Sigma → Weights → Schedule; full tracing and metrics.
- **Calibration & monotonicity:** outputs align with ordered difficulty/step
  levels; stable after calibration across domains.
- **Maintainability & TCO:** viable lifecycle with monitoring, retraining, and
  quantization; fallbacks available.

## 4. Options considered

1. **Transformer-ordinal (selected)** — Compact Transformer encoder (e.g.,
   DistilBERT-class) exported to ONNX; ordinal regression head over ordered
   depth levels; scalar depth derived from ordinal probabilities.
2. **Hybrid Transformer + Explicit Features (roadmap)** — Fuse a small set of
   engineered syntactic/lexical features with Transformer representations;
   ordinal head as in Option 1.
3. **Fixed-feature multilayer perceptron (MLP) (fallback)** — Feed-forward
   network over a pure-Rust feature vector predicting a continuous log-steps
   target; quantized ONNX; used where tokenization is constrained.
4. **Hashed n-gram linear/FFN (baseline)** — Extremely small model to validate
   regressions and act as an emergency backup.

## 5. Decision

Adopt **Transformer-ordinal** as the default architecture for
`DepthClassifierOnnx`, exported to ONNX and executed with ONNX Runtime on CPU.
Commit to a near-term optimization plan (post-training static INT8
quantization; intermediate-layer pooling ablation) and a longer-term **Hybrid**
roadmap for increased robustness and interpretability. Ship the **Fixed-feature
MLP** as a supported fallback under a feature flag.

## 6. Selected architecture — Transformer-ordinal

### I/O signature

- Inputs: `input_ids: int64[batch, seq]`, `attention_mask: int64[batch, seq]`.
  The Rust tokenizer right-pads and right-truncates every request to a fixed
  `seq = max_seq_len` (512 tokens for the launch build). `batch` is dynamic in
  the ONNX graph, but the provider currently executes single-query batches.
- Outputs: `logits_ord: float32[batch, K]` where `K` is the number of ordered
  depth thresholds. The runtime reads the first batch element and discards the
  rest, preserving compatibility with future batched inference.

### Ordinal head

- Cumulative/threshold encoding: head `k` predicts `P(depth > τ_k)` via
  `sigmoid`.
- Scalar depth for the Cognitive Clutch is computed as either:
  - **Expected value:** `E[steps] = Σ_k σ(logits_ord[k])`, yielding a value in
    the closed range `[0, K]`. The score can then be shifted and scaled onto
    the calibrated step axis before Sigma normalization.
  - **Mid-bin mapping:** count the heads with `σ(logits_ord[k]) ≥ θ` for a
    configured decision threshold `θ`. Map that integer in `0..=K` to the
    representative step value for the corresponding bin.
- This scalar is returned as the provider’s **raw depth**, then normalized by
  Sigma and combined per the design document.

### Backbone and representation

- DistilBERT-class encoder (or equivalent lightweight encoder) with
  **multi-layer representation pooling**: concatenate or pool the `[CLS]`
  hidden states from the last 4 layers before the ordinal head. This exploits
  the known stratification of linguistic information across layers.

### Graph notes

- Opset: 17.  
- Graph includes tokenization-independent components only; tokenization
  performed in Rust using a deterministic vocabulary (e.g., `tokenizers`).
- Optional calibration added as `Mul`/`Add` after the ordinal expectation;
  otherwise applied in Rust before Sigma.

## 7. Short-term optimizations (committed)

- **Static INT8 quantization:** apply post-training static quantization (weights
  and activations) for CPU; benchmark size, latency, and accuracy drift against
  FP32.
- **Intermediate-layer ablation:** compare final-layer vs multi-layer pooled
  representations; promote the best variant by validation metrics and latency.

## 8. Long-term roadmap — hybrid fusion

- **Feature selection:** identify a small, computationally inexpensive set of
  syntactic/lexical features (e.g., Average Dependency Distance,
  clause-to-sentence ratio, bracket/centre-embedding depth, limited curated
  bigrams).
- **Fusion strategies:** start with **early fusion** (concatenate pooled
  Transformer vector with engineered features before the ordinal head);
  evaluate **multi-layer fusion** and attention-based injection (attributes
  influencing attention) as advanced candidates.
- **Objective:** improve robustness to paraphrase and domain shift; increase
  interpretability while keeping latency within targets.

## 9. Training and labelling (offline)

- **Target definition:** ordered depth levels aligned to the LAG Complexity
  notion of “reasoning steps” (bins defined during dataset curation).
- **Encoding & loss:** cumulative targets over `K` thresholds; Binary
  Cross-Entropy across heads; class weighting optional.
- **Data:** multi-hop QA, mathematical word problems, code-reasoning prompts,
  and general QA. Teacher signals (e.g., chain-of-thought token counts or
  supporting-fact hops) inform binning.
- **Validation metrics:** Spearman ρ (rank correlation), MAE on mapped steps,
  and Expected Calibration Error (ECE) for the ordinal head.
- **Early stopping:** monitor ρ/MAE/ECE; snapshot best by composite score.

## 10. Calibration

- **Method:** 1-D isotonic or temperature-scaled affine mapping fitted on a
  held-out split.
- **Location:** either baked into the ONNX graph (`Mul`/`Add`) or applied in
  provider code before Sigma; both paths preserve determinism and observability.

## 11. Quantization and performance targets

- **Quantization:** post-training static INT8 for Transformer and head; retain
  FP32 outputs.
- **Targets:** p95 ≤ 10 ms (single query, CPU), ≤ 2 ms for batch of 16 short
  queries; ≤ 1% absolute accuracy degradation vs FP32; artefact size reduced by
  ≥ 50%.

## 12. Integration details

- **Provider type:** `DepthClassifierOnnx { session, input_names, output_names
  }`.
- **Trait:** implements `TextProcessor<Output = f32>`; returns scalar raw depth
  (expected steps or mapped mid-bin).
- **Tokenization:** performed in Rust with a pinned vocabulary; deterministic
  and locale-safe.
- **Runtime pinning:** the `ort` crate is fixed to the `2.0.x` series (bundling
  ONNX Runtime 1.18), ensuring the shipped runtime matches the opset required
  by the artefacts.
- **Error handling:** ONNX Runtime errors mapped to `Error::Inference` with
  model path, opset, and checksum in diagnostics.
- **Tracing & metrics:** instrument `process()`; export latency histograms,
  error counters, and token length/batch size tags; integrate with existing
  observability per the design document.
- **Configuration:** model path, enablement flags, quantization variant
  selection, and optional in-graph calibration switch; Sigma/Weights/Schedule
  remain unchanged.

## 13. Fixed-feature fallback — MLP-log

- **Use case:** environments where tokenization is undesirable or constrained.
- **I/O:** `float32[batch, F] → float32[batch, 1]`, `F ≈ 64–96`.
- **Head:** `Linear(F,128) → GELU → Linear(128,64) → GELU → Linear(64,1)`.
- **Target:** `log1p(steps)`; inference maps via `expm1`.
- **Quantization:** dynamic or static INT8 for linear layers; FP32 output.
- **Output:** scalar raw depth to Sigma.

## 14. Testing strategy

- **Unit:** tokenizer determinism; ordinal head probability semantics;
  calibration correctness.
- **Golden traces:** snapshot raw ordinal-derived depth alongside heuristic and
  fallback MLP on a fixed corpus; fail on drift beyond tolerance.
- **Property tests:** monotonic response as threshold count increases;
  robustness to benign paraphrases.
- **Cross-platform:** identical outputs across Linux/macOS/Windows for pinned
  opset and artefact.
- **Performance benches:** `criterion` p50/p95 on CPU for FP32 and INT8;
  publish artefacts in CI.
- **Ablations:** last-layer vs multi-layer pooling; early vs advanced fusion
  (when enabled).

## 15. Rollout plan

- Ship Transformer-ordinal behind the `onnx` feature.
- Retain heuristic depth as a safety net; enable A/B comparison in selected
  services and compare composite scores and traces.
- Promote to default after benchmarks, calibration, and acceptance criteria are
  met; keep Fixed-feature MLP as a documented fallback.

## 16. Risks and mitigations

- **Ordinality mismatch:** addressed by ordinal head and calibration.
- **Domain shift & brittleness:** mitigated by hybrid roadmap, diverse training
  data, and periodic recalibration.
- **Quantization drift:** maintain FP32 reference; CI checks compare INT8 vs
  FP32.
- **TCO:** invest in monitoring, anomaly detection, and scheduled retraining;
  keep fallback model available.
- **Opset/runtime compatibility:** pin opset; validate ORT versions; verify
  model checksum at load.

## 17. Compliance with LAG Complexity design

- Respects provider abstraction and returns a scalar depth compatible with
  Sigma normalization and weighting.
- Integrates with tracing, metrics, and golden testing.
- Keeps split logic (threshold schedules) unchanged; only the depth signal
  source varies.
- Provides a documented fallback path and a roadmap for hybrid improvements.

## 18. Operational details and versioning

- **Filename & checksum:** `depth_transformer_ordinal.onnx` (and
  `depth_mlp_log.onnx` fallback) with SHA-256 recorded and verified at load.
- **Tokenizer artefacts:** `tokenizer.json`, `vocab.txt`/`merges.txt`, or
  `spiece.model` hashes are pinned alongside the model. Provider start-up
  recomputes SHA-256 digests for each artefact and fails closed on mismatch.
- **Versioning:** semantic version in ONNX metadata; minor for calibration
  changes, patch for weight updates without interface changes, major for I/O
  shape changes.
- **Licensing:** include training-data and model licence notices alongside the
  artefacts.

## 19. Open questions

- Optimal number of ordinal thresholds `K` and bin boundaries for alignment with
  the LAG depth scale.
- Final choice of multi-layer pooling vs single-layer for the backbone.
- Preferred fusion mechanism and feature set for the hybrid variant.

______________________________________________________________________

### Appendix A — Minimal ONNX graph summary (Transformer-ordinal)

- Nodes: encoder subgraph (attention, FFN), `Sigmoid` × K for ordinal heads,
  optional `Mul`/`Add` calibration, small `Reduce`/`Add` to compute expectation.
- Inputs: `input_ids: int64[batch, max_seq_len]`,
  `attention_mask: int64[batch, max_seq_len]`.
- Outputs: `logits_ord: float32[batch, K]` and/or `depth_scalar: float32[batch,
  1]` if expectation is embedded.

### Appendix B — Acceptance criteria

- Spearman ρ ≥ 0.60 on held-out mixed-domain set; MAE ≤ 1.0 steps (after scalar
  mapping); ECE within target for ordinal heads.
- p95 latency ≤ 10 ms CPU (single query), ≤ 2 ms for batch of 16; artefact size
  reduced by ≥ 50% with INT8; accuracy drop ≤ 1% absolute vs FP32.
- No change required to Sigma/Weights/Schedule interfaces; golden drift ≤ 2%
  across supported platforms for identical inputs.
