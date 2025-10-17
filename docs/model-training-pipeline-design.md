# Revised Design: Self-Hosted Prefect Orchestration for End-to-End Model Training Pipeline

## Executive Summary

This document describes a revised architectural design for the model training
pipeline, focusing on **local/self-hosted orchestration with Prefect** and
**cloud-agnostic ephemeral compute**. The end-to-end system automates
everything from data ingestion and model fine-tuning to ONNX export,
quantization, and deployment artifact assembly. We replace any dependency on
managed cloud workflow services with a **self-hosted Prefect Orion** engine,
which coordinates **Prefect flows and tasks** running on local infrastructure.
Model training and evaluation are executed on interruptible **spot instances**
or on-demand VMs across cloud providers, provisioned on-the-fly with
**Terraform (via OpenTofu)**. This approach retains aggressive cost
optimization and fault tolerance via robust checkpointing and resumable
training. The ultimate deliverable remains a **highly optimized, CPU-centric
ONNX model** (with INT8 quantization) packaged with its tokenizer and metadata
for seamless integration into the LAG Complexity Rust application.

Key improvements in this design include:

- **Local Prefect Orchestration:** We implement the pipeline as a code-first
  Prefect workflow, running on a self-hosted Prefect Orion server. This
  eliminates reliance on cloud-managed orchestrators and provides fine-grained
  control, easy monitoring, and integration with our codebase. All pipeline
  stages (data prep, training, export, etc.) are Prefect tasks within a single
  flow, ensuring end-to-end automation and observability.

- **Dynamic Ephemeral Compute via Terraform:** Each heavy compute stage
  (training, evaluation, etc.) is executed on dedicated ephemeral cloud
  instances. Prefect tasks use Terraform (OpenTofu distribution) to provision
  the required **spot or on-demand VM instances** (with GPU for training, CPU
  for export/evaluation) and tear them down afterward. This ensures
  **cloud-agnostic deployment** – by swapping Terraform modules, the pipeline
  can target AWS, GCP, or other providers uniformly. It also maximizes cost
  efficiency by using spot instances and releasing resources immediately when
  tasks complete.

- **Fault Tolerance and Checkpointing:** The pipeline is architected to handle
  preemptions or failures gracefully. During training, checkpoints are saved
  frequently to a centralized object store (e.g. S3). If a spot VM is
  terminated or any error occurs, the Prefect flow can automatically launch a
  new instance and **resume training from the last checkpoint**. This design
  ensures that using cheaper interruptible instances does not compromise
  reliability – a core requirement to reduce training costs by up to 90% using
  spot pricing.

- **Reproducibility and Auditability:** Every component of the pipeline is
  **version-controlled and logged** for audit. Pipeline configuration (data
  sources, model hyperparameters, instance types, etc.) is stored in code or
  config files under version control. Prefect flows record run metadata (e.g.
  flow run IDs, timestamps, parameters) in the Orion database, providing a
  detailed audit trail for each model build. All artifacts are persisted with
  **unique version identifiers** (such as an experiment ID or semantic version)
  and accompanied by
  checksums([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L238-L246))
   and metadata. This guarantees that any model artifact can be traced back to
  the exact code, data, and environment that produced it, ensuring strict
  reproducibility. The ONNX model files include a version tag in their metadata
  and are stored alongside their tokenizer and a `metadata.json` manifest
  containing training details and metrics. The Rust inference code verifies the
  artifact integrity via checksum before
  use([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L240-L247)),
   closing the loop on end-to-end reliability.

Overall, this revised design maintains the original pipeline’s modularity,
performance focus, and production readiness, while embedding a **flexible,
self-hosted orchestration layer** and robust infrastructure automation. In the
following sections, we detail the architecture and implementation plan,
including configuration practices, fault-tolerance strategies, and how this
pipeline interfaces with the existing LAG Complexity infrastructure.

## Section 1: Foundational Architecture and Infrastructure

The success of this machine learning pipeline depends on a robust and efficient
infrastructure foundation. We outline the core architectural principles and
infrastructure components that underpin the pipeline, ensuring it meets the
needs of **fault tolerance**, **cost efficiency**, and **ease of integration**
with LAG Complexity systems. Key decisions include the choice of **Prefect for
orchestration** and **Terraform-managed compute** resources, as well as
containerization and artifact storage strategies. These infrastructure choices
directly influence how we implement reproducibility, configuration, and
fault-handling in the pipeline.

### 1.1 Pipeline Orchestration and Design Philosophy

A production-grade pipeline must be more than a collection of scripts; it
should function as a **resilient, reproducible, and automated system**. We
adhere to several core principles in designing the workflow:

- **Modularity:** The pipeline is structured as a Directed Acyclic Graph (DAG)
  of distinct stages: **Data Ingestion**, **Model Training**, **ONNX Export**,
  **Verification**, **Quantization**, and **Deployment Packaging**. Each stage
  is a self-contained unit with clearly defined inputs and outputs,
  communicated via a centralized artifact store. This modular design allows
  individual stages to be re-run or modified independently without affecting
  the entire pipeline, simplifies debugging, and lets different team members
  work in parallel on separate parts of the workflow.

- **Reproducibility:** Every component of the pipeline – from OS libraries and
  Python packages to training hyperparameters – is explicitly pinned or
  versioned. We use **containerization** (Docker images) and
  infrastructure-as-code to ensure that a given pipeline run can be exactly
  replicated in the future. This is critical for auditing and for guaranteeing
  consistent model behavior between training and production. The pipeline will
  incorporate configuration files (or Prefect configuration blocks) for all
  adjustable parameters (dataset version, model architecture, learning rates,
  instance types, etc.), which are stored in version control. By fixing these
  inputs or recording them with each run, we make each training execution
  deterministic and traceable.

- **Automation:** The entire pipeline – from the initial trigger (which could
  be a code release, new dataset availability, or a manual kickoff) to the
  final publishing of model artifacts – is executed **without manual
  intervention**. Prefect flows handle task scheduling, retries, and
  notifications. This automation not only reduces human error but also enables
  rapid iteration and consistent deployments. For example, an engineer can
  trigger a new model training run with a single command or CI/CD event, and
  Prefect will orchestrate all steps to produce a ready-to-deploy model package.

To enforce reproducibility and isolate dependencies, each stage runs inside a
**Docker container** (Containerization Strategy). We maintain separate
container images for different stages: e.g., a GPU-enabled image with PyTorch
and CUDA for training, and a lightweight CPU image with `onnxruntime` and
`optimum` for export/quantization. Containerization ensures that the code
behaves identically whether run on a developer’s machine, a CI runner, or a
cloud VM, eliminating issues of environment drift.

#### Orchestration with Prefect Orion (Self-Hosted)

In this revised design, we adopt **Prefect Orion** as the workflow
orchestration engine, running in a self-hosted mode. Prefect is an open-source
orchestration framework that lets us define the pipeline in Python code and
manage it via a local Prefect server/UI. This choice aligns with our need for
**cloud-agnostic, in-house control** and leverages the pipeline’s
containerized, modular structure (which was originally designed with
compatibility for orchestrators in mind). Key aspects of our Prefect-based
orchestration include:

- **Flow & Task Definition:** Each pipeline stage is implemented as a Prefect
  **task** (a Python function or shell operation). These tasks are composed
  into a single **Prefect flow** that encapsulates the entire end-to-end
  process. For example, tasks might include: `fetch_data()`,
  `provision_training_vm()`, `run_training()`, `export_onnx()`,
  `quantize_model()`, `evaluate_model()`, and `package_artifacts()`. Prefect’s
  DAG scheduler ensures these run in the correct order with specified
  dependencies (e.g., training must finish successfully before export starts).

- **Local Orchestration Engine:** We deploy a Prefect Orion server (the
  orchestration backend) on our infrastructure. This could be a small VM or
  container that hosts Prefect’s API and UI. The **Prefect agent**, which
  actually executes tasks, can run on the same server or on dedicated machines.
  In our design, the agent primarily executes control tasks locally (like
  coordinating Terraform or data transfer), while heavy ML tasks run on remote
  compute – as described below. Using Prefect locally gives us a web UI to
  monitor flows, visualize the DAG, and inspect logs for each task, improving
  the **observability** of the pipeline for engineers.

- **No Cloud Dependencies:** Importantly, by self-hosting Prefect we avoid any
  vendor lock-in or external service dependency. The orchestrator connects to
  our own infrastructure (e.g., Terraform, S3, etc.) and does not require AWS
  Step Functions, Google Vertex Pipelines, or Prefect’s cloud service. This
  keeps the solution **agnostic to any cloud provider** and fully under our
  control, aligning with the requirement of no managed cloud workflow services.

- **Retry and Resumption Logic:** Prefect allows us to implement custom retry
  logic at the flow or task level. We leverage this to enhance fault tolerance.
  For instance, the training task will be configured to **retry on failure**,
  but instead of simply re-running from scratch, it calls a custom routine to
  resume from the latest checkpoint (more details in Section 2.3). Similarly,
  if a Terraform provisioning task fails (e.g., due to a spot instance not
  available), Prefect can retry with exponential backoff or fall back to an
  on-demand instance. This dynamic handling is coded into the flow, making the
  pipeline robust against various failure modes.

- **Configuration and Secrets:** Prefect flows can use configuration blocks or
  environment variables for sensitive info and settings. Cloud credentials (for
  Terraform or S3) are stored securely (e.g., in Prefect’s secret store or a
  vault accessible to the runner), and pipeline options (like the target cloud
  provider, region, instance type, dataset name, etc.) are fed as
  **parameters** to the flow. For example, an engineer can specify
  `provider="aws"`, `instance_class="g4dn.xlarge"` for a run, or switch to
  `provider="gcp"` with equivalent settings without code changes. All such
  config is centralized and logged. Best practices like not hard-coding
  credentials in the pipeline code are followed – Prefect’s configuration helps
  inject these at runtime.

By using Prefect in this manner, we ensure that the orchestration layer itself
is reproducible and under version control (the pipeline code is in our
repository), and we gain fine control over how tasks are executed and
recovered. The result is a **data-and-code-driven orchestration** that matches
the pipeline’s needs for flexibility and reliability.

### 1.2 Compute Provisioning and Cloud Infrastructure

Provisioning the right compute environment for each stage is critical for both
performance and cost management. The pipeline is designed to be
**cloud-agnostic**, using Terraform to manage infrastructure on any provider.
We perform a data-driven analysis to choose instance types for each stage and
automate their lifecycle:

- **Instance Selection for Training:** Training is the most compute-intensive
  stage. Based on our analysis of GPU options (AWS g4dn vs g6 vs p4d, etc.) and
  their cost/performance trade-offs, we target an **NVIDIA T4 or L4 GPU**
  instance for initial model fine-tuning. For example, AWS’s `g4dn.xlarge`
  (Tesla T4, ~0.526 USD/hour on-demand) provides a cost-effective baseline, and
  the newer `g6.xlarge` (L4 GPU) offers more performance at higher cost. The
  pipeline can be configured to use either, or even scale up to larger
  multi-GPU instances (like `p4d.24xlarge`) if a bigger model or faster
  training is needed. Similar options from other providers (GCP, Azure, or even
  on-prem) are also supported via Terraform modules. The **choice is
  abstracted** so that switching cloud or instance type is a config change
  rather than code change.

- **Cost Optimization with Spot Instances:** To minimize cost, the pipeline
  defaults to using **spot/preemptible instances** for training whenever
  possible. Spot instances can reduce costs by 70-90%, making the pipeline
  economically efficient. However, because they can be terminated at any time,
  our design doubles down on fault tolerance via checkpointing (see Section
  2.3). The Terraform provisioning task can request a spot VM (e.g., AWS EC2
  Spot or GCP Preemptible VM) for training; if the request is not fulfilled or
  the instance is reclaimed mid-training, the Prefect flow will handle
  launching a replacement and resuming training. The **cost modeling** takes
  into account the expected interruptions and restart overhead, ensuring that
  even with occasional restarts, using spot instances yields net savings. For
  cases where spot is not available or for final runs, the pipeline can easily
  switch to on-demand instances by configuration.

- **Instance Selection for Export & Evaluation:** After training, subsequent
  stages (ONNX export, quantization, evaluation) are **CPU-bound** and less
  intensive. It is wasteful to keep a GPU machine for these. Therefore, the
  pipeline will **deprovision the GPU VM after training** and spin up a smaller
  (and cheaper) CPU instance for the export and quantization stages. For
  example, a standard 4 or 8 vCPU VM (possibly a spot instance as well) is
  sufficient to run ONNX conversion and quantization. Terraform templates for
  these stages may use a different instance type (e.g., AWS m5.large or
  c6i.xlarge) optimized for cost. This strategy of tailoring the instance to
  the stage ensures we **pay only for what we need** at each step. It also
  improves reproducibility, as the ONNX export and INT8 quantization will run
  on a consistent CPU architecture environment for which we can precisely
  validate performance (important for numerical parity checks).

- **Terraform Integration:** We manage cloud resources with Terraform
  (OpenTofu), called programmatically from Prefect tasks. Each compute stage
  has corresponding Terraform configuration files (or modules) describing the
  required infrastructure (VM instance, networking, any needed storage or
  permissions). Prefect tasks use a Terraform CLI command (e.g.,
  `terraform apply` with appropriate variables) to provision
  resources([2](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/roadmap.md#L51-L58))([2](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/roadmap.md#L59-L67)).
   Once the task finishes, another Terraform call (`terraform destroy`) is used
  to tear down the resources, ensuring no idle costs. Terraform state is stored
  in a remote backend (or locally on the Prefect server) for consistency – this
  could be an S3 bucket or local file, as appropriate, given our
  no-managed-services constraint (an S3 bucket for state is acceptable since
  it’s just storage). Using Terraform provides an immutable, declarative
  description of our infrastructure, contributing to pipeline **auditability**.
  We can track infrastructure changes over time and be confident that each
  run’s environment matched what was intended (Terraform plan outputs are
  logged for inspection).

- **Network and Storage Configuration:** Each ephemeral VM is configured to
  access the central artifact store (e.g., S3 or an on-prem MinIO bucket) where
  data and model artifacts reside. Terraform attaches appropriate IAM roles or
  credentials so that the VM can read/write to the artifact store securely. No
  other long-lived infrastructure (like a permanent Kubernetes cluster or
  workflow engine) is required – the Prefect orchestrator and short-lived VMs
  are the only compute resources used. This minimalist approach reduces
  complexity and avoids maintaining additional services.

In summary, the infrastructure layer uses **Terraform-orchestrated, right-sized
VMs** to execute pipeline tasks on the optimal hardware, and Prefect to glue
these stages together. This yields a highly efficient setup: for example,
launching a spot `g4dn.xlarge` for training an ordinal regression model and
automatically terminating it upon completion, then spinning up a cheap CPU VM
for export and quantization. All these transitions are seamless to the user –
from their perspective, they run a Prefect flow and get a finished model, while
under the hood the orchestrator handled all provisioning. **Figure 1** below
illustrates the architecture:

([2](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/roadmap.md#L51-L58))

*(Figure 1: High-level pipeline orchestration. A Prefect flow running on a
self-hosted Orion server coordinates tasks: provisioning cloud VMs via
Terraform, running containerized training on a GPU spot instance (with periodic
checkpoints saved to S3), exporting and quantizing the model on a separate CPU
instance, then assembling the final artifacts in the artifact store. Each
stage’s resources are torn down after use. This design balances cost (using
spot instances and appropriate hardware per task) with reliability (via
checkpointing and automated retries).)*

### 1.3 Artifact Storage and Configuration Management

All intermediate and final artifacts are stored in a **central artifact
repository**, for example an S3 bucket (`lag-complexity-artifacts`) accessible
to all pipeline components. This storage acts as the single source of truth for
data and model artifacts, reinforcing reproducibility:

- **Data Artifacts:** Raw datasets, processed datasets, and calibration subsets
  are stored under versioned paths (e.g., `/datasets/raw/<name>/` and
  `/datasets/processed/<name>/v1/`). The pipeline’s data ingestion stage will
  retrieve data from these paths, ensuring that training is based on a fixed
  snapshot of data. If data preparation is handled by a separate process, it
  will drop the prepared data in this store, and our pipeline simply consumes
  it. Alternatively, Prefect can also orchestrate data processing as a
  preliminary flow/task if needed.

- **Model Artifacts and Checkpoints:** During training, checkpoints and final
  model weights are saved to S3 (see Section 2.3 for structure). After
  training, we have a fine-tuned PyTorch model saved under, e.g.,
  `/models/fine-tuned/<experiment_id>/final/`. The export stage will read from
  this location. ONNX models are exported to
  `/models/onnx/<experiment_id>/fp32/model.onnx`, and quantized models to
  `/models/onnx/<experiment_id>/int8/model_quantized.onnx`. By using the
  `experiment_id` (or a semantic version identifier) in the path, we isolate
  each run’s outputs. We also maintain a **checksum manifest** for each model
  (recording SHA-256 of the ONNX files and
  tokenizer)([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L238-L246)),
   stored alongside the model, to be used by the Rust service for verification
  at load time.

- **Configuration and Code Versioning:** The exact configuration of each run
  (git commit of code, Docker image tags, hyperparameters, etc.) is captured.
  For example, the flow can log the git commit hash of the training code and
  include it in the metadata file. The `metadata.json` in the deployment
  package (see Section 4.1) will contain fields for experiment ID, base model
  used, and possibly references to the code version or config used. This ties
  back into auditability – one can always reconstruct the training context from
  the artifacts alone.

- **Secure Access:** Access to the artifact store is controlled via credentials
  that the Prefect tasks and VMs have. The Prefect agent (running the
  orchestration) might use an AWS IAM role or keys (if on AWS) to manage S3.
  The ephemeral instances receive limited-permission credentials (by Terraform
  attaching an IAM role or by injecting short-lived tokens) that allow them to
  pull only the needed dataset and push back checkpoints and models. This
  principle of least privilege secures our pipeline against accidental or
  malicious misuse.

With the foundational architecture set, we now turn to the detailed
implementation of each pipeline stage, explaining how the Prefect-orchestrated
tasks perform data ingestion, training with checkpointing, model export, and
artifact packaging.

## Section 2: The Fine-Tuning Pipeline Implementation (Training Phase)

This section provides a step-by-step walkthrough of the pipeline’s core model
training stages, highlighting how Prefect orchestration and our fault-tolerant
design come into play. The model in question is an ordinal regression
classifier (as per the LAG Complexity project’s depth classifier requirements),
fine-tuned from a Transformer. We describe each stage from initializing the
environment on the ephemeral VM, through the custom training loop with
checkpointing, to saving the final model. The focus is on how these steps are
wrapped in Prefect tasks and made resilient to failures.

### 2.1 Stage 1: Environment Setup and Data Ingestion

Each training job runs in an isolated environment on a **Terraform-provisioned
VM** (with a GPU for model training). When the Prefect flow reaches the
training stage, it executes the following steps:

**Provisioning and Bootstrapping**: The **`provision_training_vm`** task in the
Prefect flow applies our Terraform template for a GPU instance (as decided in
Section 1.2). Once the instance is up, the flow proceeds to an
**`execute_training`** task. We have two main strategies for running the
training code on the remote VM:

- **Container Startup via User Data:** The VM’s user-data script (or Terraform
  remote-exec provisioner) automatically pulls the appropriate Docker image and
  launches the training script inside the container. For example, on AWS we
  attach a user-data that does
  <!-- markdownlint-disable-next-line MD013 -->
  `docker run -e PREFECT_RUN_ID=... -v /tmp/outputs:/outputs myregistry/lag-trainer:latest python train.py --experiment <ID> ...`.
   The Prefect flow will wait until the training container signals completion
  (this could be done by the training script calling back to Prefect or simply
  by monitoring cloud instance status/logs).

- **SSH and Prefect Remote Calls:** Alternatively, the Prefect task can SSH
  into the instance and invoke the training script (either directly on the host
  or via Docker). Prefect’s ability to run tasks on remote infrastructure can
  be extended by having a lightweight Prefect agent on the VM, but for
  simplicity, SSH + Docker CLI is sufficient. The flow would look like:
  provision VM → (over SSH) start container → monitor logs.

Regardless of approach, **the training job begins as a self-contained Docker
container on the new VM**. The container image includes all necessary
dependencies (Ubuntu base with CUDA drivers, PyTorch, HuggingFace Transformers,
our training code, etc.). When the training script starts, it performs
environment initialization and data ingestion:

- It loads configuration (passed via command-line args or env vars), including
  the model checkpoint to start from (if any), dataset name, and
  hyperparameters.

- It establishes a secure connection to the artifact store (e.g., using AWS SDK
  with credentials available on the VM).

- **Data Download:** The script downloads the required data artifacts from the
  centralized store. This includes the processed training dataset and
  validation dataset (e.g., under `/datasets/processed/<dataset_name>/vX/`) and
  possibly a pre-trained base model checkpoint if we start from a published
  base model. By pulling from versioned artifact storage, we ensure the job
  isn’t using any stale or local data – it’s using exactly the data that’s been
  prepared and approved for this training run. This makes the run deterministic
  and reproducible. The downloaded data is stored locally on the VM (inside the
  container’s filesystem) for fast access during training.

- **Model & Tokenizer Loading:** With data in place, the script loads the model
  architecture. We fine-tune a pre-trained transformer (e.g., DistilBERT) for
  ordinal regression. Using Hugging Face’s APIs, the code either:

- Loads a base model checkpoint (if starting from a pre-trained weight, e.g.,
  `distilbert-base-uncased`) and then adapts it for our task.

- Or if resuming a partially trained model (in case of a retry after a
  failure), it will load the last checkpoint’s model state (more on this in
  Stage 3).

The **tokenizer** is loaded similarly, either from the Hugging Face Hub or from
the saved files corresponding to the base model. We ensure the **tokenizer
files are pinned** (the vocab and merges for BPE, etc.) so that the same
tokenizer is used during training and later in production. In fact, these files
will later be included in the deployment package. At this point, the training
container’s environment is fully set up with code, data, model, and tokenizer
ready.

### 2.2 Stage 2: Model Fine-Tuning for Ordinal Regression

The core training logic fine-tunes the model to predict an ordinal output (for
example, the “depth” of a question’s reasoning chain). Standard classification
approaches are insufficient for ordinal data, so we implement a custom solution
as described in the project’s design:

- **Model Architecture Adaptation:** Instead of the typical multi-class
  classification head (which would output logits for each class and use
  softmax), we use a single scalar regression head plus a set of learned
  cutpoint thresholds for ordinal categories. In practice, we load the
  pre-trained transformer (e.g., DistilBERT) up to the final hidden layer, then
  replace its classifier head with a new head: a single linear layer that
  outputs one raw score *f(X)* per input. This score will later be interpreted
  against learned thresholds to produce class probabilities.

- **Custom Loss Function (Cumulative Link Loss):** We override the training
  loop’s loss computation to implement an ordinal regression loss.
  Specifically, we introduce K-1 trainable cutpoints (for K ordinal classes)
  and use the **cumulative link** (logistic) function to map the model’s output
  f(X) and these cutpoints to class probabilities. The loss is the negative
  log-likelihood of the true class under this probability distribution. We
  realize this by creating a custom `Trainer` subclass (e.g.,
  `OrdinalRegressionTrainer`) that overrides `compute_loss`. This subclass
  cleanly encapsulates the ordinal-specific logic without needing to rewrite
  the entire training loop. It ensures that mis-predictions are penalized in
  proportion to their distance from the true class (e.g., predicting class 0
  when true class is 4 incurs more loss than predicting class 3 when true is 4).

- **Training Loop and HF Trainer Integration:** We leverage Hugging Face’s
  `Trainer` API for the main loop, which handles batching, optimizer steps, and
  scheduling. Our custom trainer is configured with appropriate
  `TrainingArguments`. Notably, we enable frequent evaluation and checkpoint
  saving: `evaluation_strategy="steps"` and `save_strategy="steps"` with a
  relatively frequent `save_steps` (for example, every 100 steps). We also set
  `save_total_limit` to keep only the last few checkpoints locally (to save
  disk), but since we upload them to S3, nothing is truly lost. These settings
  are crucial for fault tolerance on spot instances – by saving state every N
  steps, we limit how far back we would need to roll in case of interruption.

- **Real-time Metrics and Logging:** The training script records key metrics
  (loss, accuracy per class, etc.) on each evaluation cycle (or per epoch if
  using epoch-based eval). These metrics are both logged to stdout (and thus
  visible in Prefect’s task logs if streamed) and saved to a file (or directly
  sent to the Prefect Orion server as a heartbeat). This visibility allows the
  team to monitor training progress in the Prefect UI live. Moreover, after
  training, these metrics (especially on a validation set) can be saved as part
  of the `metadata.json` for the model artifact, giving a summary of model
  performance for later review.

The Prefect flow doesn’t interfere during the training epoch loop – it simply
monitors the remote training task’s status. If training completes successfully,
the next stage will begin. If training fails or is halted (e.g., VM
interruption), Prefect will catch that and trigger the fault recovery process
as described next.

### 2.3 Stage 3: Checkpointing and Fault-Tolerant Training

Fault tolerance via checkpointing is a **cornerstone of this pipeline’s
design**, enabling us to use transient cheap instances without risking training
progress. Here’s how it works:

**Frequent Checkpointing to Object Storage:** We configure the Hugging Face
`Trainer` to save checkpoints every few hundred steps. Each checkpoint is a
folder containing the model’s weights, optimizer state, scheduler state, etc.
We implement a custom `TrainerCallback` (integrated in the training script) to
**sync checkpoints to S3** (artifact store) whenever one is saved. For example,
after each `save_steps` interval, the callback triggers and uses `boto3` (or
the cloud SDK) to upload the checkpoint directory to a path like
`s3://lag-complexity-artifacts/models/checkpoints/<experiment_id>/step-XXXX/`.
This happens asynchronously in the background thread of the training process to
minimize delay. Only the latest few checkpoints might be kept to avoid spamming
storage, but the final checkpoint will definitely be there. By writing to
durable storage off the VM, we ensure that even if the VM disappears, our
progress up to the last checkpoint is safe.

**Interruption Handling and Resume:** Prefect monitors the training task. If
the VM is terminated (spot interruption) or the training script crashes due to
some non-fatal error, the Prefect task for training will fail. On failure, our
flow’s logic kicks in:

- The **`execute_training`** task is set to retry (with a limit, e.g., 3
  retries). Before retrying, we run a recovery sub-routine: the Prefect flow
  queries the artifact store to find the latest checkpoint for this experiment
  (by listing `.../checkpoints/<experiment_id>/` and finding the highest step
  or latest timestamp).

- The flow then **provisions a new VM** (another spot instance or on-demand if
  urgent). On this new VM, it restarts the training container but this time
  passes an argument to resume from the checkpoint path. Our training script
  supports a `--resume_from <checkpoint_path>` parameter.

- At startup, if `resume_from` is provided, the script downloads that
  checkpoint from S3 onto the VM, and then calls
  `trainer.train(resume_from_checkpoint=local_checkpoint_dir)`. Hugging Face’s
  Trainer will then load the model weights and optimizer states from that
  checkpoint and continue training as if nothing happened. Thanks to this
  mechanism, an interruption results in at most the loss of a few hundred
  training steps (since last checkpoint).

- The pipeline thus transforms a potentially catastrophic spot interruption
  into a minor hiccup – training simply continues after a short delay to
  reprovision.

The **fault-tolerance strategy** is configured and tested to ensure that if,
for instance, a spot instance averages 6 hours before interruption and training
takes 12 hours, the job will likely resume once or twice and finish
successfully within, say, 13 hours total, while costing a fraction of an
on-demand instance.

**Final Model Save:** Once training finishes (either in the first attempt or
after resumes), the script performs a final save of the model using
`trainer.save_model()`. This produces the complete fine-tuned model files
(PyTorch `pytorch_model.bin` or similar, config.json, etc.) in the container.
We upload this final model artifact to a designated S3 path: e.g.,
`s3://.../models/fine-tuned/<experiment_id>/final/`. This final artifact is
what the next stages (export & quantization) will consume. We also tag this
with a version number or commit hash if needed, and ensure the **checkpoint
callback** has flushed any last checkpoint (though final model is usually more
important than intermediate checkpoints now).

At this point, the training VM’s job is done. The Prefect flow will signal
Terraform to destroy the GPU VM to save cost. We have a fine-tuned model in the
artifact store and are ready to transition to the model optimization phase.

## Section 3: Model Export, Verification, and Optimization for Inference

After obtaining the trained PyTorch model, the pipeline transitions to
preparing it for efficient production use. This involves converting the model
to ONNX format, verifying the conversion’s correctness, and applying INT8
quantization for performance. We orchestrate these as separate stages, each
running in a controlled environment (here, a CPU instance as discussed).
Prefect ensures these steps happen sequentially and only proceed if the
previous step succeeds (especially the verification step acts as a gate – any
parity mismatch stops the pipeline).

### 3.1 Export to ONNX with Hugging Face Optimum

We use the **Hugging Face Optimum** library to export the PyTorch model to
ONNX. The rationale is that Optimum provides a high-level, validated pathway
for conversion, including graph optimizations, which is superior to manual
`torch.onnx.export` approaches.

**Execution Environment:** The Prefect flow now runs an **`provision_cpu_vm`**
task to launch a CPU-only VM (likely with Docker installed). On this VM, we run
a container (from a CPU-based image that has `optimum` and `onnxruntime`
installed). This container will perform the export. By isolating this in a
fresh environment, we ensure that the ONNX export uses a clean state and the
correct versions of ONNX and Optimum, and we avoid any chance that GPU-specific
dependencies interfere (since ONNX export doesn’t need a GPU).

**Export Process:** The `export_model` task (running in the container on the
CPU VM) will do roughly:

```bash
optimum-cli export onnx \
    --model /tmp/model/final/ \   # path where we downloaded the fine-tuned PT model
    --task text-classification \
    --opset 17 \
    --optimized_model_dir /tmp/model/onnx/fp32/
```

We first download the final fine-tuned model from S3 to the local filesystem
(e.g., under `/tmp/model/final/`). We then invoke the optimum CLI as above. Key
points:

- We specify `--task text-classification` because our ordinal regression model
  is essentially a classification model in structure (one hidden state to
  scalar output). This ensures Optimum knows how to wrap the model’s forward
  pass appropriately.

- We choose `opset_version=17` for ONNX, which is aligned with our inference
  environment (ONNX Runtime 1.22 as per the
  ADR([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L170-L178))).
   Opset 17 includes support for all needed ops (like LayerNormalization) and
  ensures portability across platforms.

- Optimum can also apply graph optimizations. We may use an `--optimize O3`
  level, which fuses operations and etc., as suggested by HF documentation
  (this corresponds to e.g. constant folding, layer norm fusion, etc., without
  requiring hardware-specific optimizations).

- The output is a directory (we named it `onnx/fp32`) containing `model.onnx`
  and some Optimum metadata (and possibly an `optimum_config.json` capturing
  how the export was done for reproducibility).

Once the export command runs, we have an ONNX model (floating-point precision)
stored locally. We upload this to the artifact store:
`s3://.../models/onnx/<experiment_id>/fp32/model.onnx`. Alongside, we save the
Optimum export config if available. This completes the conversion, but **we do
not yet tear down the CPU VM**, because we will continue to use it for
verification and quantization.

### 3.2 Post-Export Verification (Parity Check between PyTorch and ONNX)

Before moving on, we must verify that the ONNX model behaves identically to the
original PyTorch model. This stage is implemented to ensure **numerical
parity** – a critical quality gate to catch any conversion issues. The Prefect
flow runs a task `verify_onnx_parity` on the same CPU VM (in the same container
or a new one with necessary libraries):

**Procedure:**

- **Load Models:** Load the fine-tuned PyTorch model in memory (we can either
  reload it from the saved weights or still have it from before export) and
  load the ONNX model using ONNX Runtime (`onnxruntime.InferenceSession`).

- **Prepare Test Data:** We use a sample of validation data (or training data)
  for testing. The pipeline can retrieve a small batch (say 100 examples) from
  the dataset for this purpose. Ideally, the **calibration dataset** for
  quantization can double as this test set, or we take random samples from
  validation. These samples are preprocessed (tokenized) exactly as in training.

- **Run Inference on Both:** For each sample input, run a forward pass through
  the PyTorch model (in PyTorch) and through the ONNX model (using
  `session.run`). Collect the outputs.

- **Compare Outputs:** We compute the difference between outputs. Since
  floating point arithmetic and ONNX transformations might introduce tiny
  differences, we do not expect bit-for-bit identical outputs. Instead, we use
  a tolerance-based comparison like
  `numpy.allclose(pytorch_output, onnx_output, atol=1e-5)`. We also ensure the
  shapes and general structure of outputs are as expected. Given our model
  outputs a single scalar (plus maybe the cutpoint logits for internal use),
  the comparison is straightforward.

- **Pass/Fail Gate:** If **any** sample’s output differs beyond the tolerance
  threshold, we consider the verification failed. This triggers the pipeline to
  halt. Prefect will mark the flow run as failed and notify us. Such a failure
  indicates something went wrong in the export (e.g., an unsupported operation
  or numerical instability). Engineers would need to investigate, possibly
  adjust the export parameters or fix an issue in the model code, and re-run
  the pipeline. This gate ensures we do not deploy a model that doesn’t match
  the training outcomes.

If the outputs match within tolerance for all test inputs, we log a success.
The Prefect UI can record this as a checkpoint that the parity test passed. We
may also save a summary of this test (like max difference observed) into the
model’s metadata.

With a verified ONNX FP32 model, we proceed to optimize it.

### 3.3 Performance Optimization via INT8 Quantization

The final optimization step is to quantize the ONNX model from FP32 to INT8, to
achieve lower latency and smaller model size for CPU inference. We employ
**post-training static quantization** using ONNX Runtime’s quantization tools,
as decided in the
ADR([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L66-L71))([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L156-L164)).

**Calibration Data:** Static quantization requires a small calibration dataset
to compute activation scaling factors. During Stage 1 or earlier, we ensure
such a dataset is available. Often, this is a few hundred representative
examples from the training set. Our pipeline either expects this to be present
in the artifact store (e.g., under `/datasets/processed/<name>/calibration/`),
or if not, we can have a step that samples from training data to create it. In
our design, we assume the calibration data has been prepared offline or in an
earlier flow and uploaded.

**Quantization Process:** The `quantize_model` Prefect task (still on the CPU
VM container) will:

- Load the FP32 ONNX model (or have the path to it).

- Use `onnxruntime.quantization.quantize_static()` function to quantize. We
  supply:

- The path to the FP32 model as input.

- An output path for the INT8 model (e.g.,
  `/tmp/model/onnx/int8/model_quantized.onnx`).

- A data reader that iterates over the calibration dataset and feeds inputs to
  the model. We will implement this reader in Python: it will load each sample,
  tokenize it, create the input tensors (input IDs, attention mask arrays) as
  numpy arrays matching the model’s expected input format, and yield them.

- Quantization parameters: we choose **QuantFormat** = QDQ (Quantize/Dequantize
  nodes), which is recommended for compatibility and performance;
  **activation_type** = **weight_type** = `QuantType.QInt8` for symmetric 8-bit
  quantization of both weights and activations. These settings align with best
  practices for int8 quantization on CPUs (and match what our ADR suggested).

- The quantization routine will run inference on the calibration data
  internally to gather stats, then produce a quantized model.

After this, we perform another **verification** similar to Stage 3.2: compare
the outputs of the INT8 model to the FP32 ONNX model on a small test set. We
expect slightly larger differences (so we might use a tolerance like `1e-2`),
but the accuracy drop should be within acceptable limits (our target was ≤1%
degradation([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L160-L167))).
 If the INT8 model shows unacceptable deviation, the pipeline could decide to
fail or warn. Assuming all is well, we proceed to save the quantized model.

**Publishing the Quantized Model:** We upload `model_quantized.onnx` to the
artifact store under `/models/onnx/<experiment_id>/int8/model_quantized.onnx`.
We also collect all necessary pieces for deployment:

- The tokenizer files (vocab, merges, tokenizer.json) – these might already be
  stored separately (perhaps under base model data), but to be safe we copy
  them to, say, `/models/onnx/<experiment_id>/tokenizer/` in the artifact store
  so the Rust service can easily fetch the exact tokenizer used.

- The `metadata.json` – we create this deployment metadata file containing:

- Model identifier (experiment ID, version number).

- Base model name or version.

- Training details: number of epochs, date, any hyperparameters of note.

- Cutpoint values learned (if our ordinal model’s cutpoints need to be known at
  inference, we include them here).

- A mapping of class indices to labels (if applicable).

- Evaluation metrics: if we evaluated on a test set or have validation metrics
  (accuracy, MAE, correlation), include them.

- Checksums of the ONNX files (for integrity verification in production).

- Possibly the commit hash of the training code or image version, for
  traceability.

- A checksum manifest (if not included in metadata) listing the SHA-256 of
  `model_quantized.onnx` (and maybe of the FP32 model for reference).

All these files are stored in the artifact repository as the **deployment
package** for this model version.

Finally, the Prefect flow signals Terraform to destroy the CPU VM as well,
since all artifacts are now safely stored.

To summarize Stage 3, we have taken the fine-tuned model, converted it to ONNX
(for portable, runtime-efficient format), verified its correctness, and applied
quantization to meet performance targets (e.g., p95 latency ≤ 10ms on CPU as
required([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L26-L34))([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L38-L46))).
 The result is a production-ready model artifact set.

## Section 4: Integration with LAG Complexity Infrastructure (Deployment Phase)

The last phase is how the outputs of this pipeline integrate into the existing
LAG Complexity system – primarily the Rust inference component that will use
the ONNX model. We describe how the deployment package is consumed and the
interfaces ensuring a smooth deployment.

### 4.1 Deployment Package and Configuration

For each successful pipeline run, a **versioned deployment package** is
produced and stored (as detailed above). Integration with LAG Complexity
involves retrieving this package and configuring the Rust service to use it:

- We follow a convention that each model has a **semantic version** or unique
  ID. For example, depth classifier v1.0.0 might correspond to
  `experiment_id = depth-ordinal-1.0.0`. This version can be encoded in the
  artifact path or in the ONNX model’s
  metadata([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L242-L249)).
   The **Rust application** (or its config) will specify which version of the
  model to use (likely the latest stable).

- During deployment of the Rust service, a CI/CD step will fetch the needed
  artifacts from S3. This can be automated: e.g., a build script uses AWS CLI
  to download `model_quantized.onnx`, `tokenizer.json`, etc., from the known S3
  path (the path can be constructed from the version, or a manifest file can
  list the latest version).

- All artifacts are placed in the container or file system where the Rust
  service can access them. The pipeline ensures backward compatibility – the
  ONNX model adheres to the interface expected by the Rust code (same
  input/output tensor schema) and includes any metadata needed.

We also maintain **strict auditability** here: the Rust service is configured
to verify the SHA-256 checksum of the ONNX model at startup against the
checksum recorded by the
pipeline([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L240-L247)).
 Our pipeline provided that checksum (in the metadata). If they don't match,
the service will refuse to load the model, preventing any tampering or mismatch
between code and model. This was an explicit design in the ADR and we uphold it
in our artifact generation.

### 4.2 Rust Inference Integration (Using the `ort` Crate)

The LAG Complexity Rust component uses the `ort` crate (ONNX Runtime for Rust)
to load and run the model. Our pipeline’s output is tailored for easy use with
`ort`:

- **ORT Dynamic Loading:** The Rust service, as per best practices, will use
  `ORT_DYLIB_PATH` to dynamically load the ONNX Runtime library. We ensure the
  deployed environment has ONNX Runtime 1.22 (matching our model’s opset 17)
  installed, to guarantee
  compatibility([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L170-L178)).

- **Model Loading:** The Rust code will load `model_quantized.onnx` at startup
  via `Session::new()` (or similar). Thanks to quantization, this model is
  optimized for CPU and should meet latency requirements (we targeted <10ms per
  query([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L28-L35))([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L158-L165))).

- **Tokenizer and Preprocessing:** The pipeline provided the exact tokenizer
  files. The Rust service uses these (with a Rust tokenizer library or by
  calling the same tokenization as in Python, possibly via `tokenizers` crate)
  to preprocess incoming text into `input_ids` and `attention_mask` exactly as
  expected by the model. The tokenizer config (vocab size, special tokens,
  etc.) is exactly the same as used in training, ensuring no drift.

- **Inference and Postprocessing:** When a query comes in:

- Rust code tokenizes the text.

- Feeds the token IDs and attention mask to the ONNX Runtime session,
  performing inference.

- The output is a scalar (the f(X) value). The Rust code then applies the
  *ordinal logic* to this scalar to produce probabilities for each class and
  the predicted class. For this, it uses the **cutpoint values** that were
  saved in `metadata.json` by the pipeline. We ensure the metadata includes
  these thresholds.

- The Rust code computes, for example, `P(y > k)` for each cutpoint k via the
  sigmoid function and derives class probabilities (mirroring the formula from
  training). This mirrors exactly the Python’s logic (essentially Section 2.2’s
  equations) to guarantee consistent interpretation.

- The final classification result (e.g., a depth score or category) is then
  used by the LAG system’s logic (like deciding whether to use the complex
  reasoning path or not).

- **Performance and Threading:** The `ort` session is created once and reused.
  Because the model is quantized and small, the Rust service can handle a high
  throughput of requests on commodity CPU. We also included in the pipeline’s
  ADR that multiple threads or batch inference can be used; the model supports
  dynamic batch and sequence lengths, so the Rust code could batch multiple
  queries for efficiency if needed.

By providing a well-documented deployment package and ensuring the Rust code
has matching logic and config, integration is straightforward. The pipeline’s
output is essentially plug-and-play with the Rust `ort` consumer.

## Conclusion

This comprehensive design achieves an **automated, reproducible, and resilient
model training pipeline** tailored for the LAG Complexity project’s needs. By
leveraging **self-hosted Prefect orchestration**, we orchestrate complex
workflows entirely under our control, removing external dependencies while
gaining fine control over execution and failure recovery. The use of
**Terraform-managed ephemeral instances** allows us to optimize hardware usage
and costs, scaling resources up or down for each stage and utilizing spot
instances with confidence thanks to robust checkpointing. All stages of the ML
lifecycle – data ingestion, training, evaluation, conversion, and deployment
prep – are covered, with modular tasks that can be maintained or improved
independently.

Key highlights of this design include:

- **Fault-Tolerant, Cost-Efficient Training:** The pipeline embraces the use of
  interruptible spot instances to save costs (up to 90% savings) and
  counterbalances their unreliability with frequent automated checkpointing and
  seamless resumption. This design turns infrastructure cost considerations
  into first-class design parameters, achieving economical training without
  sacrificing progress or model quality.

- **Prefect-Orchestrated Modularity:** By structuring the pipeline as a series
  of containerized stages (DAG of tasks) and using Prefect flows, we obtain a
  clear **separation of concerns** and ease of monitoring. Each stage (data
  prep, train, export, etc.) can be retried or modified in isolation.
  Automation covers the entire path from raw data to deployable model, ensuring
  that no manual steps can introduce variability or error. The Prefect UI and
  logs provide transparency into each run for engineers, and failures trigger
  well-defined recovery logic rather than ad-hoc handling.

- **Reproducibility & Auditability:** Every run is deterministic given the same
  inputs, and every output artifact is versioned and checksummed. From
  container images that freeze the software stack to configuration files that
  capture all parameters, we can recreate any model build. Additionally, the
  pipeline records metadata (like model metrics, version, cutpoints, etc.) that
  not only aids deployment but also enables auditing model improvements over
  time. This addresses compliance and traceability requirements in production
  ML workflows.

- **Seamless Deployment Integration:** The final ONNX model and accompanying
  artifacts are packaged in alignment with the LAG Complexity inference
  infrastructure. The Rust service can readily load the model and knows how to
  trust its integrity and interpret its outputs, because the pipeline ensured
  consistency in tokenization and ordinal logic between training and inference.
  This minimizes the friction when promoting a new model to production – it is
  a drop-in replacement for the previous version, with only a version number
  change in config.

Implementing this design will equip the engineering team with a robust MLOps
pipeline that can be run on-demand or on schedule to refresh the LAG Complexity
models (e.g., retraining the depth or ambiguity classifiers when new data is
available or improvements are made). The use of open-source orchestration and
infrastructure-as-code fits our organizational preference for transparency and
control. By following the guidelines and architecture outlined in this
document, the team can confidently build and deploy the next generation of LAG
Complexity models with **full end-to-end automation, minimal cost, and maximum
reliability**.

**Sources:**

- LAG Complexity Roadmap and ADRs – detailing model export, quantization, and
  verification
  requirements([2](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/roadmap.md#L51-L58))([1](https://github.com/leynos/lag-complexity/blob/56bc0cd28b3f7fa31591063f23b8298151917a52/docs/adr-depth-classifier-onnx.md#L238-L246)).

- Prior Design Proposal – provided the initial pipeline structure, emphasizing
  modularity, checkpointing, and integration with Rust.

- Prefect & Terraform Documentation – informing the use of Prefect Orion for
  local orchestration and Terraform (OpenTofu) for cloud-agnostic provisioning
  (no direct excerpt, applied as per design).
