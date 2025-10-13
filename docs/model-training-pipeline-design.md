# Architecting a Production-Grade Fine-Tuning Pipeline: From Cloud-Based Training to Optimized ONNX Deployment in Rust

## Executive Summary

This document presents a comprehensive architectural design for a
production-grade, end-to-end machine learning pipeline. The system is
engineered to facilitate the fine-tuning of pre-trained transformer models for
a specialized ordinal regression task, leveraging a Python-based training
environment on cloud infrastructure. The ultimate deliverable is a highly
optimized, CPU-centric ONNX (Open Neural Network eXchange) model, designed for
seamless integration and high-performance inference within a downstream Rust
application.

The proposed architecture is founded on principles of modularity,
reproducibility, and aggressive cost optimization. Each stage of the
pipeline—from data ingestion and model training to ONNX export and
quantization—is designed as a distinct, containerized, and automated step. This
approach ensures environmental consistency and facilitates integration with
modern MLOps orchestration frameworks.

A central theme of this design is the strategic selection and utilization of
cloud resources to maximize economic efficiency without compromising
performance. The analysis mandates the use of interruptible compute resources,
such as AWS Spot Instances, which necessitates a core architectural focus on
fault tolerance through robust checkpointing and automated resumption. The
design details a data-driven methodology for selecting compute instances,
balancing hourly cost against the total time and cost required for model
convergence.

Technically, the pipeline addresses the nuanced challenge of ordinal regression
by moving beyond standard classification frameworks. It specifies a custom
model architecture and a bespoke training loop, implemented by subclassing the
Hugging Face `Trainer` class to incorporate a cumulative link loss function.
This ensures the model correctly learns the inherent order of the target labels.

The final stages of the pipeline focus on preparing the model for a
high-performance, CPU-bound production environment. This involves a rigorous,
multi-step process: exporting the fine-tuned PyTorch model to the ONNX format
using the Hugging Face `optimum` library, performing mandatory numerical parity
checks to guarantee model integrity, and applying static INT8 quantization to
dramatically reduce model size and accelerate inference speed.

Finally, the report provides a clear integration path for the consuming Rust
application. It outlines best practices for using the `ort` crate, including a
critical strategy for managing shared library dependencies, and provides a code
pattern for loading the quantized ONNX model, processing inputs, and correctly
interpreting the model’s output to perform ordinal classification. The result
is a complete, actionable blueprint for building a sophisticated,
cost-effective, and robust fine-tuning and deployment system.

## Section 1: Foundational Architecture and Cloud Infrastructure Selection

The success of any machine learning pipeline is determined not only by the
quality of its models but also by the robustness, efficiency, and economic
viability of its underlying infrastructure. This section establishes the
foundational architectural principles and presents a data-driven analysis for
selecting the optimal cloud compute and storage environment. The design choices
articulated here are interconnected, with infrastructure decisions directly
influencing code-level implementation requirements for fault tolerance and cost
management.

### 1.1 Pipeline Orchestration and Design Philosophy

A production-grade pipeline must be more than a sequence of scripts; it must be
a resilient, reproducible, and automated system. The following principles form
the bedrock of this design.

#### Core Principles: Modularity, Reproducibility, and Automation

The pipeline is architected as a Directed Acyclic Graph (DAG) of distinct,
modular stages: Data Ingestion, Model Training, ONNX Export, Parity
Verification, and Quantization. Each stage is a self-contained unit with
clearly defined inputs and outputs, which are exclusively passed via a
centralized artifact store. This modularity provides several key advantages: it
allows for individual stages to be re-run without executing the entire
pipeline, simplifies debugging, and enables parallel development by different
team members on different parts of the workflow.

Reproducibility is paramount. Every component, from system-level dependencies
to Python package versions, must be explicitly version-controlled to guarantee
that a given pipeline run can be perfectly replicated at any point in the
future. This is essential for auditing, debugging production issues, and
ensuring consistent model behavior.

Automation is the mechanism that binds these principles together. The entire
pipeline, from triggering a new training run to the final publication of a
deployable model artifact, should be executable without manual intervention.

#### Containerization Strategy

To enforce the principle of reproducibility and eliminate environmental drift,
every stage of the pipeline will be executed within a Docker container.
Containerization encapsulates the entire runtime environment—including the
operating system, system libraries (such as CUDA and cuDNN), Python
interpreter, and all package dependencies—into a single, immutable artifact.
This approach provides a hermetic seal around each pipeline step, ensuring that
the code executes identically on a developer’s local machine, a CI/CD runner,
and the production cloud environment. Separate Dockerfiles will be maintained
for the GPU-intensive training stage and the CPU-bound export/quantization
stages to create optimized, lean images for each task.

#### Orchestration Framework Compatibility

While this document does not mandate a specific workflow orchestration tool,
the modular, container-based design is intentionally architected for
compatibility with industry-standard MLOps orchestrators such as Kubeflow
Pipelines, Argo Workflows, or cloud-native solutions like AWS Step Functions.
These platforms can consume the containerized stages as individual tasks,
manage dependencies between them, handle retries, and provide a centralized
interface for monitoring and management. The adoption of this container-centric
design is the critical prerequisite that enables future integration with such a
sophisticated orchestration layer.

### 1.2 Compute Environment Analysis and Cost Modeling

The most significant operational expenditure for a model fine-tuning pipeline
is typically compute resources. Therefore, a rigorous, data-driven approach to
selecting the compute environment is essential for building a sustainable and
economically viable system. The analysis must extend beyond a simple comparison
of hourly instance prices to consider the total cost required to achieve a
converged model. AWS publishes canonical on-demand pricing tables that form the
baseline for this analysis and should be reviewed periodically for
updates.[^12]

#### The Cost-Performance Trade-off in GPU Selection

The cloud market offers a wide array of GPU-accelerated instances, each
presenting a different balance of performance and cost. For the fine-tuning
task, several instance families across major cloud providers are viable
candidates.

- **Amazon Web Services (AWS):**
- `g4dn.xlarge`: Featuring an NVIDIA T4 GPU, 16 GiB of GPU memory, and an
  on-demand price of approximately $0.526 per hour, this instance represents a
  strong baseline for cost-effective experimentation and training of moderately
  sized models.[^1][^2]
- `g6.xlarge`: A newer generation instance with an NVIDIA L4 GPU and 24 GiB of
  GPU memory, offering superior performance for a higher on-demand price of
  around $0.805 per hour.[^3][^4]
- `p4d.24xlarge`: A high-performance powerhouse equipped with eight NVIDIA A100
  GPUs (40 GiB each), this instance is designed for large-scale training. Its
  on-demand price of approximately $21.96 per hour is substantial, but its
  massive parallelism can drastically reduce training time.[^5]
- **Alternative Cloud Providers:**
- **DigitalOcean:** Presents a competitive offering with on-demand GPU
  Droplets, such as an NVIDIA L40S instance for $1.57 per hour or an NVIDIA
  H100 instance for $3.39 per hour.[^7]
- **Scaleway:** A European provider that offers compelling pricing, including
  an L40S instance at approximately €1.61 per hour.[^8] Providers like Scaleway
  and Ubicloud are particularly attractive for their generous data egress
  policies, which can significantly reduce costs compared to AWS.[^9]

A naive comparison of these hourly rates is insufficient. The optimal choice is
not the instance with the lowest hourly cost, but the one that minimizes the
_total cost to convergence_. A `p4d.24xlarge` instance, while over 40 times
more expensive per hour than a `g4dn.xlarge`, may complete a training job more
than 40 times faster, resulting in a lower total bill. The pipeline design must
therefore accommodate a flexible strategy: use cheaper, single-GPU instances
like the `g4dn.xlarge` for iterative development and experimentation, and
leverage powerful, multi-GPU instances for final, large-scale production
training runs where time-to-completion is a critical factor.

#### Strategic Recommendation: Leveraging AWS Spot Instances

The single most impactful strategy for reducing compute costs is the
utilization of AWS Spot Instances. Spot Instances allow access to spare EC2
capacity at discounts of up to 90% compared to on-demand prices.[^11] For example,
a `g4dn.xlarge` instance can see its price drop from $0.526 to as low as
$0.2175 per hour, a saving of over 58%.[^13] A `p4d.24xlarge` can drop from over
$21/hour to around $6.40/hour.[^6]

Given these dramatic potential savings, the use of Spot Instances is mandated
as the default for all training jobs within this pipeline. However, this
economic decision has a profound architectural consequence: Spot Instances can
be reclaimed by AWS with only a two-minute warning. This inherent unreliability
means that the entire training process _must_ be designed for fault tolerance.
The pipeline must be capable of automatically checkpointing its progress
frequently and resuming from the last valid checkpoint upon interruption. This
elevates the role of checkpointing from a convenient feature to a core,
non-negotiable architectural requirement, a concept that will be detailed
further in Section 2.

#### Data Transfer and Ancillary Costs

Compute costs are not the only factor. Data transfer (egress) fees can become a
significant expense, particularly if large model artifacts are moved between
cloud providers or out to the public internet. AWS charges approximately $0.09
per GB for the first 10 TB of data egress per month.[^14] In contrast, data
transfer _within_ the same AWS region (e.g., from an S3 bucket to an EC2
instance) is free. This strongly incentivizes a design where all pipeline
components and data assets are co-located within a single cloud region to
eliminate egress charges. If the final model must be distributed to a
multi-cloud or on-premises environment, the more generous egress policies of
providers like Scaleway should be considered during the initial infrastructure
selection phase.[^10]

#### CPU Instances for Post-Training Tasks

The ONNX export, verification, and quantization stages of the pipeline are
computationally less demanding and do not require GPU acceleration. Executing
these steps on an expensive GPU instance is wasteful and inefficient.
Therefore, the design specifies that these post-training tasks should be run on
cost-effective, general-purpose CPU instances. An AWS `t4g.large` instance at
approximately $0.0672 per hour[^15] or a similarly priced DigitalOcean Basic
Droplet[^16] provides more than sufficient compute power for these tasks at a
fraction of the cost of a GPU instance.

The following table summarizes the key characteristics of suitable GPU
instances for the fine-tuning stage, providing a clear basis for data-driven
selection.

| Provider     | Instance Name  | GPU(s)         | GPU RAM (Total) | vCPUs | System RAM | On-Demand Price/hr | Est. Spot Price/hr |
| ------------ | -------------- | -------------- | --------------- | ----- | ---------- | ------------------ | ------------------ |
| AWS          | `g4dn.xlarge`  | 1x NVIDIA T4   | 16 GiB          | 4     | 16 GiB     | $0.526             | ~$0.22             |
| AWS          | `g6.xlarge`    | 1x NVIDIA L4   | 24 GiB          | 4     | 16 GiB     | $0.805             | ~$0.35             |
| AWS          | `p4d.24xlarge` | 8x NVIDIA A100 | 320 GiB         | 96    | 1152 GiB   | $21.958            | ~$6.40             |
| DigitalOcean | GPU Droplet    | 1x NVIDIA L40S | 48 GB           | 8     | 64 GiB     | $1.57              | N/A                |
| DigitalOcean | GPU Droplet    | 1x NVIDIA H100 | 80 GB           | 20    | 240 GiB    | $3.39              | N/A                |
| Scaleway     | GPU Instance   | 1x NVIDIA L40S | 48 GB           | 8     | 48 GB      | €1.61 (~$1.72)     | N/A                |

### 1.3 Data and Model Storage Strategy

A robust and well-organized storage strategy is critical for managing the
various artifacts generated and consumed by the pipeline. A centralized cloud
object storage service (such as AWS S3, DigitalOcean Spaces, or Scaleway Object
Storage) will serve as the single source of truth for all data, models, and
metadata.

#### Centralized Artifact Repository and Bucket Structure

To ensure clarity, version control, and ease of access, a standardized and
strictly enforced directory structure will be implemented within the designated
object storage bucket. This structure logically separates raw data from
processed data, base models from fine-tuned artifacts, and intermediate
checkpoints from final, deployable models.

- `/datasets/raw/{dataset_name}/`: Contains the original, immutable datasets as
  they are first acquired.
- `/datasets/processed/{dataset_name}/`: Stores datasets that have been
  cleaned, tokenized, and prepared for consumption by the training script. This
  will also house the calibration dataset required for quantization.
- `/models/base/{model_name}/`: A cache for the pre-trained models and
  tokenizers downloaded from the Hugging Face Hub. This prevents redundant
  downloads on subsequent pipeline runs.
- `/models/fine-tuned/{experiment_id}/checkpoints/`: The location for storing
  intermediate model checkpoints during training. This directory is essential
  for the fault-tolerant Spot Instance strategy.
- `/models/fine-tuned/{experiment_id}/final/`: The final, fully trained PyTorch
  model artifact.
- `/models/onnx/{experiment_id}/fp32/`: The exported ONNX model in its initial
  32-bit floating-point format.
- `/models/onnx/{experiment_id}/int8/`: The final, quantized, and verified
  8-bit integer ONNX model, which is the ultimate deployable artifact of the
  pipeline.

This structured approach transforms the object store from a simple file
repository into a versioned and auditable MLOps artifact store, forming the
connective tissue that links each modular stage of the pipeline.

## Section 2: The Python Fine-Tuning Pipeline: A Step-by-Step Implementation Guide

This section provides a detailed, implementation-focused blueprint for the core
fine-tuning stage of the pipeline. The implementation is designed in Python,
leveraging the Hugging Face ecosystem. A significant focus is placed on
addressing the specific challenge of ordinal regression, which requires a
departure from standard classification techniques and necessitates a custom
implementation within the `Trainer` framework.

### 2.1 Stage 1: Environment Setup and Data Ingestion

Each training job begins as a self-contained, ephemeral process within its
Docker container. The first responsibility of the training script is to
initialize its environment and securely fetch all necessary artifacts from the
centralized object storage repository.

#### Script Initialization and Artifact Download

The Python script will be initiated with a set of parameters defining the
experiment, such as the base model name, the dataset identifier, and a unique
experiment ID. It will use a cloud-specific SDK (e.g., `boto3` for AWS) to
establish a connection with the object storage service. The script will then
programmatically download the required assets based on the predefined bucket
structure: the pre-trained base model from `/models/base/` and the processed
dataset from `/datasets/processed/`. This practice ensures that the training
job is deterministic and relies solely on the versioned artifacts in the
central repository, rather than on potentially mutable local files or direct
internet downloads.

#### Model and Tokenizer Loading

Once the artifacts are downloaded to the local filesystem of the container, the
script will use the highly flexible Hugging Face `AutoModel` and
`AutoTokenizer` classes. These classes can infer the correct model architecture
and tokenizer configuration directly from the downloaded files, providing a
robust mechanism for loading the components into memory and preparing them for
the fine-tuning process.[^17]

### 2.2 Stage 2: Model Fine-Tuning for Ordinal Regression

The core of the fine-tuning process lies in adapting a standard transformer
model to the specific requirements of ordinal regression. A naive approach
using a standard classification head and cross-entropy loss is fundamentally
flawed for this task. Such a setup treats all misclassifications as equally
severe; for example, it would penalize the misclassification of “very positive”
as “positive” with the same magnitude as misclassifying it as “very negative.”
This ignores the inherent ordered relationship between the labels, which is the
defining characteristic of an ordinal problem.[^18] Practical guides that adapt
transformer regressors for text data surface the same challenge and motivate
bespoke ordinal treatments.[^19] To address this, both the model’s architecture
and the training loss calculation must be fundamentally altered.

#### Model Head Modification

The standard `AutoModelForSequenceClassification` from Hugging Face typically
includes a classification head composed of a linear layer that outputs a vector
of logits, with one logit for each class. For ordinal regression, this head is
unsuitable. Instead, the base transformer model (e.g., BERT, RoBERTa) will be
loaded, and its classification head will be replaced with a new, simpler head
consisting of a single linear layer. This layer will output a single, unbounded
scalar value for each input example.[^20] Community implementations of
regression-ready ViT heads follow the same simplification strategy, replacing
the multi-logit classifier with a scalar regressor.[^21] This scalar, which can
be denoted as f(X), represents the model’s learned position for the input X on
a continuous latent scale. It does not directly represent a class probability
but rather a score that will be used to derive those probabilities.

#### Custom Loss Function and Trainer Subclass

The most robust and maintainable way to implement the specialized logic for
ordinal regression is to encapsulate it within a custom subclass of the Hugging
Face `Trainer`. The `Trainer` API is explicitly designed to be extensible,
allowing for methods like `compute_loss` to be overridden to introduce custom
training behaviour without rewriting the entire training loop from
scratch.[^22][^23]
This approach is superior to using external callbacks or complex data
transformations as it cleanly integrates the domain-specific logic into the
core training machinery.

The implementation will take the form of a new class,
`OrdinalRegressionTrainer(Trainer)`.

- **Initialization (**`__init__`**):** The constructor of this class will be
  extended to accept the number of ordinal classes, `num_classes`. It will then
  initialize a set of K−1 trainable parameters, known as “cutpoints” or
  “thresholds,” represented as `c_0, c_1,..., c_{K-2}`. These will be
  initialized as a `torch.nn.Parameter` tensor.[^20]
- **Loss Computation (**`compute_loss`**):** The core logic will reside in the
  overridden `compute_loss` method. For each batch of inputs, this method will
  perform the following steps:

1. Execute the model’s forward pass to obtain the batch of scalar outputs, f(X).
2. Use these scalar outputs and the learned cutpoints to compute the
   probability of each of the K ordinal classes. This is achieved using a
   cumulative link function, typically the logistic (sigmoid) function,
   $\sigma(z) = \frac{1}{1 + e^{-z}}$. The probability of an input belonging to
   class $k$ is calculated as the difference between the cumulative probabilities
   defined by the cutpoints[^24]:

   $$
   P(y = k \mid X) =
   \begin{cases}
   \sigma(c_0 - f(X)) & \text{if } k = 0 \\
   \sigma(c_k - f(X)) - \sigma(c_{k-1} - f(X)) & \text{if } 0 < k < K - 1 \\
   1 - \sigma(c_{K-2} - f(X)) & \text{if } k = K - 1
   \end{cases}
   $$

3. With these
calculated probabilities, the standard negative log-likelihood loss is computed
against the true labels. This becomes the primary loss term that drives the
learning of both the base model’s weights and the cutpoint parameters.
4. A critical constraint in ordinal regression is that the cutpoints must
   remain in ascending order (i.e., $c_0 < c_1 < \dots < c_{K-2}$). This
   constraint must be enforced. This can be achieved either by adding a penalty
   term to the loss function that penalizes out-of-order cutpoints or, more
   directly, by implementing a post-gradient-update step (similar to the
   `AscensionCallback` in the `spacecutter` library) that re-sorts or clips the
   cutpoint values after each optimizer step.[^20]

#### Training Configuration

The standard `TrainingArguments` class will be used to configure the training
run. To support the fault-tolerant architecture required for Spot Instances,
key parameters will be set to ensure frequent state saving.
`evaluation_strategy` and `save_strategy` will be set to `"steps"`, and
`save_steps` will be configured to a reasonably small value (e.g., every 100 or
500 steps). This ensures that a recent checkpoint is always available in the
event of an interruption.[^22]

### 2.3 Stage 3: Checkpointing and Model Versioning

The ability to save and resume training is not merely a convenience but a
cornerstone of the pipeline’s economic viability and robustness. The
interaction between the `Trainer`’s built-in checkpointing capabilities and the
centralized object store is what enables the use of cost-effective but
unreliable Spot Instances.

#### Fault-Tolerant Checkpointing to Object Storage

The `Trainer`’s `output_dir` will be pointed to a temporary directory within
the running container. When the `Trainer` saves a checkpoint (as dictated by
the `save_strategy`), it writes the model weights, optimizer state, and trainer
state to this local directory. To persist this state beyond the life of the
container, a custom `TrainerCallback` will be implemented. This callback will
trigger on the `on_save` event, and its sole responsibility will be to
synchronize the contents of the local checkpoint directory with the appropriate
versioned path in the S3 bucket
(`/models/fine-tuned/{experiment_id}/checkpoints/`).

#### Resuming from a Checkpoint

The training script will be designed to be resumable. It will accept an
optional parameter specifying a checkpoint from which to resume. If this
parameter is provided, the script will first download the specified checkpoint
directory from S3 to the container’s local filesystem. This local path is then
passed to the `trainer.train(resume_from_checkpoint=...)` method.[^22] The
`Trainer` will handle the rest, correctly loading the model weights, optimizer
state, and learning rate scheduler state, and seamlessly continuing the
training run from where it left off. This mechanism is the key that transforms
a potentially catastrophic Spot Instance interruption into a minor, recoverable
delay.

#### Final Model Upload

Upon the successful completion of the entire training process (i.e.,
`trainer.train()` returns without error), the script will execute one final
save operation using `trainer.save_model()`. The resulting final model
artifact, which includes the model weights and configuration files, will be
uploaded to the designated final model path in the S3 artifact store:
`/models/fine-tuned/{experiment_id}/final/`. This marks the successful
completion of the fine-tuning stage and signals that the artifact is ready for
the subsequent export and optimization stages.

## Section 3: Model Export and Optimization for Production Inference

With a fine-tuned PyTorch model artifact secured in the central repository, the
pipeline transitions from training to production preparation. This phase is
critical for transforming the large, framework-dependent training artifact into
a lightweight, portable, and highly optimized inference engine. The process
involves exporting the model to the ONNX format, rigorously verifying its
numerical integrity, and applying quantization to maximize performance on CPU
hardware.

### 3.1 Exporting from PyTorch to ONNX with Hugging Face Optimum

The ONNX format serves as a universal intermediate representation for machine
learning models, enabling them to be executed across a wide variety of hardware
and software platforms.[^25] For this pipeline, the recommended and most robust
tool for converting Hugging Face models is the `optimum` library. It is an
official extension of the `transformers` library specifically designed for
performance optimization and deployment, offering a more stable and
feature-rich interface than legacy export methods.[^27][^28]

#### The Export Process

The export will be performed using the `optimum` command-line interface (CLI),
which provides a streamlined and declarative way to handle the conversion. The
process will be executed in a new pipeline stage on a cost-effective CPU
instance.

The core command is as follows:

optimum-cli export onnx --model /path/to/fine-tuned/pytorch/model --task
text-classification onnx/fp32/

#### Key Configuration Parameters

Several parameters in the export command are crucial for generating a correct
and flexible ONNX model.

- `--model`: This argument points to the local directory containing the final,
  fine-tuned PyTorch model saved in Section 2.3.
- `--task`: This parameter guides `optimum` on how to configure the model’s
  graph for a specific use case. Even though the task is ordinal regression,
  the underlying architecture is that of a sequence classifier. Specifying
  `text-classification` will correctly configure the inputs and the base
  transformer architecture.
- `opset_version`: The ONNX opset version determines the set of available
  operators in the exported graph. A modern but widely supported version, such
  as 17, is recommended. This version includes native support for complex
  operators like `LayerNormalization`, which can lead to a cleaner and more
  efficient graph.[^29] The choice must be cross-referenced with the compatibility
  matrix of the target ONNX Runtime version to ensure support.[^30]
- `dynamic_axes`: A critical feature for production is the ability to handle
  inputs of varying sizes (e.g., different batch sizes or sentence lengths).
  The `optimum` exporter automatically configures dynamic axes for the
  `batch_size` and `sequence_length` dimensions for standard tasks, which is
  essential for a real-world inference service.[^26]

The resulting `model.onnx` file, along with its configuration, will be uploaded
to the designated path in the artifact store:
`/models/onnx/{experiment_id}/fp32/`.

Optimum also exposes dedicated configuration classes that persist the export
settings, allowing reproducible regeneration of ONNX artefacts without manual
flag management in future runs.[^31]

### 3.2 Post-Export Verification and Parity Analysis

The conversion from PyTorch to ONNX is a complex translation process that can,
in some cases, introduce subtle numerical discrepancies or structural errors.
Deploying a model without verifying its correctness is a significant
operational risk. Therefore, an automated verification step is a mandatory
quality gate in this pipeline. The deployable artifact is not merely the
`.onnx` file itself, but the `.onnx` file accompanied by a guarantee of
numerical parity with its PyTorch parent.

#### Verification Procedure

This automated step will execute immediately after the ONNX export and will
programmatically compare the behavior of the two models.

1. **Load Models:** The script will load the original fine-tuned PyTorch model
   and the newly created ONNX model into memory. The ONNX model will be loaded
   using an `onnxruntime.InferenceSession`.
2. **Generate Test Inputs:** A representative sample of inputs (e.g., 100-1,000
   examples from the validation set) will be prepared.
3. **Dual Inference:** For each input, inference will be run through both the
   PyTorch model and the ONNX Runtime session.
4. **Compare Outputs:** The output tensors from both models will be compared
   element-wise. Due to the nature of floating-point arithmetic, a direct
   equality check is too strict. Instead, a comparison with a small tolerance
   is used, for example, via
   `numpy.allclose(pytorch_output, onnx_output, atol=1e-5)`. The
   `torch.onnx.verification` module also offers utilities for this purpose.[^32]
5. **Gate:** If the maximum absolute difference between the outputs exceeds the
   predefined tolerance for any of the test inputs, the verification step
   fails. This failure will halt the entire pipeline, preventing a potentially
   corrupt model from proceeding to the quantization and deployment stages.

### 3.3 Performance Optimization via INT8 Quantization

The final optimization step is designed to prepare the model for
high-performance inference on CPU hardware, which is the target environment for
the Rust service. Quantization is the process of converting the model’s weights
and/or activations from high-precision 32-bit floating-point numbers (FP32) to
low-precision 8-bit integers (INT8). This conversion yields two major benefits:
a significant reduction in model size (up to 4x) and a substantial increase in
inference speed, as integer arithmetic is much faster on modern CPUs than
floating-point arithmetic.[^33] Real-world benchmarks of static INT8
quantisation further demonstrate the throughput gains available on commodity
CPUs.[^34]

#### Methodology: Static Quantization

There are two primary methods of quantization: dynamic and static. Dynamic
quantization converts weights offline but calculates the scaling factors for
activations on-the-fly during inference. While simpler to implement, this
on-the-fly calculation introduces a small performance overhead. Static
quantization, by contrast, pre-calculates the scaling factors for activations
using a calibration dataset. This eliminates the runtime overhead, resulting in
the fastest possible inference speed on CPUs, making it the superior choice for
latency-sensitive production services.[^33]

The choice of static quantization introduces a new dependency into the
pipeline. This optimization stage now requires not only the FP32 ONNX model but
also a small, representative “calibration dataset.” This dataset, typically
consisting of 100-500 examples from the training set, must be created during
the initial data processing stage and stored in the artifact repository (e.g.,
at `/datasets/processed/{dataset_name}/calibration/`). This illustrates how a
downstream performance requirement can impose new upstream data preparation
tasks.

#### Implementation with ONNX Runtime

The quantization process will be implemented using the tools provided in the
`onnxruntime` Python package.

1. **Calibration Data Reader:** A Python class will be created to serve as a
   data reader. This class will iterate through the calibration dataset,
   preprocess each example in the same way as during training, and yield the
   inputs as a dictionary of NumPy arrays.
2. **`quantize_static` Function:** The core of the process is the
   `onnxruntime.quantization.quantize_static` function. It will be configured
   with the following key arguments:

- `model_input`: The path to the verified `fp32/model.onnx` file.
- `model_output`: The destination path for the new quantized model,
  `int8/model.onnx`.
- `calibration_data_reader`: An instance of the data reader class created in
  the previous step.
- `quant_format`: This will be set to `QuantFormat.QDQ`. The QDQ
  (Quantize/Dequantize) format is the modern standard, which inserts explicit
  `QuantizeLinear` and `DequantizeLinear` nodes into the graph. This format is
  more flexible and is the standard for models that have undergone
  Quantization-Aware Training (QAT).[^35]
- `activation_type` and `weight_type`: Both will be set to `QuantType.QInt8` to
  perform symmetric 8-bit integer quantization, which provides a robust balance
  of performance and accuracy on most CPU architectures.[^36]

#### Final Verification and Artifact Publication

The quantized INT8 model is a new representation and must undergo its own
parity check. The same verification procedure from Section 3.2 is repeated,
comparing the outputs of the INT8 ONNX model against the FP32 ONNX model. A
slightly higher tolerance (`atol`) may be necessary to account for the
precision loss inherent in quantization. Once this final check passes, the
`model_quantized.onnx` file is uploaded to
`/models/onnx/{experiment_id}/int8/`. This artifact, along with a metadata file
containing its final accuracy metrics and the learned ordinal cutpoints, is now
the official, versioned, and deployable output of the entire Python pipeline.

The following table summarizes the key configuration choices for the export and
quantization stages, providing a clear and actionable guide.

| Stage        | Tool/Function             | Parameter         | Recommended Value | Rationale                                                                                                                    |
| ------------ | ------------------------- | ----------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Export       | `optimum-cli export onnx` | `opset_version`   | 17                | Balances modern features (e.g., LayerNorm 29) with broad compatibility in ONNX Runtime versions.[^30]                           |
| Export       | `optimum-cli export onnx` | `optimize`        | O3                | Applies extensive graph fusions and GELU approximation for performance without requiring GPU-specific features.[^37]            |
| Quantization | `quantize_static`         | `quant_format`    | `QuantFormat.QDQ` | Modern, flexible format that inserts Quantize/Dequantize nodes, compatible with QAT models and cross-framework conversion.[^35] |
| Quantization | `quantize_static`         | `activation_type` | `QuantType.QInt8` | Symmetric quantization for activations.                                                                                      |
| Quantization | `quantize_static`         | `weight_type`     | `QuantType.QInt8` | Symmetric quantization for weights, providing a good balance of performance and accuracy on CPU.[^36]                           |

## Section 4: Integration with the Rust ,,`ort`,, Inference Component

The final phase of the end-to-end process is the consumption and execution of
the optimized ONNX model within the target Rust application. This section
provides a guide for the Rust development team, focusing on best practices for
using the `ort` crate to build a high-performance, reliable inference service.

### 4.1 Preparing the ONNX Artifact for Deployment

The output of the Python pipeline is a “deployment package” that contains more
than just the model file. A robust deployment process relies on having all
necessary components versioned and bundled together.

#### The Deployment Package

For each successful pipeline run, the following artifacts will be versioned and
stored together in the S3 artifact repository:

1. **The ONNX Model:** The `model_quantized.onnx` file from
   `/models/onnx/{experiment_id}/int8/`. This is the core inference engine.
2. **The Tokenizer Configuration:** The `tokenizer.json` and associated files
   from the base model, necessary for correctly preprocessing input text in the
   Rust application.
3. **Model Metadata:** A `metadata.json` file containing critical information
   for inference, including:

- The final, learned values of the ordinal regression cutpoints
  (`c_0, c_1,...`).
- The mapping from class indices to human-readable labels.
- The experiment ID and base model name for traceability.
- Final evaluation metrics (e.g., accuracy, mean absolute error) from the
  pipeline’s verification stages.

#### Distribution and Consumption

The Rust application’s CI/CD pipeline will be responsible for fetching this
complete deployment package from the artifact store. The application should be
configured to load a specific version of the package, ensuring that deployments
are deterministic and rollback-capable.

### 4.2 Rust ,,`ort`,, Crate: A Guide to High-Performance Inference

The `ort` crate provides idiomatic and safe Rust bindings for the ONNX Runtime
C API, enabling high-performance inference.[^38][^39] Proper configuration and
usage are key to building a stable and efficient service.

#### Project Setup and Dependency Management

The `ort` crate should be added as a dependency in the Rust project’s
`Cargo.toml` file.[^40] A crucial configuration choice is how the application
links to the underlying ONNX Runtime shared library.

The default behavior of the `ort` crate can lead to what is known as “shared
library hell,” where the build process hardcodes a dependency on the
`libonnxruntime.so` file being in a specific location. This is brittle and can
easily break when moving from a build environment to a production container.

The strongly recommended best practice is to enable the `load-dynamic` feature
for the `ort` crate in `Cargo.toml`. This feature decouples the compiled Rust
binary from the ONNX Runtime library. Instead of linking at compile time, the
application loads the library dynamically at runtime from a path specified by
the `ORT_DYLIB_PATH` environment variable.[^38] This approach shifts the
responsibility of providing the dependency from the compiler to the deployment
environment, which is a much more robust and flexible DevOps practice. It
allows the same compiled binary to run in any environment, as long as the
environment variable is set correctly.

#### Configuring the Deployment Environment

The Dockerfile for the Rust service must be configured to support this dynamic
loading strategy. It should include the following steps:

1. **Download ONNX Runtime:** Download the appropriate pre-compiled,
   CPU-optimized ONNX Runtime shared library (`libonnxruntime.so` for Linux)
   for the target architecture.
2. **Install Library:** Copy the shared library to a standard location within
   the container, such as `/opt/onnxruntime/lib/`.
3. **Set Environment Variable:** Use the `ENV` instruction in the Dockerfile to
   set the required environment variable:
   `ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so`.

#### Rust Inference Code Pattern

The following provides a conceptual pattern for implementing the inference
logic in Rust.

1. **Initialization:**

   - Initialize the `ort` environment. This is typically done once at
     application
     startup.
   - Load the `model_quantized.onnx` file and the `metadata.json` file from
     disk.
   - Create an `ort::Session` from the loaded model file. This session object is
     thread-safe and can be shared across multiple requests.
   - Load the tokenizer from its configuration file using a Rust-based tokenizer
     library like `tokenizers-rs`.

2. **Per-Request Inference:**

   - **Tokenization:** Preprocess the raw input text using the loaded tokenizer.
     This will produce `input_ids` and an `attention_mask`.
   - **Input Tensor Creation:** Convert the tokenized outputs into `ndarray`
     tensors. The `ort` crate expects inputs to be in this format. The tensors
     must have the correct shape (e.g., `[1, sequence_length]` for a single
     input) and data type (typically `i64` for token IDs).
   - **Execution:** Pass the input tensors to the `session.run()` method. This
     executes the ONNX graph and returns the model’s outputs.
   - **Output Parsing and Ordinal Logic:** The crucial final step is to
     correctly
     interpret the model’s output. The ONNX model, as designed in Section 2,
     will output a single scalar tensor, f(X). This is not the final
     prediction. The Rust code must now replicate the ordinal regression logic
     from the Python trainer:

3. Extract the scalar float value from the output tensor.
4. Retrieve the learned cutpoint values (`c_0, c_1,...`) from the loaded
   metadata.
5. Apply the cumulative link (sigmoid) function to calculate the probability of
   each ordinal class, exactly as specified in the formula in Section 2.2.
6. The final predicted class is the one with the highest calculated
   probability. The service can then return this class label or the full
   probability distribution.

This final step highlights a critical aspect of the end-to-end design: the
custom logic introduced in the Python training phase must be perfectly mirrored
in the Rust inference phase. The deployment package, containing both the model
and its metadata (including the cutpoints), is what makes this possible,
ensuring a consistent and correct implementation across the entire system.

## Conclusion

The architecture detailed in this report provides a robust, scalable, and
cost-effective blueprint for a complete model fine-tuning and deployment
pipeline. By integrating strategic infrastructure choices with tailored
software implementation, the design addresses the full lifecycle of a machine
learning model, from initial training to high-performance production inference.

The key recommendations and architectural pillars are as follows:

1. **Embrace Cost-Driven Architecture:** The pipeline’s design is fundamentally
   shaped by the economic imperative to minimize compute costs. The mandated
   use of interruptible AWS Spot Instances, which can reduce training expenses
   by up to 90%, directly necessitates the core architectural requirement of
   fault tolerance. This is achieved through frequent, automated checkpointing
   to a centralized object store and a training script designed for seamless
   resumption.
2. **Adopt a Modular, Containerized Workflow:** Structuring the pipeline as a
   series of distinct, containerized stages ensures reproducibility, simplifies
   debugging, and prepares the system for integration with sophisticated MLOps
   orchestration platforms. This modularity, combined with a centralized
   artifact store in cloud object storage, creates a clean, auditable, and
   maintainable workflow.
3. **Implement Custom Logic for Specialized Tasks:** The design demonstrates
   how to move beyond generic solutions to address the specific problem of
   ordinal regression. By subclassing the Hugging Face `Trainer` and
   implementing a custom loss function, the model is correctly trained to
   understand the ordered nature of the target labels. This tailored approach
   is critical for achieving high performance on non-standard machine learning
   tasks.
4. **Prioritize Production Performance with Optimization:** The pipeline’s
   ultimate goal is a deployable artifact optimized for a CPU-bound Rust
   environment. The prescribed use of the Hugging Face `optimum` library for
   ONNX export, followed by mandatory numerical parity verification and static
   INT8 quantization, ensures the final model is not only correct but also
   maximally performant.
5. **Ensure Seamless Integration with the Downstream Application:** The design
   bridges the gap between the Python training world and the Rust production
   environment. By recommending best practices for the `ort` crate, such as the
   `load-dynamic` feature for dependency management, and by explicitly
   packaging the model with its necessary metadata (including tokenizer files
   and learned ordinal cutpoints), the design guarantees that the Rust service
   has everything it needs to perform inference correctly and efficiently.

By following this blueprint, an organization can build a sophisticated MLOps
system that is not only technically sound but also strategically aligned with
the operational realities of production deployment, balancing cutting-edge
modeling techniques with the pragmatic requirements of cost, reliability, and
performance.

## Works cited

[^1]: g4dn.xlarge Pricing and Specs: AWS EC2, accessed on October 9, 2025,
   [https://costcalc.cloudoptimo.com/aws-pricing-calculator/ec2/g4dn.xlarge](https://costcalc.cloudoptimo.com/aws-pricing-calculator/ec2/g4dn.xlarge)
[^2]: g4dn.xlarge specs and pricing | AWS | CloudPrice, accessed on October 9,
   2025,
   [https://cloudprice.net/aws/ec2/instances/g4dn.xlarge](https://cloudprice.net/aws/ec2/instances/g4dn.xlarge)
[^3]: g6.xlarge pricing and specs - Vantage Instances, accessed on October 9,
   2025,
   [https://instances.vantage.sh/aws/ec2/g6.xlarge](https://instances.vantage.sh/aws/ec2/g6.xlarge)
[^4]: g6.xlarge Instance Specs And Pricing - CloudZero Advisor, accessed on
   October 9, 2025,
   [https://advisor.cloudzero.com/aws/ec2/g6.xlarge](https://advisor.cloudzero.com/aws/ec2/g6.xlarge)
[^5]: p4d.24xlarge specs and pricing | AWS | CloudPrice, accessed on October 9,
   2025,
   [https://cloudprice.net/aws/ec2/instances/p4d.24xlarge](https://cloudprice.net/aws/ec2/instances/p4d.24xlarge)
[^6]: Amazon EC2 Instance Type p4d.24xlarge, accessed on October 9, 2025,
   [https://aws-pricing.com/p4d.24xlarge.html](https://aws-pricing.com/p4d.24xlarge.html)
[^7]: GPU Droplets Pricing | DigitalOcean, accessed on October 9, 2025,
   [https://www.digitalocean.com/pricing/gpu-droplets](https://www.digitalocean.com/pricing/gpu-droplets)
[^8]: Scaleway | Review, Pricing & Alternatives - GetDeploying, accessed on
   October 9, 2025,
   [https://getdeploying.com/scaleway](https://getdeploying.com/scaleway)
[^9]: Ubicloud Compute, accessed on October 9, 2025,
   [https://www.ubicloud.com/use-cases/ubicloud-compute](https://www.ubicloud.com/use-cases/ubicloud-compute)
[^10]: Scaleway vs AWS, accessed on October 9, 2025,
    [https://www.scaleway.com/en/scaleway-vs-aws/](https://www.scaleway.com/en/scaleway-vs-aws/)
[^11]: Amazon EC2 Spot Instances Pricing - AWS, accessed on October 9, 2025,
    [https://aws.amazon.com/ec2/spot/pricing/](https://aws.amazon.com/ec2/spot/pricing/)
[^12]: Amazon EC2 Pricing - AWS, accessed on October 9, 2025,
    [https://aws.amazon.com/ec2/pricing/](https://aws.amazon.com/ec2/pricing/)
[^13]: AWS GPU Instance Pricing | P5, G6, G5 Spot Price Comparison, accessed on
    October 9, 2025,
    [https://compute.doit.com/gpu](https://compute.doit.com/gpu)
[^14]: EC2 On-Demand Instance Pricing - AWS, accessed on October 9, 2025,
    [https://aws.amazon.com/ec2/pricing/on-demand/](https://aws.amazon.com/ec2/pricing/on-demand/)
[^15]: t4g.large Pricing and Specs: AWS EC2, accessed on October 9, 2025,
    [https://costcalc.cloudoptimo.com/aws-pricing-calculator/ec2/t4g.large](https://costcalc.cloudoptimo.com/aws-pricing-calculator/ec2/t4g.large)
[^16]: Droplet Pricing | DigitalOcean, accessed on October 9, 2025,
    [https://www.digitalocean.com/pricing/droplets](https://www.digitalocean.com/pricing/droplets)
[^17]: Quickstart - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/transformers/quicktour](https://huggingface.co/docs/transformers/quicktour)
[^18]: How to Perform Ordinal Regression / Classification in PyTorch | Towards
    Data Science, accessed on October 9, 2025,
    [https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/](https://towardsdatascience.com/how-to-perform-ordinal-regression-classification-in-pytorch-361a2a095a99/)
[^19]: Regression with Text Input Using BERT and Transformers | by La Javaness R&D
    \| Medium, accessed on October 9, 2025,
    [https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13](https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13)
     \|
[^20]: EthanRosenthal/spacecutter: Ordinal regression models in PyTorch - GitHub,
    accessed on October 9, 2025,
    [https://github.com/EthanRosenthal/spacecutter](https://github.com/EthanRosenthal/spacecutter)
[^21]: How can modify ViT Pytorch transformer model for regression task - Stack
    Overflow, accessed on October 9, 2025,
    [https://stackoverflow.com/questions/75642865/how-can-modify-vit-pytorch-transformer-model-for-regression-task](https://stackoverflow.com/questions/75642865/how-can-modify-vit-pytorch-transformer-model-for-regression-task)
[^22]: Trainer - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/transformers/main/trainer](https://huggingface.co/docs/transformers/main/trainer)
[^23]: How to Use the Trainer API in Hugging Face for Custom Training Loops -
    KDnuggets, accessed on October 9, 2025,
    [https://www.kdnuggets.com/how-to-trainer-api-hugging-face-custom-training-loops](https://www.kdnuggets.com/how-to-trainer-api-hugging-face-custom-training-loops)
[^24]: spacecutter: Ordinal Regression Models in PyTorch | Ethan Rosenthal,
    accessed on October 9, 2025,
    [https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/](https://www.ethanrosenthal.com/2018/12/06/spacecutter-ordinal-regression/)
[^25]: ONNX - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/transformers/serialization](https://huggingface.co/docs/transformers/serialization)
[^26]: From PyTorch to ONNX: How Performance and Accuracy Compare | by Claudia
    Yao, accessed on October 9, 2025,
    [https://medium.com/@claudia.yao2012/from-pytorch-to-onnx-how-performance-and-accuracy-compare-a6f4747c1171](https://medium.com/@claudia.yao2012/from-pytorch-to-onnx-how-performance-and-accuracy-compare-a6f4747c1171)
[^27]: Export to ONNX - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/transformers/v4.38.0/serialization](https://huggingface.co/docs/transformers/v4.38.0/serialization)
[^28]: Export to ONNX - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/transformers/v4.49.0/serialization](https://huggingface.co/docs/transformers/v4.49.0/serialization)
[^29]: [docs] [onnx] Update the max admissible `opset_version` in the docs of
    `torch.onnx.export` and maybe update the default `opset_version` to 17 in
    release 2.1? · Issue #107446 · pytorch/pytorch - GitHub, accessed on
    October 9, 2025,
    [https://github.com/pytorch/pytorch/issues/107446](https://github.com/pytorch/pytorch/issues/107446)
[^30]: ONNX Runtime compatibility, accessed on October 9, 2025,
    [https://onnxruntime.ai/docs/reference/compatibility.html](https://onnxruntime.ai/docs/reference/compatibility.html)
[^31]: Configuration classes for ONNX exports - Hugging Face, accessed on October
    9, 2025,
    [https://huggingface.co/docs/optimum-onnx/onnx/package_reference/configuration](https://huggingface.co/docs/optimum-onnx/onnx/package_reference/configuration)
[^32]: torch.onnx.verification — PyTorch 2.8 documentation, accessed on October 9,
    2025,
    [https://docs.pytorch.org/docs/stable/onnx_verification.html](https://docs.pytorch.org/docs/stable/onnx_verification.html)
[^33]: Onnx Model Quantization | by Nashrakhan | Medium, accessed on October 9,
    2025,
    [https://medium.com/@nashrakhan1008/model-quantization-8f10c537e0eb](https://medium.com/@nashrakhan1008/model-quantization-8f10c537e0eb)
[^34]: Boost Your AI Models with INT8 Quantization ONNX Static vs Dynamic + Python
    & C++ Speed Test - YouTube, accessed on October 9, 2025,
    [https://www.youtube.com/watch?v=l9gyN1J5CCM](https://www.youtube.com/watch?v=l9gyN1J5CCM)
[^35]: Quantize ONNX Models - ONNXRuntime - GitHub Pages, accessed on October 9,
    2025,
    [https://iot-robotics.github.io/ONNXRuntime/docs/performance/quantization.html](https://iot-robotics.github.io/ONNXRuntime/docs/performance/quantization.html)
[^36]: Quantize ONNX models | onnxruntime, accessed on October 9, 2025,
    [https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
[^37]: Optimization - Hugging Face, accessed on October 9, 2025,
    [https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/optimization](https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/optimization)
[^38]: Crate ort - Rust bindings for ONNX Runtime - [Docs.rs](http://Docs.rs),
    accessed on October 9, 2025, [https://docs.rs/ort](https://docs.rs/ort)
[^39]: pykeio/ort: Fast ML inference & training for ONNX models in Rust - GitHub,
    accessed on October 9, 2025,
    [https://github.com/pykeio/ort](https://github.com/pykeio/ort)
[^40]: The Manifest Format - The Cargo Book - Rust Documentation, accessed on
    October 9, 2025,
    [https://doc.rust-lang.org/cargo/reference/manifest.html](https://doc.rust-lang.org/cargo/reference/manifest.html)
