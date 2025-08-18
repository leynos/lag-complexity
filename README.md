# LAG Complexity: The Cognitive Clutch for Advanced AI Agents

`lag-complexity` is a high-performance, modular, and production-grade Rust
implementation of the Cognitive Load metric, CL(q), as defined in the
Logic-Augmented Generation (LAG) research paper.

Modern Large Language Models (LLMs) excel at many tasks but often struggle with
complex, multi-step questions, leading to "hallucination". The LAG paradigm
addresses this by introducing a "reasoning-first" pipeline that intelligently
decomposes complex questions into simpler, manageable steps.

This crate provides the core mechanism for that intelligence: the **Cognitive
Load metric**. It acts as a "cognitive clutch" for AI agents, giving them a
real-time signal to dynamically switch between fast, direct answers for simple
queries and deliberate, structured decomposition for complex ones.

## What is Cognitive Load?

The CL(q) score is a single floating-point number that quantifies a query's
intrinsic complexity. It is an aggregation of three distinct signals:

1. **Semantic Scope (**`scope`**)**: Measures the conceptual breadth of a
   query. A high scope suggests the query touches on many different topics.
2. **Reasoning Steps (**`depth`**)**: Estimates the number of latent logical
   steps required to answer the query. A high depth is a primary trigger for
   decomposition.
3. **Ambiguity (**`ambiguity`**)**: Quantifies the semantic uncertainty in a
   query. A high ambiguity score suggests the query might have multiple
   interpretations and should be clarified.

By evaluating this score, a LAG-based system can make intelligent, real-time
decisions about how to best approach a problem.

## Getting Started

### Installation

Add `lag-complexity` to your project's dependencies:

```null
cargo add lag-complexity

```

By default, the crate includes fast, lightweight heuristic-based providers and
local ONNX model support for a "batteries-included" experience. For more
advanced use cases, you can enable additional features.

### Quick Example

Here is a simple example of how to calculate the complexity score for a query
using the default, general-purpose configuration.

```null
use lag_complexity::api::{Complexity, ComplexityFn, Error};
use lag_complexity::providers::{
    // Using default, fast providers
    ApiEmbedding, // For this example, we'll use a mock or a real API provider
    DepthHeuristic,
    AmbiguityHeuristic,
};
use lag_complexity::config::ScoringConfig;
use lag_complexity::DefaultComplexity;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // In a real application, this would be loaded from a file (e.g., config.toml)
    let config = ScoringConfig::default();

    // Set up the providers for each complexity signal.
    // Here we use the fast heuristic providers for depth and ambiguity.
    // The embedding provider would typically be configured to point to an API
    // or a local model.
    let embedding_provider = ApiEmbedding::new("YOUR_API_KEY", "https://api.openai.com/v1/embeddings");
    let depth_provider = DepthHeuristic::default();
    let ambiguity_provider = AmbiguityHeuristic::default();

    // Compose the providers into the main scoring engine.
    let scorer = DefaultComplexity::new(
        &embedding_provider,
        &depth_provider,
        &ambiguity_provider,
        &config,
    );

    let simple_query = "What is the capital of France?";
    let complex_query = concat!(
        "What are the main differences between the economic policies ",
        "of the UK and Japan since the 2008 financial crisis?",
    );

    // Score the simple query
    match scorer.score(simple_query) {
        Ok(complexity) => {
            println!("--- Score for: '{}' ---", simple_query);
            println!("Total Complexity: {:.4}", complexity.total());
            println!("  - Scope:     {:.4}", complexity.scope());
            println!("  - Depth:     {:.4}", complexity.depth());
            println!("  - Ambiguity: {:.4}", complexity.ambiguity());
        }
        Err(e) => eprintln!("Error scoring simple query: {}", e),
    }

    println!("\n");

    // Score the complex query
    match scorer.score(complex_query) {
        Ok(complexity) => {
            println!("--- Score for: '{}' ---", complex_query);
            println!("Total Complexity: {:.4}", complexity.total());
            println!("  - Scope:     {:.4}", complexity.scope());
            println!("  - Depth:     {:.4}", complexity.depth());
            println!("  - Ambiguity: {:.4}", complexity.ambiguity());

            // Use the score to make a decision
            if config.is_split_recommended(complexity.total(), 0) {
                println!("\nDecision: Complexity is high. Recommend decomposition.");
            } else {
                println!("\nDecision: Complexity is low. Proceed with direct answer.");
            }
        }
        Err(e) => eprintln!("Error scoring complex query: {}", e),
    }

    Ok(())
}

```

## Features

The `lag-complexity` crate is highly modular and uses feature flags to keep the
core library lean. You can opt into additional functionality as needed:

| **Feature Flag**  | **Purpose**                                                                           | **Default** |
| ----------------- | ------------------------------------------------------------------------------------- | ----------- |
| `provider-api`    | Enables providers that call external HTTP APIs for embeddings or LLM-based estimates. | Off         |
| `provider-tch`    | Enables local transformer models via the `tch` crate (LibTorch backend).              | Off         |
| `provider-candle` | Enables local transformer models via the pure-Rust `candle` framework.                | On          |
| `onnx`            | Enables ONNX Runtime for lightweight classifier models.                               | On          |
| `rayon`           | Enables parallel execution for batch scoring and concurrent provider calls.           | On          |
| `python`          | Builds Python bindings for the crate.                                                 | Off         |
| `wasm`            | Builds a WebAssembly module for browser/JS environments.                              | Off         |
| `cli`             | Builds the `lagc` command-line interface binary.                                      | On          |

## Demonstrations

To see the crate in action and get an intuitive feel for the complexity metric,
check out our live demonstrations:

- **Interactive Complexity Meter (WASM)**: [Coming soon!] - Type any
  question and see its cognitive load scores update in real-time, right in your
  browser!
- **Jupyter Notebook Walkthroughs**: [Coming soon!] - Explore
  detailed, narrative examples showcasing how `lag-complexity` enables smarter,
  safer agent behaviour.

## Contributing

This project adheres to the
[Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).

## Licence

lag-complexity is distributed under the terms of the ISC licence. See
[LICENSE](LICENSE) for details.
