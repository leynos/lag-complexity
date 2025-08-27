//! Heuristic provider implementations.

pub mod ambiguity;
pub mod depth;

pub use ambiguity::{AmbiguityHeuristic, AmbiguityHeuristicError};
pub use depth::{DepthHeuristic, DepthHeuristicError};
