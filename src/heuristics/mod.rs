//! Heuristic provider implementations.

pub mod ambiguity;
pub mod depth;
pub mod text;

pub use ambiguity::{AmbiguityHeuristic, AmbiguityHeuristicError};
pub use depth::{DepthHeuristic, DepthHeuristicError};
