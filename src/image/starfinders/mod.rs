//! Star finding algorithms for astronomical image analysis
//!
//! This module contains implementations of various star finding algorithms
//! commonly used in astronomical data reduction.

pub mod dao;
pub mod iraf;

// Re-export commonly used types
pub use dao::{DAOStarFinder, DAOStarFinderConfig, DaoStar};
pub use iraf::{IRAFStar, IRAFStarFinder, IRAFStarFinderConfig};
