//! Python-Rust bridge for testing against the Python Skyfield implementation
//!
//! This module is only included when the `python-tests` feature is enabled and
//! provides utilities for comparing Rust calculations with the reference Python
//! implementation.
//!
//! The bridge allows bidirectional communication between Rust and Python:
//! 1. Executing Python code from Rust
//! 2. Converting Python data types (strings, bytes, numpy arrays) to Rust
//!
//! The bridge is primarily used for validating astronomical calculations against
//! the Python Skyfield library, which serves as a reference implementation.

pub mod bridge;
pub mod helpers;

// Re-export main components
pub use bridge::PyRustBridge;
pub use helpers::{BridgeError, PythonResult};

// Re-export conversion trait for easier use
pub use std::convert::TryFrom;
