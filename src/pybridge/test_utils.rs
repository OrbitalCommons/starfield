//! Shared test utilities for Python comparison tests
//!
//! Provides common parsing helpers for converting Python bridge results
//! into Rust types, and a shared DE421 kernel constructor.

use crate::jplephem::kernel::SpiceKernel;
use crate::pybridge::helpers::PythonResult;

/// Path to DE421 test data BSP file
pub const DE421_PATH: &str = "src/jplephem/test_data/de421.bsp";

/// Open the DE421 kernel for testing
pub fn de421_kernel() -> SpiceKernel {
    SpiceKernel::open(DE421_PATH).expect("Failed to open DE421")
}

/// Parse a single f64 from a Python string result
pub fn parse_f64(result: &str) -> f64 {
    let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
    match parsed {
        PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
        _ => panic!("Expected String result, got {:?}", parsed),
    }
}

/// Parse a comma-separated triple of f64 values
pub fn parse_f64_triple(result: &str) -> (f64, f64, f64) {
    let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
    match parsed {
        PythonResult::String(s) => {
            let parts: Vec<f64> = s.split(',').map(|p| p.trim().parse().unwrap()).collect();
            assert_eq!(parts.len(), 3, "Expected 3 values, got {}", parts.len());
            (parts[0], parts[1], parts[2])
        }
        _ => panic!("Expected String result, got {:?}", parsed),
    }
}

/// Parse a comma-separated list of f64 values
pub fn parse_f64_list(result: &str) -> Vec<f64> {
    let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
    match parsed {
        PythonResult::String(s) => s.split(',').map(|v| v.trim().parse().unwrap()).collect(),
        _ => panic!("Expected String result, got {:?}", parsed),
    }
}

/// Parse a comma-separated list of i64 values
pub fn parse_i64_list(result: &str) -> Vec<i64> {
    let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
    match parsed {
        PythonResult::String(s) => s.split(',').map(|v| v.trim().parse().unwrap()).collect(),
        _ => panic!("Expected String result, got {:?}", parsed),
    }
}

/// Parse a pipe-delimited "jds|events" string into separate JD and event type lists
pub fn parse_events(result: &str) -> (Vec<f64>, Vec<i64>) {
    let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
    let s = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected String result, got {:?}", parsed),
    };
    let parts: Vec<&str> = s.split('|').collect();
    assert_eq!(parts.len(), 2, "Expected 'jds|events' format");
    let jds: Vec<f64> = parts[0]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    let events: Vec<i64> = parts[1]
        .split(',')
        .map(|v| v.trim().parse().unwrap())
        .collect();
    (jds, events)
}

/// Compute angular difference accounting for wrapping at 360°
pub fn angular_diff_deg(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    d.min((d - 360.0).abs())
}
