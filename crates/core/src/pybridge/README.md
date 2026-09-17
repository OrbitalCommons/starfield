# Python-Rust Bridge

This module provides a bridge between Rust and Python, enabling comparison testing against the Python [Skyfield](https://rhodesmill.org/skyfield/) library which serves as a reference implementation for astronomical calculations.

## Purpose

The primary purpose of this bridge is to validate Starfield's implementation against the Python Skyfield library to ensure accuracy and correctness. This helps:

1. Verify our calculations match the reference implementation
2. Debug discrepancies between implementations
3. Benchmark Rust performance against Python

## Architecture

The bridge consists of three main components:

### 1. Python Execution Engine (`bridge.rs`)

- Initializes a Python interpreter
- Loads the helper code
- Executes Python code and captures results
- Handles Python exceptions and error formatting

### 2. Data Conversion (`helpers.rs`)

- Defines the error types for the bridge
- Implements data type conversion between Python and Rust
- Supports three primary data types:
  - Bytes: Raw binary data
  - String: Text data
  - Array: NumPy arrays with shape and dtype information

### 3. Python Helper Code (`helper.py`)

- Provides a `ResultCollector` class that serves as an interface between Python and Rust
- Handles serialization of Python objects to JSON
- Supports bytes, strings, and NumPy arrays with proper serialization

## Usage

The bridge is only active when the `python-tests` feature flag is enabled. To use it:

```rust
// Enable the feature in your tests or examples
#[cfg(feature = "python-tests")]
fn example() -> anyhow::Result<()> {
    use starfield::pybridge::{PyRustBridge, PythonResult, TryFrom};
    
    // Create a bridge instance
    let bridge = PyRustBridge::new()?;
    
    // Execute Python code
    let code = r#"
    import numpy as np
    # Create a simple array with known values
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    rust.collect_array(arr)  # Send to Rust
    "#;
    
    // Get the result as JSON
    let json_result = bridge.run_py_to_json(code)?;
    
    // Convert to a PythonResult enum
    let python_result = PythonResult::try_from(json_result.as_str())?;
    
    // Process the result
    match python_result {
        PythonResult::Array { dtype, shape, data } => {
            // ... process array data
        }
        _ => panic!("Expected Array result"),
    }
    
    Ok(())
}
```

## Tests

The bridge includes tests for:

1. Basic Python execution
2. Bytes serialization and deserialization
3. Array serialization and deserialization
4. Error handling for Python exceptions

Run tests with:

```bash
cargo test --features python-tests
```

## Dependencies

- [pyo3](https://github.com/PyO3/pyo3): Python bindings for Rust
- [numpy](https://github.com/PyO3/rust-numpy): NumPy support for Rust
- [base64](https://github.com/marshallpierce/rust-base64): Base64 encoding/decoding
- [serde_json](https://github.com/serde-rs/json): JSON serialization/deserialization