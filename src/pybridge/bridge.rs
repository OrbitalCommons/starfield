//! Bridge for communication between Rust and Python
//!
//! This module is only active when the `python-tests` feature is enabled
//! and provides functionality for executing Python code and getting results.

use crate::pybridge::helpers::BridgeError;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList},
};

/// Python-Rust bridge for testing against the Python Skyfield implementation
pub struct PyRustBridge {
    py_globals: Py<PyDict>,
}

/// Returns the helper code that will be loaded into the Python environment
pub fn get_helper_code() -> &'static str {
    include_str!("helper.py")
}

impl PyRustBridge {
    /// Create a new bridge instance
    pub fn new() -> Result<Self, BridgeError> {
        pyo3::prepare_freethreaded_python();

        Python::with_gil(|py| {
            let locals = PyDict::new(py);

            // Import commonly used modules
            let builtins = PyModule::import(py, "builtins")?;
            locals.set_item("__builtins__", builtins)?;

            Ok(Self {
                py_globals: locals.into(),
            })
        })
    }

    /// Run Python code and return the result as a JSON value
    pub fn run_py_to_json(&self, code: &str) -> Result<String, BridgeError> {
        Python::with_gil(|py| {
            let globals = self.py_globals.as_ref(py);

            // load the helper code
            let helper_code = get_helper_code();

            let get_result_code = "_result = rust.get_result()";

            for code_block in [helper_code, code, get_result_code] {
                println!("Running code block: \n{}", code_block);
                match py.run(code_block, Some(globals), None) {
                    Ok(_) => {}
                    Err(e) => {
                        // Format Python exception nicely
                        let py_err = format_py_error(py, &e);
                        return Err(BridgeError::PythonError(py_err));
                    }
                }
            }

            // Check if there's a result variable defined
            match globals.get_item("_result") {
                Some(value) => {
                    if let Ok(value) = value.extract::<String>() {
                        Ok(value)
                    } else {
                        Err(BridgeError::TypeConversionError(
                            "Failed to convert Python result to string".into(),
                        ))
                    }
                }
                None => Err(BridgeError::PythonError(
                    "No 'result' variable defined in Python code".into(),
                )),
            }
        })
    }
}

/// Format Python exception in a readable way
fn format_py_error(py: Python, error: &PyErr) -> String {
    // Get exception type and message
    let exc_type = error.get_type(py);
    let exc_name = exc_type.name().unwrap_or("Unknown");

    // Get value and traceback as strings if available
    let value = match error.value(py).str() {
        Ok(s) => s.to_string(),
        Err(_) => String::from("(no error message)"),
    };

    // Try to get traceback information
    let mut tb_info = String::new();
    if let Some(tb) = error.traceback(py) {
        if let Ok(tb_module) = PyModule::import(py, "traceback") {
            if let Ok(format_exc) = tb_module.getattr("format_tb") {
                if let Ok(tb_list) = format_exc.call1((tb,)) {
                    if let Ok(tb_list) = tb_list.downcast::<PyList>() {
                        for line in tb_list.iter() {
                            if let Ok(line_str) = line.extract::<String>() {
                                tb_info.push_str(&line_str);
                            }
                        }
                    }
                }
            }
        }
    }

    // Format the error nicely
    if tb_info.is_empty() {
        format!("{}: {}", exc_name, value)
    } else {
        format!("{}: {}\n\nTraceback:\n{}", exc_name, value, tb_info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pybridge::helpers::PythonResult;
    use std::convert::TryFrom;

    use anyhow::Result;

    #[test]
    fn test_python_print_ld_library_path() {
        // print the path after breaking it up by :
        if let Ok(val) = std::env::var("LD_LIBRARY_PATH") {
            let paths: Vec<&str> = val.split(':').collect();
            for p in paths {
                println!("Path: {}", p);
            }
        } else {
            println!("LD_LIBRARY_PATH is not set.");
        }
    }

    #[test]
    fn test_python_get_interp_info() {
        let bridge = PyRustBridge::new().unwrap();

        let code = r#"import sys; rust(sys.executable)"#;

        let result = bridge.run_py_to_json(code).unwrap();
        println!("Result Interp: {}", result);
    }

    #[test]
    fn test_python_helper_code_loaded() -> Result<()> {
        // Print the helper code to see what's being loaded
        println!("Helper code: {}", get_helper_code());

        Ok(())
    }

    #[test]
    fn test_python_bytes_serialization() -> Result<()> {
        let bridge = PyRustBridge::new()?;

        let code = r#"rust.collect_bytes(b"abcd")"#;

        let json_result = bridge.run_py_to_json(code)?;
        println!("JSON result: {}", json_result);

        // Parse the JSON result into our PythonResult enum
        let python_result = PythonResult::try_from(json_result.as_str())?;

        // Verify it's the correct type
        match python_result {
            PythonResult::Bytes(bytes) => {
                assert_eq!(bytes, b"abcd");
            }
            _ => panic!("Expected Bytes type, got {:?}", python_result),
        }

        Ok(())
    }

    #[test]
    fn test_python_error_handling() {
        let bridge = PyRustBridge::new().unwrap();

        // This code will raise a ZeroDivisionError
        let code = r#"result = 1 / 0"#;

        let result = bridge.run_py_to_json(code);
        assert!(result.is_err());

        if let Err(err) = result {
            let err_string = err.to_string();
            assert!(err_string.contains("ZeroDivisionError"));
        }
    }

    #[test]
    fn test_python_array_serialization() -> Result<()> {
        let bridge = PyRustBridge::new()?;

        // Create a numpy array and collect it
        let code = r#"
import numpy as np
# Create a simple 2x3 array with known values
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
rust.collect_array(arr)
        "#;

        let json_result = bridge.run_py_to_json(code)?;
        println!("JSON result: {}", json_result);

        // Parse the JSON result into our PythonResult enum
        let python_result = PythonResult::try_from(json_result.as_str())?;

        // Verify it's the correct type and shape
        match python_result {
            PythonResult::Array { dtype, shape, data } => {
                // Check the dtype is float64
                assert_eq!(dtype, "float64");

                // Check the shape is 2x3
                assert_eq!(shape, vec![2, 3]);

                // The data should be 48 bytes (2x3 array of float64, 8 bytes each)
                assert_eq!(data.len(), 48);

                // Convert bytes to f64 values to verify content
                // Note: We're assuming little-endian byte order here
                let mut values = Vec::new();
                for chunk in data.chunks_exact(8) {
                    let value = f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]);
                    values.push(value);
                }

                // Check specific values (row-major order)
                assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            }
            _ => panic!("Expected Array type, got {:?}", python_result),
        }

        Ok(())
    }
}
