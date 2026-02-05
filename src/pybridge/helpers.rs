//! Helper types and functions for Python-Rust data conversion
//!
//! This module contains the types for deserialization of Python values and
//! error handling for the Python-Rust bridge.

use base64::Engine;
use pyo3::PyErr;
use serde_json::Value;
use std::convert::TryFrom;
use thiserror::Error;

/// Errors that can occur in the Python-Rust bridge
#[derive(Error, Debug)]
pub enum BridgeError {
    #[error("Python error: {0}")]
    PythonError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Type conversion error: {0}")]
    TypeConversionError(String),

    #[error("Array shape mismatch: {0}")]
    ArrayShapeMismatch(String),
}

impl From<PyErr> for BridgeError {
    fn from(err: PyErr) -> Self {
        BridgeError::PythonError(err.to_string())
    }
}

/// Types of results we can get from Python
#[derive(Debug, PartialEq)]
pub enum PythonResult {
    /// Bytes data (base64 encoded)
    Bytes(Vec<u8>),

    /// String data
    String(String),

    /// Array data with shape and dtype
    ///
    /// The `data` field contains the raw bytes of the array in row-major order.
    /// The `shape` field describes the dimensions of the array.
    /// The `dtype` field describes the data type of the array elements.
    Array {
        dtype: String,
        shape: Vec<usize>,
        data: Vec<u8>,
    },
}

impl TryFrom<&str> for PythonResult {
    type Error = BridgeError;

    fn try_from(json_str: &str) -> Result<Self, Self::Error> {
        let value: Value = serde_json::from_str(json_str)
            .map_err(|e| BridgeError::SerializationError(e.to_string()))?;

        let obj = value
            .as_object()
            .ok_or_else(|| BridgeError::SerializationError("Expected JSON object".into()))?;

        let result_type = obj
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| BridgeError::SerializationError("Missing 'type' field".into()))?;

        let data = obj
            .get("data")
            .ok_or_else(|| BridgeError::SerializationError("Missing 'data' field".into()))?;

        match result_type {
            "bytes" => {
                let base64_data = data.as_str().ok_or_else(|| {
                    BridgeError::SerializationError("'data' field should be a string".into())
                })?;

                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(base64_data)
                    .map_err(|e| {
                        BridgeError::SerializationError(format!("Invalid base64: {}", e))
                    })?;

                Ok(PythonResult::Bytes(bytes))
            }
            "string" => {
                let string_data = data.as_str().ok_or_else(|| {
                    BridgeError::SerializationError("'data' field should be a string".into())
                })?;

                Ok(PythonResult::String(string_data.to_string()))
            }
            "array" => {
                let dtype = obj
                    .get("dtype")
                    .and_then(Value::as_str)
                    .ok_or_else(|| BridgeError::SerializationError("Missing 'dtype' field".into()))?
                    .to_string();

                let shape = obj
                    .get("shape")
                    .and_then(Value::as_array)
                    .ok_or_else(|| BridgeError::SerializationError("Missing 'shape' field".into()))?
                    .iter()
                    .map(|v| {
                        v.as_u64().ok_or_else(|| {
                            BridgeError::SerializationError(
                                "Shape should be array of integers".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<u64>, _>>()?
                    .into_iter()
                    .map(|v| v as usize)
                    .collect();

                let base64_data = data.as_str().ok_or_else(|| {
                    BridgeError::SerializationError("'data' field should be a string".into())
                })?;

                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(base64_data)
                    .map_err(|e| {
                        BridgeError::SerializationError(format!("Invalid base64: {}", e))
                    })?;

                Ok(PythonResult::Array {
                    dtype,
                    shape,
                    data: bytes,
                })
            }
            _ => Err(BridgeError::SerializationError(format!(
                "Unknown result type: {}",
                result_type
            ))),
        }
    }
}
