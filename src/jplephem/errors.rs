//! Error types for the jplephem module

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for jplephem functionality
#[derive(Error, Debug)]
pub enum JplephemError {
    /// Error when a file I/O operation fails
    #[error("File I/O error on {path:?}: {source}")]
    FileError {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Error when a date is outside the range covered by the ephemeris
    #[error("Date JD {jd} is outside ephemeris range ({start_jd}..{end_jd})")]
    OutOfRangeError { jd: f64, start_jd: f64, end_jd: f64 },

    /// Error when the file format is invalid or unsupported
    #[error("Invalid file format: {0}")]
    InvalidFormat(String),

    /// Error when a memory mapping operation fails
    #[error("Memory mapping error: {0}")]
    MemoryMapError(String),

    /// Error when the requested body is not found in the ephemeris
    #[error("Body not found: center={center}, target={target}")]
    BodyNotFound { center: i32, target: i32 },

    /// Error when the data type is not supported
    #[error("Unsupported SPK data type: {0}")]
    UnsupportedDataType(i32),

    /// Other errors
    #[error("{0}")]
    Other(String),
}

/// Result type for jplephem operations
pub type Result<T> = std::result::Result<T, JplephemError>;

/// Convert a std::io::Error to JplephemError with path context
pub fn io_err(path: impl Into<PathBuf>, err: std::io::Error) -> JplephemError {
    JplephemError::FileError {
        path: path.into(),
        source: err,
    }
}
