//! Shared utilities for starfield datasource crates
//!
//! Common helpers for HTTP clients, file caching, and downloads
//! used across multiple datasource implementations.

pub mod artifact;
pub mod cache;
pub mod checksum;
pub mod curve;
pub mod download;
pub mod http;

pub use artifact::{adopt_legacy, artifact_from_url, datastore_error, resolve_artifact};
pub use starfield_datastore as datastore;

pub use cache::{cache_dir, ensure_cache_dir, ensure_cache_subdir, file_exists_and_not_empty};
pub use checksum::{sha256_bytes, sha256_file, verify_sha256};
pub use curve::SampledCurve;
pub use download::download_to_file;
pub use http::{
    assert_endpoint_reachable, build_http_client, check_endpoint_status, check_response_status,
    EndpointStatus,
};
