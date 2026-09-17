//! Artifact downloads through the configured pull-through cache.

use crate::{
    artifact::{artifact_from_url, materialize},
    datastore::DatastoreBuilder,
    datastore_error,
};
use starfield_core::Result;
use std::path::Path;
use std::time::Duration;

/// Resolve a URL through the datastore and atomically publish a filename alias.
/// Upstream requires `STARFIELD_ALLOW_UPSTREAM=1`; `timeout_secs` sets the
/// connection timeout, not a total transfer deadline. Use a datasource's named
/// artifact API when available so it matches the server manifest's stable key.
pub fn download_to_file<P: AsRef<Path>>(url: &str, path: P, timeout_secs: u64) -> Result<()> {
    let artifact = artifact_from_url(url)?;
    let store = DatastoreBuilder::from_env()
        .map_err(datastore_error)?
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(datastore_error)?;
    crate::adopt_legacy(&store, &artifact, path.as_ref())?;
    let blob = store.get(&artifact).map_err(datastore_error)?;
    materialize(&blob, path.as_ref())
}
