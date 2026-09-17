//! Artifact resolution shared by archive clients. Query APIs use `http` instead.

use starfield_core::{Result, StarfieldError};
use starfield_datastore::{Artifact, ContentCheck, Datastore, DatastoreError};
use std::path::{Path, PathBuf};

/// Convert datastore errors at the shared boundary.
pub fn datastore_error(error: DatastoreError) -> StarfieldError {
    StarfieldError::Datastore(error)
}

/// Construct an artifact for a caller-supplied URL without storing query secrets
/// in its key. Archive APIs should prefer their own stable, named constructors.
pub fn artifact_from_url(url: &str) -> Result<Artifact> {
    let mut artifact = starfield_core::data::url_artifact(url)?;
    let parsed = url::Url::parse(url)
        .map_err(|_| StarfieldError::DataError("invalid artifact URL".into()))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(StarfieldError::DataError(
            "artifact URL must be HTTP(S) without userinfo".into(),
        ));
    }
    // Preserve the core constructor's stronger checks for recognized kernels.
    if !artifact.key.as_str().starts_with("url/") {
        return Ok(artifact);
    }
    let check = match parsed
        .path()
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "gz" => ContentCheck::magic(vec![vec![0x1f, 0x8b]], false),
        "fits" => ContentCheck::magic(vec![b"SIMPLE  =".to_vec()], false),
        "bsp" => ContentCheck::magic(vec![b"DAF/SPK".to_vec(), b"NAIF/DAF".to_vec()], false),
        _ => ContentCheck::default_binary(),
    };
    artifact.check = check;
    Ok(artifact)
}

/// Adopt an explicitly supplied legacy path, validating it before publication.
/// Does not discover files or consult environment variables. Rejected legacy
/// content is left untouched and normal resolution may try another layer.
pub fn adopt_legacy(store: &Datastore, artifact: &Artifact, path: &Path) -> Result<()> {
    if !store.contains(&artifact.key) && path.is_file() {
        match store.import(artifact, path) {
            Ok(_) | Err(DatastoreError::ContentRejected { .. }) => {}
            Err(error) => return Err(datastore_error(error)),
        }
    }
    Ok(())
}

/// Convenience resolution using environment/config settings with optional
/// validated adoption. Caller-configured APIs should call `store.get` directly
/// and never search the global legacy cache.
pub fn resolve_artifact(artifact: &Artifact, legacy: Option<&Path>) -> Result<PathBuf> {
    let store = Datastore::from_env().map_err(datastore_error)?;
    if let Some(path) = legacy {
        adopt_legacy(&store, artifact, path)?;
    }
    store.get(artifact).map_err(datastore_error)
}

/// Publish a filename alias for consumers whose parsers use the file extension.
/// A hard link avoids duplicating multi-GB blobs; cross-device destinations use
/// a streamed copy. Replacing/removing the alias never removes the store entry.
pub fn materialize(blob: &Path, destination: &Path) -> Result<()> {
    if blob == destination {
        return Ok(());
    }
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temp = tempfile::NamedTempFile::new_in(parent)?;
    let path = temp.into_temp_path();
    let staging_path = path.to_path_buf();
    std::fs::remove_file(&path)?;
    if std::fs::hard_link(blob, &path).is_err() {
        std::fs::copy(blob, &path)?;
    }
    let result = path
        .persist(destination)
        .map_err(|error| StarfieldError::IoError(error.error));
    // POSIX rename of hard links to the same inode succeeds without removing
    // the source. Repeated or concurrent publication must clean that link up.
    let _ = std::fs::remove_file(staging_path);
    result
}
