//! Hipparcos and kernel downloads share starfield's datastore identities.

use starfield_core::Result;
use std::path::{Path, PathBuf};

/// Legacy cache root, retained for importing existing user downloads.
pub fn get_cache_dir() -> PathBuf {
    starfield_datasource_utils::cache_dir()
}

/// Resolve known archive filenames using the same table as `starfield_core::Loader`.
pub fn resolve_url(filename: &str) -> Option<String> {
    starfield_core::data::resolve_url(filename)
}

/// Resolve through the configured pull-through cache, validating legacy files
/// before adoption. An explicit directory selects the local cache root.
pub fn download_or_cache(filename: &str, data_dir: Option<&Path>) -> Result<PathBuf> {
    starfield_core::data::download_or_cache(filename, data_dir)
}

/// Resolve the Hipparcos catalogue through starfield's shared datastore path.
pub fn download_hipparcos() -> Result<PathBuf> {
    starfield_core::data::download_hipparcos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_urls_share_the_loader_table() {
        for name in [
            "de421.bsp",
            "jup365.bsp",
            "pck00011.tpc",
            "moon_080317.tf",
            "naif0012.tls",
        ] {
            assert_eq!(resolve_url(name), starfield_core::data::resolve_url(name));
            assert!(resolve_url(name).is_some());
        }
        assert!(resolve_url("unknown.xyz").is_none());
    }
}
