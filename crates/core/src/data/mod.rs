//! Data module for downloading and managing astronomical data
//!
//! This module provides functionality for downloading, caching, and loading
//! astronomical datasets like star catalogs.

#[cfg(feature = "datastore")]
pub mod artifacts;
mod downloader;
mod gaia_downloader;

#[cfg(feature = "datastore")]
pub use artifacts::{
    artifact_for, download_or_cache_with, gaia_artifact, gaia_md5sums_artifact, hipparcos_artifact,
    kernel_artifact, url_artifact, GaiaRelease,
};
pub use downloader::{
    download_hipparcos, download_or_cache, ensure_cache_dir, get_cache_dir, resolve_url,
    HIPPARCOS_URL, JPL_BSP_URL, NAIF_LSK_URL, NAIF_PLANETS_OLD_URL, NAIF_PLANETS_URL,
    NAIF_SATELLITES_URL,
};
pub use gaia_downloader::{
    download_gaia_catalog, download_gaia_file, ensure_gaia_cache_dir, get_gaia_cache_dir,
    list_cached_gaia_files,
};
