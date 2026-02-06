//! Data module for downloading and managing astronomical data
//!
//! This module provides functionality for downloading, caching, and loading
//! astronomical datasets like star catalogs.

mod downloader;
mod gaia_downloader;

pub use downloader::{
    download_hipparcos, download_or_cache, ensure_cache_dir, get_cache_dir, resolve_url,
};
pub use gaia_downloader::{
    download_gaia_catalog, download_gaia_file, ensure_gaia_cache_dir, get_gaia_cache_dir,
    list_cached_gaia_files,
};
