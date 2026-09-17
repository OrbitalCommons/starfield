//! Hipparcos star catalog loader and downloader.
//!
//! This crate provides functionality for loading the Hipparcos star catalog
//! and downloading the catalog data file from the CDS archive.

pub mod catalog;
pub mod downloader;

pub use catalog::{HipparcosCatalog, HipparcosEntry};
pub use downloader::download_hipparcos;
