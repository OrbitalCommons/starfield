//! Gaia DR2 (Data Release 2) loader, downloader, and entry type.

pub mod catalog;
pub mod entry;
pub mod schema;
pub mod supplement;

pub use catalog::{download_all, download_file, list_cached, Dr2Catalog};
pub use entry::{AstroParams, AstrometricExtra, BpRpPhotometry, Dr2Entry, RadialVelocity};
pub use schema::Dr2;
pub use supplement::{
    decode_supplement_hip, encode_supplement_source_id, is_supplement_source_id,
    SUPPLEMENT_REF_EPOCH, SUPPLEMENT_SOURCE_ID_BIT,
};
