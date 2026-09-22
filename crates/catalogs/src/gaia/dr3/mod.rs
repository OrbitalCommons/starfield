//! Gaia DR3 (Data Release 3) loader, downloader, and entry type.

pub mod catalog;
pub mod entry;
pub mod schema;
pub mod supplement;

pub use catalog::{download_all, download_file, list_cached, Dr3Catalog};
pub use entry::{
    AstrometricExtra, BpRpPhotometry, Classifications, DataLinks, Dr3Entry, GspPhot, IpdQuality,
    RadialVelocityDr3,
};
pub use schema::Dr3;
pub use supplement::{
    decode_supplement_hip, encode_supplement_source_id, is_supplement_source_id,
    SUPPLEMENT_REF_EPOCH, SUPPLEMENT_SOURCE_ID_BIT,
};
