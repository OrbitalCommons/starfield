//! Gaia DR1 (Data Release 1) loader, downloader, and entry type.

pub mod catalog;
pub mod entry;
pub mod schema;
pub mod supplement;

pub use catalog::{download_all, download_file, list_cached, load_tgas_block_map, Dr1Catalog};
pub use entry::{AstrometricExtra, Dr1Entry, ScanDirection, TgasBlock};
pub use schema::Dr1;
pub use supplement::{
    decode_supplement_hip, encode_supplement_source_id, is_supplement_source_id,
    SUPPLEMENT_REF_EPOCH, SUPPLEMENT_SOURCE_ID_BIT,
};
