//! Shared types and machinery used by every Gaia data release.

pub mod catalog;
pub mod cone;
pub mod core;
pub mod format;
pub mod lazy;
pub mod parse;
pub mod reader;
pub mod supplement;
pub mod traits;

pub use catalog::{GaiaCatalog, MemoryResidentCatalog};
pub use cone::Cone;
pub use core::{GaiaCore, VarFlag};
pub use lazy::LazyLoadingCatalog;
pub use reader::CsvSourceReader;
pub use supplement::{
    decode_supplement_hip, encode_supplement_source_id, is_supplement_source_id, SupplementRow,
    SUPPLEMENT_SOURCE_ID_BIT,
};
pub use traits::{GaiaRelease, GaiaSource, Release};
