//! ESA Gaia star catalog loaders and downloaders for DR1, DR2, and DR3.
//!
//! Each data release has its own entry type exposing every field that release
//! publishes, and its own catalog type — none of them share a "lowest common
//! denominator" row format. The shared pieces (astrometry core, HTTP client, MD5
//! verifier, Arrow CSV reader, in-memory catalog) live in [`common`] and [`download`]
//! and are generic over the release marker via the [`GaiaRelease`] trait.
//!
//! # Feature flags
//!
//! - `dr1`: enable [`dr1`] module (DR1 `gaia_source`, optional TGAS cross-id attachment)
//! - `dr2`: enable [`dr2`] module (DR2 `gaia_source`, adds BP/RP, RV, Apsis)
//! - `dr3`: enable [`dr3`] module (DR3 `gaia_source`, adds RUWE, IPD, GSP-Phot, datalink flags)
//! - `all-releases`: enable all three
//!
//! Enabling `gaia` includes `dr3`; the catalogs package has no default sources.
//!
//! # Example
//!
//! ```no_run
//! use starfield_catalogs::gaia::Dr3Catalog;
//! use starfield_core::catalogs::StarCatalog;
//!
//! let catalog = Dr3Catalog::from_csv_file("GaiaSource_000000-003111.csv.gz", 18.0)?;
//! println!("loaded {} stars at mag <= 18.0", catalog.len());
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

pub mod common;
pub mod download;
pub mod excerpt;

#[cfg(feature = "dr1")]
pub mod dr1;
#[cfg(feature = "dr2")]
pub mod dr2;
#[cfg(feature = "dr3")]
pub mod dr3;

pub use common::{
    Cone, GaiaCatalog, GaiaCore, GaiaRelease, GaiaSource, LazyLoadingCatalog,
    MemoryResidentCatalog, Release, VarFlag,
};
pub use download::Downloader;

#[cfg(feature = "dr1")]
pub use dr1::{Dr1, Dr1Catalog, Dr1Entry, TgasBlock};
#[cfg(feature = "dr2")]
pub use dr2::{Dr2, Dr2Catalog, Dr2Entry};
#[cfg(feature = "dr3")]
pub use dr3::{Dr3, Dr3Catalog, Dr3Entry};

/// Convenience prelude for Gaia consumers.
pub mod prelude {
    #[cfg(feature = "dr1")]
    pub use crate::gaia::{Dr1Catalog, Dr1Entry};
    #[cfg(feature = "dr2")]
    pub use crate::gaia::{Dr2Catalog, Dr2Entry};
    #[cfg(feature = "dr3")]
    pub use crate::gaia::{Dr3Catalog, Dr3Entry};
    pub use crate::gaia::{GaiaCore, GaiaSource, Release};
}
