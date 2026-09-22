//! NASA-Sloan Atlas (NSA) galaxy catalog loader and downloader.
//!
//! The NSA is the canonical SDSS-derived extragalactic catalog. Two releases
//! exist in the wild and this crate handles both:
//!
//! - **v0_1_2** (~145k galaxies, 5-band `u, g, r, i, z`, ~0.5 GB) — the
//!   currently-downloadable file from `sdss.physics.nyu.edu`.
//! - **v1_0_1** (~640k galaxies, 7-band adds GALEX `FUV, NUV`, ~3 GB) — used
//!   to live under `data.sdss.org/sas/dr17/manga/atlas/`; that path 404s as
//!   of the 2026 SDSS DR17 reorg. The loader still parses it correctly if
//!   you have one on disk.
//!
//! [`NsaCatalog::from_fits_file`] auto-detects which version it's reading
//! from the `SERSIC_FLUX` column shape and remaps the v0_1_2 5-band data
//! into the canonical 7-slot in-memory layout (FUV/NUV slots zeroed). Use
//! [`NsaCatalog::version`] to ask which file you got.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::catalogs::StarCatalog;
//! use starfield_catalogs::nsa::{download_nsa, NsaCatalog};
//! let path = download_nsa()?;
//! let cat = NsaCatalog::from_fits_file(&path)?;
//! println!("{} galaxies (NSA {:?})", cat.len(), cat.version());
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

pub mod catalog;
pub mod downloader;
pub mod traits;

#[cfg(feature = "radial-profiles")]
pub use catalog::N_PROFILE_RADII;
pub use catalog::{NsaCatalog, NsaEntry, NsaVersion, BANDS, N_BANDS};
pub use downloader::download_nsa;
