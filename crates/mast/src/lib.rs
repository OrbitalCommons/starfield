//! MAST (Mikulski Archive for Space Telescopes) client.
//!
//! Wraps the public Mashup JSON-RPC endpoint at
//! `https://mast.stsci.edu/api/v0/invoke` and the data-product download
//! URLs to give Rust callers:
//!
//! - **Observation discovery** — cone-search the CAOM observation table
//!   to find what HST (or any other collection) imaging covers a region
//!   of sky. See [`MastClient::observations_in_cone`] /
//!   [`MastClient::hst_observations_in_cone`].
//! - **Data-product listing** — for any observation, list the FITS files
//!   and other artifacts associated with it. See
//!   [`MastClient::data_products`].
//! - **FITS download** — fetch a data product to a per-crate cache dir
//!   under `~/.cache/starfield/mast/`. See
//!   [`MastClient::download_product`].
//! - **WCS parsing** — read CRVAL/CRPIX/CD/CTYPE/NAXIS from a FITS
//!   header, and (for TAN projections) compute pixel↔world transforms +
//!   the four image-corner footprint in world coordinates. See
//!   [`Wcs`].
//!
//! HSC v3 source-catalog queries are deliberately out of scope for v1;
//! this crate focuses on the observation / image / WCS path.
//!
//! # Distortion caveat
//!
//! HST FLT (flatfielded but undistorted) products carry SIP polynomial
//! distortion coefficients that this crate does **not** apply.
//! `Wcs::pixel_to_world` and `Wcs::footprint` use the linear CD-matrix
//! TAN projection only — accurate to ~arcsec for typical HST FOV, fine
//! for "what region does this image cover" but not for sub-pixel
//! astrometry. DRZ / DRC drizzled products have already been distortion-
//! corrected upstream and the linear WCS is exact.
//!
//! # Example
//!
//! ```no_run
//! use starfield_gaia::Cone;
//! use starfield_mast::MastClient;
//!
//! let client = MastClient::new()?;
//! let cone = Cone::from_degrees(210.802, 54.349, 0.05); // M101 nucleus, 3'
//! let hst = client.hst_observations_in_cone(&cone)?;
//! println!("found {} HST observations", hst.len());
//!
//! if let Some(obs) = hst.first() {
//!     let products = client.data_products(&obs.obs_id)?;
//!     for p in products.iter().filter(|p| p.is_science_fits()).take(3) {
//!         let path = client.download_product(p)?;
//!         let wcs = starfield_mast::Wcs::read_from_fits(&path)?;
//!         println!("{} pointing → ({:.4}, {:.4})", p.filename, wcs.crval1, wcs.crval2);
//!     }
//! }
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

pub mod client;
pub mod observations;
pub mod products;
pub mod wcs;

pub use client::MastClient;
pub use observations::{CaomObservation, ObsCollection};
pub use products::{DataProduct, ProductType};
pub use wcs::Wcs;
