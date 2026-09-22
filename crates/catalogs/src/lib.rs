//! Catalogs implementations, each enabled by its matching feature.
//!
//! No sources are enabled by default. Prefer the `starfield` facade for stable paths.

extern crate self as starfield_catalogs;

#[cfg(feature = "gaia")]
pub mod gaia;

#[cfg(feature = "gaia-extended")]
pub mod gaia_extended;

#[cfg(feature = "bright-galaxies")]
pub mod bright_galaxies;

#[cfg(feature = "hipparcos")]
pub mod hipparcos;

#[cfg(feature = "mast")]
pub mod mast;

#[cfg(feature = "nsa")]
pub mod nsa;
