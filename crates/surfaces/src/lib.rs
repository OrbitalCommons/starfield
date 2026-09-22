//! Surfaces implementations, each enabled by its matching feature.
//!
//! No sources are enabled by default. Prefer the `starfield` facade for stable paths.

extern crate self as starfield_surfaces;

#[cfg(feature = "planet-maps")]
pub mod planet_maps;

#[cfg(feature = "reflectance-library")]
pub mod reflectance_library;

#[cfg(feature = "planet-spectra")]
pub mod planet_spectra;

#[cfg(feature = "solar-spectrum")]
pub mod solar_spectrum;
