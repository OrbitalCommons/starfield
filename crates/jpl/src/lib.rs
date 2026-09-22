//! Jpl implementations, each enabled by its matching feature.
//!
//! No sources are enabled by default. Prefer the `starfield` facade for stable paths.

extern crate self as starfield_jpl;

#[cfg(feature = "horizons")]
pub mod horizons;

#[cfg(feature = "sbdb")]
pub mod sbdb;

#[cfg(feature = "mpc")]
pub mod mpc;

#[cfg(feature = "rubin")]
pub mod rubin;
