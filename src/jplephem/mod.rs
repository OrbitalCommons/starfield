//! JPL Ephemeris module for high-precision planetary positions
//!
//! This module provides functionality for reading and interpreting JPL Development
//! Ephemerides (DE) files, which contain position and velocity data for solar
//! system bodies stored as Chebyshev polynomial coefficients in SPICE SPK format.
//!
//! # Main Components
//!
//! - [`daf`] - Double Array File format reader (underlying binary container)
//! - [`spk`] - Spacecraft Planet Kernel format reader
//! - [`kernel`] - High-level SpiceKernel API with named body access
//! - [`chebyshev`] - Chebyshev polynomial interpolation
//! - [`names`] - NAIF body name/ID mappings
//! - [`calendar`] - Julian date and calendar conversions

pub mod calendar;
pub mod chebyshev;
pub mod daf;
pub mod errors;
pub mod kernel;
pub mod names;
pub mod pck;
pub mod spk;

#[cfg(test)]
mod tests;

#[cfg(all(test, feature = "python-tests"))]
mod python_tests;

pub use self::chebyshev::{normalize_time, rescale_derivative, ChebyshevPolynomial};
pub use self::errors::JplephemError;
pub use self::kernel::SpiceKernel;
pub use self::pck::PCK;
pub use self::spk::SPK;
