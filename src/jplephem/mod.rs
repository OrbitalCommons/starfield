//! JPL Ephemeris module for high-precision planetary positions
//!
//! This module provides functionality for reading and interpreting JPL Development
//! Ephemerides (DE) files, which contain position and velocity data for solar
//! system bodies stored as Chebyshev polynomial coefficients in SPICE SPK format.
//!
//! # Supported Ephemerides
//!
//! Any SPK/BSP file using data types 2 (Chebyshev position), 3 (Chebyshev
//! position + velocity), or 21 (Modified Difference Array) is supported.
//! This includes all standard JPL planetary ephemerides:
//!
//! | Ephemeris | Time Span | Size | Notes |
//! |-----------|-----------|------|-------|
//! | DE405 | 1599–2201 | ~55 MB | Legacy, widely used |
//! | DE421 | 1900–2050 | ~17 MB | Compact, good for near-term work |
//! | DE430t | 1550–2650 | ~115 MB | Truncated DE430 |
//! | DE440 | 1550–2650 | ~114 MB | Current standard, fits to modern data |
//! | DE441 | −13200–17191 | ~3.1 GB | Extended time span for deep-past/future |
//!
//! NAIF satellite kernels (e.g. `jup365.bsp`) are also supported.
//!
//! # Main Components
//!
//! - [`daf`] - Double Array File format reader (underlying binary container)
//! - [`spk`] - Spacecraft Planet Kernel format reader
//! - [`kernel`] - High-level SpiceKernel API with named body access
//! - [`chebyshev`] - Chebyshev polynomial interpolation
//! - [`spk_type21`] - SPK Type 21 Modified Difference Array interpolation
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
pub mod spk_type21;

#[cfg(test)]
mod tests;

pub use self::chebyshev::{normalize_time, rescale_derivative, ChebyshevPolynomial};
pub use self::errors::JplephemError;
pub use self::kernel::{PlanetState, SpiceKernel, AU_KM, S_PER_DAY};
pub use self::pck::PCK;
pub use self::spk::SPK;
