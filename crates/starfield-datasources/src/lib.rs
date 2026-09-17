//! Astronomical data source clients for the starfield ecosystem.
//!
//! This crate re-exports all starfield datasource crates behind feature flags.
//! By default, all datasources are enabled.
//!
//! # Feature flags
//!
//! - `horizons` — NASA JPL HORIZONS API client
//! - `sbdb` — NASA JPL Small-Body Database API client
//! - `gaia` — ESA Gaia DR3 loader (default)
//! - `gaia-all` — adds DR1 and DR2 alongside DR3
//! - `gaia-extended` — DR3 galaxy_candidates + qso_candidates loaders
//! - `bright-galaxies` — hand-curated supplement of nearby bright galaxies (M31, LMC, Virgo, etc.)
//! - `hipparcos` — Hipparcos star catalog loader
//! - `mast` — MAST (Mikulski Archive for Space Telescopes): HST observations, data products, FITS downloads, WCS
//! - `mpc` — Minor Planet Center client (MPCORB, observatory codes, observations)
//! - `nsa` — NASA-Sloan Atlas (NSA) galaxy catalog loader
//! - `planet-spectra` — spectral geometric albedos of the giant planets and Titan (Karkoschka / PDS)
//! - `planet-maps` — equirectangular albedo maps in body-fixed coordinates (USGS mosaics)
//! - `reflectance-library` — endmember surface reflectance spectra (USGS splib07)
//! - `solar-spectrum` — TSIS-1 HSRS v2 solar reference spectrum at 1 nm
//! - `rubin` — Vera C. Rubin Observatory LSST alert broker clients

#[cfg(feature = "horizons")]
pub use starfield_horizons as horizons;

#[cfg(feature = "sbdb")]
pub use starfield_sbdb as sbdb;

#[cfg(feature = "gaia")]
pub use starfield_gaia as gaia;

#[cfg(feature = "gaia-extended")]
pub use starfield_gaia_extended as gaia_extended;

#[cfg(feature = "bright-galaxies")]
pub use starfield_bright_galaxies as bright_galaxies;

#[cfg(feature = "hipparcos")]
pub use starfield_hipparcos as hipparcos;

#[cfg(feature = "mast")]
pub use starfield_mast as mast;

#[cfg(feature = "mpc")]
pub use starfield_mpc as mpc;

#[cfg(feature = "nsa")]
pub use starfield_nsa as nsa;

#[cfg(feature = "planet-spectra")]
pub use starfield_planet_spectra as planet_spectra;

#[cfg(feature = "planet-maps")]
pub use starfield_planet_maps as planet_maps;

#[cfg(feature = "reflectance-library")]
pub use starfield_reflectance_library as reflectance_library;

#[cfg(feature = "solar-spectrum")]
pub use starfield_solar_spectrum as solar_spectrum;

#[cfg(feature = "rubin")]
pub use starfield_rubin as rubin;
