//! Astronomical calculations and optional catalogs, JPL clients, and surface data.
//!
//! Enable `catalogs`, `jpl`, or `surfaces` for a whole group, or select an
//! individual source such as `hipparcos`. Core types retain their existing paths.

pub use starfield_core::*;

/// Catalog traits, existing loaders, and optional datasource implementations.
pub mod catalogs {
    #[cfg(feature = "bright-galaxies")]
    pub use starfield_bright_galaxies as bright_galaxies;
    pub use starfield_core::catalogs::*;
    #[cfg(feature = "gaia")]
    pub use starfield_gaia as gaia;
    #[cfg(feature = "gaia-extended")]
    pub use starfield_gaia_extended as gaia_extended;
    #[cfg(feature = "mast")]
    pub use starfield_mast as mast;
    #[cfg(feature = "nsa")]
    pub use starfield_nsa as nsa;
    /// Existing Hipparcos types and the optional datasource loader module.
    pub mod hipparcos {
        pub use starfield_core::catalogs::hipparcos::*;
        #[cfg(feature = "hipparcos")]
        pub use starfield_hipparcos::{catalog, download_hipparcos, downloader};
    }
}

/// JPL and other Solar System data clients.
pub mod jpl {
    #[cfg(feature = "horizons")]
    pub use starfield_horizons as horizons;
    #[cfg(feature = "mpc")]
    pub use starfield_mpc as mpc;
    #[cfg(feature = "rubin")]
    pub use starfield_rubin as rubin;
    #[cfg(feature = "sbdb")]
    pub use starfield_sbdb as sbdb;
}

/// Planetary maps and reference spectra.
pub mod surfaces {
    #[cfg(feature = "planet-maps")]
    pub use starfield_planet_maps as planet_maps;
    #[cfg(feature = "planet-spectra")]
    pub use starfield_planet_spectra as planet_spectra;
    #[cfg(feature = "reflectance-library")]
    pub use starfield_reflectance_library as reflectance_library;
    #[cfg(feature = "solar-spectrum")]
    pub use starfield_solar_spectrum as solar_spectrum;
}

#[cfg(feature = "horizons")]
pub use starfield_horizons as horizons;
#[cfg(feature = "sbdb")]
pub use starfield_sbdb as sbdb;

/// Artifact loading and compatibility paths for optional clients.
pub mod data {
    pub use starfield_core::data::*;
    #[cfg(feature = "horizons")]
    pub use starfield_horizons as horizons;
    #[cfg(feature = "sbdb")]
    pub use starfield_sbdb as sbdb;
}
