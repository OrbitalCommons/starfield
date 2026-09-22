//! Astronomical calculations and optional catalogs, JPL clients, and surface data.
//!
//! Enable `catalogs`, `jpl`, or `surfaces` for a whole group, or select an
//! individual source such as `hipparcos`. Core types retain their existing paths.

pub use starfield_core::*;

/// Catalog traits, existing loaders, and optional datasource implementations.
pub mod catalogs {
    #[cfg(feature = "bright-galaxies")]
    pub use starfield_catalogs::bright_galaxies;
    #[cfg(feature = "gaia")]
    pub use starfield_catalogs::gaia;
    #[cfg(feature = "gaia-extended")]
    pub use starfield_catalogs::gaia_extended;
    #[cfg(feature = "mast")]
    pub use starfield_catalogs::mast;
    #[cfg(feature = "nsa")]
    pub use starfield_catalogs::nsa;
    pub use starfield_core::catalogs::*;
    /// Existing Hipparcos types and the optional datasource loader module.
    pub mod hipparcos {
        #[cfg(feature = "hipparcos")]
        pub use starfield_catalogs::hipparcos::{catalog, download_hipparcos, downloader};
        pub use starfield_core::catalogs::hipparcos::*;
    }
}

/// JPL and other Solar System data clients.
pub mod jpl {
    #[cfg(feature = "horizons")]
    pub use starfield_jpl::horizons;
    #[cfg(feature = "mpc")]
    pub use starfield_jpl::mpc;
    #[cfg(feature = "rubin")]
    pub use starfield_jpl::rubin;
    #[cfg(feature = "sbdb")]
    pub use starfield_jpl::sbdb;
}

/// Planetary maps and reference spectra.
pub mod surfaces {
    #[cfg(feature = "planet-maps")]
    pub use starfield_surfaces::planet_maps;
    #[cfg(feature = "planet-spectra")]
    pub use starfield_surfaces::planet_spectra;
    #[cfg(feature = "reflectance-library")]
    pub use starfield_surfaces::reflectance_library;
    #[cfg(feature = "solar-spectrum")]
    pub use starfield_surfaces::solar_spectrum;
}

#[cfg(feature = "horizons")]
pub use starfield_jpl::horizons;
#[cfg(feature = "sbdb")]
pub use starfield_jpl::sbdb;

/// Artifact loading and compatibility paths for optional clients.
pub mod data {
    pub use starfield_core::data::*;
    #[cfg(feature = "horizons")]
    pub use starfield_jpl::horizons;
    #[cfg(feature = "sbdb")]
    pub use starfield_jpl::sbdb;
}
