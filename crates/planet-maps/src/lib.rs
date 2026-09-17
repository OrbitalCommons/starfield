//! Equirectangular planetary albedo maps in body-fixed coordinates.
//!
//! The spatial half of resolved-body appearance: where the light comes from on
//! the disk. Spectral albedos ([`starfield-planet-spectra`]) and endmember
//! reflectances ([`starfield-reflectance-library`]) say what colour it is.
//!
//! # Conventions are data, not assumptions
//!
//! The archives disagree about longitude handedness and latitude definition,
//! and a map registered with the wrong handedness is *mirrored* — which looks
//! entirely plausible until it is checked against an ephemeris. Every product
//! therefore carries a [`MapGrid`] describing how it is stored, and all
//! conversion happens inside this crate.
//!
//! Callers pass **east-positive planetocentric radians**, which is exactly what
//! `starfield`'s body-fixed frames and `SubPoint::to_planetocentric` produce.
//! Do not apply a `LongitudeSense` first; that is for matching Horizons and IAU
//! cartographic output, not for sampling a map.
//!
//! # What the embedded tiers can and cannot support
//!
//! Earth is real per-texel composition, derived from MODIS sub-pixel land-cover
//! fractions. Mars is **one endmember**: brightness varies across the disk and
//! colour does not, pending CRISM/OMEGA terrain units.
//!
//! The Mars source is also contrast-normalised. Syrtis Major and Arabia Terra
//! differ by 1.10x in the Viking colour mosaic against roughly 2.7x on the real
//! planet, and Viking's polar coverage is sparse enough that the poles are
//! interpolated rather than measured. The dichotomy is present and correctly
//! oriented — Solis Lacus to Amazonis is 2x — but absolute contrast is
//! understated, which is one more reason its product entry is `calibrated:
//! false`. Appearance, not photometry.
//!
//! [`starfield-planet-spectra`]: https://docs.rs/starfield-planet-spectra
//! [`starfield-reflectance-library`]: https://docs.rs/starfield-reflectance-library

pub mod grid;
pub mod map;
pub mod products;
pub mod tier;

pub use tier::{
    earth_tier, mars_tier, moon_tier, AbundanceTier, Registration, EARTH_TIER_GZ, MARS_TIER_GZ,
    MOON_TIER_GZ,
};

pub use products::{require_product_for, MapProduct, PRODUCTS, USGS_MOSAIC_BASE_URL};

pub use map::{AlbedoMap, PhotometricBand, SampleFootprint, SurfaceSampler, MU_FLOOR};

pub use grid::{
    planetocentric_to_planetographic, planetographic_to_planetocentric, Latitude, Longitude,
    MapGrid, RowOrder,
};
