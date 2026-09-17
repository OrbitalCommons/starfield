//! Spectral geometric albedos of the planets, Titan and the major moons.
//!
//! Renderers that place a *resolved* solar-system body on a focal plane need
//! three separable things: where the light comes from on the disk (an albedo
//! map), how it scatters with illumination geometry (a photometric function),
//! and what colour it is (a spectral albedo). This crate is the third.
//!
//! # What this crate is not
//!
//! An albedo is a dimensionless reflectance. It is not a spectrum, a radiance
//! or a flux, and [`SpectralAlbedo`] deliberately implements no trait that
//! would let it be mistaken for one. Turning an albedo into photons at a
//! detector needs the solar spectrum and the heliocentric and observer
//! distances — geometry that belongs to `starfield` and radiometry that
//! belongs to the consumer. Consumers multiply through themselves.
//!
//! # Geometric versus full-disk albedo
//!
//! The archive this crate embeds does not report one single quantity, and the
//! difference is worth several percent on the two brightest targets. Jupiter
//! and Saturn are tabulated as full-disk albedos at 6.8° and 5.7° phase; Uranus,
//! Neptune and Titan as geometric albedos at zero phase. Every spectrum carries
//! its [`AlbedoKind`] so the distinction cannot be lost by accident. Converting
//! a full-disk albedo to a geometric one needs a phase function, which this
//! crate does not yet ship.
//!
//! Saturn's column is the globe at zero ring tilt — the rings are excluded, and
//! they dominate Saturn's real appearance.
//!
//! # Source
//!
//! Karkoschka (1994, 1998), PDS Atmospheres volume `gbat_0001`, dataset
//! `ESO-J/S/N/U-SPECTROPHOTOMETER-4-V2.0`, DOI `10.17189/2bp8-k793`. The 1995
//! low-resolution table — 300–1050 nm at 0.4 nm sampling — is embedded, so the
//! common path needs no network and spans the whole silicon detector response.
//! The higher-resolution and 1993 tables are fetched on demand into the
//! starfield cache.
//!
//! # Example
//!
//! ```
//! use starfield_planet_spectra::{KarkoschkaTable, SpectralBody};
//!
//! let table = KarkoschkaTable::load_embedded()?;
//! let neptune = table.albedo(SpectralBody::Neptune).expect("Neptune is tabulated");
//!
//! // Neptune is blue: it reflects strongly at 450 nm and is swallowed by the
//! // 619 nm methane band.
//! let blue = neptune.at_nm(450.0).unwrap();
//! let red = neptune.at_nm(620.0).unwrap();
//! assert!(blue > 4.0 * red, "{blue} vs {red}");
//!
//! // Mean albedo across a Johnson V-ish band.
//! let v = neptune.mean_over(500.0, 590.0).unwrap();
//! println!("Neptune mean albedo over 500-590 nm: {v:.3}");
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

mod albedo;
mod body;
mod karkoschka;

pub use albedo::{AlbedoKind, SpectralAlbedo};
pub use body::SpectralBody;
pub use karkoschka::{KarkoschkaTable, Product, PDS_DATA_BASE_URL};
