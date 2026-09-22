//! Trait impls wiring [`crate::nsa::NsaEntry`] into the upstream `starfield`
//! catalog-side traits added in starfield#118 / #119 / #120.
//!
//! - [`Photometry`] is unconditional. Maps `Band::Galex{Fuv,Nuv}` and
//!   `Band::Sdss{U,G,R,I,Z}` onto the broadband `NMGY` measurements; other
//!   bands return `None`. `EXTINCTION` and `KCORRECT` populate the default
//!   [`Photometry::ab_magnitude`] composer.
//! - [`IsophoteSeries`] is unconditional and returns a 2-sample series at
//!   the Petrosian 50% / 90% light radii (derived from
//!   `BA50`/`PHI50`/`BA90`/`PHI90` + `PETROTH50`/`PETROTH90`). The samples
//!   are pre-materialized at load time, so the trait's borrow returns a
//!   slice straight off the entry without any allocation.
//! - [`RadialProfile`] is gated behind the `radial-profiles` feature and
//!   uses the per-band PROFTHETA / PROFMEAN arrays. Since the trait wants
//!   `&[f64]` but NSA stores `f32`, the conversion is cached lazily per
//!   entry inside `OnceLock`s. Without the feature, the impl is omitted
//!   entirely (no per-entry storage cost).

use crate::nsa::catalog::NsaEntry;

use starfield_core::catalogs::{Band, IsophoteSample, IsophoteSeries, Photometry};

#[cfg(feature = "radial-profiles")]
use starfield_core::catalogs::RadialProfile;

/// Map a [`Band`] to its index in `NsaEntry`'s canonical 7-slot band layout
/// `[FUV, NUV, u, g, r, i, z]`. Returns `None` for bands NSA doesn't carry —
/// 2MASS, Gaia, Hipparcos, Johnson all return `None`.
pub(crate) fn band_idx(band: Band) -> Option<usize> {
    match band {
        Band::GalexFuv => Some(0),
        Band::GalexNuv => Some(1),
        Band::SdssU => Some(2),
        Band::SdssG => Some(3),
        Band::SdssR => Some(4),
        Band::SdssI => Some(5),
        Band::SdssZ => Some(6),
        _ => None,
    }
}

impl Photometry for NsaEntry {
    fn flux_nmgy(&self, band: Band) -> Option<f64> {
        let i = band_idx(band)?;
        let f = self.nmgy.as_ref()?[i];
        // Zero is NSA's sentinel for "unmeasured" (and for FUV/NUV padding
        // when the source is v0_1_2). Treat as missing.
        if f == 0.0 {
            None
        } else {
            Some(f as f64)
        }
    }

    fn flux_ivar(&self, band: Band) -> Option<f64> {
        let i = band_idx(band)?;
        let v = self.nmgy_ivar.as_ref()?[i];
        if v == 0.0 {
            None
        } else {
            Some(v as f64)
        }
    }

    fn extinction_mag(&self, band: Band) -> Option<f64> {
        let i = band_idx(band)?;
        Some(self.extinction.as_ref()?[i] as f64)
    }

    fn k_correction_mag(&self, band: Band) -> Option<f64> {
        let i = band_idx(band)?;
        Some(self.kcorrect.as_ref()?[i] as f64)
    }
}

impl IsophoteSeries for NsaEntry {
    /// Two-sample panchromatic isophote series at the 50%/90% light radii.
    /// `band` is ignored — NSA's `BA50`/`PHI50`/`BA90`/`PHI90` are derived
    /// from r-band and there's no per-band variant in the cheap surface.
    /// Callers that want per-band twist should use the full Stokes series
    /// (gated behind `radial-profiles`).
    ///
    /// Returns `None` if any of the six required scalars
    /// (`BA50`/`PHI50`/`BA90`/`PHI90`/`PETROTH50`/`PETROTH90`) is missing
    /// from the source FITS file.
    fn isophote_samples(&self, _band: Band) -> Option<&[IsophoteSample]> {
        self.isophote_samples.as_ref().map(|arr| arr.as_slice())
    }
}

#[cfg(feature = "radial-profiles")]
impl RadialProfile for NsaEntry {
    fn profile_radii_arcsec(&self) -> Option<&[f64]> {
        let arr = self.proftheta.as_ref()?;
        let cell = self
            .radii_cache
            .get_or_init(|| arr.iter().map(|&x| x as f64).collect::<Vec<f64>>());
        Some(cell.as_slice())
    }

    fn profile_surface_brightness(&self, band: Band) -> Option<&[f64]> {
        let i = band_idx(band)?;
        let arr = self.profmean.as_ref()?;
        let cell = self.brightness_cache[i]
            .get_or_init(|| arr.iter().map(|row| row[i] as f64).collect::<Vec<f64>>());
        Some(cell.as_slice())
    }

    fn profile_surface_brightness_ivar(&self, band: Band) -> Option<&[f64]> {
        let i = band_idx(band)?;
        let arr = self.profmean_ivar.as_ref()?;
        let cell = self.brightness_ivar_cache[i]
            .get_or_init(|| arr.iter().map(|row| row[i] as f64).collect::<Vec<f64>>());
        Some(cell.as_slice())
    }
}
