//! Per-band photometric measurements for catalog entries.
//!
//! This module is gated behind the `photometry` feature.
//!
//! [`Band`] enumerates the photometric pass-bands a catalog entry may
//! report a measurement in, and [`Photometry`] is the corresponding
//! trait. All fluxes are in **nanomaggies** (1 nMgy = AB mag 22.5);
//! extinction and K-corrections are in **magnitudes** (additive in mag
//! space, multiplicative in flux space).
//!
//! Catalogs that don't carry photometric data implement the trait by
//! accepting the default `None`-returning methods. Catalogs that do
//! override the relevant methods band-by-band, so a renderer asks for
//! the bands it wants and skips the ones the catalog doesn't carry.

/// A photometric pass-band the catalog may report a measurement in.
///
/// `Band` is `#[non_exhaustive]` so additional bands (LSST `ugrizy`,
/// JWST NIRCam, etc.) can be added without breaking downstream `match`
/// arms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Band {
    /// GALEX far-UV (~1528 Å).
    GalexFuv,
    /// GALEX near-UV (~2271 Å).
    GalexNuv,
    /// SDSS u (~3551 Å).
    SdssU,
    /// SDSS g (~4686 Å).
    SdssG,
    /// SDSS r (~6166 Å).
    SdssR,
    /// SDSS i (~7480 Å).
    SdssI,
    /// SDSS z (~8932 Å).
    SdssZ,
    /// 2MASS J (~1.235 µm).
    TwoMassJ,
    /// 2MASS H (~1.662 µm).
    TwoMassH,
    /// 2MASS Ks (~2.159 µm).
    TwoMassK,
    /// Gaia G broad-band.
    GaiaG,
    /// Gaia BP (blue photometer).
    GaiaBp,
    /// Gaia RP (red photometer).
    GaiaRp,
    /// Hipparcos Hp.
    HipparcosHp,
    /// Johnson V (visual).
    JohnsonV,
    /// Johnson B (blue).
    JohnsonB,
}

/// Per-band photometric measurements for a catalog entry.
///
/// All flux methods return values in nanomaggies (1 nMgy = AB mag 22.5);
/// renderers can convert to AB via `mag = 22.5 - 2.5 * log10(flux)`.
/// Extinction and K-correction are returned in magnitudes (additive in
/// mag space, multiplicative in flux space).
///
/// All methods default to `None`, so any catalog entry trivially
/// implements the trait. Implementors override only the methods for
/// bands they actually carry.
pub trait Photometry {
    /// Pre-extinction, observer-frame flux in this band, in nanomaggies.
    /// Returns `None` if this catalog doesn't measure the band, or if
    /// the measurement is invalid for this entry.
    fn flux_nmgy(&self, _band: Band) -> Option<f64> {
        None
    }

    /// Inverse variance of [`Photometry::flux_nmgy`] in nMgy⁻².
    fn flux_ivar(&self, _band: Band) -> Option<f64> {
        None
    }

    /// Galactic extinction in this band, in magnitudes (positive = dimmer).
    fn extinction_mag(&self, _band: Band) -> Option<f64> {
        None
    }

    /// K-correction in this band, in magnitudes.
    fn k_correction_mag(&self, _band: Band) -> Option<f64> {
        None
    }

    /// AB magnitude in this band, optionally extinction- and
    /// K-corrected. Default impl composes
    /// [`Photometry::flux_nmgy`], [`Photometry::extinction_mag`],
    /// and [`Photometry::k_correction_mag`].
    ///
    /// Returns `None` if the flux is missing or non-positive.
    fn ab_magnitude(&self, band: Band, deredden: bool, k_correct: bool) -> Option<f64> {
        let f = self.flux_nmgy(band)?;
        if f <= 0.0 {
            return None;
        }
        let mut m = 22.5 - 2.5 * f.log10();
        if deredden {
            m -= self.extinction_mag(band).unwrap_or(0.0);
        }
        if k_correct {
            m -= self.k_correction_mag(band).unwrap_or(0.0);
        }
        Some(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A type that doesn't override anything — exercises the default
    /// `None` returns and confirms the trait is trivially implementable.
    struct NoPhotometry;
    impl Photometry for NoPhotometry {}

    #[test]
    fn default_impl_returns_none_everywhere() {
        let s = NoPhotometry;
        assert!(s.flux_nmgy(Band::SdssR).is_none());
        assert!(s.flux_ivar(Band::SdssR).is_none());
        assert!(s.extinction_mag(Band::SdssR).is_none());
        assert!(s.k_correction_mag(Band::SdssR).is_none());
        assert!(s.ab_magnitude(Band::SdssR, true, true).is_none());
    }

    /// A minimal implementor that carries only an SDSS r flux.
    struct OneBand {
        r_flux: f64,
        r_ext: f64,
        r_kcorr: f64,
    }

    impl Photometry for OneBand {
        fn flux_nmgy(&self, band: Band) -> Option<f64> {
            match band {
                Band::SdssR => Some(self.r_flux),
                _ => None,
            }
        }

        fn extinction_mag(&self, band: Band) -> Option<f64> {
            match band {
                Band::SdssR => Some(self.r_ext),
                _ => None,
            }
        }

        fn k_correction_mag(&self, band: Band) -> Option<f64> {
            match band {
                Band::SdssR => Some(self.r_kcorr),
                _ => None,
            }
        }
    }

    #[test]
    fn ab_magnitude_inverts_nanomaggies_at_zeropoint() {
        // 1 nMgy is exactly AB mag 22.5 by definition.
        let s = OneBand {
            r_flux: 1.0,
            r_ext: 0.0,
            r_kcorr: 0.0,
        };
        let m = s.ab_magnitude(Band::SdssR, false, false).unwrap();
        assert!((m - 22.5).abs() < 1e-12, "expected 22.5, got {m}");
    }

    #[test]
    fn ab_magnitude_applies_corrections_only_when_requested() {
        let s = OneBand {
            r_flux: 1.0,
            r_ext: 0.10,
            r_kcorr: 0.25,
        };
        let raw = s.ab_magnitude(Band::SdssR, false, false).unwrap();
        let dered = s.ab_magnitude(Band::SdssR, true, false).unwrap();
        let kcorr = s.ab_magnitude(Band::SdssR, false, true).unwrap();
        let both = s.ab_magnitude(Band::SdssR, true, true).unwrap();

        // Subtracting extinction in mag space makes a source brighter
        // (smaller mag) by exactly the extinction value.
        assert!((raw - 22.5).abs() < 1e-12);
        assert!((dered - (22.5 - 0.10)).abs() < 1e-12);
        assert!((kcorr - (22.5 - 0.25)).abs() < 1e-12);
        assert!((both - (22.5 - 0.10 - 0.25)).abs() < 1e-12);
    }

    #[test]
    fn ab_magnitude_returns_none_for_missing_or_nonpositive_flux() {
        let missing = OneBand {
            r_flux: 1.0,
            r_ext: 0.0,
            r_kcorr: 0.0,
        };
        assert!(missing.ab_magnitude(Band::SdssG, false, false).is_none());

        let nonpositive = OneBand {
            r_flux: -0.1,
            r_ext: 0.0,
            r_kcorr: 0.0,
        };
        assert!(nonpositive
            .ab_magnitude(Band::SdssR, false, false)
            .is_none());

        let zero = OneBand {
            r_flux: 0.0,
            r_ext: 0.0,
            r_kcorr: 0.0,
        };
        assert!(zero.ab_magnitude(Band::SdssR, false, false).is_none());
    }
}
