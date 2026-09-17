//! Radius-resolved isophote shape (axis ratio + position angle).
//!
//! This module is gated behind the `photometry` feature.
//!
//! [`IsophoteSeries`] reports how a galaxy's apparent shape changes
//! with radius — the *isophote twist* that a single
//! [`crate::catalogs::ExtendedSource`] Sérsic fit averages away.
//! Renderers that want to capture this twist can draw multiple nested
//! ellipses with different orientations rather than a single Sérsic.
//!
//! Catalogs typically expose this two ways:
//!
//! - **Cheap**: a 2-sample series at the 50% and 90% light radii (e.g.
//!   NSA's `BA50` / `PHI50` / `BA90` / `PHI90`).
//! - **Full**: a many-radius, per-band series derived from Stokes Q/U
//!   moments (e.g. NSA's `BASTOKES` / `PHISTOKES`, 15 radii × 7 bands).
//!
//! The trait shape is the same for both — renderers that want full
//! fidelity get more samples, naive renderers just iterate however
//! many they get.

use super::Band;

/// One sample of the isophote shape at a given radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IsophoteSample {
    /// Radius in arcseconds.
    pub radius_arcsec: f64,
    /// Axis ratio b/a, where b ≤ a. Range `[0, 1]`.
    pub axis_ratio: f64,
    /// Position angle of the major axis, degrees east of north (J2000).
    pub position_angle_deg: f64,
}

/// Trait for catalog entries that report isophote shape vs. radius.
///
/// A renderer that just wants the parametric fit should use
/// [`crate::catalogs::ExtendedSource::sersic_profile`]. This trait is
/// for renderers that want to capture isophote twist.
///
/// For catalogs that report per-band series (NSA's Stokes-derived
/// arrays), `band` selects the band. Catalogs that report a single
/// panchromatic series should ignore `band` and return the same slice
/// for any value.
///
/// Returns a borrowed slice, sorted by increasing radius, or `None` if
/// no series exists for this entry / band.
///
/// There is intentionally no default impl — point sources aren't
/// extended, so `IsophoteSeries` should only be implemented on entry
/// types that actually have isophote data.
pub trait IsophoteSeries {
    /// Borrowed slice of isophote samples for this band, sorted by
    /// increasing radius.
    fn isophote_samples(&self, band: Band) -> Option<&[IsophoteSample]>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-element series in NSA's cheap-tier shape: BA50/PHI50 and
    /// BA90/PHI90.
    struct CheapTier {
        samples: [IsophoteSample; 2],
    }

    impl IsophoteSeries for CheapTier {
        fn isophote_samples(&self, _band: Band) -> Option<&[IsophoteSample]> {
            // Panchromatic series — ignore band.
            Some(&self.samples)
        }
    }

    #[test]
    fn cheap_tier_returns_two_samples_in_radius_order() {
        let s = CheapTier {
            samples: [
                IsophoteSample {
                    radius_arcsec: 1.5,
                    axis_ratio: 0.7,
                    position_angle_deg: 45.0,
                },
                IsophoteSample {
                    radius_arcsec: 4.0,
                    axis_ratio: 0.6,
                    position_angle_deg: 60.0,
                },
            ],
        };
        let series = s.isophote_samples(Band::SdssR).unwrap();
        assert_eq!(series.len(), 2);
        assert!(series[0].radius_arcsec < series[1].radius_arcsec);
        // Panchromatic implementor returns the same slice for any band.
        let other = s.isophote_samples(Band::GaiaG).unwrap();
        assert_eq!(series.as_ptr(), other.as_ptr());
    }

    /// Per-band series in NSA's Stokes-tier shape: separate slices per
    /// band. Includes a band that's measured (SdssR) and one that isn't
    /// (SdssU) to exercise the `None` path.
    struct StokesTier {
        r_band: Vec<IsophoteSample>,
        g_band: Vec<IsophoteSample>,
    }

    impl IsophoteSeries for StokesTier {
        fn isophote_samples(&self, band: Band) -> Option<&[IsophoteSample]> {
            match band {
                Band::SdssR => Some(&self.r_band),
                Band::SdssG => Some(&self.g_band),
                _ => None,
            }
        }
    }

    #[test]
    fn stokes_tier_distinguishes_per_band_series_and_borrows() {
        let s = StokesTier {
            r_band: vec![
                IsophoteSample {
                    radius_arcsec: 0.5,
                    axis_ratio: 0.8,
                    position_angle_deg: 10.0,
                },
                IsophoteSample {
                    radius_arcsec: 2.0,
                    axis_ratio: 0.7,
                    position_angle_deg: 25.0,
                },
                IsophoteSample {
                    radius_arcsec: 5.0,
                    axis_ratio: 0.6,
                    position_angle_deg: 40.0,
                },
            ],
            g_band: vec![
                IsophoteSample {
                    radius_arcsec: 0.5,
                    axis_ratio: 0.85,
                    position_angle_deg: 12.0,
                },
                IsophoteSample {
                    radius_arcsec: 2.0,
                    axis_ratio: 0.75,
                    position_angle_deg: 28.0,
                },
            ],
        };

        let r = s.isophote_samples(Band::SdssR).unwrap();
        assert_eq!(r.len(), 3);
        // Slices borrow from per-band storage.
        assert_eq!(r.as_ptr(), s.r_band.as_ptr());

        let g = s.isophote_samples(Band::SdssG).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.as_ptr(), s.g_band.as_ptr());

        // Different bands return distinct slices.
        assert_ne!(r.as_ptr(), g.as_ptr());

        // Unmeasured band returns None.
        assert!(s.isophote_samples(Band::SdssU).is_none());
    }
}
