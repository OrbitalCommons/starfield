//! Measured azimuthally-averaged surface-brightness profiles for
//! catalog entries.
//!
//! This module is gated behind the `photometry` feature.
//!
//! [`RadialProfile`] complements (does not replace)
//! [`crate::catalogs::ExtendedSource`]: the parametric Sérsic fit on
//! `ExtendedSource` works well for ~80% of galaxies, but the rest
//! (bulge+disk decompositions, irregulars, mergers) are poorly captured
//! by a single Sérsic. Catalogs like NSA additionally store a measured
//! radial profile sampled at fixed radii in each band; renderers that
//! want maximum fidelity can use the measured profile when it exists
//! and fall back to the parametric fit otherwise.
//!
//! All slices are returned as borrowed `&[f64]` so the per-pixel inner
//! loop is allocation-free; implementors back the slices with whatever
//! they store internally.

use super::Band;

/// Measured azimuthally-averaged surface-brightness samples.
///
/// Implementors return borrowed slices into their per-entry storage so
/// renderers don't allocate per source. A return value of `None` from
/// any method means the data isn't available for this entry / band; it
/// is *not* an error.
pub trait RadialProfile {
    /// Radii at which the surface brightness is sampled, in arcseconds,
    /// strictly increasing. Returns `None` if no measured profile
    /// exists for this entry.
    fn profile_radii_arcsec(&self) -> Option<&[f64]>;

    /// Mean surface brightness in `band` at each radius from
    /// [`RadialProfile::profile_radii_arcsec`], in nanomaggies / arcsec².
    /// The returned slice has the same length as the radii slice, or
    /// `None` if `band` isn't measured.
    fn profile_surface_brightness(&self, band: Band) -> Option<&[f64]>;

    /// Inverse variance of [`RadialProfile::profile_surface_brightness`],
    /// same shape. Defaults to `None`; most catalogs don't carry this.
    fn profile_surface_brightness_ivar(&self, _band: Band) -> Option<&[f64]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal NSA-shaped fixture: radii in arcsec, a single SDSS r
    /// surface-brightness array, and an inverse-variance array.
    struct ProfileEntry {
        radii: Vec<f64>,
        sb_r: Vec<f64>,
        ivar_r: Vec<f64>,
    }

    impl RadialProfile for ProfileEntry {
        fn profile_radii_arcsec(&self) -> Option<&[f64]> {
            Some(&self.radii)
        }

        fn profile_surface_brightness(&self, band: Band) -> Option<&[f64]> {
            match band {
                Band::SdssR => Some(&self.sb_r),
                _ => None,
            }
        }

        fn profile_surface_brightness_ivar(&self, band: Band) -> Option<&[f64]> {
            match band {
                Band::SdssR => Some(&self.ivar_r),
                _ => None,
            }
        }
    }

    #[test]
    fn radii_and_surface_brightness_round_trip_as_borrowed_slices() {
        let entry = ProfileEntry {
            radii: vec![0.5, 1.0, 2.0, 4.0],
            sb_r: vec![100.0, 60.0, 30.0, 5.0],
            ivar_r: vec![0.01, 0.02, 0.04, 0.1],
        };
        let radii = entry.profile_radii_arcsec().unwrap();
        let sb = entry.profile_surface_brightness(Band::SdssR).unwrap();
        let ivar = entry.profile_surface_brightness_ivar(Band::SdssR).unwrap();

        assert_eq!(radii.len(), 4);
        assert_eq!(sb.len(), radii.len());
        assert_eq!(ivar.len(), radii.len());
        assert!(radii.windows(2).all(|w| w[1] > w[0]), "radii must increase");
        // Slices borrow from the entry — no allocation.
        assert!(std::ptr::eq(radii.as_ptr(), entry.radii.as_ptr()));
        assert!(std::ptr::eq(sb.as_ptr(), entry.sb_r.as_ptr()));
    }

    #[test]
    fn unmeasured_band_returns_none() {
        let entry = ProfileEntry {
            radii: vec![0.5, 1.0],
            sb_r: vec![100.0, 50.0],
            ivar_r: vec![0.01, 0.02],
        };
        assert!(entry.profile_surface_brightness(Band::GaiaG).is_none());
        assert!(entry.profile_surface_brightness_ivar(Band::GaiaG).is_none());
    }

    /// A type that doesn't carry any profile — must still implement
    /// the trait without overhead. Returning `None` from the required
    /// methods is the conventional "no data" path.
    struct NoProfile;
    impl RadialProfile for NoProfile {
        fn profile_radii_arcsec(&self) -> Option<&[f64]> {
            None
        }
        fn profile_surface_brightness(&self, _band: Band) -> Option<&[f64]> {
            None
        }
    }

    #[test]
    fn no_profile_implementor_returns_none_and_uses_default_ivar() {
        let s = NoProfile;
        assert!(s.profile_radii_arcsec().is_none());
        assert!(s.profile_surface_brightness(Band::SdssR).is_none());
        // Default ivar impl returns None unconditionally.
        assert!(s.profile_surface_brightness_ivar(Band::SdssR).is_none());
    }
}
