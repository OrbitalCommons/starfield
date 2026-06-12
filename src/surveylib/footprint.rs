//! Survey sky footprints in equatorial coordinates.
//!
//! A footprint answers two questions: *is this (RA, dec) inside the
//! surveyed region?* and *how much sky does the region cover?* Both are
//! always posed in **equatorial** coordinates (RA/dec in degrees). Survey
//! footprints are instrumental — they live on the equatorial sky — so any
//! position computed in ecliptic coordinates must be rotated through a
//! typed frame transform (see [`super::geometry`]) before a membership
//! test. Substituting ecliptic latitude for declination misclassifies
//! exactly the survey-boundary regions that drive coverage estimates.

use crate::constants::RAD2DEG;

/// One declination band with a (possibly zero-wrapping) right-ascension
/// range, the building block of box-union footprints.
///
/// The RA range runs **eastward** from `ra_start_deg` to `ra_end_deg` and
/// may wrap through 0°/360° (e.g. `ra_start_deg = 300`, `ra_end_deg = 105`
/// covers RA 300°→360°→105°). Membership is half-open in both axes:
/// `dec_min_deg <= dec < dec_max_deg` and RA in `[ra_start, ra_end)`
/// eastward, so abutting boxes tile without double counting.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkyBox {
    /// Minimum declination (degrees, inclusive).
    pub dec_min_deg: f64,
    /// Maximum declination (degrees, exclusive).
    pub dec_max_deg: f64,
    /// RA range start (degrees); the range runs eastward from here.
    pub ra_start_deg: f64,
    /// RA range end (degrees, exclusive); may be numerically smaller than
    /// `ra_start_deg`, in which case the range wraps through 0°.
    pub ra_end_deg: f64,
}

impl SkyBox {
    /// Whether the equatorial position falls inside the box.
    pub fn contains(&self, ra_deg: f64, dec_deg: f64) -> bool {
        (self.dec_min_deg..self.dec_max_deg).contains(&dec_deg)
            && ra_in_range(ra_deg, self.ra_start_deg, self.ra_end_deg)
    }

    /// Eastward RA span of the box in degrees, in `[0°, 360°)`.
    pub fn ra_span_deg(&self) -> f64 {
        (self.ra_end_deg - self.ra_start_deg).rem_euclid(360.0)
    }

    /// Exact solid angle of the box in square degrees:
    /// `ΔRA · (sin dec_max − sin dec_min) · 180/π`.
    pub fn solid_angle_deg2(&self) -> f64 {
        let dsin = self.dec_max_deg.to_radians().sin() - self.dec_min_deg.to_radians().sin();
        self.ra_span_deg() * dsin * RAD2DEG
    }
}

/// Whether `ra` lies in the eastward range `[start, end)`, wrapping
/// through 0° when `start > end`.
fn ra_in_range(ra: f64, start: f64, end: f64) -> bool {
    let ra = ra.rem_euclid(360.0);
    if start <= end {
        (start..end).contains(&ra)
    } else {
        ra >= start || ra < end
    }
}

/// A survey's sky coverage in equatorial coordinates.
///
/// The variants cover the footprint shapes that real wide-field surveys
/// declare:
///
/// * [`Footprint::AllSky`] — the whole celestial sphere (toy surveys,
///   completeness baselines).
/// * [`Footprint::DecRange`] — a declination band, the natural model for
///   hemispheric surveys driven by a single site's horizon (e.g. a
///   "dec > −30°" cut).
/// * [`Footprint::BoxUnion`] — a union of RA/dec boxes approximating an
///   irregular contiguous footprint, with an exact analytic solid angle.
/// * [`Footprint::Custom`] — an arbitrary membership predicate (polygon
///   tests, HEALPix masks, ...) with a caller-declared solid angle.
///
/// # Solid angle
///
/// [`Footprint::solid_angle_deg2`] is **exact** for `AllSky` and
/// `DecRange`, and exact for `BoxUnion` **only when the boxes are
/// disjoint** (overlapping boxes are double counted — keep the union a
/// tiling). For `Custom` the value is whatever the caller declared; it is
/// the caller's responsibility that it integrates the predicate. A good
/// validation pattern is a Monte-Carlo cross-check: sample uniformly on
/// the sphere (`ra ~ U[0°, 360°)`, `dec = asin(U[−1, 1])`) and compare the
/// hit fraction times 4π sr against the declared value.
#[derive(Debug, Clone)]
pub enum Footprint {
    /// The full celestial sphere.
    AllSky,
    /// All right ascensions within a declination band
    /// (`dec_min_deg <= dec <= dec_max_deg`, inclusive).
    DecRange {
        /// Minimum declination (degrees).
        dec_min_deg: f64,
        /// Maximum declination (degrees).
        dec_max_deg: f64,
    },
    /// A union of RA/dec boxes. The solid angle is the sum of the box
    /// solid angles, exact only for disjoint boxes.
    BoxUnion(Vec<SkyBox>),
    /// An arbitrary membership test with a caller-declared solid angle.
    Custom {
        /// Membership predicate: `contains(ra_deg, dec_deg)`.
        contains: fn(f64, f64) -> bool,
        /// The region's solid angle in square degrees, as declared by the
        /// caller (not derived from the predicate).
        solid_angle_deg2: f64,
    },
}

/// Solid angle of the full sphere in square degrees (≈ 41 252.96).
pub const FULL_SKY_DEG2: f64 = 4.0 * std::f64::consts::PI * RAD2DEG * RAD2DEG;

impl Footprint {
    /// Whether the equatorial position (degrees) falls inside the
    /// footprint. RA is reduced modulo 360°.
    pub fn contains(&self, ra_deg: f64, dec_deg: f64) -> bool {
        match self {
            Footprint::AllSky => true,
            Footprint::DecRange {
                dec_min_deg,
                dec_max_deg,
            } => (*dec_min_deg..=*dec_max_deg).contains(&dec_deg),
            Footprint::BoxUnion(boxes) => boxes.iter().any(|b| b.contains(ra_deg, dec_deg)),
            Footprint::Custom { contains, .. } => contains(ra_deg, dec_deg),
        }
    }

    /// The footprint's solid angle in square degrees. See the type-level
    /// docs for exactness caveats per variant.
    pub fn solid_angle_deg2(&self) -> f64 {
        match self {
            Footprint::AllSky => FULL_SKY_DEG2,
            Footprint::DecRange {
                dec_min_deg,
                dec_max_deg,
            } => {
                let dsin = dec_max_deg.to_radians().sin() - dec_min_deg.to_radians().sin();
                360.0 * dsin * RAD2DEG
            }
            Footprint::BoxUnion(boxes) => boxes.iter().map(SkyBox::solid_angle_deg2).sum(),
            Footprint::Custom {
                solid_angle_deg2, ..
            } => *solid_angle_deg2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_sky_contains_everything_and_integrates_to_full_sphere() {
        let fp = Footprint::AllSky;
        assert!(fp.contains(0.0, 90.0));
        assert!(fp.contains(359.9, -90.0));
        assert!((fp.solid_angle_deg2() - 41_252.96).abs() < 0.01);
    }

    #[test]
    fn dec_range_solid_angle() {
        // Northern hemisphere: exactly half the sphere.
        let north = Footprint::DecRange {
            dec_min_deg: 0.0,
            dec_max_deg: 90.0,
        };
        assert!((north.solid_angle_deg2() - FULL_SKY_DEG2 / 2.0).abs() < 1e-9);
        // dec > -30: 360 * (1 + sin 30) * 180/pi = 3/4 of the sphere.
        let band = Footprint::DecRange {
            dec_min_deg: -30.0,
            dec_max_deg: 90.0,
        };
        assert!((band.solid_angle_deg2() - 0.75 * FULL_SKY_DEG2).abs() < 1e-9);
        assert!(band.contains(123.0, -29.9));
        assert!(!band.contains(123.0, -30.1));
    }

    #[test]
    fn sky_box_ra_wrap() {
        let b = SkyBox {
            dec_min_deg: -65.0,
            dec_max_deg: -40.0,
            ra_start_deg: 300.0,
            ra_end_deg: 105.0,
        };
        // The eastward range 300 -> 105 wraps through 0.
        assert!((b.ra_span_deg() - 165.0).abs() < 1e-12);
        assert!(b.contains(350.0, -50.0));
        assert!(b.contains(30.0, -55.0));
        assert!(b.contains(100.0, -45.0));
        assert!(!b.contains(150.0, -50.0));
        // RA outside [0, 360) is reduced.
        assert!(b.contains(360.0 + 30.0, -55.0));
        // Half-open declination edges.
        assert!(b.contains(30.0, -65.0));
        assert!(!b.contains(30.0, -40.0));
    }

    #[test]
    fn box_union_solid_angle_monte_carlo() {
        // Cross-check the analytic solid angle of a disjoint union with
        // uniform-sphere sampling.
        use rand::{Rng, SeedableRng};
        let fp = Footprint::BoxUnion(vec![
            SkyBox {
                dec_min_deg: -65.0,
                dec_max_deg: -40.0,
                ra_start_deg: 300.0,
                ra_end_deg: 105.0,
            },
            SkyBox {
                dec_min_deg: -40.0,
                dec_max_deg: -20.0,
                ra_start_deg: 335.0,
                ra_end_deg: 55.0,
            },
            SkyBox {
                dec_min_deg: -20.0,
                dec_max_deg: 5.0,
                ra_start_deg: 355.0,
                ra_end_deg: 40.0,
            },
        ]);
        let analytic = fp.solid_angle_deg2();
        let mut rng = rand::rngs::StdRng::seed_from_u64(2026);
        let n = 200_000;
        let mut hits = 0u32;
        for _ in 0..n {
            let ra = rng.random::<f64>() * 360.0;
            let z: f64 = rng.random::<f64>() * 2.0 - 1.0;
            if fp.contains(ra, z.asin().to_degrees()) {
                hits += 1;
            }
        }
        let mc = FULL_SKY_DEG2 * hits as f64 / n as f64;
        assert!(
            (mc - analytic).abs() < 150.0,
            "MC = {mc:.0} deg^2 vs analytic {analytic:.0} deg^2"
        );
    }

    #[test]
    fn custom_footprint_uses_declared_values() {
        fn south_cap(_ra: f64, dec: f64) -> bool {
            dec < -60.0
        }
        let fp = Footprint::Custom {
            contains: south_cap,
            solid_angle_deg2: 1000.0,
        };
        assert!(fp.contains(10.0, -70.0));
        assert!(!fp.contains(10.0, 0.0));
        assert_eq!(fp.solid_angle_deg2(), 1000.0);
    }
}
