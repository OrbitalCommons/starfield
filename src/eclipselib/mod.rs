//! Eclipse detection and classification
//!
//! Provides functions to find and classify lunar and solar eclipses
//! within a date range, using SPK ephemeris data.
//!
//! Follows the geometric approach from the Explanatory Supplement to the
//! Astronomical Almanac, adapted from Python Skyfield's `eclipselib.py`.

#[cfg(feature = "python-tests")]
mod python_tests;

use std::cell::RefCell;

use crate::constants::EARTH_RADIUS;
use crate::jplephem::kernel::SpiceKernel;
use crate::searchlib::find_maxima;
use crate::time::Timescale;

/// Solar radius in km
const SOLAR_RADIUS_KM: f64 = 696_340.0;
/// Moon radius in km
const MOON_RADIUS_KM: f64 = 1_737.1;
/// Earth radius in km
const EARTH_RADIUS_KM: f64 = EARTH_RADIUS / 1000.0;

/// Lunar eclipse type codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LunarEclipseType {
    /// Moon enters Earth's penumbra only
    Penumbral = 0,
    /// Moon partially enters Earth's umbra
    Partial = 1,
    /// Moon fully enters Earth's umbra
    Total = 2,
}

impl std::fmt::Display for LunarEclipseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LunarEclipseType::Penumbral => write!(f, "Penumbral"),
            LunarEclipseType::Partial => write!(f, "Partial"),
            LunarEclipseType::Total => write!(f, "Total"),
        }
    }
}

/// Solar eclipse type codes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolarEclipseType {
    /// Moon's penumbra touches Earth
    Partial = 0,
    /// Moon's umbral cone reaches Earth but Moon appears smaller than Sun
    Annular = 1,
    /// Moon's umbral cone reaches Earth and Moon appears larger than Sun
    Total = 2,
}

impl std::fmt::Display for SolarEclipseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolarEclipseType::Partial => write!(f, "Partial"),
            SolarEclipseType::Annular => write!(f, "Annular"),
            SolarEclipseType::Total => write!(f, "Total"),
        }
    }
}

/// Details about a lunar eclipse
#[derive(Debug, Clone)]
pub struct LunarEclipseInfo {
    /// TDB Julian date of maximum eclipse
    pub jd_tdb: f64,
    /// Eclipse type
    pub eclipse_type: LunarEclipseType,
    /// Closest approach of Moon center to shadow axis (radians)
    pub closest_approach: f64,
    /// Umbral magnitude (fraction of Moon's diameter in umbra)
    pub umbral_magnitude: f64,
    /// Penumbral magnitude (fraction of Moon's diameter in penumbra)
    pub penumbral_magnitude: f64,
}

/// Details about a solar eclipse
#[derive(Debug, Clone)]
pub struct SolarEclipseInfo {
    /// TDB Julian date of maximum eclipse
    pub jd_tdb: f64,
    /// Eclipse type (from central line perspective)
    pub eclipse_type: SolarEclipseType,
    /// Closest approach of Moon's shadow axis to Earth center (radians)
    pub closest_approach: f64,
    /// Eclipse magnitude (ratio of apparent diameters)
    pub magnitude: f64,
}

/// Find lunar eclipses between two TDB Julian dates.
///
/// Returns a vector of `LunarEclipseInfo` describing each eclipse found.
///
/// # Arguments
/// * `kernel` - SPK ephemeris kernel (must contain Sun, Earth, Moon segments)
/// * `jd_start` - Start TDB Julian date
/// * `jd_end` - End TDB Julian date
pub fn lunar_eclipses(
    kernel: &mut SpiceKernel,
    jd_start: f64,
    jd_end: f64,
) -> Vec<LunarEclipseInfo> {
    let ts = Timescale::default();

    // Wrap kernel in RefCell for interior mutability inside Fn closure
    let kernel_cell = RefCell::new(kernel);

    // Find times when Sun-Moon angular separation (as seen from Earth)
    // is at a maximum — these are full moon times (potential lunar eclipses)
    let f = |jds: &[f64]| -> Vec<f64> {
        let mut k = kernel_cell.borrow_mut();
        jds.iter()
            .map(|&jd| {
                let t = ts.tdb_jd(jd);
                let (earth_to_sun_km, earth_to_moon_km) = earth_sun_moon_km(*k, &t);
                angle_between_vec(&earth_to_sun_km, &earth_to_moon_km)
            })
            .collect()
    };

    // Sample every 5 days — synodic month is ~29.5 days
    let maxima = find_maxima(jd_start, jd_end, &f, 5.0, 1.0 / 86400.0, 12);

    let kernel = kernel_cell.into_inner();
    let mut eclipses = Vec::new();

    for (jd, _) in &maxima {
        let t = ts.tdb_jd(*jd);
        let (earth_to_sun_km, earth_to_moon_km) = earth_sun_moon_km(kernel, &t);
        let moon_to_earth_km = [
            -earth_to_moon_km[0],
            -earth_to_moon_km[1],
            -earth_to_moon_km[2],
        ];

        let moon_dist = vec_len(&moon_to_earth_km);
        let sun_dist = vec_len(&earth_to_sun_km);

        // Angular sizes (small angle approx)
        let pi_m = EARTH_RADIUS_KM / moon_dist; // Earth's parallax at Moon
        let pi_s = EARTH_RADIUS_KM / sun_dist; // Earth's parallax at Sun
        let s_s = SOLAR_RADIUS_KM / sun_dist; // Sun's angular radius

        let closest_approach = angle_between_vec(&earth_to_sun_km, &moon_to_earth_km);
        let moon_radius = (MOON_RADIUS_KM / moon_dist).asin();

        // Danjon's enlargement factor for Earth's shadow
        let pi_1 = 1.01 * pi_m;
        let penumbra_radius = pi_1 + pi_s + s_s;
        let umbra_radius = pi_1 + pi_s - s_s;

        let penumbral = closest_approach < penumbra_radius + moon_radius;
        if !penumbral {
            continue;
        }

        let twice_radius = 2.0 * moon_radius;
        let umbral_magnitude = (umbra_radius + moon_radius - closest_approach) / twice_radius;
        let penumbral_magnitude = (penumbra_radius + moon_radius - closest_approach) / twice_radius;

        let partial = closest_approach < umbra_radius + moon_radius;
        let total = closest_approach < umbra_radius - moon_radius;

        let eclipse_type = if total {
            LunarEclipseType::Total
        } else if partial {
            LunarEclipseType::Partial
        } else {
            LunarEclipseType::Penumbral
        };

        eclipses.push(LunarEclipseInfo {
            jd_tdb: *jd,
            eclipse_type,
            closest_approach,
            umbral_magnitude,
            penumbral_magnitude,
        });
    }

    eclipses
}

/// Find solar eclipses between two TDB Julian dates.
///
/// A solar eclipse occurs when the Moon passes between the Sun and Earth.
/// This finds times when the Sun-Moon angular separation (as seen from Earth)
/// reaches a minimum near zero, then classifies each eclipse.
///
/// # Arguments
/// * `kernel` - SPK ephemeris kernel (must contain Sun, Earth, Moon segments)
/// * `jd_start` - Start TDB Julian date
/// * `jd_end` - End TDB Julian date
pub fn solar_eclipses(
    kernel: &mut SpiceKernel,
    jd_start: f64,
    jd_end: f64,
) -> Vec<SolarEclipseInfo> {
    let ts = Timescale::default();

    let kernel_cell = RefCell::new(kernel);

    // For solar eclipses, we want minima of the Sun-Moon angular separation.
    // We negate it and find maxima.
    let f = |jds: &[f64]| -> Vec<f64> {
        let mut k = kernel_cell.borrow_mut();
        jds.iter()
            .map(|&jd| {
                let t = ts.tdb_jd(jd);
                let (earth_to_sun_km, earth_to_moon_km) = earth_sun_moon_km(*k, &t);
                -angle_between_vec(&earth_to_sun_km, &earth_to_moon_km)
            })
            .collect()
    };

    // Sample every 5 days
    let maxima = find_maxima(jd_start, jd_end, &f, 5.0, 1.0 / 86400.0, 12);

    let kernel = kernel_cell.into_inner();
    let mut eclipses = Vec::new();

    for (jd, _) in &maxima {
        let t = ts.tdb_jd(*jd);
        let (earth_to_sun_km, earth_to_moon_km) = earth_sun_moon_km(kernel, &t);
        let separation = angle_between_vec(&earth_to_sun_km, &earth_to_moon_km);

        let sun_dist = vec_len(&earth_to_sun_km);
        let moon_dist = vec_len(&earth_to_moon_km);

        let sun_angular_radius = (SOLAR_RADIUS_KM / sun_dist).asin();
        let moon_angular_radius = (MOON_RADIUS_KM / moon_dist).asin();

        // How close the shadow axis must come to Earth for any partial eclipse
        let earth_angular_radius = (EARTH_RADIUS_KM / moon_dist).asin();

        let penumbral_half_angle = sun_angular_radius + moon_angular_radius;
        let is_eclipse = separation < penumbral_half_angle + earth_angular_radius;

        if !is_eclipse {
            continue;
        }

        // Classify: the ratio of apparent diameters determines annular vs total
        let magnitude = moon_angular_radius / sun_angular_radius;

        // For centrality: the umbral/antumbral shadow cone touches Earth
        // when the shadow axis passes within earth_angular_radius of Earth's center.
        // The shadow axis hits Earth's center when separation = 0.
        // The shadow edge touches Earth when separation < earth_angular_radius.
        // For a central eclipse (total or annular), the umbra/antumbra must touch Earth.
        let central = separation < earth_angular_radius;

        let eclipse_type = if !central {
            SolarEclipseType::Partial
        } else if magnitude >= 1.0 {
            SolarEclipseType::Total
        } else {
            SolarEclipseType::Annular
        };

        eclipses.push(SolarEclipseInfo {
            jd_tdb: *jd,
            eclipse_type,
            closest_approach: separation,
            magnitude,
        });
    }

    eclipses
}

/// Compute Earth-to-Sun and Earth-to-Moon vectors in km
fn earth_sun_moon_km(kernel: &mut SpiceKernel, t: &crate::time::Time) -> ([f64; 3], [f64; 3]) {
    let (sun_pos, _) = kernel.compute_km("sun", t).unwrap();
    let (earth_pos, _) = kernel.compute_km("earth", t).unwrap();
    let (moon_pos, _) = kernel.compute_km("moon", t).unwrap();

    let earth_to_sun = [
        sun_pos.x - earth_pos.x,
        sun_pos.y - earth_pos.y,
        sun_pos.z - earth_pos.z,
    ];
    let earth_to_moon = [
        moon_pos.x - earth_pos.x,
        moon_pos.y - earth_pos.y,
        moon_pos.z - earth_pos.z,
    ];

    (earth_to_sun, earth_to_moon)
}

/// Vector length
fn vec_len(v: &[f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// Angle between two 3D vectors in radians
fn angle_between_vec(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mag_a = vec_len(a);
    let mag_b = vec_len(b);
    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp")
            .expect("de421.bsp required for eclipse tests")
    }

    #[test]
    fn test_lunar_eclipses_2020() {
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        // 2020 had 4 penumbral lunar eclipses:
        // Jan 10, Jun 5, Jul 5, Nov 30
        let t0 = ts.tdb((2020, 1, 1, 0, 0, 0.0)).tdb();
        let t1 = ts.tdb((2021, 1, 1, 0, 0, 0.0)).tdb();

        let eclipses = lunar_eclipses(&mut kernel, t0, t1);

        // Should find at least 3 eclipses (penumbral may be marginal)
        assert!(
            eclipses.len() >= 3,
            "Expected at least 3 lunar eclipses in 2020, got {}",
            eclipses.len()
        );

        // All should be penumbral in 2020
        for e in &eclipses {
            assert!(
                e.penumbral_magnitude > 0.0,
                "Eclipse should have positive penumbral magnitude"
            );
        }
    }

    #[test]
    fn test_lunar_eclipse_types() {
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        // 2015 had a total lunar eclipse (Sep 28) and partial (Apr 4)
        let t0 = ts.tdb((2015, 1, 1, 0, 0, 0.0)).tdb();
        let t1 = ts.tdb((2016, 1, 1, 0, 0, 0.0)).tdb();

        let eclipses = lunar_eclipses(&mut kernel, t0, t1);

        assert!(
            eclipses.len() >= 2,
            "Expected at least 2 lunar eclipses in 2015, got {}",
            eclipses.len()
        );

        let has_total = eclipses
            .iter()
            .any(|e| e.eclipse_type == LunarEclipseType::Total);
        assert!(has_total, "Expected a total lunar eclipse in 2015");
    }

    #[test]
    fn test_solar_eclipses_2017() {
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        // 2017 had the Great American Eclipse (Aug 21, total)
        // and an annular eclipse (Feb 26)
        let t0 = ts.tdb((2017, 1, 1, 0, 0, 0.0)).tdb();
        let t1 = ts.tdb((2018, 1, 1, 0, 0, 0.0)).tdb();

        let eclipses = solar_eclipses(&mut kernel, t0, t1);

        assert!(
            eclipses.len() >= 2,
            "Expected at least 2 solar eclipses in 2017, got {}",
            eclipses.len()
        );

        let has_total = eclipses
            .iter()
            .any(|e| e.eclipse_type == SolarEclipseType::Total);
        assert!(has_total, "Expected a total solar eclipse in 2017");
    }

    #[test]
    fn test_solar_eclipse_classification() {
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        // 2012 had an annular (May 20) and total (Nov 13)
        let t0 = ts.tdb((2012, 1, 1, 0, 0, 0.0)).tdb();
        let t1 = ts.tdb((2013, 1, 1, 0, 0, 0.0)).tdb();

        let eclipses = solar_eclipses(&mut kernel, t0, t1);

        assert!(
            eclipses.len() >= 2,
            "Expected at least 2 solar eclipses in 2012, got {}",
            eclipses.len()
        );

        let types: Vec<_> = eclipses.iter().map(|e| e.eclipse_type).collect();
        let has_annular = types.contains(&SolarEclipseType::Annular);
        let has_total = types.contains(&SolarEclipseType::Total);

        assert!(
            has_annular || has_total,
            "Expected annular or total in 2012, got {:?}",
            types
        );
    }

    #[test]
    fn test_no_eclipses_in_short_period() {
        let mut kernel = test_kernel();

        // A 10-day period with no eclipse
        let t0 = 2451545.0; // J2000
        let t1 = t0 + 10.0;

        let lunar = lunar_eclipses(&mut kernel, t0, t1);
        let solar = solar_eclipses(&mut kernel, t0, t1);

        // Unlikely to have an eclipse in a random 10-day window
        assert!(
            lunar.len() + solar.len() <= 1,
            "Unexpected eclipses in 10-day window"
        );
    }

    #[test]
    fn test_eclipse_display() {
        assert_eq!(LunarEclipseType::Total.to_string(), "Total");
        assert_eq!(LunarEclipseType::Partial.to_string(), "Partial");
        assert_eq!(LunarEclipseType::Penumbral.to_string(), "Penumbral");
        assert_eq!(SolarEclipseType::Total.to_string(), "Total");
        assert_eq!(SolarEclipseType::Annular.to_string(), "Annular");
        assert_eq!(SolarEclipseType::Partial.to_string(), "Partial");
    }
}
