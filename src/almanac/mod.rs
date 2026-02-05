//! Astronomical almanac functions for finding celestial events
//!
//! Ported from Python Skyfield's `almanac.py`. Provides practical functions that
//! answer "when does X happen?" — seasons, moon phases, sunrise/sunset, twilight.
//!
//! All event-finding functions return closures suitable for use with
//! [`find_discrete`](crate::searchlib::find_discrete).
//!
//! # Example
//!
//! ```ignore
//! let mut kernel = SpiceKernel::open("de421.bsp")?;
//! let ts = Timescale::default();
//!
//! // Find seasons in 2005
//! let t0 = ts.tt((2005, 1, 1)).tdb();
//! let t1 = ts.tt((2006, 1, 1)).tdb();
//! let f = seasons(&mut kernel);
//! let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);
//! for (jd, season) in &events {
//!     println!("{}: {}", jd, SEASON_NAMES[*season as usize]);
//! }
//! ```

#[cfg(feature = "python-tests")]
mod python_tests;

use std::f64::consts::PI;

use crate::constants::TAU;
use crate::jplephem::kernel::SpiceKernel;
use crate::time::Timescale;

/// Sun's apparent angular radius plus standard refraction (50 arcminutes)
const SUN_HORIZON_DEGREES: f64 = -50.0 / 60.0;

/// Standard atmospheric refraction at the horizon (34 arcminutes)
pub const REFRACTION_DEGREES: f64 = -34.0 / 60.0;

/// Human-readable season names indexed by season number 0..3
pub const SEASON_NAMES: &[&str] = &[
    "Vernal Equinox",
    "Summer Solstice",
    "Autumnal Equinox",
    "Winter Solstice",
];

/// Human-readable moon phase names indexed by phase number 0..3
pub const MOON_PHASE_NAMES: &[&str] = &["New Moon", "First Quarter", "Full Moon", "Last Quarter"];

/// Human-readable twilight state names indexed by state number 0..4
pub const TWILIGHT_NAMES: &[&str] = &[
    "Night",
    "Astronomical Twilight",
    "Nautical Twilight",
    "Civil Twilight",
    "Day",
];

/// Compute the ecliptic longitude of a target body as seen from Earth at the
/// given TDB Julian dates. Returns longitudes in radians [0, 2*PI).
fn body_ecliptic_longitude(kernel: &mut SpiceKernel, target: &str, jd_tdb: &[f64]) -> Vec<f64> {
    let ts = Timescale::default();

    jd_tdb
        .iter()
        .map(|&jd| {
            let t = ts.tdb_jd(jd);
            let earth = kernel.at("earth", &t).unwrap();
            let astro = earth.observe(target, kernel, &t).unwrap();
            let app = astro.apparent(kernel, &t).unwrap();
            let (lon, _lat, _dist) = app.ecliptic_latlon(&t);
            lon
        })
        .collect()
}

/// Compute the ecliptic longitude of the Sun as seen from Earth at the given
/// TDB Julian dates.
///
/// Used internally by [`seasons`] and can be called directly for custom
/// ecliptic longitude queries. Returns longitudes in radians [0, 2*PI).
pub fn sun_ecliptic_longitude(kernel: &mut SpiceKernel, jd_tdb: &[f64]) -> Vec<f64> {
    body_ecliptic_longitude(kernel, "sun", jd_tdb)
}

/// Compute the ecliptic longitude of the Moon as seen from Earth at the given
/// TDB Julian dates. Returns longitudes in radians [0, 2*PI).
pub fn moon_ecliptic_longitude(kernel: &mut SpiceKernel, jd_tdb: &[f64]) -> Vec<f64> {
    body_ecliptic_longitude(kernel, "moon", jd_tdb)
}

/// Return a closure that computes the season index (0..3) from the Sun's
/// ecliptic longitude.
///
/// The returned function maps TDB Julian dates to season indices:
/// - 0 = Vernal Equinox (Sun enters 0° ecliptic longitude)
/// - 1 = Summer Solstice (Sun enters 90°)
/// - 2 = Autumnal Equinox (Sun enters 180°)
/// - 3 = Winter Solstice (Sun enters 270°)
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 90.0`.
pub fn seasons(kernel: &mut SpiceKernel) -> impl FnMut(&[f64]) -> Vec<i64> + '_ {
    move |jd_tdb: &[f64]| {
        let lons = sun_ecliptic_longitude(kernel, jd_tdb);
        lons.iter()
            .map(|&lon| (lon / (TAU / 4.0)).floor() as i64 % 4)
            .collect()
    }
}

/// Compute the continuous moon phase angle (0°..360°) at the given TDB
/// Julian dates.
///
/// The phase angle is the ecliptic longitude difference between Moon and Sun:
/// - 0° = New Moon
/// - 90° = First Quarter
/// - 180° = Full Moon
/// - 270° = Last Quarter
pub fn moon_phase_angle(kernel: &mut SpiceKernel, jd_tdb: &[f64]) -> Vec<f64> {
    let ts = Timescale::default();

    jd_tdb
        .iter()
        .map(|&jd| {
            let t = ts.tdb_jd(jd);
            let earth = kernel.at("earth", &t).unwrap();

            let sun_astro = earth.observe("sun", kernel, &t).unwrap();
            let sun_app = sun_astro.apparent(kernel, &t).unwrap();
            let (sun_lon, _, _) = sun_app.ecliptic_latlon(&t);

            let moon_astro = earth.observe("moon", kernel, &t).unwrap();
            let moon_app = moon_astro.apparent(kernel, &t).unwrap();
            let (moon_lon, _, _) = moon_app.ecliptic_latlon(&t);

            let mut phase = moon_lon - sun_lon;
            if phase < 0.0 {
                phase += TAU;
            }
            phase.to_degrees()
        })
        .collect()
}

/// Return a closure that computes the moon phase index (0..3) from the
/// Moon-Sun ecliptic longitude difference.
///
/// The returned function maps TDB Julian dates to phase indices:
/// - 0 = New Moon (0°..90°)
/// - 1 = First Quarter (90°..180°)
/// - 2 = Full Moon (180°..270°)
/// - 3 = Last Quarter (270°..360°)
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 7.0`.
pub fn moon_phases(kernel: &mut SpiceKernel) -> impl FnMut(&[f64]) -> Vec<i64> + '_ {
    move |jd_tdb: &[f64]| {
        let angles = moon_phase_angle(kernel, jd_tdb);
        angles
            .iter()
            .map(|&deg| (deg / 90.0).floor() as i64 % 4)
            .collect()
    }
}

/// Compute the altitude of the Sun as seen from an observer at the given
/// TDB Julian dates. Returns altitude in degrees.
fn sun_altitude(
    kernel: &mut SpiceKernel,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
    jd_tdb: &[f64],
) -> Vec<f64> {
    body_altitude(
        kernel,
        "sun",
        latitude_deg,
        longitude_deg,
        elevation_m,
        jd_tdb,
    )
}

/// Return a closure that computes whether the Sun is above the horizon (1)
/// or below (0), using the standard Sun depression angle of -50 arcminutes
/// (accounting for refraction and solar radius).
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 0.5`.
pub fn sunrise_sunset(
    kernel: &mut SpiceKernel,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
) -> impl FnMut(&[f64]) -> Vec<i64> + '_ {
    move |jd_tdb: &[f64]| {
        let alts = sun_altitude(kernel, latitude_deg, longitude_deg, elevation_m, jd_tdb);
        alts.iter()
            .map(|&alt| if alt >= SUN_HORIZON_DEGREES { 1 } else { 0 })
            .collect()
    }
}

/// Return a closure that classifies the sky brightness into 5 states
/// based on the Sun's altitude below the horizon:
///
/// - 0 = Night (Sun below -18°)
/// - 1 = Astronomical Twilight (-18° to -12°)
/// - 2 = Nautical Twilight (-12° to -6°)
/// - 3 = Civil Twilight (-6° to -0.8333°)
/// - 4 = Day (Sun above -0.8333°)
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 0.5`.
pub fn dark_twilight_day(
    kernel: &mut SpiceKernel,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
) -> impl FnMut(&[f64]) -> Vec<i64> + '_ {
    move |jd_tdb: &[f64]| {
        let alts = sun_altitude(kernel, latitude_deg, longitude_deg, elevation_m, jd_tdb);
        alts.iter()
            .map(|&alt| {
                if alt >= SUN_HORIZON_DEGREES {
                    4 // Day
                } else if alt >= -6.0 {
                    3 // Civil twilight
                } else if alt >= -12.0 {
                    2 // Nautical twilight
                } else if alt >= -18.0 {
                    1 // Astronomical twilight
                } else {
                    0 // Night
                }
            })
            .collect()
    }
}

/// Compute the altitude of a target body as seen from an observer at the
/// given TDB Julian dates. Returns altitude in degrees.
fn body_altitude(
    kernel: &mut SpiceKernel,
    target_name: &str,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
    jd_tdb: &[f64],
) -> Vec<f64> {
    let ts = Timescale::default();
    let geoid = crate::toposlib::WGS84;
    let observer = geoid.latlon(latitude_deg, longitude_deg, elevation_m);

    jd_tdb
        .iter()
        .map(|&jd| {
            let t = ts.tdb_jd(jd);
            let obs_pos = observer.at(&t, kernel).unwrap();
            let target_astro = obs_pos.observe(target_name, kernel, &t).unwrap();
            let target_app = target_astro.apparent(kernel, &t).unwrap();
            let (alt, _az, _dist) = observer.altaz(&target_app, &t);
            alt
        })
        .collect()
}

/// Return a closure that computes whether a target body is above the
/// horizon (1) or below (0).
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 0.5`.
pub fn risings_and_settings<'a>(
    kernel: &'a mut SpiceKernel,
    target_name: &str,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
    horizon_degrees: f64,
) -> impl FnMut(&[f64]) -> Vec<i64> + 'a {
    let target = target_name.to_string();

    move |jd_tdb: &[f64]| {
        let alts = body_altitude(
            kernel,
            &target,
            latitude_deg,
            longitude_deg,
            elevation_m,
            jd_tdb,
        );
        alts.iter()
            .map(|&alt| if alt >= horizon_degrees { 1 } else { 0 })
            .collect()
    }
}

/// Compute the hour angle of a target body as seen from an observer
/// at the given TDB Julian dates. Returns hour angle in degrees (0-360).
fn body_hour_angle(
    kernel: &mut SpiceKernel,
    target_name: &str,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
    jd_tdb: &[f64],
) -> Vec<f64> {
    let ts = Timescale::default();
    let geoid = crate::toposlib::WGS84;
    let observer = geoid.latlon(latitude_deg, longitude_deg, elevation_m);

    jd_tdb
        .iter()
        .map(|&jd| {
            let t = ts.tdb_jd(jd);
            let obs_pos = observer.at(&t, kernel).unwrap();
            let target_astro = obs_pos.observe(target_name, kernel, &t).unwrap();
            let target_app = target_astro.apparent(kernel, &t).unwrap();
            let (ra_h, _dec_d, _dist) = target_app.radec(Some(&t));
            let lst = observer.lst_hours(&t);

            let mut ha = lst - ra_h;
            if ha < 0.0 {
                ha += 24.0;
            }
            ha * 15.0
        })
        .collect()
}

/// Return a closure that detects meridian transits.
///
/// Returns 1 when the body is west of the meridian (hour angle 0-180°),
/// 0 when east (180-360°). Transitions from 0 to 1 mark upper transits.
///
/// Use with [`find_discrete`](crate::searchlib::find_discrete) with
/// `step_days = 0.5`.
pub fn meridian_transits<'a>(
    kernel: &'a mut SpiceKernel,
    target_name: &str,
    latitude_deg: f64,
    longitude_deg: f64,
    elevation_m: f64,
) -> impl FnMut(&[f64]) -> Vec<i64> + 'a {
    let target = target_name.to_string();

    move |jd_tdb: &[f64]| {
        let has = body_hour_angle(
            kernel,
            &target,
            latitude_deg,
            longitude_deg,
            elevation_m,
            jd_tdb,
        );
        has.iter()
            .map(|&ha| if ha < 180.0 { 1 } else { 0 })
            .collect()
    }
}

/// Compute the opposition/conjunction state of a planet relative to the Sun.
///
/// Returns a closure that yields 0 when the target's ecliptic longitude
/// is within 180° ahead of the Sun, and 1 otherwise. The 0 to 1 transition
/// marks opposition, 1 to 0 marks conjunction.
pub fn oppositions_conjunctions<'a>(
    kernel: &'a mut SpiceKernel,
    target_name: &str,
) -> impl FnMut(&[f64]) -> Vec<i64> + 'a {
    let target = target_name.to_string();

    move |jd_tdb: &[f64]| {
        let ts = Timescale::default();
        jd_tdb
            .iter()
            .map(|&jd| {
                let t = ts.tdb_jd(jd);
                let earth = kernel.at("earth", &t).unwrap();

                let sun_astro = earth.observe("sun", kernel, &t).unwrap();
                let sun_app = sun_astro.apparent(kernel, &t).unwrap();
                let (sun_lon, _, _) = sun_app.ecliptic_latlon(&t);

                let target_astro = earth.observe(&target, kernel, &t).unwrap();
                let target_app = target_astro.apparent(kernel, &t).unwrap();
                let (target_lon, _, _) = target_app.ecliptic_latlon(&t);

                let mut diff = target_lon - sun_lon;
                if diff < 0.0 {
                    diff += TAU;
                }
                if diff < PI {
                    0
                } else {
                    1
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::searchlib::{find_discrete, DEFAULT_NUM, EPSILON_DISCRETE};
    use crate::time::Timescale;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    // --- Sun ecliptic longitude ---

    #[test]
    fn test_sun_ecliptic_longitude_at_j2000() {
        let mut kernel = de421_kernel();
        // At J2000 (Jan 1.5 2000), Sun ecliptic longitude should be ~280°
        let lons = sun_ecliptic_longitude(&mut kernel, &[2_451_545.0]);
        let lon_deg = lons[0].to_degrees();
        assert!(
            lon_deg > 270.0 && lon_deg < 290.0,
            "Sun ecliptic longitude at J2000 should be ~280°, got {:.1}°",
            lon_deg
        );
    }

    #[test]
    fn test_sun_ecliptic_longitude_increases() {
        let mut kernel = de421_kernel();
        let jds: Vec<f64> = (0..10).map(|i| 2_451_545.0 + i as f64 * 30.0).collect();
        let lons = sun_ecliptic_longitude(&mut kernel, &jds);

        for i in 1..lons.len() {
            let diff = (lons[i] - lons[i - 1]).rem_euclid(TAU);
            assert!(
                diff > 0.0 && diff < PI,
                "Sun longitude should increase: diff = {:.4} rad at step {}",
                diff,
                i
            );
        }
    }

    // --- Seasons ---

    #[test]
    fn test_seasons_produces_four_events_in_year() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            4,
            "Should find 4 seasonal transitions in a year, got {}",
            events.len()
        );

        let season_vals: Vec<i64> = events.iter().map(|e| e.1).collect();
        assert!(season_vals.contains(&0), "Missing Vernal Equinox");
        assert!(season_vals.contains(&1), "Missing Summer Solstice");
        assert!(season_vals.contains(&2), "Missing Autumnal Equinox");
        assert!(season_vals.contains(&3), "Missing Winter Solstice");
    }

    #[test]
    fn test_seasons_vernal_equinox_near_march_20() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);

        let ve = events
            .iter()
            .find(|e| e.1 == 0)
            .expect("No vernal equinox found");
        let march_20_jd = ts.tt((2005, 3, 20)).tdb();
        let diff_days = (ve.0 - march_20_jd).abs();
        assert!(
            diff_days < 1.5,
            "Vernal equinox should be near March 20, diff = {:.2} days",
            diff_days
        );
    }

    #[test]
    fn test_seasons_summer_solstice_near_june_21() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);

        let ss = events
            .iter()
            .find(|e| e.1 == 1)
            .expect("No summer solstice found");
        let june_21_jd = ts.tt((2005, 6, 21)).tdb();
        let diff_days = (ss.0 - june_21_jd).abs();
        assert!(
            diff_days < 1.5,
            "Summer solstice should be near June 21, diff = {:.2} days",
            diff_days
        );
    }

    // --- Moon phases ---

    #[test]
    fn test_moon_phase_angle_range() {
        let mut kernel = de421_kernel();
        let jds: Vec<f64> = (0..30).map(|i| 2_453_371.0 + i as f64).collect();
        let angles = moon_phase_angle(&mut kernel, &jds);

        for &angle in &angles {
            assert!(
                angle >= 0.0 && angle < 360.0,
                "Moon phase angle should be in [0, 360), got {}",
                angle
            );
        }
    }

    #[test]
    fn test_moon_phases_about_four_per_month() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2005, 2, 1)).tdb();

        let mut f = moon_phases(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 7.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert!(
            events.len() >= 3 && events.len() <= 5,
            "Should find 3-5 phase transitions per month, got {}",
            events.len()
        );
    }

    #[test]
    fn test_moon_phases_finds_all_types_in_two_months() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2005, 3, 1)).tdb();

        let mut f = moon_phases(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 7.0, EPSILON_DISCRETE, DEFAULT_NUM);

        let phase_vals: Vec<i64> = events.iter().map(|e| e.1).collect();
        assert!(phase_vals.contains(&0), "Missing New Moon");
        assert!(phase_vals.contains(&1), "Missing First Quarter");
        assert!(phase_vals.contains(&2), "Missing Full Moon");
        assert!(phase_vals.contains(&3), "Missing Last Quarter");
    }

    // --- Moon ecliptic longitude ---

    #[test]
    fn test_moon_ecliptic_longitude_at_j2000() {
        let mut kernel = de421_kernel();
        let lons = moon_ecliptic_longitude(&mut kernel, &[2_451_545.0]);
        let lon_deg = lons[0].to_degrees();
        assert!(
            lon_deg >= 0.0 && lon_deg < 360.0,
            "Moon ecliptic longitude should be in [0, 360), got {:.1}°",
            lon_deg
        );
    }

    // --- Sunrise / sunset ---

    #[test]
    fn test_sunrise_sunset_finds_events() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = sunrise_sunset(&mut kernel, 40.7128, -74.0060, 0.0);
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        assert!(
            events.len() >= 4 && events.len() <= 8,
            "Should find 4-8 sunrise/sunset events in 3 days, got {}",
            events.len()
        );

        for i in 1..events.len() {
            assert_ne!(
                events[i].1,
                events[i - 1].1,
                "Events should alternate, got {} then {}",
                events[i - 1].1,
                events[i].1
            );
        }
    }

    // --- Twilight ---

    #[test]
    fn test_dark_twilight_day_mid_latitude() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 22)).tdb();

        let mut f = dark_twilight_day(&mut kernel, 40.7128, -74.0060, 0.0);
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        assert!(
            events.len() >= 4,
            "Should find at least 4 twilight transitions in 2 days, got {}",
            events.len()
        );

        for (_, state) in &events {
            assert!(
                *state >= 0 && *state <= 4,
                "Twilight state should be 0-4, got {}",
                state
            );
        }
    }

    // --- Opposition/conjunction ---

    #[test]
    fn test_oppositions_conjunctions_mars() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2003, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = oppositions_conjunctions(&mut kernel, "mars");
        let events = find_discrete(t0, t1, &mut f, 30.0, EPSILON_DISCRETE, DEFAULT_NUM);

        let oppositions: Vec<_> = events.iter().filter(|e| e.1 == 1).collect();
        assert!(
            !oppositions.is_empty(),
            "Should find at least one Mars opposition in 3 years"
        );
    }

    // --- Meridian transit ---

    #[test]
    fn test_meridian_transits_sun() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = meridian_transits(&mut kernel, "sun", 40.7128, -74.0060, 0.0);
        let events = find_discrete(t0, t1, &mut f, 0.5, EPSILON_DISCRETE, DEFAULT_NUM);

        let upper_transits: Vec<_> = events.iter().filter(|e| e.1 == 1).collect();
        assert!(
            upper_transits.len() >= 2 && upper_transits.len() <= 4,
            "Should find 2-4 upper transits of Sun in 3 days, got {}",
            upper_transits.len()
        );
    }

    // --- Risings and settings ---

    #[test]
    fn test_risings_and_settings_moon() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = risings_and_settings(
            &mut kernel,
            "moon",
            40.7128,
            -74.0060,
            0.0,
            REFRACTION_DEGREES,
        );
        // step_days = 0.25 matches Skyfield's risings_and_settings default
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        assert!(
            events.len() >= 3,
            "Should find at least 3 moonrise/set events in 3 days, got {}",
            events.len()
        );
    }

    // --- Edge cases ---

    #[test]
    fn test_season_names_valid() {
        assert_eq!(SEASON_NAMES.len(), 4);
        assert_eq!(SEASON_NAMES[0], "Vernal Equinox");
        assert_eq!(SEASON_NAMES[3], "Winter Solstice");
    }

    #[test]
    fn test_moon_phase_names_valid() {
        assert_eq!(MOON_PHASE_NAMES.len(), 4);
        assert_eq!(MOON_PHASE_NAMES[0], "New Moon");
        assert_eq!(MOON_PHASE_NAMES[2], "Full Moon");
    }

    #[test]
    fn test_twilight_names_valid() {
        assert_eq!(TWILIGHT_NAMES.len(), 5);
        assert_eq!(TWILIGHT_NAMES[0], "Night");
        assert_eq!(TWILIGHT_NAMES[4], "Day");
    }
}
