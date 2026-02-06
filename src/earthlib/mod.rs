//! Earth rotation, sidereal time, polar motion, and atmospheric refraction
//!
//! Implements the Earth Rotation Angle (ERA) per IAU 2000 Resolution B1.8,
//! Greenwich Mean Sidereal Time (GMST) per USNO Circular 179, Section 2.6.2,
//! polar motion from IERS finals2000A data,
//! and atmospheric refraction per the Bennett (1982) formula.

use crate::constants::J2000;

#[cfg(feature = "python-tests")]
mod python_tests;

/// Compute the Earth Rotation Angle (ERA) for a UT1 date
///
/// Uses the expression from IAU Resolution B1.8 of 2000.
/// Returns a fraction between 0.0 and 1.0 representing whole rotations.
///
/// # Arguments
/// * `jd_ut1` - UT1 Julian date (whole part)
/// * `fraction_ut1` - UT1 Julian date (fractional part)
pub fn earth_rotation_angle(jd_ut1: f64, fraction_ut1: f64) -> f64 {
    let th = 0.7790572732640 + 0.00273781191135448 * (jd_ut1 - J2000 + fraction_ut1);
    (th.rem_euclid(1.0) + jd_ut1.rem_euclid(1.0) + fraction_ut1).rem_euclid(1.0)
}

/// Compute Greenwich Mean Sidereal Time (GMST) in hours
///
/// Follows the "equinox method" from USNO Circular 179, Section 2.6.2.
/// Precession-in-RA terms are from Capitaine et al. (2003), eq. (42).
///
/// # Arguments
/// * `jd_ut1_whole` - UT1 Julian date (whole part) for ERA
/// * `ut1_fraction` - UT1 Julian date (fractional part) for ERA
/// * `tdb_centuries` - TDB time in Julian centuries since J2000.0
pub fn sidereal_time(jd_ut1_whole: f64, ut1_fraction: f64, tdb_centuries: f64) -> f64 {
    let theta = earth_rotation_angle(jd_ut1_whole, ut1_fraction);

    // Precession-in-RA terms in mean sidereal time
    // Coefficients in arcseconds, from USNO Circular 179 / Capitaine et al. (2003)
    let t = tdb_centuries;
    let st = 0.014506
        + ((((-0.0000000368 * t - 0.000029956) * t - 0.00000044) * t + 1.3915817) * t
            + 4612.156534)
            * t;

    // Convert: st is in arcseconds, divide by 54000 to get hours
    // theta is in rotations, multiply by 24 to get hours
    (st / 54000.0 + theta * 24.0).rem_euclid(24.0)
}

/// Parse IERS finals2000A.all file to extract polar motion data.
///
/// Returns `(utc_mjd, x_arcseconds, y_arcseconds)` vectors.
/// The finals2000A format uses fixed-width columns:
/// - Columns 7-15: UTC Modified Julian Date
/// - Columns 18-27: Polar motion X (arcseconds)
/// - Columns 37-46: Polar motion Y (arcseconds)
pub fn parse_finals2000a(data: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut mjds = Vec::new();
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    for line in data.lines() {
        if line.len() < 68 {
            continue;
        }
        let mjd_str = line.get(7..15).unwrap_or("").trim();
        let x_str = line.get(18..27).unwrap_or("").trim();
        let y_str = line.get(37..46).unwrap_or("").trim();

        if let (Ok(mjd), Ok(x), Ok(y)) = (
            mjd_str.parse::<f64>(),
            x_str.parse::<f64>(),
            y_str.parse::<f64>(),
        ) {
            mjds.push(mjd);
            xs.push(x);
            ys.push(y);
        }
    }

    (mjds, xs, ys)
}

/// Convert IERS finals data to polar motion table format (TT Julian dates)
///
/// Converts MJD dates to TT Julian dates by adding the MJD epoch offset
/// and an approximate TT-UTC offset.
pub fn finals_to_polar_motion_table(
    mjds: &[f64],
    xs: &[f64],
    ys: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // MJD → JD: add 2400000.5
    // UTC → TT: approximate with 32.184 + leap_seconds (~37s in modern era)
    // For interpolation purposes, ~1 second error in the time index is negligible
    let tt_jds: Vec<f64> = mjds
        .iter()
        .map(|&mjd| mjd + 2400000.5 + 69.184 / 86400.0)
        .collect();
    (tt_jds, xs.to_vec(), ys.to_vec())
}

const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Compute atmospheric refraction for an observed altitude.
///
/// Uses the Skyfield/Bennett formula: given the altitude at which a body
/// is observed, returns the amount by which the atmosphere has raised it.
///
/// # Arguments
/// * `alt_degrees` — Observed altitude above horizon in degrees
/// * `temperature_c` — Air temperature in Celsius
/// * `pressure_mbar` — Atmospheric pressure in millibars
///
/// Returns refraction in degrees. Returns 0.0 outside the range [-1°, 89.9°].
pub fn refraction(alt_degrees: f64, temperature_c: f64, pressure_mbar: f64) -> f64 {
    if !(-1.0..=89.9).contains(&alt_degrees) {
        return 0.0;
    }
    let r = 0.016667 / ((alt_degrees + 7.31 / (alt_degrees + 4.4)) * DEG2RAD).tan();
    r * (0.28 * pressure_mbar / (temperature_c + 273.0))
}

/// Apply atmospheric refraction to a true (geometric) altitude.
///
/// Given the true altitude of a body (before refraction), returns the
/// altitude at which it appears due to atmospheric bending.
///
/// Uses iterative refinement (converges in 3-4 iterations).
///
/// # Arguments
/// * `alt_degrees` — True geometric altitude in degrees
/// * `temperature_c` — Air temperature in Celsius
/// * `pressure_mbar` — Atmospheric pressure in millibars
pub fn refract(alt_degrees: f64, temperature_c: f64, pressure_mbar: f64) -> f64 {
    let mut refracted = alt_degrees;
    for _ in 0..10 {
        let delta = refraction(refracted, temperature_c, pressure_mbar);
        let new = alt_degrees + delta;
        if (new - refracted).abs() < 3.0e-5 {
            return new;
        }
        refracted = new;
    }
    refracted
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_era_at_j2000() {
        let era = earth_rotation_angle(J2000, 0.0);
        assert_relative_eq!(era, 0.7790572732640, epsilon = 1e-10);
    }

    #[test]
    fn test_era_range() {
        for offset in &[-1000.0, -100.0, 0.0, 100.0, 1000.0] {
            let era = earth_rotation_angle(J2000 + offset, 0.0);
            assert!(
                (0.0..1.0).contains(&era),
                "ERA out of range for offset {offset}: {era}"
            );
        }
    }

    #[test]
    fn test_gmst_at_j2000() {
        let gmst = sidereal_time(J2000, 0.0, 0.0);
        assert_relative_eq!(gmst, 18.697, epsilon = 0.01);
    }

    #[test]
    fn test_gmst_range() {
        for offset in &[-1000.0, -100.0, 0.0, 100.0, 1000.0] {
            let t_centuries = offset / 36525.0;
            let gmst = sidereal_time(J2000 + offset, 0.0, t_centuries);
            assert!(
                (0.0..24.0).contains(&gmst),
                "GMST out of range for offset {offset}: {gmst}"
            );
        }
    }

    #[test]
    fn test_gmst_increases_with_time() {
        let gmst1 = sidereal_time(J2000, 0.0, 0.0);
        let gmst2 = sidereal_time(J2000, 0.01, 0.01 / 36525.0);
        let diff = (gmst2 - gmst1 + 24.0) % 24.0;
        assert!(diff > 0.2 && diff < 0.3, "GMST increase = {diff}");
    }

    #[test]
    fn test_polar_motion_matrix_identity_at_zero() {
        let ts = crate::time::Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        // Without polar motion table, matrix should be near identity
        let w = t.polar_motion_matrix();
        let det = w.determinant();
        assert_relative_eq!(det, 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_polar_motion_angles_without_table() {
        let ts = crate::time::Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let (s_prime, x, y) = t.polar_motion_angles();
        // s_prime should be near zero at J2000
        assert!(s_prime.abs() < 1e-3, "s_prime = {s_prime}");
        // Without table, x and y should be zero
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn test_polar_motion_with_table() {
        let mut ts = crate::time::Timescale::default();
        // Install a simple table: J2000 ± 100 days with constant x=0.1, y=0.2
        let tt_jd = vec![2451445.0, 2451545.0, 2451645.0];
        let x = vec![0.1, 0.1, 0.1];
        let y = vec![0.2, 0.2, 0.2];
        ts.set_polar_motion_table(tt_jd, x, y);
        assert!(ts.has_polar_motion());

        let t = ts.tdb_jd(2451545.0);
        let (_, xp, yp) = t.polar_motion_angles();
        assert_relative_eq!(xp, 0.1, epsilon = 1e-10);
        assert_relative_eq!(yp, 0.2, epsilon = 1e-10);
    }

    #[test]
    fn test_c_matrix_orthogonality() {
        let ts = crate::time::Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let c = t.c_matrix();
        let product = c.transpose() * c;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_refraction_at_horizon() {
        let r = refraction(0.0, 10.0, 1010.0);
        assert!(
            r > 0.4 && r < 0.7,
            "Horizon refraction should be ~0.5°, got {r}"
        );
    }

    #[test]
    fn test_refraction_at_zenith() {
        let r = refraction(90.0, 10.0, 1010.0);
        assert_relative_eq!(r, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_refraction_below_horizon() {
        assert_relative_eq!(refraction(-5.0, 10.0, 1010.0), 0.0);
    }

    #[test]
    fn test_refraction_at_45() {
        let r = refraction(45.0, 10.0, 1010.0);
        assert!(
            r > 0.01 && r < 0.03,
            "45° refraction should be ~0.02°, got {r}"
        );
    }

    #[test]
    fn test_refract_raises_altitude() {
        let apparent = refract(0.0, 10.0, 1010.0);
        assert!(
            apparent > 0.0,
            "Refracted altitude should be above true altitude"
        );
    }

    #[test]
    fn test_refract_roundtrip() {
        // refract(true_alt) ≈ true_alt + refraction(true_alt)
        // Exact roundtrip is not possible (inverse problem), but they should be close
        let true_alt = 10.0;
        let apparent = refract(true_alt, 15.0, 1013.25);
        assert!(apparent > true_alt);
        assert!(apparent - true_alt < 0.2);
    }
}
