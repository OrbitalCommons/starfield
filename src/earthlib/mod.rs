//! Earth rotation and sidereal time computations
//!
//! Implements the Earth Rotation Angle (ERA) per IAU 2000 Resolution B1.8,
//! and Greenwich Mean Sidereal Time (GMST) per USNO Circular 179, Section 2.6.2.

use crate::constants::J2000;

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
}
