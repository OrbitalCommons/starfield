//! TEME (True Equator Mean Equinox) frame transformations
//!
//! TEME is the reference frame used by SGP4 for satellite positions.
//! It uses the true equator (with nutation) but the mean equinox (precession only).
//! This was computationally tractable for 1970s computers when SGP4 was developed.
//!
//! Reference: AIAA 2006-6753 (Revisiting Spacetrack Report #3)

use nalgebra::{Matrix3, Vector3};
use std::f64::consts::TAU;

use crate::time::Time;

/// J2000 epoch in Julian days
const T0: f64 = 2451545.0;

/// Seconds per day
const DAY_S: f64 = 86400.0;

/// Compute Greenwich Mean Sidereal Time (1982 formulation)
///
/// Returns (theta, theta_dot) where:
/// - theta is the GMST angle in radians
/// - theta_dot is the angular velocity in radians/day
///
/// This is the GMST formula from AIAA 2006-6753, which matches the
/// formulation used in the original Spacetrack Report #3.
pub fn gmst1982(jd_ut1: f64, frac_ut1: f64) -> (f64, f64) {
    // Julian centuries from J2000 (using UT1)
    let t = (jd_ut1 - T0 + frac_ut1) / 36525.0;

    // GMST polynomial coefficients (seconds of time)
    // From AIAA 2006-6753 / Vallado
    let g = 67310.54841 + (8640184.812866 + (0.093104 + (-6.2e-6) * t) * t) * t;

    // Time derivative of g (for velocity transformation)
    let dg = 8640184.812866 + (0.093104 * 2.0 + (-6.2e-6 * 3.0) * t) * t;

    // Convert to angle in radians
    // The fractional day contributes directly to the rotation
    let theta = ((jd_ut1 % 1.0 + frac_ut1 + g / DAY_S) % 1.0) * TAU;

    // Angular velocity in radians/day
    let theta_dot = (1.0 + dg / (DAY_S * 36525.0)) * TAU;

    (theta, theta_dot)
}

/// Rotation matrix about the Z axis
fn rot_z(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0)
}

/// Compute TEME to GCRS rotation matrix
///
/// The transformation accounts for the difference between:
/// - TEME's use of the mean equinox (GMST)
/// - GCRS's use of the true equinox (GAST)
///
/// Plus the precession-nutation matrix M that takes J2000 to the date.
pub fn teme_to_gcrs(t: &Time) -> Matrix3<f64> {
    let ut1 = t.ut1();
    let jd_ut1 = ut1.floor();
    let frac_ut1 = ut1 - jd_ut1;

    let (theta, _theta_dot) = gmst1982(jd_ut1, frac_ut1);

    // The angle difference between GMST (mean equinox) and GAST (true equinox)
    // accounts for the equation of the equinoxes
    let gast_rad = t.gast() / 24.0 * TAU;
    let angle = theta - gast_rad;

    // Combine rotation with precession-nutation matrix
    rot_z(angle) * t.m_matrix()
}

/// Compute TEME to GCRS transformation for both position and velocity
///
/// Returns (R, R_dot) where R is the rotation matrix and R_dot accounts
/// for Earth's rotation when transforming velocity.
pub fn teme_to_gcrs_full(t: &Time) -> (Matrix3<f64>, f64) {
    let ut1 = t.ut1();
    let jd_ut1 = ut1.floor();
    let frac_ut1 = ut1 - jd_ut1;

    let (theta, theta_dot) = gmst1982(jd_ut1, frac_ut1);
    let gast_rad = t.gast() / 24.0 * TAU;
    let angle = theta - gast_rad;

    let r = rot_z(angle) * t.m_matrix();
    (r, theta_dot)
}

/// Transform position and velocity from TEME to GCRS
///
/// Position in km, velocity in km/s.
/// Returns transformed (position, velocity) in the same units.
pub fn transform_teme_to_gcrs(
    t: &Time,
    pos_teme: &Vector3<f64>,
    vel_teme: &Vector3<f64>,
) -> (Vector3<f64>, Vector3<f64>) {
    let r = teme_to_gcrs(t);

    // Transform position
    let pos_gcrs = r * pos_teme;

    // Transform velocity (rotation only, ignoring Earth rotation for now
    // since it's a small effect for satellite tracking)
    let vel_gcrs = r * vel_teme;

    (pos_gcrs, vel_gcrs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    #[test]
    fn test_gmst1982_at_j2000() {
        // At J2000, GMST should be approximately 280.46 degrees
        let (theta, _) = gmst1982(2451545.0, 0.0);
        let theta_deg = theta.to_degrees();

        // Normalize to 0-360
        let theta_deg = ((theta_deg % 360.0) + 360.0) % 360.0;

        // GMST at J2000.0 is approximately 280.46 degrees
        // (18h 41m 50.548s = 280.4606 degrees)
        assert_relative_eq!(theta_deg, 280.46, epsilon = 0.5);
    }

    #[test]
    fn test_rot_z_identity() {
        let r = rot_z(0.0);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let result = r * v;
        assert_relative_eq!(result.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rot_z_90_degrees() {
        let r = rot_z(std::f64::consts::FRAC_PI_2);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let result = r * v;
        assert_relative_eq!(result.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.y, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_teme_to_gcrs_returns_rotation_matrix() {
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let r = teme_to_gcrs(&t);

        // A rotation matrix should have determinant 1
        let det = r.determinant();
        assert_relative_eq!(det, 1.0, epsilon = 1e-10);

        // R * R^T should be identity
        let rrt = r * r.transpose();
        assert_relative_eq!(rrt[(0, 0)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rrt[(1, 1)], 1.0, epsilon = 1e-10);
        assert_relative_eq!(rrt[(2, 2)], 1.0, epsilon = 1e-10);
    }
}
