//! Precession matrix computation
//!
//! Implements the Capitaine et al. (2003) 4-angle precession formulation
//! for computing the precession matrix from J2000 to the equinox of date.
//! Reference: Astronomy and Astrophysics 412, 567-586.

use crate::constants::{ASEC2RAD, J2000};
use nalgebra::Matrix3;

/// Mean obliquity at J2000.0 in arcseconds
const EPS0: f64 = 84381.406;

/// Compute the precession rotation matrix for a given TDB Julian date
///
/// Uses the Capitaine et al. (2003) four-angle formulation:
/// P = R3(chi_a) * R1(-omega_a) * R3(-psi_a) * R1(eps_0)
///
/// # Arguments
/// * `jd_tdb` - TDB Julian date
pub fn compute_precession(jd_tdb: f64) -> Matrix3<f64> {
    let t = (jd_tdb - J2000) / 36525.0;

    // Precession angles from Capitaine et al. (2003), in arcseconds
    let psi_a = ((((-0.0000000951 * t + 0.000132851) * t - 0.00114045) * t - 1.0790069) * t
        + 5038.481507)
        * t;

    let omega_a =
        ((((0.0000003337 * t - 0.000000467) * t - 0.00772503) * t + 0.0512623) * t - 0.025754) * t
            + EPS0;

    let chi_a = ((((-0.0000000560 * t + 0.000170663) * t - 0.00121197) * t - 2.3814292) * t
        + 10.556403)
        * t;

    // Convert to radians
    let eps0 = EPS0 * ASEC2RAD;
    let psi_a = psi_a * ASEC2RAD;
    let omega_a = omega_a * ASEC2RAD;
    let chi_a = chi_a * ASEC2RAD;

    // Compose as R3(chi_a) * R1(-omega_a) * R3(-psi_a) * R1(eps_0)
    let (sa, ca) = eps0.sin_cos();
    let (sb, cb) = (-psi_a).sin_cos();
    let (sc, cc) = (-omega_a).sin_cos();
    let (sd, cd) = chi_a.sin_cos();

    #[rustfmt::skip]
    let m = Matrix3::new(
        cd * cb - sb * sd * cc,
        cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
        cd * sb * sa + sd * cc * cb * sa + ca * sd * sc,

        -sd * cb - sb * cd * cc,
        -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
        -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc,

        sb * sc,
        -sc * cb * ca - sa * cc,
        -sc * cb * sa + cc * ca,
    );

    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_precession_at_j2000_is_identity() {
        let p = compute_precession(J2000);
        // At J2000, the precession matrix should be very close to the identity
        // (not exactly identity due to the frame bias being separate)
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(p[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_precession_orthogonality() {
        // Precession matrix should be orthogonal (P^T * P = I)
        let p = compute_precession(J2000 + 3652.5); // ~10 years later
        let product = p.transpose() * p;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_precession_determinant() {
        let p = compute_precession(J2000 + 3652.5);
        assert_relative_eq!(p.determinant(), 1.0, epsilon = 1e-14);
    }
}
