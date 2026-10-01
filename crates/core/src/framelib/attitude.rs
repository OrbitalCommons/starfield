//! Spacecraft attitude <-> pointing conversions.
//!
//! An attitude is the rotation that carries vectors expressed in a
//! spacecraft *body* frame into the ICRF. This module converts between that
//! rotation and the `(boresight, roll)` description operators usually work
//! in.
//!
//! # Body frame convention
//!
//! - body `+Z` is the instrument boresight, pointing outward at the sky.
//! - body `+X` is local **east** at the boresight when `roll = 0`: the unit
//!   vector `(-sin ra, cos ra, 0)`, toward increasing right ascension.
//! - body `+Y` is local **north** at the boresight when `roll = 0`:
//!   `(-sin dec cos ra, -sin dec sin ra, cos dec)`, toward increasing
//!   declination.
//!
//! `(east, north, boresight)` is a right-handed triad, so the roll-zero
//! attitude matrix simply has those three vectors as its columns.
//!
//! # Roll convention
//!
//! Roll is a right-handed rotation about body `+Z` (the outward boresight),
//! applied in the body frame: `q = q_base * R_z(roll)`. A positive roll
//! rotates body `+X` from east toward north. The position angle of body `+Y`
//! (measured from north through east, the astronomical convention) is
//! therefore `-roll`.
//!
//! # Pole behaviour
//!
//! At a celestial pole east and north are not defined by the boresight
//! alone. [`attitude_from_pointing`] still produces a well-defined attitude
//! there because it uses the *supplied* right ascension to build the triad
//! (the limit approached along that RA meridian). Going the other way, when
//! the boresight lies within [`POLE_EPSILON`] radians of a pole,
//! [`boresight_of`] reports `ra = 0` and [`roll_of`] measures roll against
//! the `ra = 0` east vector, which is world `+Y`. The pair therefore always
//! round-trips: `attitude_from_pointing(&boresight_of(&q), roll_of(&q))`
//! reproduces `q` everywhere, including at the poles, to within
//! `POLE_EPSILON`.

use super::inertial::Equatorial;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use std::f64::consts::FRAC_PI_2;

/// Angular distance (radians) from a celestial pole inside which a
/// boresight is treated as lying on the pole: its right ascension is
/// reported as `0` and roll is measured against the `ra = 0` east vector.
pub const POLE_EPSILON: f64 = 1e-9;

/// Local `(east, north, radial)` unit vectors in ICRF at `eq`.
///
/// `east` points toward increasing right ascension, `north` toward
/// increasing declination and `radial` along the direction itself. The
/// triad is right-handed (`east x north = radial`). At a pole `east` and
/// `north` are those of the limit along the meridian of `eq.ra`.
pub fn local_triad(eq: &Equatorial) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let (sin_ra, cos_ra) = eq.ra.sin_cos();
    let (sin_dec, cos_dec) = eq.dec.sin_cos();
    let east = Vector3::new(-sin_ra, cos_ra, 0.0);
    let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
    let radial = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
    (east, north, radial)
}

/// Body -> ICRF rotation matrix for a boresight `eq` and `roll_rad`.
///
/// Columns of the result are body `+X`, `+Y`, `+Z` expressed in ICRF. See
/// the [module documentation](self) for the axis and roll conventions.
pub fn rotation_from_pointing(eq: &Equatorial, roll_rad: f64) -> Rotation3<f64> {
    let (east, north, bore) = local_triad(eq);
    let base = Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[east, north, bore]));
    base * Rotation3::from_axis_angle(&Vector3::z_axis(), roll_rad)
}

/// Body -> ICRF attitude quaternion for a boresight `eq` and `roll_rad`.
///
/// Equivalent to [`rotation_from_pointing`] as a unit quaternion:
/// `q * Vector3::z()` is the boresight unit vector and `q * Vector3::x()`
/// is local east rotated by `roll_rad` toward north.
pub fn attitude_from_pointing(eq: &Equatorial, roll_rad: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&rotation_from_pointing(eq, roll_rad))
}

/// Boresight (body `+Z` in ICRF) of a body -> ICRF attitude.
///
/// Within [`POLE_EPSILON`] of a pole the right ascension is reported as `0`.
pub fn boresight_of(q: &UnitQuaternion<f64>) -> Equatorial {
    let z = q * Vector3::z();
    let r_xy = z.x.hypot(z.y);
    let dec = z.z.atan2(r_xy);
    if FRAC_PI_2 - dec.abs() < POLE_EPSILON {
        Equatorial::new(0.0, dec)
    } else {
        Equatorial::new(z.y.atan2(z.x), dec)
    }
}

/// Roll (radians, in `[-pi, pi]`) of a body -> ICRF attitude.
///
/// The signed angle about the outward boresight from local east at
/// [`boresight_of`]`(q)` to body `+X`. Inverse of the `roll_rad` argument of
/// [`attitude_from_pointing`]; see the module documentation for the pole
/// convention.
pub fn roll_of(q: &UnitQuaternion<f64>) -> f64 {
    let (east, _north, bore) = local_triad(&boresight_of(q));
    let x_world = q * Vector3::x();
    let cos_roll = east.dot(&x_world);
    let sin_roll = east.cross(&x_world).dot(&bore);
    sin_roll.atan2(cos_roll)
}

/// Boresight and roll of a body -> ICRF attitude, the inverse of
/// [`attitude_from_pointing`].
pub fn pointing_of(q: &UnitQuaternion<f64>) -> (Equatorial, f64) {
    (boresight_of(q), roll_of(q))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framelib::inertial::InertialFrame;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn eq_deg(ra_deg: f64, dec_deg: f64) -> Equatorial {
        Equatorial::from_degrees(ra_deg, dec_deg)
    }

    fn assert_vec_eq(a: &Vector3<f64>, b: &Vector3<f64>, eps: f64) {
        assert_abs_diff_eq!(a.x, b.x, epsilon = eps);
        assert_abs_diff_eq!(a.y, b.y, epsilon = eps);
        assert_abs_diff_eq!(a.z, b.z, epsilon = eps);
    }

    fn unit(eq: &Equatorial) -> Vector3<f64> {
        eq.to_cartesian().to_vector3()
    }

    const ROLLS: [f64; 7] = [-PI * 0.9, -FRAC_PI_2, -0.5, 0.0, 0.5, FRAC_PI_2, PI * 0.9];

    #[test]
    fn local_triad_is_right_handed_orthonormal() {
        for eq in [eq_deg(0.0, 0.0), eq_deg(123.0, -42.0), eq_deg(300.0, 90.0)] {
            let (e, n, r) = local_triad(&eq);
            assert_abs_diff_eq!(e.norm(), 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(n.norm(), 1.0, epsilon = 1e-15);
            assert_abs_diff_eq!(e.dot(&n), 0.0, epsilon = 1e-15);
            assert_vec_eq(&e.cross(&n), &r, 1e-15);
            assert_vec_eq(&r, &unit(&eq), 1e-15);
        }
    }

    #[test]
    fn roll_zero_maps_body_axes_to_east_north_boresight() {
        let eq = eq_deg(45.0, 20.0);
        let q = attitude_from_pointing(&eq, 0.0);
        let (e, n, r) = local_triad(&eq);
        assert_vec_eq(&(q * Vector3::x()), &e, 1e-12);
        assert_vec_eq(&(q * Vector3::y()), &n, 1e-12);
        assert_vec_eq(&(q * Vector3::z()), &r, 1e-12);
        // East at (45, 20) has positive world +Y and negative world +X.
        assert!(e.x < 0.0 && e.y > 0.0);
    }

    #[test]
    fn positive_roll_turns_body_x_toward_north() {
        let eq = eq_deg(80.0, -30.0);
        let q = attitude_from_pointing(&eq, FRAC_PI_2);
        let (e, n, _) = local_triad(&eq);
        assert_vec_eq(&(q * Vector3::x()), &n, 1e-12);
        assert_vec_eq(&(q * Vector3::y()), &(-e), 1e-12);
    }

    #[test]
    fn body_y_position_angle_is_minus_roll() {
        let eq = eq_deg(200.0, 35.0);
        let (e, n, _) = local_triad(&eq);
        for roll in ROLLS {
            let y = attitude_from_pointing(&eq, roll) * Vector3::y();
            let pa = y.dot(&e).atan2(y.dot(&n));
            assert_abs_diff_eq!(pa, -roll, epsilon = 1e-12);
        }
    }

    #[test]
    fn matrix_and_quaternion_agree() {
        let eq = eq_deg(12.0, 34.0);
        let m = rotation_from_pointing(&eq, 0.7);
        let q = attitude_from_pointing(&eq, 0.7);
        for v in [Vector3::x(), Vector3::y(), Vector3::z()] {
            assert_vec_eq(&(m * v), &(q * v), 1e-14);
        }
    }

    #[test]
    fn pointing_round_trips_off_pole() {
        let samples = [
            eq_deg(0.0, 0.0),
            eq_deg(45.0, 30.0),
            eq_deg(123.5, -42.0),
            eq_deg(180.0, -15.0),
            eq_deg(359.9, 89.9),
            eq_deg(10.0, -89.9),
        ];
        for eq in samples {
            for roll in ROLLS {
                let q = attitude_from_pointing(&eq, roll);
                let (b, r) = pointing_of(&q);
                assert_vec_eq(&unit(&b), &unit(&eq), 1e-12);
                assert_abs_diff_eq!(b.ra, eq.ra, epsilon = 1e-9);
                assert_abs_diff_eq!(b.dec, eq.dec, epsilon = 1e-12);
                assert_abs_diff_eq!(r, roll, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn roll_of_handles_half_turn() {
        let q = attitude_from_pointing(&eq_deg(30.0, 10.0), PI);
        assert_abs_diff_eq!(roll_of(&q).abs(), PI, epsilon = 1e-12);
    }

    #[test]
    fn pole_reports_zero_ra_and_reconstructs_attitude() {
        for dec in [FRAC_PI_2, -FRAC_PI_2, FRAC_PI_2 - 1e-12] {
            for ra in [0.0, 1.2, 4.0] {
                for roll in ROLLS {
                    let q = attitude_from_pointing(&Equatorial::new(ra, dec), roll);
                    let (b, r) = pointing_of(&q);
                    assert_eq!(b.ra, 0.0);
                    let rebuilt = attitude_from_pointing(&b, r);
                    assert_abs_diff_eq!(rebuilt.angle_to(&q), 0.0, epsilon = 1e-8);
                }
            }
        }
    }

    #[test]
    fn north_pole_roll_absorbs_ra() {
        // At the north pole the ra = 0 east vector is world +Y, and an
        // attitude built at (ra, roll) equals one built at (0, ra + roll).
        let q = attitude_from_pointing(&Equatorial::new(1.2, FRAC_PI_2), 0.3);
        assert_abs_diff_eq!(roll_of(&q), 1.5, epsilon = 1e-12);
        let q = attitude_from_pointing(&Equatorial::new(1.2, -FRAC_PI_2), 0.3);
        assert_abs_diff_eq!(roll_of(&q), 0.3 - 1.2, epsilon = 1e-12);
    }
}
