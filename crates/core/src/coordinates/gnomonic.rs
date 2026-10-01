//! Gnomonic (tangent-plane, FITS `TAN`) projection of the celestial sphere.
//!
//! A gnomonic projection maps a direction on the sky onto the plane tangent
//! to the unit sphere at a chosen centre, by extending the ray from the sphere
//! centre until it meets that plane. Great circles become straight lines, the
//! centre maps to the origin, and a point at angular distance `c` from the
//! centre lands at radius `tan(c)`. Only the hemisphere in front of the plane
//! (`c < 90°`) has an image.
//!
//! # Standard coordinates
//!
//! With the default orientation the plane coordinates are the classical
//! astrometric *standard coordinates* `(xi, eta)`, in radians (dimensionless
//! tangent-plane units, equal to angle for small offsets):
//!
//! - `+xi` points **east**, toward increasing right ascension;
//! - `+eta` points **north**, toward increasing declination.
//!
//! In closed form, for a centre `(a0, d0)` and a target `(a, d)`:
//!
//! ```text
//! cos c = sin d0 sin d + cos d0 cos d cos(a - a0)
//! xi    = cos d sin(a - a0) / cos c
//! eta   = (cos d0 sin d - sin d0 cos d cos(a - a0)) / cos c
//! ```
//!
//! These are the FITS WCS intermediate world coordinates of a `TAN`
//! projection with the default `LONPOLE = 180°`. Plotted with `+xi` to the
//! right and `+eta` up they show the sky **mirrored** (east-right); a sky
//! chart or a detector image, which shows the sky as an observer sees it,
//! has north up and east **left**, i.e. its horizontal axis is `-xi`. That
//! mirror belongs to the pixel mapping (a FITS `CD1_1 < 0`), not here.
//!
//! # Position angle
//!
//! [`GnomonicProjection::with_position_angle`] rotates the plane axes about
//! the centre. The position angle `theta` is the direction of the `+v` axis
//! measured from north through east (the usual astronomical position-angle
//! sense), and `+u` points to position angle `theta + 90°`, so the frame
//! keeps the handedness of `(xi, eta)`:
//!
//! ```text
//! u = xi cos(theta) - eta sin(theta)
//! v = xi sin(theta) + eta cos(theta)
//! ```
//!
//! At `theta = 0`, `(u, v) = (xi, eta)`.
//!
//! # Poles
//!
//! The east/north basis is built analytically from the centre's right
//! ascension, so a centre exactly at a celestial pole is well defined: north
//! is taken as the direction along the meridian `ra_centre + 180°`, which is
//! the limit of the basis as the centre approaches the pole along its own
//! meridian.
//!
//! # Scope
//!
//! The projection is instrument-agnostic. Plate scale, pixel origin, axis
//! flips, and detector bounds are a linear map applied by the caller (see
//! `starfield_catalogs::mast::Wcs` for a FITS CD-matrix example).

use nalgebra::{Matrix3, Vector3};

use crate::coordinates::cartesian::Cartesian3;
use crate::framelib::inertial::{Equatorial, InertialFrame};

/// Gnomonic projection about a fixed centre, with an optional position-angle
/// rotation of the plane axes.
///
/// See the [module documentation](self) for the sign conventions. The
/// projection is immutable after construction and cheap to copy; projecting
/// a point costs one 3x3 matrix-vector product and two divisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnomonicProjection {
    center: Equatorial,
    position_angle: f64,
    /// Rows are the `+u` axis, the `+v` axis, and the centre direction: a
    /// right-handed orthonormal triad in the equatorial (ICRS) frame.
    basis: Matrix3<f64>,
}

impl GnomonicProjection {
    /// Projection centred on `center` with `(u, v) = (xi, eta)`: `+u` east,
    /// `+v` north.
    pub fn new(center: Equatorial) -> Self {
        Self::with_position_angle(center, 0.0)
    }

    /// Projection centred on `center` with the plane axes rotated so that
    /// `+v` points to `position_angle` (radians, north through east) and
    /// `+u` to `position_angle + 90°`.
    pub fn with_position_angle(center: Equatorial, position_angle: f64) -> Self {
        let (sin_ra, cos_ra) = center.ra.sin_cos();
        let (sin_dec, cos_dec) = center.dec.sin_cos();

        let boresight = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
        let east = Vector3::new(-sin_ra, cos_ra, 0.0);
        let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);

        let (sin_pa, cos_pa) = position_angle.sin_cos();
        let u_axis = east * cos_pa - north * sin_pa;
        let v_axis = east * sin_pa + north * cos_pa;

        let basis = Matrix3::from_rows(&[
            u_axis.transpose(),
            v_axis.transpose(),
            boresight.transpose(),
        ]);

        Self {
            center,
            position_angle,
            basis,
        }
    }

    /// The tangent point, which projects to `(0, 0)`.
    pub fn center(&self) -> Equatorial {
        self.center
    }

    /// Position angle of the `+v` axis in radians, north through east.
    pub fn position_angle(&self) -> f64 {
        self.position_angle
    }

    /// Unit vector of the `+u` axis in the equatorial frame.
    pub fn u_axis(&self) -> Vector3<f64> {
        self.basis.row(0).transpose()
    }

    /// Unit vector of the `+v` axis in the equatorial frame.
    pub fn v_axis(&self) -> Vector3<f64> {
        self.basis.row(1).transpose()
    }

    /// Unit vector toward the projection centre in the equatorial frame.
    pub fn boresight(&self) -> Vector3<f64> {
        self.basis.row(2).transpose()
    }

    /// Project a sky position to plane coordinates `(u, v)`.
    ///
    /// Returns `None` when the target is 90° or more from the centre, where
    /// the ray never meets the tangent plane. No other bound is applied:
    /// points just inside 90° map to arbitrarily large coordinates.
    pub fn project(&self, target: &Equatorial) -> Option<(f64, f64)> {
        self.project_vector(&target.to_cartesian().to_vector3())
    }

    /// Project a direction given as an equatorial-frame vector of any
    /// non-zero length to plane coordinates `(u, v)`.
    ///
    /// Returns `None` when the direction is 90° or more from the centre.
    pub fn project_vector(&self, direction: &Vector3<f64>) -> Option<(f64, f64)> {
        let local = self.basis * direction;
        if local.z <= 0.0 {
            return None;
        }
        Some((local.x / local.z, local.y / local.z))
    }

    /// Inverse projection: the sky position whose image is `(u, v)`.
    ///
    /// Every finite plane point has exactly one preimage, in the hemisphere
    /// centred on the projection centre.
    pub fn deproject(&self, u: f64, v: f64) -> Equatorial {
        Equatorial::from_cartesian(Cartesian3::from_vector3(self.deproject_vector(u, v)))
    }

    /// Inverse projection returning the equatorial-frame unit vector whose
    /// image is `(u, v)`.
    pub fn deproject_vector(&self, u: f64, v: f64) -> Vector3<f64> {
        (self.basis.transpose() * Vector3::new(u, v, 1.0)).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn assert_same_direction(a: &Equatorial, b: &Equatorial, epsilon: f64) {
        assert_abs_diff_eq!(a.angle_between(b), 0.0, epsilon = epsilon);
    }

    /// Closed-form standard coordinates, independent of the matrix basis.
    fn standard_coordinates(center: &Equatorial, target: &Equatorial) -> (f64, f64) {
        let da = target.ra - center.ra;
        let cos_c =
            center.dec.sin() * target.dec.sin() + center.dec.cos() * target.dec.cos() * da.cos();
        let xi = target.dec.cos() * da.sin() / cos_c;
        let eta = (center.dec.cos() * target.dec.sin()
            - center.dec.sin() * target.dec.cos() * da.cos())
            / cos_c;
        (xi, eta)
    }

    #[test]
    fn test_center_maps_to_origin() {
        for &(ra, dec) in &[(0.0, 0.0), (120.0, -15.0), (359.0, 89.0), (45.0, -90.0)] {
            let center = Equatorial::from_degrees(ra, dec);
            let proj = GnomonicProjection::new(center);
            let (u, v) = proj.project(&center).unwrap();
            assert_abs_diff_eq!(u, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_east_is_positive_xi_north_is_positive_eta() {
        let proj = GnomonicProjection::new(Equatorial::from_degrees(0.0, 0.0));

        let (xi, eta) = proj.project(&Equatorial::new(0.1, 0.0)).unwrap();
        assert_abs_diff_eq!(xi, 0.1_f64.tan(), epsilon = 1e-12);
        assert_abs_diff_eq!(eta, 0.0, epsilon = 1e-12);

        let (xi, eta) = proj.project(&Equatorial::new(0.0, 0.1)).unwrap();
        assert_abs_diff_eq!(xi, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(eta, 0.1_f64.tan(), epsilon = 1e-12);
    }

    #[test]
    fn test_matches_closed_form_standard_coordinates() {
        let center = Equatorial::from_degrees(210.0, 37.0);
        let proj = GnomonicProjection::new(center);
        for &(ra, dec) in &[(211.0, 37.5), (205.0, 30.0), (230.0, 50.0), (190.0, 20.0)] {
            let target = Equatorial::from_degrees(ra, dec);
            let (xi, eta) = proj.project(&target).unwrap();
            let (xi_ref, eta_ref) = standard_coordinates(&center, &target);
            assert_abs_diff_eq!(xi, xi_ref, epsilon = 1e-12);
            assert_abs_diff_eq!(eta, eta_ref, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_radius_is_tangent_of_separation() {
        let center = Equatorial::from_degrees(80.0, -60.0);
        let proj = GnomonicProjection::with_position_angle(center, 0.7);
        let target = Equatorial::from_degrees(95.0, -48.0);
        let (u, v) = proj.project(&target).unwrap();
        assert_abs_diff_eq!(
            u.hypot(v),
            center.angle_between(&target).tan(),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_hemisphere_behind_plane_is_rejected() {
        let proj = GnomonicProjection::new(Equatorial::from_degrees(0.0, 0.0));
        assert!(proj.project(&Equatorial::new(PI, 0.0)).is_none());
        assert!(proj.project(&Equatorial::new(FRAC_PI_2, 0.0)).is_none());
        assert!(proj.project(&Equatorial::new(0.0, -FRAC_PI_2)).is_none());
        assert!(proj.project(&Equatorial::new(1.5, 0.0)).is_some());
    }

    #[test]
    fn test_position_angle_rotates_axes() {
        let center = Equatorial::from_degrees(150.0, 20.0);
        let target = Equatorial::from_degrees(150.4, 20.3);
        let (xi, eta) = GnomonicProjection::new(center).project(&target).unwrap();

        let theta = 30.0_f64.to_radians();
        let (u, v) = GnomonicProjection::with_position_angle(center, theta)
            .project(&target)
            .unwrap();
        assert_abs_diff_eq!(u, xi * theta.cos() - eta * theta.sin(), epsilon = 1e-12);
        assert_abs_diff_eq!(v, xi * theta.sin() + eta * theta.cos(), epsilon = 1e-12);
    }

    #[test]
    fn test_position_angle_ninety_puts_v_east() {
        let proj = GnomonicProjection::with_position_angle(Equatorial::new(0.0, 0.0), FRAC_PI_2);
        let (u, v) = proj.project(&Equatorial::new(0.1, 0.0)).unwrap();
        assert_abs_diff_eq!(u, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(v, 0.1_f64.tan(), epsilon = 1e-12);

        let (u, v) = proj.project(&Equatorial::new(0.0, 0.1)).unwrap();
        assert_abs_diff_eq!(u, -(0.1_f64.tan()), epsilon = 1e-12);
        assert_abs_diff_eq!(v, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_basis_is_right_handed_orthonormal() {
        let proj =
            GnomonicProjection::with_position_angle(Equatorial::from_degrees(33.0, 71.0), 2.1);
        let (u, v, z) = (proj.u_axis(), proj.v_axis(), proj.boresight());
        assert_abs_diff_eq!(u.norm(), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(v.norm(), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(u.dot(&v), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!(u.dot(&z), 0.0, epsilon = 1e-14);
        assert_abs_diff_eq!((u.cross(&v) - z).norm(), 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_round_trip() {
        let centers = [
            Equatorial::from_degrees(0.0, 0.0),
            Equatorial::from_degrees(266.4, -29.0),
            Equatorial::from_degrees(10.0, 89.9),
            Equatorial::from_degrees(10.0, 90.0),
            Equatorial::from_degrees(300.0, -90.0),
        ];
        for center in centers {
            for &pa in &[0.0, 0.4, -2.5] {
                let proj = GnomonicProjection::with_position_angle(center, pa);
                for &(u, v) in &[(0.0, 0.0), (0.01, -0.02), (-0.3, 0.2), (1.5, 2.0)] {
                    let sky = proj.deproject(u, v);
                    let (u2, v2) = proj.project(&sky).unwrap();
                    assert_abs_diff_eq!(u2, u, epsilon = 1e-12);
                    assert_abs_diff_eq!(v2, v, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_deproject_origin_is_center() {
        let center = Equatorial::from_degrees(123.0, -45.0);
        let proj = GnomonicProjection::with_position_angle(center, 1.0);
        assert_same_direction(&proj.deproject(0.0, 0.0), &center, 1e-14);
    }

    #[test]
    fn test_pole_center_is_limit_of_near_pole_center() {
        let ra = 40.0_f64.to_radians();
        let at_pole = GnomonicProjection::new(Equatorial::new(ra, FRAC_PI_2));
        let near_pole = GnomonicProjection::new(Equatorial::new(ra, FRAC_PI_2 - 1e-9));
        let target = Equatorial::from_degrees(100.0, 88.0);
        let (u0, v0) = at_pole.project(&target).unwrap();
        let (u1, v1) = near_pole.project(&target).unwrap();
        assert_abs_diff_eq!(u0, u1, epsilon = 1e-8);
        assert_abs_diff_eq!(v0, v1, epsilon = 1e-8);
        assert_abs_diff_eq!(u0.hypot(v0), 2.0_f64.to_radians().tan(), epsilon = 1e-12);
    }

    #[test]
    fn test_project_vector_ignores_length() {
        let proj = GnomonicProjection::new(Equatorial::from_degrees(60.0, 10.0));
        let target = Equatorial::from_degrees(61.0, 11.0);
        let unit = target.to_cartesian().to_vector3();
        let a = proj.project_vector(&unit).unwrap();
        let b = proj.project_vector(&(unit * 4.2e8)).unwrap();
        assert_abs_diff_eq!(a.0, b.0, epsilon = 1e-14);
        assert_abs_diff_eq!(a.1, b.1, epsilon = 1e-14);
    }

    #[cfg(feature = "python-tests")]
    #[test]
    fn test_matches_astropy_tan_wcs() {
        use crate::pybridge::{bridge::PyRustBridge, helpers::PythonResult};

        let bridge = PyRustBridge::new().expect("bridge");
        let raw = bridge
            .run_py_to_json(
                r#"
import numpy as np
from astropy.wcs import WCS

# Identity CD matrix in degrees with CRPIX at the origin makes astropy's
# zero-based pixel coordinates the TAN intermediate world coordinates.
w = WCS(naxis=2)
w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
w.wcs.crval = [210.0, 37.0]
w.wcs.crpix = [1.0, 1.0]
w.wcs.cd = np.eye(2)
ra = np.array([211.0, 205.0, 230.0, 190.0])
dec = np.array([37.5, 30.0, 50.0, 20.0])
x, y = w.wcs_world2pix(ra, dec, 0)
rust.collect_array(np.concatenate([x, y]).astype(np.float64))
"#,
            )
            .expect("bridge run failed");

        let parsed = PythonResult::try_from(raw.as_str()).expect("bad json from bridge");
        let values: Vec<f64> = match parsed {
            PythonResult::Array { data, .. } => data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => panic!("expected Array, got {other:?}"),
        };

        let proj = GnomonicProjection::new(Equatorial::from_degrees(210.0, 37.0));
        let targets = [(211.0, 37.5), (205.0, 30.0), (230.0, 50.0), (190.0, 20.0)];
        for (i, &(ra, dec)) in targets.iter().enumerate() {
            let (xi, eta) = proj.project(&Equatorial::from_degrees(ra, dec)).unwrap();
            assert_abs_diff_eq!(xi.to_degrees(), values[i], epsilon = 1e-9);
            assert_abs_diff_eq!(eta.to_degrees(), values[i + targets.len()], epsilon = 1e-9);
        }
    }
}
