//! Apparent angular size of a resolved body.
//!
//! Two methods on [`Position`], both answering "how big does the target look
//! from here?":
//!
//! * [`Position::angular_semi_diameter`] treats the body as a sphere of its
//!   largest equatorial radius, which is what an ephemeris table means by
//!   "angular diameter";
//! * [`Position::apparent_ellipse`] projects the real triaxial ellipsoid onto
//!   the sky and returns the outline the observer would resolve, which for an
//!   oblate planet such as Jupiter is visibly flattened.
//!
//! Radii come from [`planetarylib`](crate::planetarylib) — either
//! [`Body::radii_km`](crate::planetlib::Body::radii_km) or
//! [`PlanetaryConstants::radii`](crate::planetarylib::PlanetaryConstants::radii)
//! — and the orientation of the ellipsoid from any
//! [`Frame`](crate::framelib::Frame), in practice the body-fixed frame that
//! [`PlanetaryConstants::frame_for`](crate::planetarylib::PlanetaryConstants::frame_for)
//! returns.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::SpiceKernel;
//! use starfield_core::jplephem_ext::SpiceKernelExt;
//! use starfield_core::planetlib::Body;
//! use starfield_core::time::Timescale;
//!
//! let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
//! let ts = Timescale::default();
//! let t = ts.utc((2007, 10, 3, 8, 30, 0.0));
//!
//! let mars = kernel.at("mars", &t).unwrap();
//! let earth = mars.observe("earth", &mut kernel, &t).unwrap();
//! let arcsec = earth.angular_semi_diameter(Body::Earth.radii_km()).to_degrees() * 7200.0;
//! println!("Earth is {:.2} arcseconds across, seen from Mars", arcsec);
//! ```

use nalgebra::Vector3;

use crate::constants::AU_KM;
use crate::framelib::Frame;
use crate::positions::Position;
use crate::time::Time;

/// Relative spread of the two eigenvalues below which the projected outline
/// counts as a circle and its position angle as zero. A few times the machine
/// epsilon: an ellipse this round is one part in 10¹⁵ from a circle, so its
/// axes are rounding error rather than geometry.
const CIRCLE_TOLERANCE: f64 = 8.0 * f64::EPSILON;

impl Position {
    /// The angular radius of the body, in radians, taken as a sphere.
    ///
    /// `asin(a / d)`, where `a` is the larger of the two equatorial radii of
    /// `radii_km` and `d` the distance from the observer to the body, taken
    /// from [`position`](Position::position). Doubling the result gives the
    /// angular diameter that ephemeris tables quote.
    ///
    /// An observer inside the body — `a` greater than `d` — gets a quarter
    /// turn rather than a NaN, and so does a body at zero distance.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use starfield_core::planetlib::Body;
    /// use starfield_core::positions::Position;
    ///
    /// // One AU away, the Earth is 17.59 arcseconds across.
    /// let p = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 399);
    /// let arcsec = p.angular_semi_diameter(Body::Earth.radii_km()).to_degrees() * 7200.0;
    /// assert!((arcsec - 17.59).abs() < 0.01);
    /// ```
    pub fn angular_semi_diameter(&self, radii_km: [f64; 3]) -> f64 {
        let distance_km = self.position.norm() * AU_KM;
        semi_angle(radii_km[0].max(radii_km[1]), distance_km)
    }

    /// The outline of the body's ellipsoid as projected onto the sky.
    ///
    /// Returns `(semi_major, semi_minor, position_angle)`: the two angular
    /// semi-axes in radians, and the position angle of the major axis in
    /// radians east of celestial north, folded into `[0, π)` because the axis
    /// of an ellipse has no head or tail.
    ///
    /// `frame` is the body-fixed frame the radii are given in — the one
    /// [`PlanetaryConstants::frame_for`](crate::planetarylib::PlanetaryConstants::frame_for)
    /// returns — and is evaluated at the light-time-corrected epoch
    /// `t − self.light_time`, since what the observer sees is the orientation
    /// the body had when the light left it.
    ///
    /// # The projection
    ///
    /// The body is the ellipsoid `xᵀ A x = 1` in body-fixed coordinates, with
    /// `A = diag(1/a², 1/b², 1/c²)` for `radii_km = [a, b, c]`. Rotate into a
    /// frame whose first two axes `u`, `v` span the sky plane (east and north)
    /// and whose third axis `n` is the line of sight, using the orthogonal
    /// matrix `C` whose rows are those three vectors written in body-fixed
    /// coordinates. In that frame the ellipsoid is `yᵀ B y = 1` with
    /// `B = C A Cᵀ`, which splits into blocks
    ///
    /// ```text
    /// B = ⎡ B₁₁  b₁₂ ⎤     B₁₁ is 2×2, b₁₂ is 2×1, b₂₂ is a scalar.
    ///     ⎣ b₁₂ᵀ b₂₂ ⎦
    /// ```
    ///
    /// The observer is far enough away that the projection is orthographic, so
    /// the outline is the shadow the solid `yᵀ B y ≤ 1` casts along `n`: a
    /// sky-plane point `p` lies inside the outline exactly when some `y₃`
    /// satisfies
    ///
    /// ```text
    /// pᵀ B₁₁ p + 2 y₃ b₁₂ᵀ p + b₂₂ y₃² ≤ 1.
    /// ```
    ///
    /// The left side is a quadratic in `y₃`, minimised at
    /// `y₃ = −(b₁₂ᵀ p) / b₂₂`, where it takes the value
    /// `pᵀ (B₁₁ − b₁₂ b₁₂ᵀ / b₂₂) p`. The outline is therefore the conic
    /// `pᵀ S p = 1` whose matrix
    ///
    /// ```text
    /// S = B₁₁ − b₁₂ b₁₂ᵀ / b₂₂
    /// ```
    ///
    /// is the Schur complement of `b₂₂` in `B`. `S` is symmetric and positive
    /// definite; its eigenvectors give the directions of the projected axes,
    /// and the semi-axis along an eigenvector of eigenvalue `λ` is `1/√λ`, so
    /// the smaller eigenvalue belongs to the major axis. Each semi-axis is
    /// then turned into an angle the way [`angular_semi_diameter`] turns a
    /// radius into one.
    ///
    /// Nothing here assumes `a = b`: a triaxial body is projected exactly, and
    /// a sphere gives two equal axes and a position angle of zero.
    ///
    /// [`angular_semi_diameter`]: Position::angular_semi_diameter
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::Vector3;
    /// use starfield_core::framelib::ICRS;
    /// use starfield_core::planetlib::Body;
    /// use starfield_core::positions::Position;
    /// use starfield_core::time::Timescale;
    ///
    /// let t = Timescale::default().tdb_jd(2451545.0);
    /// // Five AU away along the x axis, so for a body whose pole is the ICRF
    /// // z axis the line of sight lies in its equatorial plane.
    /// let p = Position::barycentric(Vector3::new(5.0, 0.0, 0.0), Vector3::zeros(), 599);
    /// let (major, minor, pa) = p.apparent_ellipse(&ICRS, Body::Jupiter.radii_km(), &t);
    /// assert!((minor / major - (1.0 - Body::Jupiter.flattening())).abs() < 1e-6);
    /// // The pole is projected north, so the equator runs east and west.
    /// assert!((pa.to_degrees() - 90.0).abs() < 1e-9);
    /// ```
    pub fn apparent_ellipse(
        &self,
        frame: &dyn Frame,
        radii_km: [f64; 3],
        t: &Time,
    ) -> (f64, f64, f64) {
        let distance_km = self.position.norm() * AU_KM;
        let (east, north, line_of_sight) = sky_basis(&self.position);

        // The body-fixed orientation at the moment the light left the body.
        let rotation = frame.rotation_at(&(t.clone() - self.light_time));
        let u = rotation * east;
        let v = rotation * north;
        let n = rotation * line_of_sight;

        // B = C A Cᵀ, one entry at a time; A is diagonal, so `quadratic` is
        // all the linear algebra the Schur complement needs.
        let b11 = quadratic(&u, &u, radii_km);
        let b12 = quadratic(&u, &v, radii_km);
        let b13 = quadratic(&u, &n, radii_km);
        let b22 = quadratic(&v, &v, radii_km);
        let b23 = quadratic(&v, &n, radii_km);
        let b33 = quadratic(&n, &n, radii_km);

        let s11 = b11 - b13 * b13 / b33;
        let s12 = b12 - b13 * b23 / b33;
        let s22 = b22 - b23 * b23 / b33;

        // Eigenvalues of the symmetric 2×2 conic matrix, smaller one first.
        let mean = 0.5 * (s11 + s22);
        let half_difference = 0.5 * (s11 - s22);
        let radius = half_difference.hypot(s12);
        let semi_major_km = 1.0 / (mean - radius).sqrt();
        let semi_minor_km = 1.0 / (mean + radius).sqrt();

        // Eigenvector of the smaller eigenvalue, in (east, north) components.
        // Both rows of `S − λI` give the same direction; the longer of the two
        // is the numerically safe one. When the two eigenvalues differ by no
        // more than rounding the outline is a circle, its axes are arbitrary,
        // and the eigenvector is noise: report a position angle of zero
        // rather than that noise.
        let lambda = mean - radius;
        let (first, second) = ((s12, lambda - s11), (lambda - s22, s12));
        let (east_part, north_part) = if first.0.hypot(first.1) >= second.0.hypot(second.1) {
            first
        } else {
            second
        };
        let position_angle = if radius <= CIRCLE_TOLERANCE * mean {
            0.0
        } else {
            east_part.atan2(north_part).rem_euclid(std::f64::consts::PI)
        };

        (
            semi_angle(semi_major_km, distance_km),
            semi_angle(semi_minor_km, distance_km),
            position_angle,
        )
    }
}

/// The angle a radius subtends at a distance, saturating at a quarter turn.
fn semi_angle(radius_km: f64, distance_km: f64) -> f64 {
    if distance_km <= 0.0 {
        return std::f64::consts::FRAC_PI_2;
    }
    (radius_km / distance_km).min(1.0).asin()
}

/// `wᵀ A z` for the diagonal ellipsoid matrix `A = diag(1/a², 1/b², 1/c²)`.
fn quadratic(w: &Vector3<f64>, z: &Vector3<f64>, radii_km: [f64; 3]) -> f64 {
    w.x * z.x / (radii_km[0] * radii_km[0])
        + w.y * z.y / (radii_km[1] * radii_km[1])
        + w.z * z.z / (radii_km[2] * radii_km[2])
}

/// The right-handed sky triad `(east, north, line_of_sight)` at a direction.
///
/// East is `ẑ × r̂`, the direction of increasing right ascension, and north
/// completes the triad as `r̂ × east`. A target exactly at a celestial pole
/// leaves east undefined; there the x axis stands in for the pole, so the
/// triad is still orthonormal and the position angle is merely arbitrary, as
/// it must be.
fn sky_basis(position: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
    let line_of_sight = position.normalize();
    let mut east = Vector3::z().cross(&line_of_sight);
    if east.norm() < 1e-12 {
        east = Vector3::x().cross(&line_of_sight);
    }
    let east = east.normalize();
    let north = line_of_sight.cross(&east);
    (east, north, line_of_sight)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framelib::ICRS;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::planetarylib::pck_frame::{rot_x, rot_z};
    use crate::planetarylib::PlanetaryConstants;
    use crate::planetlib::Body;
    use crate::time::Timescale;
    use nalgebra::Matrix3;

    /// Radians to arcseconds.
    const RAD2ASEC: f64 = 206_264.806_247_096_36;

    /// A body-fixed frame that never moves, for tests that place the line of
    /// sight exactly on a body's equator or pole.
    struct FixedFrame(Matrix3<f64>);

    impl Frame for FixedFrame {
        fn rotation_at(&self, _t: &Time) -> Matrix3<f64> {
            self.0
        }
    }

    fn t_j2000() -> Time {
        Timescale::default().tdb_jd(2451545.0)
    }

    /// The epoch of the HiRISE Earth-and-Moon portrait, 2007 October 3 at
    /// 08:30 UTC, when Mars was 0.9507 AU from the Earth.
    fn hirise_epoch() -> Time {
        Timescale::default().utc((2007, 10, 3, 8, 30, 0.0))
    }

    /// The epoch of the earlier Earth-and-Moon portrait, the one the Mars
    /// Orbiter Camera of Mars Global Surveyor took on 2003 May 8 at 13:00 UTC
    /// from 0.9304 AU.
    fn moc_epoch() -> Time {
        Timescale::default().utc((2003, 5, 8, 13, 0, 0.0))
    }

    fn at_distance(au: f64, direction: Vector3<f64>) -> Position {
        Position::barycentric(direction.normalize() * au, Vector3::zeros(), 0)
    }

    fn de421() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("test_data/de421.bsp")
    }

    #[test]
    fn test_semi_diameter_is_asin_of_the_ratio() {
        let p = at_distance(1.0, Vector3::x());
        let expected = (6378.1366 / AU_KM).asin();
        assert!((p.angular_semi_diameter(Body::Earth.radii_km()) - expected).abs() < 1e-18);
    }

    #[test]
    fn test_semi_diameter_uses_the_larger_equatorial_radius() {
        let p = at_distance(1.0, Vector3::x());
        let flat = p.angular_semi_diameter([100.0, 200.0, 10.0]);
        let round = p.angular_semi_diameter([200.0, 200.0, 200.0]);
        assert_eq!(flat, round);
    }

    #[test]
    fn test_semi_diameter_saturates_inside_the_body() {
        let inside = at_distance(1e-9, Vector3::x());
        assert_eq!(
            inside.angular_semi_diameter(Body::Earth.radii_km()),
            std::f64::consts::FRAC_PI_2
        );
        let nowhere = Position::barycentric(Vector3::zeros(), Vector3::zeros(), 0);
        assert_eq!(
            nowhere.angular_semi_diameter(Body::Earth.radii_km()),
            std::f64::consts::FRAC_PI_2
        );
    }

    #[test]
    fn test_a_sphere_projects_to_a_circle() {
        // An arbitrary line of sight and an arbitrary body orientation: a
        // sphere has no orientation to reveal.
        let p = at_distance(0.5, Vector3::new(0.3, -0.7, 0.4));
        let frame = FixedFrame(rot_z(0.7) * rot_x(-0.3));
        let (major, minor, pa) = p.apparent_ellipse(&frame, [1737.4; 3], &t_j2000());
        assert!((major - minor).abs() < 1e-15);
        assert!((major - p.angular_semi_diameter([1737.4; 3])).abs() < 1e-15);
        // A circle has no major axis to report an angle for.
        assert_eq!(pa, 0.0);
    }

    #[test]
    fn test_pole_on_view_of_an_oblate_body_is_a_circle() {
        // Line of sight along the body's pole: the outline is the equator.
        let p = at_distance(5.0, Vector3::z());
        let radii = Body::Jupiter.radii_km();
        let (major, minor, _) = p.apparent_ellipse(&ICRS, radii, &t_j2000());
        assert!((major - minor).abs() < 1e-15);
        assert!((major - p.angular_semi_diameter(radii)).abs() < 1e-15);
    }

    #[test]
    fn test_jupiter_equator_on_reproduces_its_flattening() {
        // Line of sight in the body's equatorial plane, so the polar axis is
        // projected at its full length.
        let p = at_distance(5.0, Vector3::x());
        let radii = Body::Jupiter.radii_km();
        let (major, minor, pa) = p.apparent_ellipse(&ICRS, radii, &t_j2000());

        let flattening = (radii[0] - radii[2]) / radii[0];
        assert!(
            (flattening - 0.06487).abs() < 5e-6,
            "flattening {flattening}"
        );
        assert!(
            (minor / major - (1.0 - flattening)).abs() < 1e-6,
            "ratio {} against {}",
            minor / major,
            1.0 - flattening
        );
        // The major axis is the equator, which runs east and west on the sky.
        assert!((pa - std::f64::consts::FRAC_PI_2).abs() < 1e-12, "pa {pa}");
        assert!((major - p.angular_semi_diameter(radii)).abs() < 1e-15);
    }

    #[test]
    fn test_a_tilted_pole_tilts_the_ellipse() {
        // Turn the body 30° about the line of sight: the pole is projected at
        // position angle 30°, so the equator lies at 120°.
        let p = at_distance(5.0, Vector3::x());
        let frame = FixedFrame(rot_x(30f64.to_radians()));
        let (_, _, pa) = p.apparent_ellipse(&frame, Body::Jupiter.radii_km(), &t_j2000());
        assert!(
            (pa.to_degrees() - 120.0).abs() < 1e-9,
            "position angle {} degrees",
            pa.to_degrees()
        );
    }

    #[test]
    fn test_a_triaxial_body_seen_down_its_long_axis() {
        // Radii 3, 2, 1 along body x, y, z seen along body x: the outline has
        // semi-axes 2 and 1.
        let p = at_distance(1.0, Vector3::x());
        let (major, minor, _) = p.apparent_ellipse(&ICRS, [3.0, 2.0, 1.0], &t_j2000());
        assert!((major - (2.0f64 / AU_KM).asin()).abs() < 1e-18);
        assert!((minor - (1.0f64 / AU_KM).asin()).abs() < 1e-18);
    }

    /// The published 18.90″ and 5.14″ belong to this epoch, not to the 2007
    /// HiRISE one: Mars was 0.9304 AU from the Earth when the Mars Orbiter
    /// Camera took its Earth-and-Moon portrait, against 0.9507 AU four years
    /// later, and the difference is a fifth of an arcsecond on the Earth.
    #[test]
    fn test_earth_and_moon_from_mars_at_the_moc_epoch() {
        let mut kernel = de421();
        let t = moc_epoch();
        let mars = kernel.at("mars", &t).unwrap();

        let earth = mars.observe("earth", &mut kernel, &t).unwrap();
        let earth_diameter = 2.0 * earth.angular_semi_diameter(Body::Earth.radii_km()) * RAD2ASEC;
        assert!(
            (earth_diameter - 18.90).abs() < 0.02,
            "the Earth is {earth_diameter} arcseconds across, expected 18.90"
        );

        let moon = mars.observe("moon", &mut kernel, &t).unwrap();
        let moon_diameter = 2.0 * moon.angular_semi_diameter(Body::Moon.radii_km()) * RAD2ASEC;
        assert!(
            (moon_diameter - 5.14).abs() < 0.02,
            "the Moon is {moon_diameter} arcseconds across, expected 5.14"
        );
    }

    #[test]
    fn test_earth_and_moon_from_mars_at_the_hirise_epoch() {
        let mut kernel = de421();
        let t = hirise_epoch();
        let mars = kernel.at("mars", &t).unwrap();

        let earth = mars.observe("earth", &mut kernel, &t).unwrap();
        let earth_diameter = 2.0 * earth.angular_semi_diameter(Body::Earth.radii_km()) * RAD2ASEC;
        assert!(
            (earth_diameter - 18.50).abs() < 0.02,
            "the Earth is {earth_diameter} arcseconds across, expected 18.50"
        );

        let moon = mars.observe("moon", &mut kernel, &t).unwrap();
        let moon_diameter = 2.0 * moon.angular_semi_diameter(Body::Moon.radii_km()) * RAD2ASEC;
        assert!(
            (moon_diameter - 5.05).abs() < 0.02,
            "the Moon is {moon_diameter} arcseconds across, expected 5.05"
        );
    }

    #[test]
    fn test_the_earths_apparent_ellipse_from_mars_is_barely_flattened() {
        let mut kernel = de421();
        let t = hirise_epoch();
        let mars = kernel.at("mars", &t).unwrap();
        let earth = mars.observe("earth", &mut kernel, &t).unwrap();

        let radii = Body::Earth.radii_km();
        let frame = PlanetaryConstants::new().frame_for(399).unwrap();
        let (major, minor, pa) = earth.apparent_ellipse(frame.as_ref(), radii, &t);

        // The projected major axis of an oblate body is its equatorial radius,
        // whatever the line of sight.
        assert!((2.0 * major * RAD2ASEC - 18.50).abs() < 0.02);
        // The polar axis is foreshortened by between nothing and the full
        // flattening; the Earth's pole is well off the line of sight here.
        let ratio = minor / major;
        assert!(
            (radii[2] / radii[0]..=1.0).contains(&ratio),
            "axis ratio {ratio}"
        );
        assert!((ratio - 0.99718).abs() < 1e-5, "axis ratio {ratio}");
        // The Earth's pole is within a degree of celestial north from Mars, so
        // the equator — the major axis — runs east and west.
        assert!((pa.to_degrees() - 90.0).abs() < 1.0, "position angle {pa}");
    }

    /// The projection of an oblate spheroid has a closed form: seen from a
    /// direction that makes an angle `φ` with the equatorial plane, the
    /// outline has semi-axes `a` and `√(c² cos²φ + a² sin²φ)`. The general
    /// Schur-complement projection must reproduce it at every `φ`.
    #[test]
    fn test_an_oblate_body_matches_the_closed_form() {
        let radii = Body::Jupiter.radii_km();
        let (a, c) = (radii[0], radii[2]);
        let distance_au = 5.0;

        for degrees in [0.0f64, 7.5, 23.0, 45.0, 66.0, 90.0] {
            let phi = degrees.to_radians();
            // A line of sight at latitude φ above the body's equator.
            let p = at_distance(distance_au, Vector3::new(phi.cos(), 0.0, phi.sin()));
            let (major, minor, _) = p.apparent_ellipse(&ICRS, radii, &t_j2000());

            let distance_km = distance_au * AU_KM;
            let expected_minor =
                ((c * phi.cos()).powi(2) + (a * phi.sin()).powi(2)).sqrt() / distance_km;
            assert!(
                (major.sin() - a / distance_km).abs() < 1e-18,
                "semi-major at {degrees}°"
            );
            assert!(
                (minor.sin() - expected_minor).abs() < 1e-15,
                "semi-minor at {degrees}°: {} against {}",
                minor.sin(),
                expected_minor
            );
        }
    }
}

#[cfg(all(test, feature = "python-tests"))]
mod python_tests {
    use crate::jplephem::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::planetlib::Body;
    use crate::pybridge::{PyRustBridge, PythonResult};
    use crate::time::Timescale;

    /// Skyfield has no angular diameter of its own, so it supplies the
    /// light-time-corrected distance and the same `asin` is taken on both
    /// sides. Agreement to a microarcsecond means the geometry agrees.
    #[test]
    fn test_angular_diameter_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Python bridge");
        let python = bridge
            .run_py_to_json(
                r#"
from numpy import arcsin
from skyfield.api import load
import json

ts = load.timescale()
eph = load('test_data/de421.bsp')
t = ts.utc(2003, 5, 8, 13, 0, 0)

mars = eph['mars'].at(t)
out = {}
for name, radius_km in (('earth', 6378.1366), ('moon', 1737.4)):
    distance_km = mars.observe(eph[name]).distance().km
    out[name] = float(2.0 * arcsin(radius_km / distance_km))

rust.collect_string(json.dumps(out))
"#,
            )
            .expect("Skyfield angular diameter");

        let inner = match PythonResult::try_from(python.as_str()).expect("Python result") {
            PythonResult::String(s) => s,
            other => panic!("expected a string, got {:?}", other),
        };
        let parsed: serde_json::Value = serde_json::from_str(&inner).expect("JSON");

        let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
        let t = Timescale::default().utc((2003, 5, 8, 13, 0, 0.0));
        let mars = kernel.at("mars", &t).unwrap();

        for (name, body) in [("earth", Body::Earth), ("moon", Body::Moon)] {
            let target = mars.observe(name, &mut kernel, &t).unwrap();
            let rust = 2.0 * target.angular_semi_diameter(body.radii_km());
            let skyfield = parsed[name].as_f64().unwrap();
            let arcsec = (rust - skyfield).abs().to_degrees() * 3600.0;
            assert!(arcsec < 1e-6, "{name}: {rust} against {skyfield}");
        }
    }
}
