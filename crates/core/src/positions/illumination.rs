//! Illumination geometry: phase angle, illuminated fraction, elongation.
//!
//! Three angles describe how the Sun lights a body for a given observer:
//!
//! * the **phase angle**, Sun–target–observer, which is zero when the observer
//!   looks along the same line the sunlight arrives on and sees a full disc,
//!   and 180° when the observer is behind the target and sees only night side;
//! * the **illuminated fraction** of the apparent disc, which is a direct
//!   function of the phase angle for a spherical body; and
//! * the **solar elongation**, Sun–observer–target, the angle a planet stands
//!   away from the Sun in the observer's sky.
//!
//! All three, with the bright limb's position angle and the Sun's direction
//! on the observer's sky, are computed by [`IlluminationGeometry`] from plain
//! vectors, with no ephemeris needed: give it the observer, target and Sun and
//! it closes the triangle. The same three angles are methods on [`Position`]
//! that take the Sun from a kernel and delegate to it. All of them work for
//! any observer, not just an Earth-bound one — the guiding case for this
//! module is the Earth seen from Mars orbit.
//!
//! The [`Position`] methods take the Sun from the kernel at the observation time, exactly as
//! Skyfield's `positionlib.phase_angle` does. This is *not* the approximation
//! used inside [`magnitudelib`](crate::magnitudelib), which places the Sun at
//! the solar system barycentre; the two differ by up to about 0.005 AU, which
//! matters at the millimagnitude level and not at all for the geometry here.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::kernel::SpiceKernel;
//! use starfield_core::jplephem_ext::SpiceKernelExt;
//! use starfield_core::time::Timescale;
//!
//! let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
//! let t = Timescale::default().utc((2007, 10, 3, 0, 0, 0.0));
//!
//! let mars = kernel.at("mars", &t).unwrap();
//! let earth = mars.observe("earth", &mut kernel, &t).unwrap();
//!
//! let alpha = earth.phase_angle(&mut kernel, &t).unwrap();
//! println!("phase {:.1}°, {:.0}% lit", alpha.to_degrees(),
//!          100.0 * earth.illuminated_fraction(&mut kernel, &t).unwrap());
//! ```

use nalgebra::Vector3;

use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem_ext::SpiceKernelExt;
use crate::positions::{position_angle, sky_basis, Position};
use crate::time::Time;
use crate::{Result, StarfieldError};

/// The Sun–target–observer triangle, from which every illumination quantity
/// follows without reference to an ephemeris.
///
/// It holds two vectors in one frame (in practice the ICRF) and one length
/// unit (in practice AU): the target as seen from the observer, and the Sun as
/// seen from the target. Which epochs they belong to, and so which light-time
/// corrections they carry, is up to the caller; the kernel-based methods on
/// [`Position`] fill them in for their own conventions and then delegate here.
///
/// # Example
///
/// ```
/// use nalgebra::Vector3;
/// use starfield_core::positions::illumination::IlluminationGeometry;
///
/// // The observer at the origin, the target 1 AU along x, and the Sun off to
/// // the side, so the observer sees the target at quarter phase.
/// let geometry = IlluminationGeometry::from_barycentric(
///     Vector3::zeros(),
///     Vector3::new(1.0, 0.0, 0.0),
///     Vector3::new(1.0, 1.0, 0.0),
/// );
/// assert!((geometry.phase_angle().to_degrees() - 90.0).abs() < 1e-12);
/// assert!((geometry.illuminated_fraction() - 0.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IlluminationGeometry {
    target_from_observer: Vector3<f64>,
    sun_from_target: Vector3<f64>,
}

impl IlluminationGeometry {
    /// The geometry from the target's position relative to the observer and
    /// the Sun's position relative to the target.
    pub fn new(target_from_observer: Vector3<f64>, sun_from_target: Vector3<f64>) -> Self {
        Self {
            target_from_observer,
            sun_from_target,
        }
    }

    /// The geometry from barycentric positions of the observer, the target
    /// and the Sun.
    ///
    /// For an astrometric view, `target` is the target's position at the
    /// moment the light left it, and `sun` the Sun's at whichever epoch the
    /// caller's convention calls for.
    pub fn from_barycentric(
        observer: Vector3<f64>,
        target: Vector3<f64>,
        sun: Vector3<f64>,
    ) -> Self {
        Self::new(target - observer, sun - target)
    }

    /// The target's position relative to the observer: the line of sight.
    pub fn target_from_observer(&self) -> Vector3<f64> {
        self.target_from_observer
    }

    /// The Sun's position relative to the target.
    pub fn sun_from_target(&self) -> Vector3<f64> {
        self.sun_from_target
    }

    /// The observer–target distance, in the unit of the input vectors.
    pub fn distance(&self) -> f64 {
        self.target_from_observer.norm()
    }

    /// The Sun–target distance, in the unit of the input vectors.
    pub fn heliocentric_distance(&self) -> f64 {
        self.sun_from_target.norm()
    }

    /// The unit vector from the target toward the Sun.
    pub fn sun_direction(&self) -> Vector3<f64> {
        self.sun_from_target.normalize()
    }

    /// The unit vector from the target toward the observer.
    pub fn observer_direction(&self) -> Vector3<f64> {
        -self.target_from_observer.normalize()
    }

    /// The phase angle: the Sun–target–observer angle in radians.
    ///
    /// Zero means the target is fully lit as the observer sees it, 180° means
    /// the observer sees only its night side. Ports Skyfield's
    /// `positionlib.phase_angle`.
    pub fn phase_angle(&self) -> f64 {
        // Skyfield: u is observer → target, v is Sun → target; negating both
        // leaves the angle unchanged to the last bit.
        angle_between(&-self.target_from_observer, &self.sun_from_target)
    }

    /// The fraction of the target's disc that is lit, from 0.0 to 1.0.
    ///
    /// `(1 + cos α) / 2` for the phase angle α, which assumes the target is a
    /// sphere. Ports Skyfield's `positionlib.fraction_illuminated`.
    pub fn illuminated_fraction(&self) -> f64 {
        0.5 * (1.0 + self.phase_angle().cos())
    }

    /// The solar elongation: the Sun–observer–target angle in radians.
    ///
    /// Zero means the target lies in the same direction as the Sun, 180° means
    /// the target is opposite the Sun in the observer's sky.
    pub fn solar_elongation(&self) -> f64 {
        let sun_from_observer = self.target_from_observer + self.sun_from_target;
        angle_between(&sun_from_observer, &self.target_from_observer)
    }

    /// The position angle of the bright limb, radians east of celestial north
    /// in the observer's sky frame, in `[0, 2π)`.
    ///
    /// This is the direction from the centre of the disc toward the Sun as
    /// projected on the sky: the position angle of the sub-solar point, of
    /// the midpoint of the illuminated limb, and of the normal to the
    /// terminator. It is ill-conditioned near zero phase, where the Sun lies
    /// almost along the line of sight.
    pub fn bright_limb_position_angle(&self) -> f64 {
        position_angle(&self.target_from_observer, &self.sun_from_target)
    }

    /// The direction of the Sun from the target in the observer's sky frame,
    /// as `(east, north, toward the observer)` components of a unit vector.
    ///
    /// East and north are those of [`sky_basis`] along the line of sight, and
    /// the third axis points back at the observer: the frame in which a disc
    /// is drawn on the sky, with north up and east to the left as the observer
    /// sees it. Because `east × north` points along the line of sight, away
    /// from the observer, this `(east, north, toward the observer)` triad is
    /// left-handed. The z component is the cosine of the phase angle, and
    /// `atan2(x, y)` is the bright limb's position angle.
    pub fn sun_direction_sky_frame(&self) -> Vector3<f64> {
        let (east, north, line_of_sight) = sky_basis(&self.target_from_observer);
        let sun = self.sun_direction();
        Vector3::new(sun.dot(&east), sun.dot(&north), -sun.dot(&line_of_sight))
    }
}

impl Position {
    /// The phase angle: the Sun–target–observer angle in radians.
    ///
    /// Zero means the target is fully lit as the observer sees it, 180° means
    /// the observer sees only its night side. Ports Skyfield's
    /// `positionlib.phase_angle`, taking the Sun from the kernel at `t`
    /// without a light-time correction of its own.
    ///
    /// `self` should be an astrometric or apparent position — the output of
    /// [`observe`](Position::observe), which records the observer.
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::MissingObserver`] if `self` does not carry
    /// the observer's barycentric position, and
    /// [`StarfieldError::EphemerisError`] if the kernel cannot place the Sun
    /// at `t`.
    pub fn phase_angle(&self, kernel: &mut SpiceKernel, t: &Time) -> Result<f64> {
        Ok(self.illumination_at(kernel, t)?.phase_angle())
    }

    /// The fraction of the target's disc that is lit, from 0.0 to 1.0.
    ///
    /// `(1 + cos α) / 2` for the phase angle α of
    /// [`phase_angle`](Position::phase_angle), which assumes the target is a
    /// sphere. Ports Skyfield's `positionlib.fraction_illuminated`.
    ///
    /// # Errors
    ///
    /// The errors of [`phase_angle`](Position::phase_angle).
    pub fn illuminated_fraction(&self, kernel: &mut SpiceKernel, t: &Time) -> Result<f64> {
        Ok(self.illumination_at(kernel, t)?.illuminated_fraction())
    }

    /// The solar elongation: the Sun–observer–target angle in radians.
    ///
    /// Zero means the target lies in the same direction as the Sun, 180° means
    /// the target is opposite the Sun in the observer's sky and at opposition.
    ///
    /// # Errors
    ///
    /// The errors of [`phase_angle`](Position::phase_angle).
    pub fn solar_elongation(&self, kernel: &mut SpiceKernel, t: &Time) -> Result<f64> {
        Ok(self.illumination_at(kernel, t)?.solar_elongation())
    }

    /// The illumination geometry with the Sun taken from the kernel at `t`,
    /// the convention of Skyfield's `positionlib.phase_angle`.
    fn illumination_at(&self, kernel: &mut SpiceKernel, t: &Time) -> Result<IlluminationGeometry> {
        let observer = self.require_observer()?.position;
        let sun = kernel.at("sun", t)?.position;
        let target = observer + self.position;
        Ok(IlluminationGeometry::new(self.position, sun - target))
    }

    /// The observer's barycentric position, or an error naming what is
    /// missing.
    ///
    /// `observe()` always records it; a position built by hand may not.
    pub(crate) fn require_observer(&self) -> Result<&Position> {
        self.observer_barycentric
            .as_deref()
            .ok_or(StarfieldError::MissingObserver)
    }
}

/// The angle in radians between two vectors, by the formula of Kahan's
/// *Mindless Assessments of Roundoff in Floating-Point Computation* §12 that
/// Skyfield's `functions.angle_between` uses.
///
/// It stays accurate for nearly parallel and nearly antiparallel vectors,
/// where the textbook `acos` of a normalised dot product loses digits.
fn angle_between(u: &Vector3<f64>, v: &Vector3<f64>) -> f64 {
    let a = u * v.norm();
    let b = v * u.norm();
    2.0 * (a - b).norm().atan2((a + b).norm())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Timescale;
    use approx::assert_relative_eq;
    use nalgebra::Matrix3;
    use std::f64::consts::PI;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
    }

    /// The epoch of the HiRISE image PSP_005558_9040, in which Mars
    /// Reconnaissance Orbiter photographed a gibbous Earth.
    fn hirise_epoch() -> Time {
        Timescale::default().utc((2007, 10, 3, 0, 0, 0.0))
    }

    #[test]
    fn test_angle_between_orthogonal() {
        let u = Vector3::new(3.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 0.5, 0.0);
        assert_relative_eq!(angle_between(&u, &v), PI / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_angle_between_parallel_and_antiparallel() {
        let u = Vector3::new(1.0, 2.0, 3.0);
        assert_relative_eq!(angle_between(&u, &(u * 2.0)), 0.0, epsilon = 1e-15);
        assert_relative_eq!(angle_between(&u, &(-u)), PI, epsilon = 1e-15);
    }

    /// Observer at the origin, target 1 AU along +x, Sun 1 AU from the
    /// target along +y: a quarter phase with the Sun due east on the sky.
    fn quarter_phase() -> IlluminationGeometry {
        IlluminationGeometry::from_barycentric(
            Vector3::zeros(),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        )
    }

    #[test]
    fn test_from_barycentric_forms_the_relative_vectors() {
        let observer = Vector3::new(0.3, -1.2, 0.4);
        let target = Vector3::new(4.1, 2.2, -0.7);
        let sun = Vector3::new(0.002, -0.004, 0.001);
        let geometry = IlluminationGeometry::from_barycentric(observer, target, sun);
        assert_eq!(geometry.target_from_observer(), target - observer);
        assert_eq!(geometry.sun_from_target(), sun - target);
        assert_eq!(
            geometry,
            IlluminationGeometry::new(target - observer, sun - target)
        );
        assert_relative_eq!(geometry.distance(), (target - observer).norm());
        assert_relative_eq!(geometry.heliocentric_distance(), (sun - target).norm());
    }

    #[test]
    fn test_quarter_phase_geometry() {
        let geometry = quarter_phase();
        assert_relative_eq!(geometry.phase_angle(), PI / 2.0, epsilon = 1e-15);
        assert_relative_eq!(geometry.illuminated_fraction(), 0.5, epsilon = 1e-15);
        assert_relative_eq!(geometry.solar_elongation(), PI / 4.0, epsilon = 1e-15);
        assert_relative_eq!(geometry.heliocentric_distance(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(geometry.sun_direction(), Vector3::y(), epsilon = 1e-15);
        assert_relative_eq!(
            geometry.observer_direction(),
            -Vector3::x(),
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_full_and_new_phase() {
        let full =
            IlluminationGeometry::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(-3.0, 0.0, 0.0));
        assert_relative_eq!(full.phase_angle(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(full.illuminated_fraction(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(full.solar_elongation(), PI, epsilon = 1e-15);

        let new =
            IlluminationGeometry::new(Vector3::new(2.0, 0.0, 0.0), Vector3::new(3.0, 0.0, 0.0));
        assert_relative_eq!(new.phase_angle(), PI, epsilon = 1e-15);
        assert_relative_eq!(new.illuminated_fraction(), 0.0, epsilon = 1e-15);
        assert_relative_eq!(new.solar_elongation(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_bright_limb_of_the_cardinal_directions() {
        // Looking along +x the sky has east toward +y and north toward +z.
        let line_of_sight = Vector3::new(1.0, 0.0, 0.0);
        for (sun_from_target, expected) in [
            (Vector3::z(), 0.0),
            (Vector3::y(), PI / 2.0),
            (-Vector3::z(), PI),
            (-Vector3::y(), 3.0 * PI / 2.0),
        ] {
            let geometry = IlluminationGeometry::new(line_of_sight, sun_from_target);
            assert_relative_eq!(
                geometry.bright_limb_position_angle(),
                expected,
                epsilon = 1e-15
            );
        }
    }

    #[test]
    fn test_sun_direction_sky_frame() {
        // Quarter phase with the Sun due east: all of it along the east axis.
        assert_relative_eq!(
            quarter_phase().sun_direction_sky_frame(),
            Vector3::x(),
            epsilon = 1e-15
        );

        // Sun behind the observer: straight down the axis toward the observer.
        let full =
            IlluminationGeometry::new(Vector3::new(0.0, 2.0, 0.0), Vector3::new(0.0, -5.0, 0.0));
        assert_relative_eq!(
            full.sun_direction_sky_frame(),
            Vector3::z(),
            epsilon = 1e-15
        );

        // A general geometry: the components agree with the phase angle and
        // the bright limb's position angle.
        let geometry =
            IlluminationGeometry::new(Vector3::new(0.4, -1.3, 0.6), Vector3::new(-0.2, 0.9, 1.1));
        let sky = geometry.sun_direction_sky_frame();
        assert_relative_eq!(sky.norm(), 1.0, epsilon = 1e-15);
        assert_relative_eq!(sky.z, geometry.phase_angle().cos(), epsilon = 1e-14);
        assert_relative_eq!(
            sky.x.atan2(sky.y).rem_euclid(std::f64::consts::TAU),
            geometry.bright_limb_position_angle(),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_sun_direction_sky_frame_is_a_left_handed_decomposition() {
        let sun_from_target = Vector3::new(-0.2, 0.9, 1.1);
        for line_of_sight in [
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.4, -1.3, 0.6),
            Vector3::new(-4.0, 1.5, -2.5),
            Vector3::new(1e-7, 0.0, 1.0),
            Vector3::new(0.0, -1e-7, -1.0),
            Vector3::z(),
            -Vector3::z() * 3.0,
        ] {
            let (east, north, away) = sky_basis(&line_of_sight);
            let toward_observer = -away;
            let determinant = Matrix3::from_columns(&[east, north, toward_observer]).determinant();
            assert_relative_eq!(determinant, -1.0, epsilon = 1e-14);
            assert_relative_eq!(
                east.cross(&north),
                line_of_sight.normalize(),
                epsilon = 1e-14
            );

            let geometry = IlluminationGeometry::new(line_of_sight, sun_from_target);
            let sky = geometry.sun_direction_sky_frame();
            assert_relative_eq!(
                sky.x * east + sky.y * north + sky.z * toward_observer,
                geometry.sun_direction(),
                epsilon = 1e-14
            );
        }
    }

    #[test]
    fn test_the_three_angles_close_the_triangle() {
        let geometry =
            IlluminationGeometry::new(Vector3::new(0.4, -1.3, 0.6), Vector3::new(-0.2, 0.9, 1.1));
        let at_sun = angle_between(
            &-geometry.sun_from_target(),
            &-(geometry.target_from_observer() + geometry.sun_from_target()),
        );
        assert_relative_eq!(
            geometry.phase_angle() + geometry.solar_elongation() + at_sun,
            PI,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_position_methods_delegate_to_the_geometry() {
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        let mars = kernel.at("mars", &t).unwrap();
        let earth = mars.observe("earth", &mut kernel, &t).unwrap();
        let sun = kernel.at("sun", &t).unwrap();

        let geometry = IlluminationGeometry::from_barycentric(
            mars.position,
            mars.position + earth.position,
            sun.position,
        );
        assert_relative_eq!(
            earth.phase_angle(&mut kernel, &t).unwrap(),
            geometry.phase_angle(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            earth.illuminated_fraction(&mut kernel, &t).unwrap(),
            geometry.illuminated_fraction(),
            epsilon = 1e-14
        );
        assert_relative_eq!(
            earth.solar_elongation(&mut kernel, &t).unwrap(),
            geometry.solar_elongation(),
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_earth_from_mars_at_hirise_epoch() {
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        let mars = kernel.at("mars", &t).unwrap();
        let earth = mars.observe("earth", &mut kernel, &t).unwrap();

        let phase_deg = earth.phase_angle(&mut kernel, &t).unwrap().to_degrees();
        assert!(
            (phase_deg - 98.0).abs() < 0.5,
            "Earth from Mars phase angle should be 98°, got {phase_deg}"
        );

        let fraction = earth.illuminated_fraction(&mut kernel, &t).unwrap();
        assert!(
            (fraction - 0.43).abs() < 0.01,
            "Earth from Mars illuminated fraction should be 0.43, got {fraction}"
        );
    }

    #[test]
    fn test_illuminated_fraction_follows_phase_angle() {
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        let mars = kernel.at("mars", &t).unwrap();
        let earth = mars.observe("earth", &mut kernel, &t).unwrap();

        let alpha = earth.phase_angle(&mut kernel, &t).unwrap();
        let fraction = earth.illuminated_fraction(&mut kernel, &t).unwrap();
        assert_relative_eq!(fraction, 0.5 * (1.0 + alpha.cos()), epsilon = 1e-15);
    }

    #[test]
    fn test_sun_is_at_zero_elongation_from_itself() {
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        let earth = kernel.at("earth", &t).unwrap();
        let sun = earth.observe("sun", &mut kernel, &t).unwrap();

        // All that separates the Sun's light-time corrected direction from
        // its direction at `t` is the distance it moves in 8 minutes.
        let elongation = sun.solar_elongation(&mut kernel, &t).unwrap();
        assert!(
            elongation < 1e-6,
            "the Sun should be at zero elongation from itself, got {elongation} rad"
        );
    }

    #[test]
    fn test_phase_angle_and_elongation_close_the_triangle() {
        // Sun–target–observer, Sun–observer–target and the angle at the Sun
        // are the three angles of one plane triangle and sum to π.
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let sun = kernel.at("sun", &t).unwrap();

        let phase = mars.phase_angle(&mut kernel, &t).unwrap();
        let elongation = mars.solar_elongation(&mut kernel, &t).unwrap();

        let sun_to_observer = earth.position - sun.position;
        let sun_to_target = earth.position + mars.position - sun.position;
        let at_sun = angle_between(&sun_to_observer, &sun_to_target);

        // Light time makes the triangle inexact at the arcsecond level.
        assert_relative_eq!(phase + elongation + at_sun, PI, epsilon = 1e-4);
    }

    #[test]
    fn test_elongation_of_an_opposition_planet_is_large() {
        // Mars came to opposition on 2020-10-13 at 23:19 UT. Its ecliptic
        // latitude of about 3° keeps the elongation a few degrees short of a
        // half turn, and leaves it that same small phase angle.
        let mut kernel = de421_kernel();
        let t = Timescale::default().utc((2020, 10, 13, 23, 0, 0.0));

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        let elongation_deg = mars.solar_elongation(&mut kernel, &t).unwrap().to_degrees();
        assert!(
            elongation_deg > 176.0,
            "Mars at opposition should be nearly 180° from the Sun, got {elongation_deg}"
        );

        let phase_deg = mars.phase_angle(&mut kernel, &t).unwrap().to_degrees();
        assert!(
            phase_deg < 3.0,
            "Mars at opposition should show almost no phase, got {phase_deg}"
        );
        assert!(
            mars.illuminated_fraction(&mut kernel, &t).unwrap() > 0.999,
            "Mars at opposition should be all but fully lit"
        );
    }

    #[test]
    fn test_missing_observer_is_an_error() {
        let mut kernel = de421_kernel();
        let t = hirise_epoch();

        // A barycentric position never records an observer.
        let earth = kernel.at("earth", &t).unwrap();
        assert!(earth.observer_barycentric.is_none());

        for result in [
            earth.phase_angle(&mut kernel, &t),
            earth.illuminated_fraction(&mut kernel, &t),
            earth.solar_elongation(&mut kernel, &t),
        ] {
            match result {
                Err(StarfieldError::MissingObserver) => {}
                other => panic!("expected MissingObserver, got {:?}", other),
            }
        }
    }
}
