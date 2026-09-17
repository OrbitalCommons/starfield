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
//! All three are methods on [`Position`], all three need the Sun's position
//! and therefore take the kernel, and all three work for any observer, not
//! just an Earth-bound one — the guiding case for this module is the Earth
//! seen from Mars orbit.
//!
//! The Sun is taken from the kernel at the observation time, exactly as
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
use crate::positions::Position;
use crate::time::Time;
use crate::{Result, StarfieldError};

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
        let observer = self.require_observer()?.position;
        let sun = kernel.at("sun", t)?.position;

        // Skyfield: u is observer → target, v is Sun → target.
        let observer_to_target = self.position;
        let sun_to_target = self.position + observer - sun;
        Ok(angle_between(&observer_to_target, &sun_to_target))
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
        Ok(0.5 * (1.0 + self.phase_angle(kernel, t)?.cos()))
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
        let observer = self.require_observer()?.position;
        let sun = kernel.at("sun", t)?.position;

        let observer_to_sun = sun - observer;
        Ok(angle_between(&observer_to_sun, &self.position))
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
