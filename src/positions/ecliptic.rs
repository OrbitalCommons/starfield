//! Heliocentric ecliptic J2000 state vectors
//!
//! Provides [`EclipticStateVector`], a convenient type for working with
//! heliocentric ecliptic J2000 positions and velocities. The ecliptic
//! coordinate system is natural for solar system dynamics because
//! planetary orbits lie close to the ecliptic plane.
//!
//! # Coordinate System
//!
//! - **Origin**: Sun (heliocentric)
//! - **Reference plane**: Ecliptic plane at J2000
//! - **X-axis**: Toward vernal equinox at J2000
//! - **Z-axis**: Toward ecliptic north pole at J2000
//!
//! # Example
//!
//! ```ignore
//! let loader = starfield::Loader::new();
//! let ts = loader.timescale();
//! let mut kernel = loader.open("de421.bsp")?;
//!
//! let t = ts.tdb_jd(2451545.0);
//! let mars_ecliptic = EclipticStateVector::compute(&mut kernel, "mars", "sun", &t)?;
//! println!("Distance: {:.6} AU", mars_ecliptic.distance());
//! println!("Speed: {:.6} AU/day", mars_ecliptic.speed());
//! ```

use nalgebra::Vector3;

use crate::coordinates::cartesian::Cartesian3;
use crate::framelib::{ECLIPJ2000_MATRIX, ECLIPJ2000_MATRIX_INV};
use crate::jplephem::errors::Result;
use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem_ext::SpiceKernelExt;
use crate::time::Time;

/// A heliocentric ecliptic J2000 state vector (position + velocity).
///
/// Contains the position and velocity of a body relative to the Sun,
/// expressed in the ecliptic J2000 coordinate system. This frame is
/// widely used in solar system dynamics because planetary orbits lie
/// close to the ecliptic plane.
///
/// # Fields
///
/// - `position` -- Heliocentric ecliptic J2000 position in AU
/// - `velocity` -- Heliocentric ecliptic J2000 velocity in AU/day
/// - `target` -- NAIF ID of the target body
/// - `center` -- NAIF ID of the center body (typically 10 for Sun)
#[derive(Debug, Clone)]
pub struct EclipticStateVector {
    /// Heliocentric ecliptic J2000 position in AU
    pub position: Cartesian3,
    /// Heliocentric ecliptic J2000 velocity in AU/day
    pub velocity: Cartesian3,
    /// NAIF ID of the target body
    pub target: i32,
    /// NAIF ID of the center body
    pub center: i32,
}

impl EclipticStateVector {
    /// Compute the heliocentric ecliptic J2000 state vector of a body.
    ///
    /// Looks up both the target and center body in the kernel, subtracts
    /// to get the relative state, then rotates from equatorial J2000
    /// (the native SPK frame) into ecliptic J2000.
    ///
    /// # Arguments
    ///
    /// * `kernel` -- An open SPK/BSP kernel
    /// * `target` -- Name of the target body (e.g. "mars", "earth")
    /// * `center` -- Name of the center body (e.g. "sun")
    /// * `time` -- The computation epoch
    ///
    /// # Example
    ///
    /// ```ignore
    /// let sv = EclipticStateVector::compute(&mut kernel, "mars", "sun", &t)?;
    /// ```
    pub fn compute(
        kernel: &mut SpiceKernel,
        target: &str,
        center: &str,
        time: &Time,
    ) -> Result<Self> {
        let target_state = kernel.compute_at(target, time)?;
        let center_state = kernel.compute_at(center, time)?;

        let target_vf = kernel.get(target)?;
        let center_vf = kernel.get(center)?;

        // Relative state in equatorial J2000 (AU and AU/day)
        let rel_pos = Vector3::new(
            target_state.position.x - center_state.position.x,
            target_state.position.y - center_state.position.y,
            target_state.position.z - center_state.position.z,
        );
        let rel_vel = target_state.velocity - center_state.velocity;

        // Rotate from equatorial J2000 to ecliptic J2000
        let ecl_pos = *ECLIPJ2000_MATRIX * rel_pos;
        let ecl_vel = *ECLIPJ2000_MATRIX * rel_vel;

        Ok(EclipticStateVector {
            position: Cartesian3::from_vector3(ecl_pos),
            velocity: Cartesian3::from_vector3(ecl_vel),
            target: target_vf.target_id,
            center: center_vf.target_id,
        })
    }

    /// Convert the ecliptic state vector back to equatorial J2000 components.
    ///
    /// Returns `(position, velocity)` as `Cartesian3` values in the
    /// equatorial J2000 frame (AU and AU/day).
    pub fn to_equatorial(&self) -> (Cartesian3, Cartesian3) {
        let eq_pos = *ECLIPJ2000_MATRIX_INV * self.position.to_vector3();
        let eq_vel = *ECLIPJ2000_MATRIX_INV * self.velocity.to_vector3();
        (
            Cartesian3::from_vector3(eq_pos),
            Cartesian3::from_vector3(eq_vel),
        )
    }

    /// Distance from the center body in AU.
    pub fn distance(&self) -> f64 {
        self.position.magnitude()
    }

    /// Speed relative to the center body in AU/day.
    pub fn speed(&self) -> f64 {
        self.velocity.magnitude()
    }

    /// Ecliptic longitude in radians, in [0, 2*PI).
    pub fn longitude(&self) -> f64 {
        let mut lon = self.position.y.atan2(self.position.x);
        if lon < 0.0 {
            lon += std::f64::consts::TAU;
        }
        lon
    }

    /// Ecliptic latitude in radians, in [-PI/2, PI/2].
    pub fn latitude(&self) -> f64 {
        let r = self.distance();
        if r == 0.0 {
            return 0.0;
        }
        (self.position.z / r).asin()
    }
}

impl std::fmt::Display for EclipticStateVector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EclipticStateVector({} -> {}) pos=[{:.6}, {:.6}, {:.6}] AU  vel=[{:.6}, {:.6}, {:.6}] AU/day",
            self.center,
            self.target,
            self.position.x, self.position.y, self.position.z,
            self.velocity.x, self.velocity.y, self.velocity.z,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
    }

    fn j2000_time() -> Time {
        Timescale::default().tdb_jd(2451545.0)
    }

    #[test]
    fn test_earth_near_ecliptic_plane() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let earth = EclipticStateVector::compute(&mut kernel, "earth", "sun", &t).unwrap();

        // Earth's orbit is the ecliptic plane by definition, so its
        // ecliptic latitude should be extremely small (arcseconds).
        let lat_deg = earth.latitude().to_degrees();
        assert!(
            lat_deg.abs() < 0.01,
            "Earth ecliptic latitude should be near zero, got {lat_deg:.6} deg"
        );

        // Distance should be roughly 1 AU
        let dist = earth.distance();
        assert!(
            (0.98..1.02).contains(&dist),
            "Earth-Sun distance should be ~1 AU, got {dist:.6}"
        );
    }

    #[test]
    fn test_roundtrip_equatorial_ecliptic() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        // Get Mars ecliptic state
        let mars_ecl = EclipticStateVector::compute(&mut kernel, "mars", "sun", &t).unwrap();

        // Convert back to equatorial
        let (eq_pos, eq_vel) = mars_ecl.to_equatorial();

        // Get Mars equatorial state directly
        let mars_eq = kernel.compute_at("mars", &t).unwrap();
        let sun_eq = kernel.compute_at("sun", &t).unwrap();

        let direct_pos = Cartesian3::new(
            mars_eq.position.x - sun_eq.position.x,
            mars_eq.position.y - sun_eq.position.y,
            mars_eq.position.z - sun_eq.position.z,
        );
        let direct_vel = Cartesian3::from_vector3(mars_eq.velocity - sun_eq.velocity);

        assert_relative_eq!(eq_pos.x, direct_pos.x, epsilon = 1e-12);
        assert_relative_eq!(eq_pos.y, direct_pos.y, epsilon = 1e-12);
        assert_relative_eq!(eq_pos.z, direct_pos.z, epsilon = 1e-12);
        assert_relative_eq!(eq_vel.x, direct_vel.x, epsilon = 1e-12);
        assert_relative_eq!(eq_vel.y, direct_vel.y, epsilon = 1e-12);
        assert_relative_eq!(eq_vel.z, direct_vel.z, epsilon = 1e-12);
    }

    #[test]
    fn test_magnitude_preserved_through_rotation() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        // Get Mars state in both frames
        let mars_ecl = EclipticStateVector::compute(&mut kernel, "mars", "sun", &t).unwrap();

        let mars_eq = kernel.compute_at("mars", &t).unwrap();
        let sun_eq = kernel.compute_at("sun", &t).unwrap();

        let eq_pos_mag = ((mars_eq.position.x - sun_eq.position.x).powi(2)
            + (mars_eq.position.y - sun_eq.position.y).powi(2)
            + (mars_eq.position.z - sun_eq.position.z).powi(2))
        .sqrt();

        let eq_vel_mag = (mars_eq.velocity - sun_eq.velocity).norm();

        // Rotation preserves magnitude
        assert_relative_eq!(mars_ecl.distance(), eq_pos_mag, epsilon = 1e-12);
        assert_relative_eq!(mars_ecl.speed(), eq_vel_mag, epsilon = 1e-12);
    }

    #[test]
    fn test_mars_ecliptic_latitude_small() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let mars = EclipticStateVector::compute(&mut kernel, "mars", "sun", &t).unwrap();

        // Mars inclination is ~1.85 deg, so ecliptic latitude should be modest
        let lat_deg = mars.latitude().to_degrees();
        assert!(
            lat_deg.abs() < 5.0,
            "Mars ecliptic latitude should be small, got {lat_deg:.4} deg"
        );
    }

    #[test]
    fn test_longitude_in_range() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        for body in &["mercury", "venus", "earth", "mars"] {
            let sv = EclipticStateVector::compute(&mut kernel, body, "sun", &t).unwrap();
            let lon = sv.longitude();
            assert!(
                (0.0..std::f64::consts::TAU).contains(&lon),
                "{body} longitude {lon} out of range"
            );
        }
    }

    #[test]
    fn test_display() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let sv = EclipticStateVector::compute(&mut kernel, "earth", "sun", &t).unwrap();
        let s = format!("{sv}");
        assert!(s.contains("EclipticStateVector"));
    }

    #[test]
    fn test_speed_reasonable() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let earth = EclipticStateVector::compute(&mut kernel, "earth", "sun", &t).unwrap();

        // Earth orbital speed is ~29.78 km/s = ~0.01720 AU/day
        let speed = earth.speed();
        assert!(
            (0.015..0.020).contains(&speed),
            "Earth speed should be ~0.0172 AU/day, got {speed:.6}"
        );
    }
}
