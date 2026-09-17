//! Relativistic correction functions for astrometry
//!
//! Ported from Python Skyfield's `relativity.py`. Provides:
//! - [`light_time_difference`] — light-time projection between observer and barycenter
//! - [`add_deflection`] — gravitational light bending by a single mass (IERS 2003)
//! - [`add_aberration`] — stellar aberration correction (Klioner 2003)

use crate::constants::{AU_M, C, C_AUDAY, GS};
use nalgebra::Vector3;

/// Smallest positive normal f64, used to avoid division by zero.
const AVOID_DIVIDE_BY_ZERO: f64 = f64::MIN_POSITIVE;

/// Reciprocal masses of solar system bodies (mass of Sun / mass of body).
///
/// Used in gravitational deflection calculations.
pub const RMASSES: &[(&str, f64)] = &[
    ("sun", 1.0),
    ("jupiter", 1047.3486),
    ("saturn", 3497.898),
    ("uranus", 22902.98),
    ("neptune", 19412.24),
    ("venus", 408523.71),
    ("earth", 332946.050895),
    ("mars", 3098708.0),
    ("mercury", 6023600.0),
    ("moon", 27068700.387534),
    ("pluto", 135200000.0),
];

/// Bodies to consider for light deflection, ordered by importance.
pub const DEFLECTORS: &[&str] = &[
    "sun", "jupiter", "saturn", "moon", "venus", "uranus", "neptune",
];

/// Look up the reciprocal mass for a named body.
pub fn rmass(name: &str) -> Option<f64> {
    RMASSES.iter().find(|(n, _)| *n == name).map(|(_, m)| *m)
}

/// Compute light-time difference between barycenter and observer.
///
/// Returns the projection of the observer position onto the direction
/// toward the object, divided by the speed of light, in days.
///
/// # Arguments
/// * `position` - Position vector of the object (AU)
/// * `observer_pos` - Position vector of the observer (AU)
pub fn light_time_difference(position: &Vector3<f64>, observer_pos: &Vector3<f64>) -> f64 {
    let dis = position.norm();
    let u1 = position / (dis + AVOID_DIVIDE_BY_ZERO);
    u1.dot(observer_pos) / C_AUDAY
}

/// Apply gravitational light deflection from a single mass.
///
/// Implements the IERS 2003 Conventions formulation for the apparent
/// direction change of light passing near a gravitating body.
///
/// # Arguments
/// * `position` - Position of observed object from observer (AU), modified in-place
/// * `observer` - Observer position from solar system barycenter (AU)
/// * `deflector` - Deflector body position from solar system barycenter (AU)
/// * `rmass` - Reciprocal mass of the deflector (Sun mass / body mass)
pub fn add_deflection(
    position: &mut Vector3<f64>,
    observer: &Vector3<f64>,
    deflector: &Vector3<f64>,
    rmass: f64,
) {
    // Geometry vectors
    let pq = observer + *position - deflector; // deflector to object
    let pe = observer - deflector; // deflector to observer

    let pmag = position.norm();
    let qmag = pq.norm();
    let emag = pe.norm();

    // Unit vectors (guard against zero magnitude)
    let phat = if pmag > 0.0 {
        *position / pmag
    } else {
        Vector3::zeros()
    };
    let qhat = if qmag > 0.0 {
        pq / qmag
    } else {
        Vector3::zeros()
    };
    let ehat = if emag > 0.0 {
        pe / emag
    } else {
        Vector3::zeros()
    };

    // Dot products
    let pdotq = phat.dot(&qhat);
    let qdote = qhat.dot(&ehat);
    let edotp = ehat.dot(&phat);

    // Skip correction if deflector is nearly aligned with object (within ~1 arcsec)
    if edotp.abs() > 0.99999999999 {
        return;
    }

    // Deflection factors
    let fac1 = 2.0 * GS / (C * C * emag * AU_M * rmass);
    let fac2 = 1.0 + qdote;

    // Apply correction
    let correction = fac1 * (pdotq * ehat - edotp * qhat) / fac2 * pmag;
    *position += correction;
}

/// Apply stellar aberration correction.
///
/// Implements the Klioner (2003) formulation for the apparent shift in
/// position due to the observer's velocity.
///
/// # Arguments
/// * `position` - Relative position of object from observer (AU), modified in-place
/// * `velocity` - Observer velocity vector (AU/day)
/// * `light_time` - Light propagation time to the object (days)
pub fn add_aberration(position: &mut Vector3<f64>, velocity: &Vector3<f64>, light_time: f64) {
    let p1mag = light_time * C_AUDAY;
    let vemag = velocity.norm();

    let beta = vemag / C_AUDAY;
    let dot = position.dot(velocity);

    let cosd = dot / (p1mag * vemag + AVOID_DIVIDE_BY_ZERO);
    let gammai = (1.0 - beta * beta).sqrt();
    let p = beta * cosd;
    let q = (1.0 + p / (1.0 + gammai)) * light_time;
    let r = 1.0 + p;

    *position *= gammai;
    *position += q * velocity;
    *position /= r;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_light_time_difference_along_axis() {
        // Observer 1 AU along x-axis, object along x-axis
        let position = Vector3::new(10.0, 0.0, 0.0);
        let observer = Vector3::new(1.0, 0.0, 0.0);
        let ltt = light_time_difference(&position, &observer);
        // Should be 1.0 / C_AUDAY
        assert_relative_eq!(ltt, 1.0 / C_AUDAY, epsilon = 1e-15);
    }

    #[test]
    fn test_light_time_difference_perpendicular() {
        // Observer along y-axis, object along x-axis — projection is zero
        let position = Vector3::new(10.0, 0.0, 0.0);
        let observer = Vector3::new(0.0, 1.0, 0.0);
        let ltt = light_time_difference(&position, &observer);
        assert_relative_eq!(ltt, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_add_deflection_sun() {
        // Star at 10 AU along x, observer at 1 AU along y (90 deg from Sun)
        // Sun at origin
        let mut position = Vector3::new(10.0, 0.0, 0.0);
        let observer = Vector3::new(0.0, 1.0, 0.0);
        let deflector = Vector3::new(0.0, 0.0, 0.0);

        let pos_before = position;
        add_deflection(&mut position, &observer, &deflector, 1.0);

        // Deflection should be small but nonzero
        let shift = (position - pos_before).norm();
        assert!(shift > 0.0, "Deflection should be nonzero");
        // Solar deflection at 90 degrees is ~1.75 arcsec / (angular sep in radians)
        // This is a very rough magnitude check
        assert!(shift < 1e-6, "Deflection should be tiny in AU");
    }

    #[test]
    fn test_add_deflection_aligned_skipped() {
        // When observer-deflector line is parallel to object direction,
        // the correction is skipped
        let mut position = Vector3::new(10.0, 0.0, 0.0);
        let observer = Vector3::new(1.0, 0.0, 0.0);
        let deflector = Vector3::new(0.0, 0.0, 0.0);

        let pos_before = position;
        add_deflection(&mut position, &observer, &deflector, 1.0);

        // edotp ~1.0, so correction should be skipped
        assert_relative_eq!(position.x, pos_before.x, epsilon = 1e-15);
        assert_relative_eq!(position.y, pos_before.y, epsilon = 1e-15);
        assert_relative_eq!(position.z, pos_before.z, epsilon = 1e-15);
    }

    #[test]
    fn test_add_aberration_zero_velocity() {
        let mut position = Vector3::new(1.0, 0.0, 0.0);
        let velocity = Vector3::new(0.0, 0.0, 0.0);
        let pos_before = position;

        add_aberration(&mut position, &velocity, 0.01);

        // Zero velocity should give no aberration
        assert_relative_eq!(position.x, pos_before.x, epsilon = 1e-10);
        assert_relative_eq!(position.y, pos_before.y, epsilon = 1e-10);
    }

    #[test]
    fn test_add_aberration_earth_velocity() {
        // Earth orbital velocity ~29.78 km/s ≈ 0.0172 AU/day
        // This should produce ~20 arcsec aberration
        let mut position = Vector3::new(1.0, 0.0, 0.0);
        let earth_v_auday = 29.78e3 / AU_M * 86400.0; // ~0.0172 AU/day
        let velocity = Vector3::new(0.0, earth_v_auday, 0.0);
        let light_time = 1.0 / C_AUDAY;

        let pos_before = position;
        add_aberration(&mut position, &velocity, light_time);

        // The y-component should have shifted
        let shift_y = position.y - pos_before.y;
        assert!(
            shift_y.abs() > 1e-6,
            "Aberration should shift position in y"
        );

        // The shift angle should be on order of v/c ~ 1e-4 radians ~ 20 arcsec
        let angle = (position - pos_before).norm() / position.norm();
        assert!(
            angle > 1e-5 && angle < 1e-3,
            "Aberration angle {} should be ~1e-4 rad",
            angle
        );
    }

    #[test]
    fn test_rmass_lookup() {
        assert_relative_eq!(rmass("sun").unwrap(), 1.0);
        assert_relative_eq!(rmass("jupiter").unwrap(), 1047.3486);
        assert!(rmass("nonexistent").is_none());
    }
}
