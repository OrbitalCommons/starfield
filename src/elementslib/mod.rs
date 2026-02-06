//! Osculating orbital elements from state vectors
//!
//! Port of Skyfield's `elementslib.py` — convert position + velocity state
//! vectors into classical Keplerian orbital elements. Matches NASA HORIZONS output.
//!
//! # Example
//!
//! ```ignore
//! use starfield::elementslib::OsculatingElements;
//! use starfield::constants::GM_SUN;
//! use nalgebra::Vector3;
//!
//! // Position and velocity in km and km/s
//! let pos_km = Vector3::new(1.5e8, 0.0, 0.0);
//! let vel_km_s = Vector3::new(0.0, 29.78, 0.0);
//! let elements = OsculatingElements::new(pos_km, vel_km_s, GM_SUN);
//!
//! println!("Semi-major axis: {} km", elements.semi_major_axis_km());
//! println!("Eccentricity: {}", elements.eccentricity());
//! ```

#[cfg(all(test, feature = "python-tests"))]
mod python_tests;

use std::f64::consts::PI;

use nalgebra::Vector3;

use crate::constants::{AU_KM, DAY_S};

const TAU: f64 = 2.0 * PI;

/// Osculating orbital elements computed from a state vector
///
/// All internal computations use km and km/s.
/// Provides accessors in both km and AU units.
#[derive(Debug, Clone)]
pub struct OsculatingElements {
    /// Specific angular momentum vector (km²/s)
    h_vec: Vector3<f64>,
    /// Eccentricity vector (dimensionless)
    e_vec: Vector3<f64>,
    /// Node vector (dimensionless, normalized)
    n_vec: Vector3<f64>,
    /// Position vector (km)
    pos_km: Vector3<f64>,
    /// Velocity vector (km/s)
    vel_km_s: Vector3<f64>,
    /// Gravitational parameter (km³/s²)
    mu: f64,
}

impl OsculatingElements {
    /// Compute osculating elements from position and velocity vectors
    ///
    /// # Arguments
    /// * `pos_km` — position in km
    /// * `vel_km_s` — velocity in km/s
    /// * `mu_km3_s2` — gravitational parameter GM in km³/s²
    pub fn new(pos_km: Vector3<f64>, vel_km_s: Vector3<f64>, mu_km3_s2: f64) -> Self {
        let h_vec = pos_km.cross(&vel_km_s);
        let e_vec = eccentricity_vector(&pos_km, &vel_km_s, mu_km3_s2);
        let n_vec = node_vector(&h_vec);

        OsculatingElements {
            h_vec,
            e_vec,
            n_vec,
            pos_km,
            vel_km_s,
            mu: mu_km3_s2,
        }
    }

    /// Compute osculating elements from a Position with AU units
    ///
    /// # Arguments
    /// * `pos_au` — position in AU
    /// * `vel_au_day` — velocity in AU/day
    /// * `mu_km3_s2` — gravitational parameter GM in km³/s²
    pub fn from_au(pos_au: &Vector3<f64>, vel_au_day: &Vector3<f64>, mu_km3_s2: f64) -> Self {
        let pos_km = pos_au * AU_KM;
        let vel_km_s = vel_au_day * AU_KM / DAY_S;
        Self::new(pos_km, vel_km_s, mu_km3_s2)
    }

    /// Eccentricity (dimensionless)
    pub fn eccentricity(&self) -> f64 {
        self.e_vec.norm()
    }

    /// Inclination in radians [0, π]
    pub fn inclination_rad(&self) -> f64 {
        let k = Vector3::new(0.0, 0.0, 1.0);
        angle_between(&self.h_vec, &k)
    }

    /// Inclination in degrees
    pub fn inclination_deg(&self) -> f64 {
        self.inclination_rad().to_degrees()
    }

    /// Longitude of ascending node Ω in radians [0, 2π)
    pub fn longitude_of_ascending_node_rad(&self) -> f64 {
        let i = self.inclination_rad();
        if i.abs() < 1e-15 {
            0.0
        } else {
            self.h_vec.x.atan2(-self.h_vec.y).rem_euclid(TAU)
        }
    }

    /// Longitude of ascending node Ω in degrees
    pub fn longitude_of_ascending_node_deg(&self) -> f64 {
        self.longitude_of_ascending_node_rad().to_degrees()
    }

    /// Argument of periapsis ω in radians [0, 2π)
    pub fn argument_of_periapsis_rad(&self) -> f64 {
        let e = self.eccentricity();
        let n_len = self.n_vec.norm();

        if e < 1e-15 {
            // Circular orbit
            0.0
        } else if n_len < 1e-15 {
            // Equatorial and not circular
            let angle = self.e_vec.y.atan2(self.e_vec.x).rem_euclid(TAU);
            let h_z = self.pos_km.cross(&self.vel_km_s).z;
            if h_z >= 0.0 {
                angle
            } else {
                (-angle).rem_euclid(TAU)
            }
        } else {
            // General case
            let angle = angle_between(&self.n_vec, &self.e_vec);
            if self.e_vec.z > 0.0 {
                angle
            } else {
                (-angle).rem_euclid(TAU)
            }
        }
    }

    /// Argument of periapsis ω in degrees
    pub fn argument_of_periapsis_deg(&self) -> f64 {
        self.argument_of_periapsis_rad().to_degrees()
    }

    /// True anomaly ν in radians [0, 2π) for elliptic; [-π, π] for hyperbolic
    pub fn true_anomaly_rad(&self) -> f64 {
        let e = self.eccentricity();
        let n_len = self.n_vec.norm();

        let v = if e > 1e-15 {
            // Not circular
            let angle = angle_between(&self.e_vec, &self.pos_km);
            if self.pos_km.dot(&self.vel_km_s) > 0.0 {
                angle
            } else {
                (-angle).rem_euclid(TAU)
            }
        } else if n_len < 1e-15 {
            // Circular and equatorial
            let r = self.pos_km.norm();
            let angle = (self.pos_km.x / r).clamp(-1.0, 1.0).acos();
            if self.vel_km_s.x < 0.0 {
                angle
            } else {
                (-angle).rem_euclid(TAU)
            }
        } else {
            // Circular and not equatorial
            let angle = angle_between(&self.n_vec, &self.pos_km);
            if self.pos_km.z >= 0.0 {
                angle
            } else {
                (-angle).rem_euclid(TAU)
            }
        };

        if e > (1.0 - 1e-15) {
            normpi(v)
        } else {
            v
        }
    }

    /// True anomaly ν in degrees
    pub fn true_anomaly_deg(&self) -> f64 {
        self.true_anomaly_rad().to_degrees()
    }

    /// Eccentric anomaly E in radians
    pub fn eccentric_anomaly_rad(&self) -> f64 {
        let e = self.eccentricity();
        let v = self.true_anomaly_rad();

        if e < 1.0 {
            2.0 * (((1.0 - e) / (1.0 + e)).sqrt() * (v / 2.0).tan()).atan()
        } else if e > 1.0 {
            let val = (v / 2.0).tan() / ((e + 1.0) / (e - 1.0)).sqrt();
            normpi(2.0 * val.atanh())
        } else {
            0.0 // parabolic
        }
    }

    /// Mean anomaly M in radians [0, 2π) for elliptic
    pub fn mean_anomaly_rad(&self) -> f64 {
        let e = self.eccentricity();
        let ea = self.eccentric_anomaly_rad();

        if e < 1.0 {
            (ea - e * ea.sin()).rem_euclid(TAU)
        } else if e > 1.0 {
            normpi(e * ea.sinh() - ea)
        } else {
            0.0
        }
    }

    /// Mean anomaly M in degrees
    pub fn mean_anomaly_deg(&self) -> f64 {
        self.mean_anomaly_rad().to_degrees()
    }

    /// Semi-latus rectum p in km
    pub fn semi_latus_rectum_km(&self) -> f64 {
        let h = self.h_vec.norm();
        h * h / self.mu
    }

    /// Semi-latus rectum p in AU
    pub fn semi_latus_rectum_au(&self) -> f64 {
        self.semi_latus_rectum_km() / AU_KM
    }

    /// Semi-major axis a in km (inf for parabolic)
    pub fn semi_major_axis_km(&self) -> f64 {
        let p = self.semi_latus_rectum_km();
        let e = self.eccentricity();
        if (e - 1.0).abs() > 1e-15 {
            p / (1.0 - e * e)
        } else {
            f64::INFINITY
        }
    }

    /// Semi-major axis a in AU
    pub fn semi_major_axis_au(&self) -> f64 {
        self.semi_major_axis_km() / AU_KM
    }

    /// Semi-minor axis b in km
    pub fn semi_minor_axis_km(&self) -> f64 {
        let p = self.semi_latus_rectum_km();
        let e = self.eccentricity();
        if e < 1.0 {
            p / (1.0 - e * e).sqrt()
        } else if e > 1.0 {
            p * (e * e - 1.0).sqrt() / (1.0 - e * e)
        } else {
            0.0
        }
    }

    /// Periapsis distance q in km
    pub fn periapsis_distance_km(&self) -> f64 {
        let p = self.semi_latus_rectum_km();
        let e = self.eccentricity();
        if (e - 1.0).abs() > 1e-15 {
            p * (1.0 - e) / (1.0 - e * e)
        } else {
            p / 2.0
        }
    }

    /// Periapsis distance q in AU
    pub fn periapsis_distance_au(&self) -> f64 {
        self.periapsis_distance_km() / AU_KM
    }

    /// Apoapsis distance Q in km (inf for parabolic/hyperbolic)
    pub fn apoapsis_distance_km(&self) -> f64 {
        let p = self.semi_latus_rectum_km();
        let e = self.eccentricity();
        if e < (1.0 - 1e-15) {
            p * (1.0 + e) / (1.0 - e * e)
        } else {
            f64::INFINITY
        }
    }

    /// Apoapsis distance Q in AU
    pub fn apoapsis_distance_au(&self) -> f64 {
        self.apoapsis_distance_km() / AU_KM
    }

    /// Orbital period in seconds (inf for parabolic/hyperbolic)
    pub fn period_seconds(&self) -> f64 {
        let a = self.semi_major_axis_km();
        if a > 0.0 && a.is_finite() {
            TAU * (a.powi(3) / self.mu).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Orbital period in days
    pub fn period_days(&self) -> f64 {
        self.period_seconds() / DAY_S
    }

    /// Mean motion n in radians/second
    pub fn mean_motion_rad_per_sec(&self) -> f64 {
        let a = self.semi_major_axis_km();
        (self.mu / a.abs().powi(3)).sqrt()
    }

    /// Mean motion n in radians/day
    pub fn mean_motion_rad_per_day(&self) -> f64 {
        self.mean_motion_rad_per_sec() * DAY_S
    }

    /// Argument of latitude u = ω + ν in radians [0, 2π)
    pub fn argument_of_latitude_rad(&self) -> f64 {
        (self.argument_of_periapsis_rad() + self.true_anomaly_rad()).rem_euclid(TAU)
    }

    /// Longitude of periapsis ϖ = Ω + ω in radians [0, 2π)
    pub fn longitude_of_periapsis_rad(&self) -> f64 {
        (self.longitude_of_ascending_node_rad() + self.argument_of_periapsis_rad()).rem_euclid(TAU)
    }

    /// True longitude l = Ω + ω + ν in radians [0, 2π)
    pub fn true_longitude_rad(&self) -> f64 {
        (self.longitude_of_ascending_node_rad()
            + self.argument_of_periapsis_rad()
            + self.true_anomaly_rad())
        .rem_euclid(TAU)
    }

    /// Mean longitude L = Ω + ω + M in radians [0, 2π)
    pub fn mean_longitude_rad(&self) -> f64 {
        (self.longitude_of_ascending_node_rad()
            + self.argument_of_periapsis_rad()
            + self.mean_anomaly_rad())
        .rem_euclid(TAU)
    }
}

impl std::fmt::Display for OsculatingElements {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "OsculatingElements(a={:.4} AU, e={:.6}, i={:.2}°)",
            self.semi_major_axis_au(),
            self.eccentricity(),
            self.inclination_deg(),
        )
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Normalize angle to [-π, π]
fn normpi(x: f64) -> f64 {
    (x + PI).rem_euclid(TAU) - PI
}

/// Eccentricity vector from state vectors
fn eccentricity_vector(pos: &Vector3<f64>, vel: &Vector3<f64>, mu: f64) -> Vector3<f64> {
    let r = pos.norm();
    let v_sq = vel.norm_squared();
    ((v_sq - mu / r) * pos - pos.dot(vel) * vel) / mu
}

/// Node vector (normalized cross product of k-hat with h)
fn node_vector(h: &Vector3<f64>) -> Vector3<f64> {
    // k × h = [-h_y, h_x, 0]
    let n = Vector3::new(-h.y, h.x, 0.0);
    let len = n.norm();
    if len > 0.0 {
        n / len
    } else {
        n
    }
}

/// Angle between two vectors in radians [0, π]
fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let cos_angle = a.dot(b) / (a.norm() * b.norm());
    cos_angle.clamp(-1.0, 1.0).acos()
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Compute osculating orbital elements for a position
///
/// This is the Rust equivalent of Skyfield's `osculating_elements_of()`.
///
/// # Arguments
/// * `pos_au` — position in AU
/// * `vel_au_day` — velocity in AU/day
/// * `mu_km3_s2` — gravitational parameter GM in km³/s²
pub fn osculating_elements_of(
    pos_au: &Vector3<f64>,
    vel_au_day: &Vector3<f64>,
    mu_km3_s2: f64,
) -> OsculatingElements {
    OsculatingElements::from_au(pos_au, vel_au_day, mu_km3_s2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GM_SUN;
    use approx::assert_relative_eq;

    #[test]
    fn test_circular_orbit() {
        // Earth-like circular orbit: r = 1 AU, v = sqrt(GM/r)
        let r_km = AU_KM;
        let v_km_s = (GM_SUN / r_km).sqrt();
        let pos = Vector3::new(r_km, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_km_s, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        assert_relative_eq!(elem.eccentricity(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.semi_major_axis_km(), r_km, epsilon = 1.0);
        assert_relative_eq!(elem.inclination_rad(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.periapsis_distance_km(), r_km, epsilon = 1.0);
    }

    #[test]
    fn test_elliptic_orbit() {
        // Orbit with known eccentricity: e = 0.5, a = 1 AU
        let a = AU_KM;
        let e = 0.5;
        let r_periapsis = a * (1.0 - e);
        let v_periapsis = ((GM_SUN / a) * (1.0 + e) / (1.0 - e)).sqrt();

        let pos = Vector3::new(r_periapsis, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_periapsis, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        assert_relative_eq!(elem.eccentricity(), e, epsilon = 1e-10);
        assert_relative_eq!(elem.semi_major_axis_km(), a, epsilon = 1.0);
        assert_relative_eq!(elem.periapsis_distance_km(), r_periapsis, epsilon = 1.0);
        assert_relative_eq!(elem.apoapsis_distance_km(), a * (1.0 + e), epsilon = 1.0);
    }

    #[test]
    fn test_inclined_orbit() {
        // Orbit inclined 45° to equatorial plane
        let r_km = AU_KM;
        let v_km_s = (GM_SUN / r_km).sqrt();
        let angle = 45.0_f64.to_radians();
        let pos = Vector3::new(r_km, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_km_s * angle.cos(), v_km_s * angle.sin());

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        assert_relative_eq!(elem.inclination_deg(), 45.0, epsilon = 1e-8);
        assert_relative_eq!(elem.eccentricity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_period() {
        // Earth-like orbit: period should be ~365.25 days
        let r_km = AU_KM;
        let v_km_s = (GM_SUN / r_km).sqrt();
        let pos = Vector3::new(r_km, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_km_s, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);
        let period_days = elem.period_days();

        assert_relative_eq!(period_days, 365.256, epsilon = 0.01);
    }

    #[test]
    fn test_from_au() {
        let pos_au = Vector3::new(1.0, 0.0, 0.0);
        let vel_au_day = Vector3::new(0.0, (GM_SUN / AU_KM).sqrt() * DAY_S / AU_KM, 0.0);

        let elem = OsculatingElements::from_au(&pos_au, &vel_au_day, GM_SUN);

        assert_relative_eq!(elem.semi_major_axis_au(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(elem.eccentricity(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_anomaly_at_periapsis() {
        // At periapsis, true anomaly = 0, eccentric anomaly = 0, mean anomaly = 0
        let a = AU_KM;
        let e = 0.5;
        let r_periapsis = a * (1.0 - e);
        let v_periapsis = ((GM_SUN / a) * (1.0 + e) / (1.0 - e)).sqrt();

        let pos = Vector3::new(r_periapsis, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_periapsis, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        assert_relative_eq!(elem.true_anomaly_rad(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.eccentric_anomaly_rad(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.mean_anomaly_rad(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_display() {
        let pos = Vector3::new(AU_KM, 0.0, 0.0);
        let vel = Vector3::new(0.0, (GM_SUN / AU_KM).sqrt(), 0.0);
        let elem = OsculatingElements::new(pos, vel, GM_SUN);
        let s = format!("{elem}");
        assert!(s.contains("OsculatingElements"));
        assert!(s.contains("AU"));
    }

    #[test]
    fn test_normpi() {
        assert_relative_eq!(normpi(0.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(normpi(PI), -PI, epsilon = 1e-10);
        assert_relative_eq!(normpi(3.0 * PI), -PI, epsilon = 1e-10);
    }

    #[test]
    fn test_hyperbolic_orbit() {
        // Hyperbolic orbit: e > 1
        let r_km = AU_KM;
        // Velocity above escape velocity
        let v_escape = (2.0 * GM_SUN / r_km).sqrt();
        let v_hyp = v_escape * 1.5;
        let pos = Vector3::new(r_km, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_hyp, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        assert!(
            elem.eccentricity() > 1.0,
            "e should be > 1, got {}",
            elem.eccentricity()
        );
        assert!(
            elem.semi_major_axis_km() < 0.0,
            "a should be negative for hyperbolic"
        );
        assert!(
            elem.apoapsis_distance_km().is_infinite(),
            "Q should be infinite"
        );
        assert!(
            elem.period_days().is_infinite(),
            "Period should be infinite"
        );
    }

    #[test]
    fn test_longitude_quantities() {
        // For a simple equatorial prograde orbit at periapsis
        let a = AU_KM;
        let e = 0.3;
        let r = a * (1.0 - e);
        let v = ((GM_SUN / a) * (1.0 + e) / (1.0 - e)).sqrt();
        let pos = Vector3::new(r, 0.0, 0.0);
        let vel = Vector3::new(0.0, v, 0.0);

        let elem = OsculatingElements::new(pos, vel, GM_SUN);

        // At periapsis: ν=0, ω=0 (equatorial), Ω=0 → all longitudes should be 0
        assert_relative_eq!(elem.true_anomaly_rad(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.true_longitude_rad(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(elem.mean_longitude_rad(), 0.0, epsilon = 1e-10);
    }
}
