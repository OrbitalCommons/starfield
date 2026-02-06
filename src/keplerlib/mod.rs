//! Two-body Keplerian orbit propagation
//!
//! Port of Skyfield's `keplerlib.py` — propagates orbits using two-body
//! (Keplerian) dynamics for comets, asteroids, and other bodies defined
//! by classical orbital elements.
//!
//! # Example
//!
//! ```ignore
//! use starfield::keplerlib::KeplerOrbit;
//! use starfield::time::Timescale;
//! use starfield::constants::GM_SUN;
//!
//! let ts = Timescale::default();
//! let epoch = ts.tt_jd(2451545.0, None);
//!
//! // Halley's Comet (approximate elements)
//! let halley = KeplerOrbit::from_periapsis(
//!     1.058, 0.967, 162.26, 58.42, 111.33,
//!     &epoch, GM_SUN, Some(10), Some("1P/Halley"),
//! );
//!
//! let t = ts.tt_jd(2458000.0, None);
//! let pos = halley.at(&t);
//! ```

#[cfg(feature = "python-tests")]
mod python_tests;

use std::f64::consts::PI;

use nalgebra::{Matrix3, Vector3};

#[cfg(test)]
use crate::constants::GM_SUN;
use crate::constants::{AU_KM, DAY_S, DEG2RAD};
use crate::positions::Position;
use crate::time::Time;

/// Convert GM from km³/s² to AU³/day²
const CONVERT_GM: f64 = DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);

/// A two-body Keplerian orbit
///
/// Stores the state vector (position + velocity) at an epoch and
/// propagates to arbitrary times using universal variable formulation.
#[derive(Debug, Clone)]
pub struct KeplerOrbit {
    /// Position at epoch in AU
    pub position_au: Vector3<f64>,
    /// Velocity at epoch in AU/day
    pub velocity_au_per_day: Vector3<f64>,
    /// Epoch as TT Julian date
    pub epoch_tt: f64,
    /// Gravitational parameter in AU³/day²
    pub mu_au3_d2: f64,
    /// NAIF ID of the central body (10 = Sun)
    pub center: i32,
    /// Name of the orbiting body
    pub target_name: Option<String>,
    /// Optional rotation matrix (e.g., ECLIPJ2000 → equatorial)
    rotation: Option<Matrix3<f64>>,
}

impl KeplerOrbit {
    /// Create a KeplerOrbit from position/velocity state vectors
    ///
    /// Position in AU, velocity in AU/day, GM in AU³/day².
    pub fn new(
        position_au: Vector3<f64>,
        velocity_au_per_day: Vector3<f64>,
        epoch: &Time,
        mu_au3_d2: f64,
        center: Option<i32>,
        target_name: Option<&str>,
    ) -> Self {
        KeplerOrbit {
            position_au,
            velocity_au_per_day,
            epoch_tt: epoch.tt(),
            mu_au3_d2,
            center: center.unwrap_or(10),
            target_name: target_name.map(|s| s.to_string()),
            rotation: None,
        }
    }

    /// Build from periapsis parameters (used for comets)
    ///
    /// # Arguments
    /// * `semilatus_rectum_au` — semi-latus rectum in AU
    /// * `eccentricity` — orbital eccentricity
    /// * `inclination_deg` — inclination in degrees
    /// * `longitude_of_ascending_node_deg` — Ω in degrees
    /// * `argument_of_perihelion_deg` — ω in degrees
    /// * `epoch` — time of periapsis passage
    /// * `gm_km3_s2` — GM in km³/s²
    /// * `center` — NAIF ID of central body
    /// * `target_name` — name of orbiting body
    #[allow(clippy::too_many_arguments)]
    pub fn from_periapsis(
        semilatus_rectum_au: f64,
        eccentricity: f64,
        inclination_deg: f64,
        longitude_of_ascending_node_deg: f64,
        argument_of_perihelion_deg: f64,
        epoch: &Time,
        gm_km3_s2: f64,
        center: Option<i32>,
        target_name: Option<&str>,
    ) -> Self {
        let gm_au3_d2 = gm_km3_s2 * CONVERT_GM;
        let (pos, vel) = ele_to_vec(
            semilatus_rectum_au,
            eccentricity,
            DEG2RAD * inclination_deg,
            DEG2RAD * longitude_of_ascending_node_deg,
            DEG2RAD * argument_of_perihelion_deg,
            0.0, // true anomaly = 0 at periapsis
            gm_au3_d2,
        );
        KeplerOrbit {
            position_au: pos,
            velocity_au_per_day: vel,
            epoch_tt: epoch.tt(),
            mu_au3_d2: gm_au3_d2,
            center: center.unwrap_or(10),
            target_name: target_name.map(|s| s.to_string()),
            rotation: None,
        }
    }

    /// Build from mean anomaly (used for asteroids / MPC data)
    ///
    /// # Arguments
    /// * `semilatus_rectum_au` — semi-latus rectum in AU
    /// * `eccentricity` — orbital eccentricity
    /// * `inclination_deg` — inclination in degrees
    /// * `longitude_of_ascending_node_deg` — Ω in degrees
    /// * `argument_of_perihelion_deg` — ω in degrees
    /// * `mean_anomaly_deg` — mean anomaly M in degrees
    /// * `epoch` — epoch of elements
    /// * `gm_km3_s2` — GM in km³/s²
    /// * `center` — NAIF ID of central body
    /// * `target_name` — name of orbiting body
    #[allow(clippy::too_many_arguments)]
    pub fn from_mean_anomaly(
        semilatus_rectum_au: f64,
        eccentricity: f64,
        inclination_deg: f64,
        longitude_of_ascending_node_deg: f64,
        argument_of_perihelion_deg: f64,
        mean_anomaly_deg: f64,
        epoch: &Time,
        gm_km3_s2: f64,
        center: Option<i32>,
        target_name: Option<&str>,
    ) -> Self {
        let m = DEG2RAD * mean_anomaly_deg;
        let gm_au3_d2 = gm_km3_s2 * CONVERT_GM;

        let v = if eccentricity < 1.0 {
            let ea = eccentric_anomaly(eccentricity, m);
            true_anomaly_closed(eccentricity, ea)
        } else if eccentricity > 1.0 {
            let ea = eccentric_anomaly(eccentricity, m);
            true_anomaly_hyperbolic(eccentricity, ea)
        } else {
            true_anomaly_parabolic(semilatus_rectum_au, gm_au3_d2, m)
        };

        let (pos, vel) = ele_to_vec(
            semilatus_rectum_au,
            eccentricity,
            DEG2RAD * inclination_deg,
            DEG2RAD * longitude_of_ascending_node_deg,
            DEG2RAD * argument_of_perihelion_deg,
            v,
            gm_au3_d2,
        );
        KeplerOrbit {
            position_au: pos,
            velocity_au_per_day: vel,
            epoch_tt: epoch.tt(),
            mu_au3_d2: gm_au3_d2,
            center: center.unwrap_or(10),
            target_name: target_name.map(|s| s.to_string()),
            rotation: None,
        }
    }

    /// Set the rotation matrix to convert from ecliptic (ECLIPJ2000) to equatorial
    ///
    /// The ECLIPJ2000 frame matrix rotates equatorial → ecliptic.
    /// Its transpose rotates ecliptic → equatorial, which is what we need
    /// for orbits defined in the ecliptic plane (comets, asteroids from MPC data).
    pub fn set_ecliptic_rotation(&mut self) {
        // ECLIPJ2000 rotation matrix (equatorial → ecliptic) from SPICE/Skyfield.
        // This is Rx(obliquity) where obliquity = 23.4392911 degrees.
        #[rustfmt::skip]
        let eclip = Matrix3::new(
            1.0,  0.0,                    0.0,
            0.0,  0.917_482_062_069_181_8,  0.397_777_155_931_913_7,
            0.0, -0.397_777_155_931_913_7,  0.917_482_062_069_181_8,
        );
        // Transpose: ecliptic → equatorial
        self.rotation = Some(eclip.transpose());
    }

    /// Propagate the orbit to the given time
    ///
    /// Returns a Barycentric `Position` in the ICRF.
    pub fn at(&self, time: &Time) -> Position {
        let (mut pos, mut vel) = propagate(
            &self.position_au,
            &self.velocity_au_per_day,
            self.epoch_tt,
            time.tt(),
            self.mu_au3_d2,
        );

        if let Some(rot) = &self.rotation {
            pos = rot * pos;
            vel = rot * vel;
        }

        Position::barycentric(pos, vel, self.center)
    }
}

impl std::fmt::Display for KeplerOrbit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(name) = &self.target_name {
            write!(f, "KeplerOrbit {} → {}", self.center, name)
        } else {
            write!(
                f,
                "KeplerOrbit {} → e={:.3} a={:.2} AU",
                self.center,
                0.0, // placeholder
                0.0,
            )
        }
    }
}

/// Build a comet orbit from MPC-style parameters
///
/// This is the Rust equivalent of Skyfield's `comet_orbit()`.
#[allow(clippy::too_many_arguments)]
pub fn comet_orbit(
    perihelion_distance_au: f64,
    eccentricity: f64,
    inclination_deg: f64,
    longitude_of_ascending_node_deg: f64,
    argument_of_perihelion_deg: f64,
    t_perihelion: &Time,
    gm_km3_s2: f64,
    target_name: Option<&str>,
) -> KeplerOrbit {
    let p = if eccentricity == 1.0 {
        perihelion_distance_au * 2.0
    } else {
        let a = perihelion_distance_au / (1.0 - eccentricity);
        a * (1.0 - eccentricity * eccentricity)
    };

    let mut orbit = KeplerOrbit::from_periapsis(
        p,
        eccentricity,
        inclination_deg,
        longitude_of_ascending_node_deg,
        argument_of_perihelion_deg,
        t_perihelion,
        gm_km3_s2,
        Some(10),
        target_name,
    );
    orbit.set_ecliptic_rotation();
    orbit
}

/// Build a minor planet orbit from MPC-style parameters
///
/// This is the Rust equivalent of Skyfield's `mpcorb_orbit()`.
#[allow(clippy::too_many_arguments)]
pub fn mpcorb_orbit(
    semimajor_axis_au: f64,
    eccentricity: f64,
    inclination_deg: f64,
    longitude_of_ascending_node_deg: f64,
    argument_of_perihelion_deg: f64,
    mean_anomaly_deg: f64,
    epoch: &Time,
    gm_km3_s2: f64,
    target_name: Option<&str>,
) -> KeplerOrbit {
    let p = semimajor_axis_au * (1.0 - eccentricity * eccentricity);

    let mut orbit = KeplerOrbit::from_mean_anomaly(
        p,
        eccentricity,
        inclination_deg,
        longitude_of_ascending_node_deg,
        argument_of_perihelion_deg,
        mean_anomaly_deg,
        epoch,
        gm_km3_s2,
        Some(10),
        target_name,
    );
    orbit.set_ecliptic_rotation();
    orbit
}

// ---------------------------------------------------------------------------
// Core orbital mechanics functions
// ---------------------------------------------------------------------------

/// Normalize angle to [-π, π]
fn normpi(m: f64) -> f64 {
    let mut x = m % (2.0 * PI);
    if x > PI {
        x -= 2.0 * PI;
    }
    if x < -PI {
        x += 2.0 * PI;
    }
    x
}

/// Solve Kepler's equation for eccentric anomaly
///
/// Iterative solver following arXiv:2108.03215.
/// Works for both elliptic (e < 1) and hyperbolic (e > 1) orbits.
pub(crate) fn eccentric_anomaly(e: f64, m: f64) -> f64 {
    let m = normpi(m);
    let sign_m = m.signum();
    let m = m * sign_m;

    let ebar = 0.25 * PI / e - 1.0;
    let mut ea = 0.5 * PI * ebar * (ebar.signum() * (1.0 + m / (e * ebar * ebar)).sqrt() - 1.0);

    for _ in 0..10 {
        let f1 = 1.0 - e * ea.cos();
        let f2 = e * ea.sin();
        let f = ea - f2 - m;
        let d_ea = f * f1 / (f1 * f1 - 0.5 * f * f2);
        ea -= d_ea;
        if d_ea.abs() < 1e-14 {
            return ea * sign_m;
        }
    }

    ea * sign_m
}

/// True anomaly from eccentric anomaly for closed (elliptic) orbits
fn true_anomaly_closed(e: f64, ea: f64) -> f64 {
    2.0 * (((1.0 + e) / (1.0 - e)).sqrt() * (ea / 2.0).tan()).atan()
}

/// True anomaly from eccentric anomaly for hyperbolic orbits
fn true_anomaly_hyperbolic(e: f64, ea: f64) -> f64 {
    2.0 * (((e + 1.0) / (e - 1.0)).sqrt() * (ea / 2.0).tanh()).atan()
}

/// True anomaly for parabolic orbits
fn true_anomaly_parabolic(p: f64, gm: f64, m: f64) -> f64 {
    let delta_t = (2.0 * p.powi(3) / gm).sqrt() * m;
    let periapsis_distance = p / 2.0;
    let a = 1.5 * (gm / (2.0 * periapsis_distance.powi(3))).sqrt() * delta_t;
    let b = (a + (a * a + 1.0)).cbrt();
    2.0 * (b - 1.0 / b).atan()
}

/// Convert orbital elements to position and velocity state vectors
///
/// Based on CCAR equations:
/// <https://web.archive.org/web/*/http://ccar.colorado.edu/asen5070/handouts/kep2cart_2002.doc>
///
/// # Arguments
/// * `p` — semi-latus rectum (AU)
/// * `e` — eccentricity
/// * `i` — inclination (radians)
/// * `om` — longitude of ascending node Ω (radians)
/// * `w` — argument of periapsis ω (radians)
/// * `v` — true anomaly ν (radians)
/// * `mu` — gravitational parameter (AU³/day²)
pub(crate) fn ele_to_vec(
    p: f64,
    e: f64,
    i: f64,
    om: f64,
    w: f64,
    v: f64,
    mu: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let r = p / (1.0 + e * v.cos());
    let h = (p * mu).sqrt();
    let u = v + w;

    let (sin_om, cos_om) = om.sin_cos();
    let (sin_u, cos_u) = u.sin_cos();
    let cos_i = i.cos();
    let sin_i = i.sin();

    let x = r * (cos_om * cos_u - sin_om * sin_u * cos_i);
    let y = r * (sin_om * cos_u + cos_om * sin_u * cos_i);
    let z = r * (sin_i * sin_u);

    let he_rp = h * e / (r * p) * v.sin();
    let h_r = h / r;

    let x_dot = x * he_rp - h_r * (cos_om * sin_u + sin_om * cos_u * cos_i);
    let y_dot = y * he_rp - h_r * (sin_om * sin_u - cos_om * cos_u * cos_i);
    let z_dot = z * he_rp + h_r * sin_i * cos_u;

    (Vector3::new(x, y, z), Vector3::new(x_dot, y_dot, z_dot))
}

/// Stumpff functions c0, c1, c2, c3
///
/// Based on SPICE toolkit's `stmp03.f`.
fn stumpff(x: f64) -> (f64, f64, f64, f64) {
    let z = x.abs().sqrt();

    if x < -1.0 {
        // Hyperbolic regime
        let c0 = z.cosh();
        let c1 = z.sinh() / z;
        let c2 = (1.0 - c0) / x;
        let c3 = (1.0 - c1) / x;
        (c0, c1, c2, c3)
    } else if x > 1.0 {
        // Elliptic regime
        let c0 = z.cos();
        let c1 = z.sin() / z;
        let c2 = (1.0 - c0) / x;
        let c3 = (1.0 - c1) / x;
        (c0, c1, c2, c3)
    } else {
        // Near-zero: use series expansion for c2 and c3
        // c2 = 1/2! - x/4! + x²/6! - x³/8! + ...
        // c3 = 1/3! - x/5! + x²/7! - x³/9! + ...
        let mut c2 = 0.0;
        let mut c3 = 0.0;
        let mut xn = 1.0;
        // Even factorials for c2: 2!, 4!, 6!, 8!, ...
        let c2_facts: [f64; 8] = [
            2.0,
            24.0,
            720.0,
            40320.0,
            3628800.0,
            479001600.0,
            87178291200.0,
            20922789888000.0,
        ];
        // Odd factorials for c3: 3!, 5!, 7!, 9!, ...
        let c3_facts: [f64; 8] = [
            6.0,
            120.0,
            5040.0,
            362880.0,
            39916800.0,
            6227020800.0,
            1307674368000.0,
            355687428096000.0,
        ];
        let mut sign = 1.0;
        for k in 0..8 {
            c2 += sign * xn / c2_facts[k];
            c3 += sign * xn / c3_facts[k];
            xn *= x;
            sign = -sign;
        }
        let c1 = 1.0 - x * c3;
        let c0 = 1.0 - x * c2;
        (c0, c1, c2, c3)
    }
}

/// Propagate an orbit from t0 to t1 using universal variable formulation
///
/// Based on SPICE toolkit's `prop2b.f` via Skyfield's `propagate()`.
///
/// # Arguments
/// * `position` — position at epoch (AU)
/// * `velocity` — velocity at epoch (AU/day)
/// * `t0` — epoch TT Julian date
/// * `t1` — target TT Julian date
/// * `gm` — gravitational parameter (AU³/day²)
///
/// # Returns
/// (position, velocity) at time t1
pub(crate) fn propagate(
    position: &Vector3<f64>,
    velocity: &Vector3<f64>,
    t0: f64,
    t1: f64,
    gm: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let r0 = position.norm();
    let rv = position.dot(velocity);

    let hvec = position.cross(velocity);
    let h2 = hvec.dot(&hvec);

    let eqvec = velocity.cross(&hvec) / gm - position / r0;
    let e = eqvec.norm();
    let q = h2 / (gm * (1.0 + e));

    let f = 1.0 - e;
    let b = (q / gm).sqrt();

    let br0 = b * r0;
    let b2rv = b * b * rv;
    let bq = b * q;
    let qovr0 = q / r0;

    let maxc = br0
        .abs()
        .max(b2rv.abs())
        .max(bq.abs())
        .max((qovr0 / bq).abs());

    // Compute upper bound for x
    let bound = if f < 0.0 {
        // Hyperbolic
        let fixed = (f64::MAX / 2.0).ln() - maxc.ln();
        let rootf = (-f).sqrt();
        let logf = (-f).ln();
        (fixed / rootf).min((fixed + 1.5 * logf) / rootf)
    } else {
        // Elliptic or parabolic
        let logbound = (1.5_f64.ln() + (f64::MAX).ln() - maxc.ln()) / 3.0;
        logbound.exp()
    };

    let dt = t1 - t0;

    // Kepler function: x*(br0*c1 + x*(b2rv*c2 + x*bq*c3))
    let kepler = |x: f64| -> f64 {
        let (_, c1, c2, c3) = stumpff(f * x * x);
        x * (br0 * c1 + x * (b2rv * c2 + x * bq * c3))
    };

    // Initial guess
    let mut x = (dt / bq).clamp(-bound, bound);
    let mut kfun = kepler(x);

    // Bracket the root
    let mut lower;
    let mut upper;

    if dt < 0.0 {
        upper = 0.0;
        lower = x;
        while kfun > dt {
            upper = lower;
            lower *= 2.0;
            let old_x = x;
            x = lower.clamp(-bound, bound);
            if x == old_x {
                break;
            }
            kfun = kepler(x);
        }
    } else if dt > 0.0 {
        lower = 0.0;
        upper = x;
        while kfun < dt {
            lower = upper;
            upper *= 2.0;
            let old_x = x;
            x = upper.clamp(-bound, bound);
            if x == old_x {
                break;
            }
            kfun = kepler(x);
        }
    } else {
        // dt == 0: no propagation needed
        return (*position, *velocity);
    }

    // Bisection refinement
    x = if lower <= upper {
        (upper + lower) / 2.0
    } else {
        upper
    };

    let mut lcount = 0;
    let mut mostc = 1000;

    while lower < x && x < upper && lcount < mostc {
        kfun = kepler(x);

        if kfun > dt {
            upper = x;
        } else if kfun < dt {
            lower = x;
        } else {
            upper = x;
            lower = x;
        }

        if mostc > 64 && upper != 0.0 && lower != 0.0 {
            mostc = 64;
            lcount = 0;
        }

        x = if lower > upper {
            upper
        } else {
            (upper + lower) / 2.0
        };

        lcount += 1;
    }

    // Compute final state vectors using Lagrange coefficients
    let (c0, c1, c2, c3) = stumpff(f * x * x);
    let br = br0 * c0 + x * (b2rv * c1 + x * bq * c2);

    let pc = 1.0 - qovr0 * x * x * c2;
    let vc = dt - bq * x.powi(3) * c3;
    let pcdot = -qovr0 / br * x * c1;
    let vcdot = 1.0 - bq / br * x * x * c2;

    let pos = pc * position + vc * velocity;
    let vel = pcdot * position + vcdot * velocity;

    (pos, vel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn ts() -> Timescale {
        Timescale::default()
    }

    #[test]
    fn test_stumpff_zero() {
        let (c0, c1, c2, c3) = stumpff(0.0);
        assert_relative_eq!(c0, 1.0, epsilon = 1e-14);
        assert_relative_eq!(c1, 1.0, epsilon = 1e-14);
        assert_relative_eq!(c2, 0.5, epsilon = 1e-14);
        assert_relative_eq!(c3, 1.0 / 6.0, epsilon = 1e-14);
    }

    #[test]
    fn test_stumpff_positive() {
        // x = PI² → c0 = cos(PI) = -1, c1 = sin(PI)/PI ≈ 0
        let x = PI * PI;
        let (c0, c1, _c2, _c3) = stumpff(x);
        assert_relative_eq!(c0, -1.0, epsilon = 1e-10);
        assert_relative_eq!(c1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stumpff_negative() {
        // x = -1 → hyperbolic: c0 = cosh(1), c1 = sinh(1)
        let (c0, c1, _c2, _c3) = stumpff(-1.0);
        assert_relative_eq!(c0, 1.0_f64.cosh(), epsilon = 1e-10);
        assert_relative_eq!(c1, 1.0_f64.sinh(), epsilon = 1e-10);
    }

    #[test]
    fn test_stumpff_continuity() {
        // Verify series matches trig at boundaries
        let (c0a, c1a, c2a, c3a) = stumpff(0.99);
        let (c0b, c1b, c2b, c3b) = stumpff(1.01);
        assert!((c0a - c0b).abs() < 0.01);
        assert!((c1a - c1b).abs() < 0.01);
        assert!((c2a - c2b).abs() < 0.01);
        assert!((c3a - c3b).abs() < 0.01);
    }

    #[test]
    fn test_eccentric_anomaly_circular() {
        // For e ≈ 0, E ≈ M
        let e = 0.001;
        let m = 1.0;
        let ea = eccentric_anomaly(e, m);
        assert_relative_eq!(ea, m, epsilon = 0.01);
    }

    #[test]
    fn test_eccentric_anomaly_moderate() {
        // e = 0.5, M = 1.0 → known solution
        let e = 0.5;
        let m = 1.0;
        let ea = eccentric_anomaly(e, m);
        // Verify: E - e*sin(E) = M
        let residual = ea - e * ea.sin() - m;
        assert!(residual.abs() < 1e-13, "Kepler residual = {residual}");
    }

    #[test]
    fn test_eccentric_anomaly_high_e() {
        // e = 0.99, M = 0.5
        let e = 0.99;
        let m = 0.5;
        let ea = eccentric_anomaly(e, m);
        let residual = ea - e * ea.sin() - m;
        assert!(residual.abs() < 1e-12, "Kepler residual = {residual}");
    }

    #[test]
    fn test_ele_to_vec_circular() {
        // Circular orbit: e=0, p=a=1 AU, i=0, face-on
        let gm = GM_SUN * CONVERT_GM;
        let (pos, vel) = ele_to_vec(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, gm);
        // At periapsis of circular orbit: r = p = 1 AU, v = 0
        assert_relative_eq!(pos.norm(), 1.0, epsilon = 1e-10);
        // Velocity should be √(GM/a) ≈ √(GM) for a=1
        let expected_v = gm.sqrt();
        assert_relative_eq!(vel.norm(), expected_v, epsilon = 1e-6);
    }

    #[test]
    fn test_propagate_stationary() {
        // dt = 0 should return original state
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 0.01, 0.0);
        let gm = GM_SUN * CONVERT_GM;
        let (p2, v2) = propagate(&pos, &vel, 0.0, 0.0, gm);
        assert_relative_eq!(p2, pos, epsilon = 1e-14);
        assert_relative_eq!(v2, vel, epsilon = 1e-14);
    }

    #[test]
    fn test_propagate_half_period() {
        // Circular orbit: after half period, should be on opposite side
        let gm = GM_SUN * CONVERT_GM;
        let a = 1.0; // 1 AU
        let v_circ = (gm / a).sqrt();
        let pos = Vector3::new(a, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);

        let period = 2.0 * PI * (a.powi(3) / gm).sqrt();
        let (p2, _v2) = propagate(&pos, &vel, 0.0, period / 2.0, gm);

        // Should be at roughly (-1, 0, 0)
        assert_relative_eq!(p2.x, -a, epsilon = 1e-6);
        assert!(p2.y.abs() < 1e-6);
    }

    #[test]
    fn test_propagate_full_period() {
        // After one full period, should return to starting position
        let gm = GM_SUN * CONVERT_GM;
        let a = 1.0;
        let v_circ = (gm / a).sqrt();
        let pos = Vector3::new(a, 0.0, 0.0);
        let vel = Vector3::new(0.0, v_circ, 0.0);

        let period = 2.0 * PI * (a.powi(3) / gm).sqrt();
        let (p2, v2) = propagate(&pos, &vel, 0.0, period, gm);

        assert_relative_eq!(p2.x, pos.x, epsilon = 1e-5);
        assert_relative_eq!(p2.y, pos.y, epsilon = 1e-5);
        assert_relative_eq!(v2.x, vel.x, epsilon = 1e-5);
        assert_relative_eq!(v2.y, vel.y, epsilon = 1e-5);
    }

    #[test]
    fn test_comet_orbit_basic() {
        let ts = ts();
        let epoch = ts.tt_jd(2451545.0, None);

        // Simple comet: q=1 AU, e=0.5
        let orbit = comet_orbit(1.0, 0.5, 0.0, 0.0, 0.0, &epoch, GM_SUN, Some("TestComet"));

        // At periapsis (epoch), distance from Sun should be ~1 AU
        let pos = orbit.at(&epoch);
        let dist = pos.position.norm();
        assert!(
            dist > 0.9 && dist < 1.1,
            "Periapsis distance should be ~1 AU, got {dist}"
        );
    }

    #[test]
    fn test_mpcorb_orbit_basic() {
        let ts = ts();
        let epoch = ts.tt_jd(2451545.0, None);

        // Earth-like orbit: a=1, e=0.01
        let orbit = mpcorb_orbit(
            1.0,
            0.01,
            0.0,
            0.0,
            0.0,
            0.0,
            &epoch,
            GM_SUN,
            Some("TestAsteroid"),
        );

        let pos = orbit.at(&epoch);
        let dist = pos.position.norm();
        assert!(
            dist > 0.9 && dist < 1.1,
            "Distance should be ~1 AU, got {dist}"
        );
    }

    #[test]
    fn test_normpi() {
        assert_relative_eq!(normpi(0.0), 0.0, epsilon = 1e-14);
        assert_relative_eq!(normpi(PI), PI, epsilon = 1e-14);
        assert_relative_eq!(normpi(-PI), -PI, epsilon = 1e-14);
        assert_relative_eq!(normpi(3.0 * PI), PI, epsilon = 1e-10);
        assert_relative_eq!(normpi(-3.0 * PI), -PI, epsilon = 1e-10);
    }

    #[test]
    fn test_true_anomaly_closed_at_zero() {
        let v = true_anomaly_closed(0.5, 0.0);
        assert_relative_eq!(v, 0.0, epsilon = 1e-14);
    }

    #[test]
    fn test_true_anomaly_closed_at_pi() {
        let v = true_anomaly_closed(0.5, PI);
        assert_relative_eq!(v, PI, epsilon = 1e-10);
    }

    #[test]
    fn test_display() {
        let ts = ts();
        let epoch = ts.tt_jd(2451545.0, None);
        let orbit = comet_orbit(1.0, 0.5, 0.0, 0.0, 0.0, &epoch, GM_SUN, Some("1P/Halley"));
        let s = format!("{orbit}");
        assert!(s.contains("Halley"));
    }
}
