//! Core types for the N-body integrators: state vectors, orbital elements,
//! massive bodies, and simulation configuration.
//!
//! Units throughout: AU, days, GM in AU³/day², heliocentric ecliptic-J2000
//! frame (see the [module root](super)).
//!
//! The element conversions here are the integrators' fast internal path
//! (Murray & Dermott 1999 formulation, bisection-safeguarded Kepler solver).
//! They are cross-validated against `crate::elementslib` — an independent
//! Skyfield/SPICE-parity implementation — by the oracle tests below.

use nalgebra::Vector3;

use crate::constants::TAU as TWO_PI;
use crate::keplerlib::KeplerOrbit;
use crate::planetlib::{Body, Ephemeris, PlanetError};
use crate::time::Time;

use super::{GM_JUPITER, GM_NEPTUNE, GM_SATURN, GM_SUN, GM_URANUS};

/// 3D position + velocity state vector in heliocentric ecliptic J2000.
/// Units: AU for position, AU/day for velocity.
#[derive(Debug, Clone, Copy)]
pub struct StateVector {
    /// Heliocentric position in AU.
    pub pos: Vector3<f64>,
    /// Heliocentric velocity in AU/day.
    pub vel: Vector3<f64>,
}

impl StateVector {
    /// Build a state vector from position (AU) and velocity (AU/day).
    pub fn new(pos: Vector3<f64>, vel: Vector3<f64>) -> Self {
        Self { pos, vel }
    }

    /// The zero state (origin, at rest).
    pub fn zero() -> Self {
        Self {
            pos: Vector3::zeros(),
            vel: Vector3::zeros(),
        }
    }

    /// Heliocentric distance in AU.
    pub fn distance(&self) -> f64 {
        self.pos.norm()
    }

    /// Speed in AU/day.
    pub fn speed(&self) -> f64 {
        self.vel.norm()
    }

    /// Test-particle initial conditions from a `keplerlib::KeplerOrbit`,
    /// propagated to `time` with the two-body universal-variable propagator.
    ///
    /// The state is returned in the frame the orbit's epoch state was given
    /// in; any rotation configured on the orbit (e.g. via
    /// `set_ecliptic_rotation`) is applied by `KeplerOrbit::at` first. For
    /// use with this module's integrators the orbit should describe
    /// heliocentric ecliptic-J2000 motion with no equatorial rotation set.
    pub fn from_kepler_orbit(orbit: &KeplerOrbit, time: &Time) -> Self {
        let p = orbit.at(time);
        Self {
            pos: p.position,
            vel: p.velocity,
        }
    }
}

/// Classical Keplerian orbital elements.
/// Angles in radians, distances in AU.
#[derive(Debug, Clone, Copy)]
pub struct OrbitalElements {
    /// Semi-major axis (AU). Negative for hyperbolic orbits.
    pub a: f64,
    /// Eccentricity (dimensionless)
    pub e: f64,
    /// Inclination (radians, [0, pi])
    pub i: f64,
    /// Longitude of ascending node (radians, [0, 2pi))
    pub omega_big: f64,
    /// Argument of perihelion (radians, [0, 2pi))
    pub omega: f64,
    /// Mean anomaly (radians, [0, 2pi))
    pub mean_anomaly: f64,
}

impl OrbitalElements {
    /// Perihelion distance q = a(1-e) in AU
    pub fn perihelion(&self) -> f64 {
        self.a * (1.0 - self.e)
    }

    /// Aphelion distance Q = a(1+e) in AU
    pub fn aphelion(&self) -> f64 {
        self.a * (1.0 + self.e)
    }

    /// Semi-latus rectum p = a(1-e^2)
    pub fn semi_latus_rectum(&self) -> f64 {
        self.a * (1.0 - self.e * self.e)
    }

    /// Orbital period in days (only meaningful for bound orbits)
    pub fn period(&self, gm: f64) -> f64 {
        TWO_PI * (self.a.powi(3) / gm).sqrt()
    }

    /// Longitude of perihelion (varpi = Omega + omega)
    pub fn longitude_of_perihelion(&self) -> f64 {
        (self.omega_big + self.omega) % TWO_PI
    }

    /// Mean motion (radians/day)
    pub fn mean_motion(&self, gm: f64) -> f64 {
        (gm / self.a.powi(3)).sqrt()
    }

    /// Convert to Cartesian state vector.
    /// `gm` is the central body GM in AU^3/day^2.
    pub fn to_state_vector(&self, gm: f64) -> StateVector {
        elements_to_cartesian(self, gm)
    }
}

/// A massive body (planet, dwarf planet, hypothetical perturber).
#[derive(Debug, Clone)]
pub struct MassiveBody {
    /// Display name (diagnostics only).
    pub name: String,
    /// GM in AU^3/day^2
    pub gm: f64,
    /// Mass in solar masses
    pub mass: f64,
    /// Heliocentric ecliptic-J2000 state.
    pub state: StateVector,
}

impl MassiveBody {
    /// Build a massive body from its GM (AU³/day²) and heliocentric state.
    /// The solar-mass `mass` field is derived as `gm / GM_SUN`.
    pub fn new(name: impl Into<String>, gm: f64, state: StateVector) -> Self {
        Self {
            name: name.into(),
            gm,
            mass: gm / GM_SUN,
            state,
        }
    }

    /// Hill sphere radius: r_H = d * (m / 3 M_sun)^(1/3), with `d` the
    /// body's distance from the central mass in AU.
    pub fn hill_radius(&self, distance_from_central: f64) -> f64 {
        distance_from_central * (self.mass / 3.0).cbrt()
    }

    /// Initial conditions from a SPICE kernel: the body's heliocentric
    /// ecliptic-J2000 state at `time`, with the caller-supplied GM
    /// (AU³/day²).
    ///
    /// The ephemeris provides *states only*; GM always comes from the caller
    /// (e.g. [`super::GM_JUPITER`]) rather than the kernel header, so
    /// hard-coded and ephemeris-seeded integrations share one mass model.
    /// Note `planetlib` resolves the outer planets to their NAIF *system
    /// barycenters* — the right object here, since the `GM_*` constants are
    /// planetary-system GMs (planet + satellites).
    pub fn from_ephemeris(
        ephemeris: &mut Ephemeris,
        body: Body,
        time: &Time,
        gm: f64,
    ) -> Result<Self, PlanetError> {
        let sv = ephemeris.ecliptic_state(body, time)?;
        Ok(Self::new(
            body.name(),
            gm,
            StateVector::new(sv.position.to_vector3(), sv.velocity.to_vector3()),
        ))
    }
}

/// The four giant planets (Jupiter, Saturn, Uranus, Neptune) at `time`,
/// with kernel barycenter states and DE440 system GMs. The standard
/// starting condition for outer-solar-system integrations.
pub fn giant_planets_from_ephemeris(
    ephemeris: &mut Ephemeris,
    time: &Time,
) -> Result<Vec<MassiveBody>, PlanetError> {
    [
        (Body::Jupiter, GM_JUPITER),
        (Body::Saturn, GM_SATURN),
        (Body::Uranus, GM_URANUS),
        (Body::Neptune, GM_NEPTUNE),
    ]
    .into_iter()
    .map(|(body, gm)| MassiveBody::from_ephemeris(ephemeris, body, time, gm))
    .collect()
}

/// Configuration for an integration run.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Timestep in days (see `whm::validate_timestep`).
    pub dt: f64,
    /// Remove particles closer than this to the Sun (AU)
    pub removal_inner_au: f64,
    /// Remove particles farther than this from the Sun (AU)
    pub removal_outer_au: f64,
    /// Hill sphere multiplier for hybrid integrator switching
    pub hybrid_changeover_hill: f64,
    /// Bulirsch-Stoer relative accuracy parameter
    pub bs_epsilon: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            dt: 100.0,
            removal_inner_au: 0.1,
            removal_outer_au: 1e6,
            hybrid_changeover_hill: 3.0,
            bs_epsilon: 1e-12,
        }
    }
}

// ============================================================
// Orbital element <-> Cartesian conversion
// ============================================================

/// Convert Keplerian elements to Cartesian state vector.
/// Standard Murray & Dermott (1999) formulation.
pub fn elements_to_cartesian(elem: &OrbitalElements, gm: f64) -> StateVector {
    let p = elem.semi_latus_rectum();

    // Solve Kepler's equation for eccentric anomaly
    let ea = solve_kepler(elem.e, elem.mean_anomaly);

    // True anomaly from eccentric anomaly
    let nu = if elem.e < 1.0 {
        let sin_nu = (1.0 - elem.e * elem.e).sqrt() * ea.sin() / (1.0 - elem.e * ea.cos());
        let cos_nu = (ea.cos() - elem.e) / (1.0 - elem.e * ea.cos());
        sin_nu.atan2(cos_nu)
    } else {
        // Hyperbolic case: sin ν = +√(e²−1)·sinh H / (e·cosh H − 1), so that
        // M > 0 (outbound, r·v > 0) maps to ν > 0. A negated sin ν mirrors the
        // state about perihelion (inbound/outbound swapped).
        let sin_nu = (elem.e * elem.e - 1.0).sqrt() * ea.sinh() / (elem.e * ea.cosh() - 1.0);
        let cos_nu = (elem.e - ea.cosh()) / (elem.e * ea.cosh() - 1.0);
        sin_nu.atan2(cos_nu)
    };

    // Distance
    let r = p / (1.0 + elem.e * nu.cos());

    // Position and velocity in orbital plane
    let r_orb = Vector3::new(r * nu.cos(), r * nu.sin(), 0.0);
    let mu_p = (gm / p).sqrt();
    let v_orb = Vector3::new(-mu_p * nu.sin(), mu_p * (elem.e + nu.cos()), 0.0);

    // Rotation from orbital plane to ecliptic
    let cos_o = elem.omega.cos();
    let sin_o = elem.omega.sin();
    let cos_i = elem.i.cos();
    let sin_i = elem.i.sin();
    let cos_big_o = elem.omega_big.cos();
    let sin_big_o = elem.omega_big.sin();

    let px = cos_big_o * cos_o - sin_big_o * sin_o * cos_i;
    let py = sin_big_o * cos_o + cos_big_o * sin_o * cos_i;
    let pz = sin_o * sin_i;

    let qx = -cos_big_o * sin_o - sin_big_o * cos_o * cos_i;
    let qy = -sin_big_o * sin_o + cos_big_o * cos_o * cos_i;
    let qz = cos_o * sin_i;

    let pos = Vector3::new(
        r_orb.x * px + r_orb.y * qx,
        r_orb.x * py + r_orb.y * qy,
        r_orb.x * pz + r_orb.y * qz,
    );

    let vel = Vector3::new(
        v_orb.x * px + v_orb.y * qx,
        v_orb.x * py + v_orb.y * qy,
        v_orb.x * pz + v_orb.y * qz,
    );

    StateVector { pos, vel }
}

/// Convert Cartesian state vector to Keplerian elements.
pub fn cartesian_to_elements(state: &StateVector, gm: f64) -> OrbitalElements {
    let r = &state.pos;
    let v = &state.vel;
    let r_mag = r.norm();
    let v_mag = v.norm();

    // Specific angular momentum
    let h = r.cross(v);
    let h_mag = h.norm();

    // Node vector (z cross h)
    let n = Vector3::new(-h.y, h.x, 0.0);
    let n_mag = n.norm();

    // Eccentricity vector
    let e_vec = ((v_mag * v_mag - gm / r_mag) * r - r.dot(v) * v) / gm;
    let e = e_vec.norm();

    // Semi-major axis
    let energy = 0.5 * v_mag * v_mag - gm / r_mag;
    let a = if energy.abs() < 1e-30 {
        f64::INFINITY
    } else {
        -gm / (2.0 * energy)
    };

    // Inclination
    let i = (h.z / h_mag).clamp(-1.0, 1.0).acos();

    // Longitude of ascending node
    let omega_big = if n_mag < 1e-30 {
        0.0
    } else {
        let mut o = (n.x / n_mag).clamp(-1.0, 1.0).acos();
        if n.y < 0.0 {
            o = TWO_PI - o;
        }
        o
    };

    // Argument of perihelion
    let omega = if n_mag < 1e-30 || e < 1e-15 {
        if e >= 1e-15 {
            // Equatorial (no node): perihelion longitude measured along the
            // direction of motion; h.z's sign distinguishes prograde from
            // retrograde (a v.x or e_vec.y test alone breaks for i = π).
            let mut w = (e_vec.x / e).clamp(-1.0, 1.0).acos();
            if e_vec.y * h.z.signum() < 0.0 {
                w = TWO_PI - w;
            }
            w
        } else {
            0.0
        }
    } else {
        let cos_w = (n.dot(&e_vec) / (n_mag * e)).clamp(-1.0, 1.0);
        let mut w = cos_w.acos();
        if e_vec.z < 0.0 {
            w = TWO_PI - w;
        }
        w
    };

    // True anomaly
    let nu = if e < 1e-15 {
        if n_mag < 1e-30 {
            // Circular equatorial: true longitude along the direction of
            // motion, prograde/retrograde resolved by sign(h.z).
            let mut nu = (r.x / r_mag).clamp(-1.0, 1.0).acos();
            if r.y * h.z.signum() < 0.0 {
                nu = TWO_PI - nu;
            }
            nu
        } else {
            let cos_nu = (n.dot(r) / (n_mag * r_mag)).clamp(-1.0, 1.0);
            let mut nu = cos_nu.acos();
            if r.z < 0.0 {
                nu = TWO_PI - nu;
            }
            nu
        }
    } else {
        let cos_nu = (e_vec.dot(r) / (e * r_mag)).clamp(-1.0, 1.0);
        let mut nu = cos_nu.acos();
        if r.dot(v) < 0.0 {
            nu = TWO_PI - nu;
        }
        nu
    };

    // Mean anomaly from true anomaly
    let mean_anomaly = if e < 1.0 {
        let ea = ((1.0 - e) / (1.0 + e)).sqrt() * (nu / 2.0).tan();
        let ea = 2.0 * ea.atan();
        let ma = ea - e * ea.sin();
        if ma < 0.0 {
            ma + TWO_PI
        } else {
            ma
        }
    } else {
        // Hyperbolic
        let ha = ((e - 1.0) / (e + 1.0)).sqrt() * (nu / 2.0).tan();
        let ha = 2.0 * ha.atanh();
        e * ha.sinh() - ha
    };

    OrbitalElements {
        a,
        e,
        i,
        omega_big,
        omega,
        mean_anomaly,
    }
}

/// Solve Kepler's equation for the eccentric (e < 1) or hyperbolic (e > 1) anomaly.
///
/// Elliptic: M = E − e·sin E. Hyperbolic: M = e·sinh H − H.
///
/// Newton-Raphson with a bisection safeguard: the Kepler function is strictly
/// monotone in the anomaly, so the iterate is confined to a sign-changing
/// bracket and cannot diverge, even for e → 1. Starters: E₀ = π for e > 0.8
/// (elliptic, avoids the e·sin E ≈ M degeneracy near perihelion), Danby's
/// H₀ = ln(2M/e + 1.8) (hyperbolic).
pub fn solve_kepler(e: f64, mean_anomaly: f64) -> f64 {
    if e < 1.0 {
        // Reduce M to (-pi, pi] and exploit the odd symmetry E(-M) = -E(M).
        let mut m = mean_anomaly % TWO_PI;
        if m > std::f64::consts::PI {
            m -= TWO_PI;
        } else if m < -std::f64::consts::PI {
            m += TWO_PI;
        }
        let sign = if m < 0.0 { -1.0 } else { 1.0 };
        let m = m.abs();

        // For m in [0, pi] the root lies in [m, m + e] (since 0 <= e sin E <= e).
        let mut lo = m;
        let mut hi = (m + e).min(std::f64::consts::PI + e);
        let mut ea = if e > 0.8 {
            std::f64::consts::PI
        } else {
            m + 0.85 * e
        };
        ea = ea.clamp(lo, hi);

        let mut converged = false;
        for _ in 0..100 {
            let f = ea - e * ea.sin() - m;
            if f > 0.0 {
                hi = ea;
            } else {
                lo = ea;
            }
            let fp = 1.0 - e * ea.cos();
            let delta = f / fp;
            let mut next = ea - delta;
            if !(lo..=hi).contains(&next) {
                // Newton left the bracket: bisect instead.
                next = 0.5 * (lo + hi);
            }
            let step = (next - ea).abs();
            ea = next;
            if step < 1e-15 * ea.abs().max(1.0) {
                converged = true;
                break;
            }
        }
        debug_assert!(
            converged || (ea - e * ea.sin() - m).abs() < 1e-12,
            "Kepler solver failed to converge: e = {e}, M = {mean_anomaly}"
        );
        sign * ea
    } else {
        // Hyperbolic, odd symmetry H(-M) = -H(M).
        let sign = if mean_anomaly < 0.0 { -1.0 } else { 1.0 };
        let m = mean_anomaly.abs();

        // Danby (1988) starter.
        let mut ha = (2.0 * m / e + 1.8).ln().max(1e-12);

        // Bracket: f(0) = -m <= 0; expand hi until f(hi) >= 0
        // (f' = e cosh H - 1 > 0, so the root is unique).
        let f = |h: f64| e * h.sinh() - h - m;
        let mut lo = 0.0_f64;
        let mut hi = ha.max(1.0);
        while f(hi) < 0.0 {
            hi *= 2.0;
        }
        ha = ha.clamp(lo, hi);

        let mut converged = false;
        for _ in 0..100 {
            let fv = f(ha);
            if fv > 0.0 {
                hi = ha;
            } else {
                lo = ha;
            }
            let fp = e * ha.cosh() - 1.0;
            let mut next = ha - fv / fp;
            if !(lo..=hi).contains(&next) {
                next = 0.5 * (lo + hi);
            }
            let step = (next - ha).abs();
            ha = next;
            if step < 1e-15 * ha.abs().max(1.0) {
                converged = true;
                break;
            }
        }
        debug_assert!(
            converged || f(ha).abs() < 1e-12 * m.max(1.0),
            "hyperbolic Kepler solver failed to converge: e = {e}, M = {mean_anomaly}"
        );
        sign * ha
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AU_KM, DAY_S, J2000};
    use crate::jplephem::kernel::SpiceKernel;
    use crate::time::Timescale;

    /// This module's GM_SUN (AU³/day²) expressed in km³/s² for
    /// `elementslib`'s APIs, so both sides describe the identical two-body
    /// problem.
    fn gm_sun_km3_s2() -> f64 {
        GM_SUN * AU_KM * AU_KM * AU_KM / (DAY_S * DAY_S)
    }

    /// Wrap an angle difference to (-pi, pi].
    fn angle_diff(x: f64, y: f64) -> f64 {
        let mut d = (x - y) % TWO_PI;
        if d > std::f64::consts::PI {
            d -= TWO_PI;
        } else if d < -std::f64::consts::PI {
            d += TWO_PI;
        }
        d
    }

    /// Compare `cartesian_to_elements` against `elementslib`'s
    /// OsculatingElements for the same state vector.
    ///
    /// Tolerances by regime:
    /// - a (or q for near-parabolic), e, i: 1e-9 relative — pure algebra on
    ///   the state vector in both implementations.
    /// - Ω, ω, M: 1e-8 rad away from degeneracies. Near-degenerate cases
    ///   (e ≤ 1e-6 makes ω/M individually ill-defined; i ≤ 1e-6 or i ≥ π−1e-6
    ///   makes Ω/ω individually ill-defined) are compared through the
    ///   well-defined combinations instead, with 2e-5 rad slack since the
    ///   two codes resolve the degenerate angle from differently-rounded
    ///   tiny vectors.
    fn check_elements_against_elementslib(state: &StateVector) {
        let ours = cartesian_to_elements(state, GM_SUN);
        let sf =
            crate::elementslib::osculating_elements_of(&state.pos, &state.vel, gm_sun_km3_s2());

        let e = sf.eccentricity();
        let i = sf.inclination_rad();

        // Shape: a for bound/hyperbolic, q always. Skip a near e = 1, where
        // a = -gm/(2*energy) is ill-conditioned (the ~1e-16 relative noise of
        // the energy is amplified by 1/|e-1|); q remains well-conditioned and
        // is always compared.
        if (e - 1.0).abs() > 1e-5 {
            let a_sf = sf.semi_major_axis_au();
            assert!(
                (ours.a - a_sf).abs() <= 1e-9 * a_sf.abs(),
                "a mismatch: {} vs {a_sf}",
                ours.a
            );
        }
        let q_sf = sf.periapsis_distance_au();
        let q_ours = ours.a * (1.0 - ours.e);
        assert!(
            (q_ours - q_sf).abs() <= 1e-8 * q_sf,
            "q mismatch: {q_ours} vs {q_sf} (e = {e})"
        );
        assert!(
            (ours.e - e).abs() <= 1e-9 * e.max(1.0),
            "e mismatch: {} vs {e}",
            ours.e
        );
        assert!((ours.i - i).abs() <= 1e-9, "i mismatch: {} vs {i}", ours.i);

        let near_circular = e <= 1e-6;
        let near_equatorial = !(1e-6..=std::f64::consts::PI - 1e-6).contains(&i);

        if !near_equatorial {
            assert!(
                angle_diff(ours.omega_big, sf.longitude_of_ascending_node_rad()).abs() <= 1e-8,
                "Omega mismatch: {} vs {}",
                ours.omega_big,
                sf.longitude_of_ascending_node_rad()
            );
        }
        if !near_circular && !near_equatorial {
            assert!(
                angle_diff(ours.omega, sf.argument_of_periapsis_rad()).abs() <= 1e-8,
                "omega mismatch: {} vs {}",
                ours.omega,
                sf.argument_of_periapsis_rad()
            );
        }
        if !near_circular && e < 1.0 {
            assert!(
                angle_diff(ours.mean_anomaly, sf.mean_anomaly_rad()).abs() <= 1e-8,
                "M mismatch: {} vs {} (e = {e})",
                ours.mean_anomaly,
                sf.mean_anomaly_rad()
            );
        }

        // Degenerate regimes: compare the non-degenerate combinations.
        // ϖ-like sum Ω + ω + M is well-defined for prograde orbits whenever
        // the state itself is (elementslib folds the same convention into its
        // longitude accessors only for i < π/2, so restrict to prograde).
        if (near_circular || near_equatorial) && e < 1.0 && i < std::f64::consts::FRAC_PI_2 {
            let ours_l = ours.omega_big + ours.omega + ours.mean_anomaly;
            let sf_l = sf.longitude_of_ascending_node_rad()
                + sf.argument_of_periapsis_rad()
                + sf.mean_anomaly_rad();
            assert!(
                angle_diff(ours_l, sf_l).abs() <= 2e-5,
                "mean longitude mismatch in degenerate regime: {ours_l} vs {sf_l} \
                 (e = {e}, i = {i})"
            );
        }
    }

    /// Round trip: elements -> cartesian -> elements must reproduce the
    /// state vector (the conversions are mutual inverses), checked through
    /// the state because angle representations differ in degenerate regimes.
    fn check_round_trip(elem: &OrbitalElements) {
        let state = elements_to_cartesian(elem, GM_SUN);
        let back = cartesian_to_elements(&state, GM_SUN);
        let state2 = elements_to_cartesian(&back, GM_SUN);

        let scale_r = state.pos.norm();
        let scale_v = state.vel.norm();
        assert!(
            (state.pos - state2.pos).norm() <= 1e-9 * scale_r,
            "round-trip position drift: a = {}, e = {}, i = {}, O = {}, w = {}, M = {}: \
             |dr| = {:.3e}",
            elem.a,
            elem.e,
            elem.i,
            elem.omega_big,
            elem.omega,
            elem.mean_anomaly,
            (state.pos - state2.pos).norm()
        );
        assert!(
            (state.vel - state2.vel).norm() <= 1e-9 * scale_v,
            "round-trip velocity drift: a = {}, e = {}, i = {}, O = {}, w = {}, M = {}",
            elem.a,
            elem.e,
            elem.i,
            elem.omega_big,
            elem.omega,
            elem.mean_anomaly
        );
    }

    #[test]
    fn test_element_round_trips_vs_elementslib() {
        let e_grid = [0.0, 1e-6, 0.3, 0.95, 0.999, 1.0 + 1e-6, 1.5];
        let i_grid_deg = [0.0, 1e-6_f64.to_degrees(), 30.0, 90.0, 150.0, 180.0];
        // All quadrants of Omega, omega, nu (via mean anomaly proxies).
        let angles = [0.5, 2.0, 3.8, 5.5];

        for &e in &e_grid {
            for &i_deg in &i_grid_deg {
                for &omega_big in &angles {
                    for &omega in &angles {
                        for &m in &angles {
                            // Hyperbolic: a < 0 with q = 30 AU. Pick the
                            // anomaly through the true anomaly as a fraction
                            // of the asymptote angle so r stays O(q) — a raw
                            // M grid puts near-parabolic states
                            // (|a| = q/(e-1), huge) at absurd distances where
                            // every element is ill-conditioned. The fractions
                            // still cover inbound and outbound on both sides
                            // of perihelion.
                            let (a, mean_anomaly) = if e < 1.0 {
                                (450.0, m)
                            } else {
                                let nu_inf = (-1.0_f64 / e).acos();
                                let frac = 0.8 * (m - std::f64::consts::PI) / std::f64::consts::PI;
                                let nu = frac * nu_inf;
                                let h: f64 = 2.0
                                    * (((e - 1.0) / (e + 1.0)).sqrt() * (nu / 2.0).tan()).atanh();
                                (-30.0 / (e - 1.0), e * h.sinh() - h)
                            };
                            let elem = OrbitalElements {
                                a,
                                e,
                                i: i_deg.to_radians(),
                                omega_big,
                                omega,
                                mean_anomaly,
                            };
                            let state = elements_to_cartesian(&elem, GM_SUN);
                            check_elements_against_elementslib(&state);
                            check_round_trip(&elem);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_giant_planets_from_de421_at_j2000() {
        let kernel = SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421");
        let mut eph = Ephemeris::from_kernel(kernel);
        let ts = Timescale::default();
        let t = ts.tdb_jd(J2000);

        let giants = giant_planets_from_ephemeris(&mut eph, &t).expect("kernel lookup");
        let expected = [
            (5.2, "Jupiter"),
            (9.5, "Saturn"),
            (19.2, "Uranus"),
            (30.1, "Neptune"),
        ];
        for (body, (a, name)) in giants.iter().zip(expected) {
            assert_eq!(body.name, name);
            let r = body.state.distance();
            assert!(
                (r - a).abs() / a < 0.06,
                "{name} at r = {r:.2} AU, expected near {a} AU"
            );
            // Speed near the vis-viva value for that distance and a.
            let vis_viva = (GM_SUN * (2.0 / r - 1.0 / a)).sqrt();
            assert!((body.state.speed() - vis_viva).abs() / vis_viva < 0.05);
            // Giants are within ~2.5 deg of the ecliptic.
            let lat = (body.state.pos.z / r).asin().abs();
            assert!(lat < 0.05, "{name} ecliptic latitude {lat:.3} rad");
            // mass derived from gm
            assert!((body.mass - body.gm / GM_SUN).abs() < 1e-18);
        }
    }

    #[test]
    fn test_state_from_kepler_orbit_matches_elements() {
        // A KeplerOrbit built from an nbodylib state, propagated zero days,
        // returns the same state; propagated half a period it matches
        // elements_to_cartesian at M + pi.
        let elem = OrbitalElements {
            a: 40.0,
            e: 0.2,
            i: 0.3,
            omega_big: 1.1,
            omega: 0.7,
            mean_anomaly: 0.4,
        };
        let state = elements_to_cartesian(&elem, GM_SUN);

        let ts = Timescale::default();
        let epoch = ts.tt_jd(J2000, None);
        let orbit = KeplerOrbit::new(state.pos, state.vel, &epoch, GM_SUN, None, None);

        let s0 = StateVector::from_kepler_orbit(&orbit, &epoch);
        assert!((s0.pos - state.pos).norm() < 1e-12);
        assert!((s0.vel - state.vel).norm() < 1e-12);

        let period = elem.period(GM_SUN);
        let t_half = ts.tt_jd(J2000 + 0.5 * period, None);
        let s_half = StateVector::from_kepler_orbit(&orbit, &t_half);

        let mut elem_half = elem;
        elem_half.mean_anomaly = (elem.mean_anomaly + std::f64::consts::PI) % TWO_PI;
        let expect = elements_to_cartesian(&elem_half, GM_SUN);
        assert!(
            (s_half.pos - expect.pos).norm() < 1e-8 * expect.pos.norm(),
            "half-period mismatch: {:.3e}",
            (s_half.pos - expect.pos).norm()
        );
    }
}
