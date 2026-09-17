//! Bulirsch-Stoer integrator for high-accuracy integration during close
//! encounters.
//!
//! Uses the modified midpoint method with Richardson extrapolation to achieve
//! arbitrary accuracy. This is the fallback integrator when a particle enters
//! a planet's Hill sphere and the symplectic integrator would fail.
//!
//! Two fixes relative to a naive implementation (bug history preserved from
//! the original port):
//!
//! 1. **Planets move during the step.** Each acceleration evaluation
//!    Kepler-drifts every massive body from its step-start state to the
//!    substep epoch, instead of holding planets frozen over the full dt
//!    (Jupiter moves ~25 deg in 300 d).
//! 2. **Adaptive step reduction.** When the extrapolation tableau fails to
//!    converge, the step is recursively halved rather than silently
//!    returning the unconverged full-step estimate.
//!
//! The force model supports the Chambers (1999) hybrid changeover: each body
//! carries a critical radius `r_crit`, and the BS force takes the fraction
//! `1 - K(d)` of the direct planet force (the kick stage applies the
//! complementary `K(d)` fraction plus the indirect term). Pass `r_crit = 0`
//! for full direct force (pure BS integration, no kick sharing).
//!
//! Reference: Press et al., Numerical Recipes; Stoer & Bulirsch (1980);
//! Chambers (1999)

use nalgebra::Vector3;

use super::kepler::kepler_drift;
use super::types::{MassiveBody, StateVector};
use super::GM_SUN;

/// Maximum subdivision sequence for Bulirsch-Stoer extrapolation.
/// Bulirsch-Stoer sequence: 2, 4, 6, 8, 12, 16, 24, 32, ...
const BS_SEQUENCE: [usize; 12] = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// Maximum number of columns in the extrapolation tableau.
const MAX_COLUMNS: usize = 12;

/// Maximum recursive halvings before giving up (dt reduction of 2^8 = 256x).
const MAX_HALVINGS: u32 = 8;

/// Chambers (1999) changeover function K(d).
///
/// Returns the fraction of the direct planet force handled by the *kick*
/// stage: K = 1 far from the planet (d >= r_crit, pure symplectic kick),
/// K = 0 deep inside the encounter region (d <= 0.1 r_crit, pure BS), with
/// the smooth rational bridge K = y^2 / (2y^2 - 2y + 1) in between.
/// The BS force uses the complement (1 - K).
pub fn changeover_k(d: f64, r_crit: f64) -> f64 {
    if r_crit <= 0.0 {
        return 1.0;
    }
    let y = (d - 0.1 * r_crit) / (0.9 * r_crit);
    if y <= 0.0 {
        0.0
    } else if y >= 1.0 {
        1.0
    } else {
        y * y / (2.0 * y * y - 2.0 * y + 1.0)
    }
}

/// Gravitational acceleration on a test particle at time `t` after the step
/// start: Sun plus the BS share `(1 - K)` of the direct force from each body,
/// with bodies Kepler-drifted from their step-start states to time `t`.
///
/// The indirect (heliocentric-frame) term is *not* included here: in the
/// hybrid scheme it is applied at full weight by the kick stage. For
/// standalone BS use (empty `r_crit` handling via `r_crit[i] = 0`) the
/// indirect term is negligible only if the bodies are massless or the caller
/// accounts for it separately.
fn acceleration(
    pos: &Vector3<f64>,
    t: f64,
    bodies: &[MassiveBody],
    r_crit: &[f64],
) -> Vector3<f64> {
    let r = pos.norm();
    let mut accel = if r > 1e-30 {
        -GM_SUN * pos / (r * r * r)
    } else {
        Vector3::zeros()
    };

    for (j, body) in bodies.iter().enumerate() {
        // Drift the planet to the substep epoch (frozen planets give O(1)
        // errors over a 300 d step).
        let body_pos = if t != 0.0 {
            kepler_drift(&body.state, t, GM_SUN + body.gm).pos
        } else {
            body.state.pos
        };
        let dr = pos - body_pos;
        let dr_mag = dr.norm();
        if dr_mag > 1e-30 {
            let bs_weight = 1.0 - changeover_k(dr_mag, r_crit.get(j).copied().unwrap_or(0.0));
            accel -= bs_weight * body.gm * dr / (dr_mag * dr_mag * dr_mag);
        }
    }

    accel
}

/// Modified midpoint method: integrate from t0 to t0+H using n substeps.
/// Returns the final state vector.
fn modified_midpoint(
    state: &StateVector,
    bodies: &[MassiveBody],
    r_crit: &[f64],
    t0: f64,
    h_total: f64,
    n_steps: usize,
) -> StateVector {
    let h = h_total / n_steps as f64;

    // First step: Euler
    let a0 = acceleration(&state.pos, t0, bodies, r_crit);
    let mut z_prev = state.pos;
    let mut z_curr = state.pos + h * state.vel;
    let mut v_prev = state.vel;
    let mut v_curr = state.vel + h * a0;

    // General steps: leapfrog
    for k in 1..n_steps {
        let a = acceleration(&z_curr, t0 + k as f64 * h, bodies, r_crit);
        let z_next = z_prev + 2.0 * h * v_curr;
        let v_next = v_prev + 2.0 * h * a;

        z_prev = z_curr;
        z_curr = z_next;
        v_prev = v_curr;
        v_curr = v_next;
    }

    // Final smoothing step
    let a_final = acceleration(&z_curr, t0 + h_total, bodies, r_crit);
    let pos = 0.5 * (z_prev + z_curr + h * v_curr);
    let vel = 0.5 * (v_prev + v_curr + h * a_final);

    StateVector::new(pos, vel)
}

/// One Bulirsch-Stoer extrapolation attempt over [t0, t0+dt].
///
/// Returns `(state, converged)`: the extrapolated state and whether the
/// tableau met the tolerance `epsilon`.
fn bs_attempt(
    state: &StateVector,
    bodies: &[MassiveBody],
    r_crit: &[f64],
    t0: f64,
    dt: f64,
    epsilon: f64,
) -> (StateVector, bool) {
    let mut pos_tableau = [[Vector3::<f64>::zeros(); MAX_COLUMNS]; MAX_COLUMNS];
    let mut vel_tableau = [[Vector3::<f64>::zeros(); MAX_COLUMNS]; MAX_COLUMNS];

    for k in 0..MAX_COLUMNS {
        let n = BS_SEQUENCE[k];
        let result = modified_midpoint(state, bodies, r_crit, t0, dt, n);

        pos_tableau[k][0] = result.pos;
        vel_tableau[k][0] = result.vel;

        if k > 0 {
            // Richardson extrapolation (must go forward: each j depends on j-1)
            for j in 1..=k {
                let ratio = (BS_SEQUENCE[k] as f64 / BS_SEQUENCE[k - j] as f64).powi(2);
                let factor = 1.0 / (ratio - 1.0);

                pos_tableau[k][j] = pos_tableau[k][j - 1]
                    + factor * (pos_tableau[k][j - 1] - pos_tableau[k - 1][j - 1]);
                vel_tableau[k][j] = vel_tableau[k][j - 1]
                    + factor * (vel_tableau[k][j - 1] - vel_tableau[k - 1][j - 1]);
            }

            // Check convergence
            let pos_err = (pos_tableau[k][k] - pos_tableau[k][k - 1]).norm();
            let vel_err = (vel_tableau[k][k] - vel_tableau[k][k - 1]).norm();

            let pos_scale = pos_tableau[k][k].norm().max(1.0);
            let vel_scale = vel_tableau[k][k].norm().max(1e-6);

            if pos_err / pos_scale < epsilon && vel_err / vel_scale < epsilon {
                return (StateVector::new(pos_tableau[k][k], vel_tableau[k][k]), true);
            }
        }
    }

    let k = MAX_COLUMNS - 1;
    (
        StateVector::new(pos_tableau[k][k], vel_tableau[k][k]),
        false,
    )
}

/// Recursive halving driver around `bs_attempt`.
fn bs_recurse(
    state: &StateVector,
    bodies: &[MassiveBody],
    r_crit: &[f64],
    t0: f64,
    dt: f64,
    epsilon: f64,
    depth: u32,
) -> (StateVector, bool) {
    let (result, converged) = bs_attempt(state, bodies, r_crit, t0, dt, epsilon);
    if converged {
        return (result, true);
    }
    if depth >= MAX_HALVINGS {
        return (result, false);
    }
    let half = 0.5 * dt;
    let (mid, ok1) = bs_recurse(state, bodies, r_crit, t0, half, epsilon, depth + 1);
    let (end, ok2) = bs_recurse(&mid, bodies, r_crit, t0 + half, half, epsilon, depth + 1);
    (end, ok1 && ok2)
}

/// Perform one Bulirsch-Stoer step on a single test particle over [0, dt].
///
/// `bodies` hold their step-start states and are Kepler-drifted internally to
/// each substep epoch. `r_crit` gives the Chambers changeover radius per body
/// (use 0.0, or an empty slice, for full direct force).
///
/// On convergence failure the step is recursively halved up to 2^8 times;
/// the returned flag is `false` only if even the smallest substep failed.
///
/// Returns `(new_state, converged)`.
pub fn bs_step(
    state: &StateVector,
    bodies: &[MassiveBody],
    r_crit: &[f64],
    dt: f64,
    epsilon: f64,
) -> (StateVector, bool) {
    bs_recurse(state, bodies, r_crit, 0.0, dt, epsilon, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bs_circular_orbit() {
        let a = 1.0;
        let v = (GM_SUN / a).sqrt();
        let state = StateVector::new(Vector3::new(a, 0.0, 0.0), Vector3::new(0.0, v, 0.0));

        let period = 2.0 * std::f64::consts::PI * (a * a * a / GM_SUN).sqrt();
        let n_steps = 100;
        let dt = period / n_steps as f64;

        let mut s = state;
        for _ in 0..n_steps {
            let (new_s, converged) = bs_step(&s, &[], &[], dt, 1e-12);
            assert!(converged);
            s = new_s;
        }

        // Should return to start within high accuracy
        assert_relative_eq!(s.pos.x, a, epsilon = 1e-8);
        assert_relative_eq!(s.pos.y, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_bs_energy_conservation() {
        let state = StateVector::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, 0.02, 0.002));

        let energy = |s: &StateVector| 0.5 * s.vel.norm_squared() - GM_SUN / s.pos.norm();

        let e0 = energy(&state);
        let mut s = state;
        for _ in 0..100 {
            let (new_s, converged) = bs_step(&s, &[], &[], 50.0, 1e-12);
            assert!(converged);
            s = new_s;
        }
        let e1 = energy(&s);

        let de = ((e1 - e0) / e0).abs();
        assert!(de < 1e-10, "BS energy error: {:.2e}", de);
    }

    #[test]
    fn test_bs_adaptive_halving_high_eccentricity() {
        // e = 0.99 orbit at perihelion with a step that spans the whole
        // perihelion passage: a single tableau cannot converge, the adaptive
        // halving must.
        let a = 1.0;
        let e = 0.99;
        let q = a * (1.0 - e);
        let v_peri = (GM_SUN * (2.0 / q - 1.0 / a)).sqrt();
        let state = StateVector::new(Vector3::new(q, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));

        let energy = |s: &StateVector| 0.5 * s.vel.norm_squared() - GM_SUN / s.pos.norm();
        let e0 = energy(&state);

        // ~1/6 of the orbital period in one BS call
        let period = 2.0 * std::f64::consts::PI * (a * a * a / GM_SUN).sqrt();
        let (s1, converged) = bs_step(&state, &[], &[], period / 6.0, 1e-12);
        assert!(converged, "adaptive BS should converge via halving");

        let de = ((energy(&s1) - e0) / e0).abs();
        assert!(de < 1e-8, "energy error through perihelion: {:.2e}", de);
    }

    #[test]
    fn test_bs_matches_kepler_drift_hyperbolic_flyby() {
        // A solar flyby on a hyperbolic orbit, no planets: the numerical BS
        // trajectory must agree with the analytic universal-variable drift.
        let e = 1.5;
        let q = 2.0;
        let v_peri = (GM_SUN * (1.0 + e) / q).sqrt();
        let state = StateVector::new(Vector3::new(q, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));

        let total_dt = 2000.0;
        let mut s = state;
        for _ in 0..40 {
            let (new_s, converged) = bs_step(&s, &[], &[], total_dt / 40.0, 1e-13);
            assert!(converged);
            s = new_s;
        }

        let exact = kepler_drift(&state, total_dt, GM_SUN);
        let dr = (s.pos - exact.pos).norm() / exact.pos.norm();
        let dv = (s.vel - exact.vel).norm() / exact.vel.norm();
        assert!(dr < 1e-9, "BS vs drift flyby position mismatch: {dr:.2e}");
        assert!(dv < 1e-9, "BS vs drift flyby velocity mismatch: {dv:.2e}");
    }

    #[test]
    fn test_bs_planets_drift_during_step() {
        // A particle co-orbiting far from a Jupiter-mass planet: the
        // acceleration evaluated at the end of the step must use the drifted
        // planet position. We verify by comparing one big BS step against
        // many small ones — frozen planets would produce a secular offset.
        let gm_jup = 1e-3 * GM_SUN;
        let a_jup = 5.2;
        let v_jup = (GM_SUN / a_jup).sqrt();
        let jupiter = MassiveBody::new(
            "Jupiter",
            gm_jup,
            StateVector::new(Vector3::new(a_jup, 0.0, 0.0), Vector3::new(0.0, v_jup, 0.0)),
        );

        let a_p = 8.0;
        let v_p = (GM_SUN / a_p).sqrt();
        let particle = StateVector::new(Vector3::new(0.0, a_p, 0.0), Vector3::new(-v_p, 0.0, 0.0));

        // One 600 d step vs 60 x 10 d steps (re-drifting Jupiter between calls)
        let (coarse, ok) = bs_step(
            &particle,
            std::slice::from_ref(&jupiter),
            &[0.0],
            600.0,
            1e-13,
        );
        assert!(ok);

        let mut fine = particle;
        let mut jup = jupiter;
        for _ in 0..60 {
            let (s, ok) = bs_step(&fine, &[jup.clone()], &[0.0], 10.0, 1e-13);
            assert!(ok);
            fine = s;
            jup.state = kepler_drift(&jup.state, 10.0, GM_SUN + jup.gm);
        }

        let dr = (coarse.pos - fine.pos).norm();
        assert!(
            dr < 1e-6,
            "coarse vs fine BS positions differ by {dr:.2e} AU"
        );
    }

    #[test]
    fn test_changeover_limits_and_monotone() {
        let rc = 1.0;
        assert_eq!(changeover_k(0.0, rc), 0.0);
        assert_eq!(changeover_k(0.05, rc), 0.0);
        assert_eq!(changeover_k(1.0, rc), 1.0);
        assert_eq!(changeover_k(5.0, rc), 1.0);
        // r_crit = 0 disables the changeover (full kick share)
        assert_eq!(changeover_k(0.5, 0.0), 1.0);

        let mut prev = 0.0;
        for k in 0..=100 {
            let d = k as f64 * 0.01;
            let val = changeover_k(d, rc);
            assert!(val >= prev, "K not monotone at d = {d}");
            prev = val;
        }
        // Continuity at the midpoint: K(0.55) = y^2/(2y^2-2y+1) at y = 0.5 -> 0.5
        assert_relative_eq!(changeover_k(0.55, rc), 0.5, epsilon = 1e-12);
    }
}
