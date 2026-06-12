//! Kepler drift step: advance each body along its two-body Keplerian orbit.
//!
//! Uses the universal variable formulation (Battin 1999) for robust
//! propagation of elliptic, parabolic, and hyperbolic orbits.

use crate::constants::TAU as TWO_PI;

use super::types::StateVector;

/// Advance a state vector along its Keplerian orbit for time dt.
/// Uses Stumpff function formulation with bisection-safeguarded
/// Newton-Raphson iteration on the universal variable s. Bound orbits are
/// reduced modulo the orbital period so the root is bracketed within one
/// revolution of the universal anomaly; hyperbolic brackets expand outward
/// but never past the Stumpff overflow cap. Cross-validated against
/// `crate::keplerlib` (the SPICE prop2b port) by the oracle tests below.
///
/// `gm` is the central body GM in AU^3/day^2.
pub fn kepler_drift(state: &StateVector, dt: f64, gm: f64) -> StateVector {
    let r0 = state.pos;
    let v0 = state.vel;
    let r0_mag = r0.norm();
    let v0_mag2 = v0.norm_squared();

    if r0_mag < 1e-30 {
        return *state;
    }

    // Specific orbital energy -> semi-major axis
    let energy = 0.5 * v0_mag2 - gm / r0_mag;
    let alpha = -2.0 * energy / gm; // = 1/a (positive for elliptic)

    let r0_dot_v0 = r0.dot(&v0);

    // Bound orbits are periodic: reduce dt by whole orbital periods so the
    // universal anomaly stays within one revolution. Without this, multi-orbit
    // dt makes every term of the universal Kepler equation ~ sqrt(gm)*|dt|;
    // the rounding noise of the residual, divided by f' = r (small near
    // perihelion at high e), then exceeds the 1e-15*s step tolerance and
    // safeguarded Newton dithers inside a noise-width bracket forever
    // ("convergence plateau"). Within one period every term is O(one orbit)
    // and the noise floor sits below the tolerance.
    let mut elliptic_period = None;
    let dt = if alpha > 0.0 {
        let period = TWO_PI / (gm * alpha.powi(3)).sqrt();
        if period.is_finite() {
            elliptic_period = Some(period);
            dt.rem_euclid(period)
        } else {
            dt
        }
    } else {
        dt
    };
    if dt == 0.0 {
        return *state;
    }

    // Initial guess for universal variable s
    let mut s = if alpha > 0.0 {
        // Elliptic: s ~ n*dt where n = sqrt(gm * alpha^3)
        let n = (gm * alpha.powi(3)).sqrt();
        dt * n / (1.0 + 0.85 * (r0_dot_v0 / r0_mag).abs() * dt * n)
    } else if alpha < -1e-15 {
        // Hyperbolic
        let a = 1.0 / alpha;
        let sign_dt = dt.signum();
        sign_dt
            * (-a).sqrt()
            * ((-2.0 * gm * alpha * dt)
                / (r0_dot_v0 + sign_dt * (-gm / alpha).sqrt() * (1.0 - r0_mag * alpha)))
                .abs()
                .ln()
    } else {
        // Near-parabolic
        dt / r0_mag
    };

    // Newton-Raphson iteration on the universal Kepler equation
    // f(s) = r0*s + r0_dot_v0/sqrt(gm) * s^2 * c2(alpha*s^2)
    //        + (1 - r0*alpha) * s^3 * c3(alpha*s^2) - sqrt(gm)*dt = 0
    //
    // f is strictly monotone in s (f' = r > 0), so a sign-changing bracket
    // always exists and bisection-safeguarded Newton cannot diverge.
    let sqrt_gm = gm.sqrt();

    let kepler_f = |s: f64| -> f64 {
        let psi = alpha * s * s;
        let (c2, c3) = stumpff_c2c3(psi);
        let s2 = s * s;
        r0_mag * s + r0_dot_v0 / sqrt_gm * s2 * c2 + (1.0 - r0_mag * alpha) * s2 * s * c3
            - sqrt_gm * dt
    };

    // Hyperbolic cap: |psi| = -alpha*s^2 beyond ~580^2 overflows cosh in the
    // Stumpff terms, turning f into inf/NaN and corrupting the bracket.
    // Physical roots sit exponentially far inside the cap (H = 580 means
    // r ~ e^580 * |a|), so clamping loses nothing.
    let s_cap = if alpha < 0.0 {
        580.0 / (-alpha).sqrt()
    } else {
        f64::INFINITY
    };

    // Bracket the root: f(0) = -sqrt(gm)*dt has the opposite sign of dt.
    // Elliptic: dt was reduced to [0, period), so the root lies within one
    // revolution of the universal anomaly, s in [0, 2*pi/sqrt(alpha)]
    // (expand only to absorb rounding at the endpoint). Hyperbolic /
    // near-parabolic: expand outward from the starter until the sign
    // changes, never past the overflow cap.
    let (mut lo, mut hi) = if elliptic_period.is_some() {
        let mut hi = TWO_PI / alpha.sqrt();
        while kepler_f(hi) < 0.0 {
            hi *= 1.5;
        }
        (0.0, hi)
    } else if dt >= 0.0 {
        let mut hi = s.abs().max(dt / r0_mag).max(1e-8).min(s_cap);
        while kepler_f(hi) < 0.0 && hi < s_cap {
            hi = (hi * 2.0).min(s_cap);
        }
        (0.0, hi)
    } else {
        let mut lo = (-s.abs().max(-dt / r0_mag).max(1e-8)).max(-s_cap);
        while kepler_f(lo) > 0.0 && lo > -s_cap {
            lo = (lo * 2.0).max(-s_cap);
        }
        (lo, 0.0)
    };
    s = s.clamp(lo, hi);

    // Residual scale at the root (the constant term of the universal Kepler
    // equation). When |f| dwarfs this, the iterate is deep in the Stumpff
    // exponential tail where Newton only creeps additively (~1/sqrt(-alpha)
    // per step without leaving the bracket); bisection gets there in
    // O(log) steps instead.
    let f_scale = sqrt_gm * dt.abs();

    let mut converged = false;
    let mut s_prev = f64::NAN;
    for _ in 0..100 {
        let psi = alpha * s * s;
        let (c2, c3) = stumpff_c2c3(psi);

        let s2 = s * s;
        let s3 = s2 * s;

        let t1 = r0_mag * s;
        let t2 = r0_dot_v0 / sqrt_gm * s2 * c2;
        let t3 = (1.0 - r0_mag * alpha) * s3 * c3;
        let f_val = t1 + t2 + t3 - sqrt_gm * dt;

        // The residual cannot be resolved below the rounding noise of its
        // constituent terms; reaching that floor is convergence.
        let f_floor = 4.0 * f64::EPSILON * (t1.abs() + t2.abs() + t3.abs() + sqrt_gm * dt.abs());
        if f_val.abs() <= f_floor {
            converged = true;
            break;
        }

        if f_val > 0.0 {
            hi = s;
        } else {
            lo = s;
        }

        // F'(χ) = r(χ) — the distance at the current iterate
        let df =
            r0_mag + r0_dot_v0 / sqrt_gm * s * (1.0 - psi * c3) + (1.0 - r0_mag * alpha) * s2 * c2;

        let mut next = s - f_val / df;
        if !(lo..=hi).contains(&next) || f_val.abs() > 1e8 * f_scale {
            next = 0.5 * (lo + hi);
        }
        let ds = (next - s).abs();
        // A bit-exact period-2 cycle means the iterate is alternating
        // between the two adjacent representable values that straddle the
        // root: the residual is pure rounding noise and the answer is as
        // accurate as f64 permits.
        let cycled = next == s_prev;
        s_prev = s;
        s = next;

        // Converged when the step or the remaining bracket is at relative
        // machine precision (the bracket collapses onto the root even when
        // residual noise makes the Newton step dither).
        let tol = 1e-15 * s.abs().max(1.0);
        if ds < tol || (hi - lo) < tol || cycled {
            converged = true;
            break;
        }
    }
    debug_assert!(
        converged,
        "universal Kepler solver failed to converge: dt = {dt}, gm = {gm}"
    );

    // Compute f, g, fdot, gdot Lagrange coefficients
    let psi = alpha * s * s;
    let (c2, c3) = stumpff_c2c3(psi);

    let s2 = s * s;
    let s3 = s2 * s;

    let r_mag =
        r0_mag + r0_dot_v0 / sqrt_gm * s * (1.0 - psi * c3) + (1.0 - r0_mag * alpha) * s2 * c2;

    let f = 1.0 - s2 * c2 / r0_mag;
    let g = dt - s3 * c3 / sqrt_gm;

    let pos = f * r0 + g * v0;

    let f_dot = sqrt_gm / (r_mag * r0_mag) * s * (alpha * s2 * c3 - 1.0);
    let g_dot = 1.0 - s2 * c2 / r_mag;

    let vel = f_dot * r0 + g_dot * v0;

    StateVector::new(pos, vel)
}

/// Stumpff functions c2(psi) and c3(psi).
/// c2(psi) = (1 - cos(sqrt(psi))) / psi     for psi > 0
/// c3(psi) = (sqrt(psi) - sin(sqrt(psi))) / psi^(3/2)  for psi > 0
///
/// Uses 4-fold argument reduction: psi is divided by 4 until |psi| <= 0.1,
/// the series is evaluated there, and the quadruple-angle recurrences
///   c2(4x) = c1(x)^2 / 2,  c3(4x) = (c2(x) + c0(x) c3(x)) / 4
/// (with c0 = 1 - x c2, c1 = 1 - x c3) recover the original argument.
/// This avoids the catastrophic cosh overflow / cancellation the direct
/// formulas suffer for large hyperbolic |psi|.
fn stumpff_c2c3(psi: f64) -> (f64, f64) {
    let mut x = psi;
    let mut depth = 0u32;
    while x.abs() > 0.1 {
        x *= 0.25;
        depth += 1;
    }

    // Maclaurin series, |x| <= 0.1: truncation error < 1e-16.
    let mut c2 = (1.0 / 2.0) + x * (-1.0 / 24.0 + x * (1.0 / 720.0 + x * (-1.0 / 40_320.0)));
    let mut c3 = (1.0 / 6.0) + x * (-1.0 / 120.0 + x * (1.0 / 5_040.0 + x * (-1.0 / 362_880.0)));

    let mut c0 = 1.0 - x * c2;
    let mut c1 = 1.0 - x * c3;

    for _ in 0..depth {
        c3 = (c2 + c0 * c3) * 0.25;
        c2 = c1 * c1 * 0.5;
        c1 *= c0;
        c0 = 2.0 * c0 * c0 - 1.0;
    }

    (c2, c3)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::J2000;
    use crate::keplerlib::KeplerOrbit;
    use crate::nbodylib::types::{elements_to_cartesian, OrbitalElements};
    use crate::nbodylib::GM_SUN;
    use crate::time::Timescale;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    #[test]
    fn test_circular_orbit() {
        // Circular orbit at 1 AU: v = sqrt(GM/r)
        let v_circ = (GM_SUN / 1.0_f64).sqrt();
        let state = StateVector::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, v_circ, 0.0));

        // One full period
        let period = 2.0 * std::f64::consts::PI / (GM_SUN).sqrt();
        let final_state = kepler_drift(&state, period, GM_SUN);

        assert_relative_eq!(final_state.pos.x, 1.0, epsilon = 1e-10);
        assert_relative_eq!(final_state.pos.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(final_state.pos.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_energy_conservation() {
        // Eccentric orbit
        let state = StateVector::new(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.02, 0.002), // AU/day
        );

        let energy =
            |s: &StateVector| -> f64 { 0.5 * s.vel.norm_squared() - GM_SUN / s.pos.norm() };

        let e0 = energy(&state);
        let mut s = state;
        for _ in 0..10_000 {
            s = kepler_drift(&s, 10.0, GM_SUN);
        }
        let e1 = energy(&s);

        assert_relative_eq!(e0, e1, epsilon = 1e-12);
    }

    #[test]
    fn test_half_orbit() {
        // Start at perihelion of e=0.5 orbit at 1 AU perihelion
        // a = q/(1-e) = 2 AU
        let a: f64 = 2.0;
        let e = 0.5;
        let v_peri = (GM_SUN * (1.0 + e) / (a * (1.0 - e))).sqrt();
        let state = StateVector::new(Vector3::new(1.0, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));

        let period = 2.0 * std::f64::consts::PI * (a.powi(3) / GM_SUN).sqrt();
        let half = kepler_drift(&state, period / 2.0, GM_SUN);

        // At aphelion: x should be -Q = -a(1+e) = -3
        assert_relative_eq!(half.pos.x, -3.0, epsilon = 1e-8);
        assert_relative_eq!(half.pos.y, 0.0, epsilon = 1e-8);
    }

    #[test]
    fn test_hyperbolic_round_trip() {
        // Drift out and back along hyperbolic orbits: must return to the
        // starting state to relative rounding.
        for &e in &[1.0 + 1e-6, 1.5, 3.0] {
            for &q in &[5.0, 800.0] {
                let v_peri = (GM_SUN * (1.0 + e) / q).sqrt();
                let s0 =
                    StateVector::new(Vector3::new(q, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));
                for &dt in &[1.0e3, 2.0e5] {
                    let out = kepler_drift(&s0, dt, GM_SUN);
                    let back = kepler_drift(&out, -dt, GM_SUN);
                    // Rounding error scales with the largest distance the
                    // trajectory reaches, not the perihelion distance, and
                    // the return leg loses a couple of digits to
                    // cancellation in the f,g reconstruction near
                    // perihelion (~2.5e-9 relative for e = 1.5, q = 5 AU).
                    let scale = out.pos.norm().max(q);
                    assert!(
                        (back.pos - s0.pos).norm() <= 1e-8 * scale,
                        "round-trip position drift: e = {e}, q = {q}, dt = {dt}"
                    );
                    // The perihelion velocity is the most sensitive quantity
                    // (the v-field gradient is steepest there), so its error
                    // also grows with the length of the round trip.
                    assert!(
                        (back.vel - s0.vel).norm() <= 1e-8 * v_peri * (scale / q),
                        "round-trip velocity drift: e = {e}, q = {q}, dt = {dt}"
                    );
                }
            }
        }
    }

    // ===================================================================
    // Oracle tests: kepler_drift vs keplerlib (SPICE prop2b port)
    // ===================================================================

    /// Propagate a state with keplerlib's prop2b port (universal variables +
    /// Stumpff functions, SPICE parity) and return (pos AU, vel AU/day).
    fn keplerlib_propagate(state: &StateVector, dt: f64) -> (Vector3<f64>, Vector3<f64>) {
        let ts = Timescale::default();
        let epoch = ts.tt_jd(J2000, None);
        let target = ts.tt_jd(J2000 + dt, None);
        let orbit = KeplerOrbit::new(state.pos, state.vel, &epoch, GM_SUN, None, None);
        let p = orbit.at(&target);
        (p.position, p.velocity)
    }

    /// Compare kepler_drift to keplerlib over one (state, dt) case.
    /// `tol` is a relative tolerance scaled by the orbit size (AU) and by
    /// the number of orbits covered: the two independent universal-variable
    /// formulations accumulate along-track phase differences of a few ulps
    /// per revolution, so multi-orbit dt amplifies an irreducible rounding
    /// disagreement linearly.
    fn check_drift_against_keplerlib(elem: &OrbitalElements, dt: f64, tol: f64) {
        let state = elements_to_cartesian(elem, GM_SUN);
        let ours = kepler_drift(&state, dt, GM_SUN);
        let (sf_pos, sf_vel) = keplerlib_propagate(&state, dt);

        let period = TWO_PI * (elem.a.abs().powi(3) / GM_SUN).sqrt();
        let tol = tol * (1.0 + (dt / period).abs());

        let scale_r = state.pos.norm().max(sf_pos.norm());
        let scale_v = state.vel.norm().max(sf_vel.norm());

        let dpos = (ours.pos - sf_pos).norm();
        let dvel = (ours.vel - sf_vel).norm();
        assert!(
            dpos <= tol * scale_r,
            "position mismatch vs keplerlib: |dr| = {dpos:.3e} AU (scale {scale_r:.3e}), \
             a = {}, e = {}, M = {}, dt = {dt}",
            elem.a,
            elem.e,
            elem.mean_anomaly
        );
        assert!(
            dvel <= tol * scale_v,
            "velocity mismatch vs keplerlib: |dv| = {dvel:.3e} AU/d (scale {scale_v:.3e}), \
             a = {}, e = {}, M = {}, dt = {dt}",
            elem.a,
            elem.e,
            elem.mean_anomaly
        );
    }

    /// The documented failing corner of an earlier universal-variable
    /// iteration: a = 800 AU, e = 0.95, M ≈ 5.97 (just before perihelion),
    /// dt = 4e5 days. In debug builds the unhardened solver hit a
    /// convergence-plateau debug_assert; the hardened solver must converge
    /// AND agree with keplerlib.
    #[test]
    fn test_kepler_drift_known_failing_corner() {
        let elem = OrbitalElements {
            a: 800.0,
            e: 0.95,
            i: 0.3,
            omega_big: 1.0,
            omega: 2.0,
            mean_anomaly: 5.97,
        };
        check_drift_against_keplerlib(&elem, 4.0e5, 1e-8);
    }

    /// Property grid: (a, e, M, dt) sweep spanning the integrators'
    /// operating envelope (Kuiper belt through inner Oort cloud, fractions
    /// of an orbit through many orbits, both time directions). Tolerance
    /// 1e-8 relative: both solvers are independent universal-variable
    /// formulations, and near-perihelion high-e states lose a few digits to
    /// cancellation in either implementation.
    #[test]
    fn test_kepler_drift_grid_vs_keplerlib() {
        let a_grid = [5.0, 40.0, 250.0, 800.0, 3000.0];
        let e_grid = [0.0, 0.3, 0.7, 0.95, 0.99];
        let m_grid = [0.0, 1.0, 3.1, 5.97];
        // dt as a fraction of the orbital period, plus fixed offsets that
        // put long-period orbits in the "fraction of an orbit across
        // perihelion" regime where the plateau was observed.
        let dt_fracs = [0.05, 0.5, 1.0, 3.7, -0.5];

        for &a in &a_grid {
            let period = TWO_PI * (a * a * a / GM_SUN).sqrt();
            for &e in &e_grid {
                for &m in &m_grid {
                    let elem = OrbitalElements {
                        a,
                        e,
                        i: 0.4,
                        omega_big: 0.7,
                        omega: 2.1,
                        mean_anomaly: m,
                    };
                    for &frac in &dt_fracs {
                        check_drift_against_keplerlib(&elem, frac * period, 1e-8);
                    }
                    // Fixed dt = 4e5 d (a typical Oort-cloud step size).
                    check_drift_against_keplerlib(&elem, 4.0e5, 1e-8);
                }
            }
        }
    }

    /// Hyperbolic states: kepler_drift vs keplerlib for e > 1.
    #[test]
    fn test_kepler_drift_hyperbolic_vs_keplerlib() {
        for &e in &[1.0 + 1e-6, 1.1, 1.5, 3.0] {
            for &q in &[5.0, 100.0, 800.0] {
                // Build a perihelion state directly: r = q x̂, v = v_peri ŷ.
                let v_peri = (GM_SUN * (1.0 + e) / q).sqrt();
                let state =
                    StateVector::new(Vector3::new(q, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));
                for &dt in &[-2.0e5, -1.0e3, 1.0e3, 2.0e5] {
                    let ours = kepler_drift(&state, dt, GM_SUN);
                    let (sf_pos, sf_vel) = keplerlib_propagate(&state, dt);
                    let scale_r = sf_pos.norm();
                    assert!(
                        (ours.pos - sf_pos).norm() <= 1e-9 * scale_r,
                        "hyperbolic position mismatch: e = {e}, q = {q}, dt = {dt}, \
                         |dr| = {:.3e} (scale {scale_r:.3e})",
                        (ours.pos - sf_pos).norm()
                    );
                    assert!(
                        (ours.vel - sf_vel).norm() <= 1e-9 * sf_vel.norm(),
                        "hyperbolic velocity mismatch: e = {e}, q = {q}, dt = {dt}"
                    );
                }
            }
        }
    }
}
