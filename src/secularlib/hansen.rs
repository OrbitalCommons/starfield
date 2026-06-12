//! Hansen coefficients X_j^{n,m}(e) for the disturbing function.
//!
//! The numerical evaluator integrates in *eccentric* anomaly with the
//! Jacobian dM = (1 − e cos E) dE — the integrand is smooth and periodic in E,
//! so the uniform midpoint rule converges spectrally — and doubles the node
//! count until the requested tolerance is met (the required resolution grows
//! like j (1 − e)^{−3/2}).
//!
//! Units: AU, radians (see the [`crate::secularlib`] docs).
//!
//! Reference: Hughes (1981); Mardling (2013) Appendix B

use crate::constants::TAU;
use crate::secularlib::A_NEPTUNE_AU;
use std::f64::consts::PI;

/// Largest node count attempted before giving up on the tolerance.
const MAX_POINTS: usize = 1 << 18;

/// Numerical Hansen coefficient
///
///   X_j^{n,m}(e) = (1/2π) ∮ (r/a)^n cos(m f − j M) dM
///
/// evaluated with automatic convergence control: the quadrature node count
/// doubles from 64 until successive estimates agree to `tol` (relative, with
/// an absolute floor of 1e-13 so that coefficients that are exactly zero
/// converge), panicking in debug builds if 2^18 nodes do not suffice.
///
/// Reference: Hughes (1981), Celestial Mechanics 25, 101
///
/// # Example
///
/// ```
/// use starfield::secularlib::hansen_coefficient;
///
/// // X_0^{-2,0}(e) = (1 - e²)^{-1/2} in closed form.
/// let num = hansen_coefficient(0, -2, 0, 0.4, 1e-10);
/// let exact = (1.0_f64 - 0.4 * 0.4).powf(-0.5);
/// assert!((num - exact).abs() < 1e-8);
/// ```
pub fn hansen_coefficient(j: i64, n: i32, m: i32, e: f64, tol: f64) -> f64 {
    assert!((0.0..1.0).contains(&e), "eccentricity out of range: {e}");

    let mut n_points = 64usize;
    let mut prev = hansen_fixed(j, n, m, e, n_points);
    while n_points < MAX_POINTS {
        n_points *= 2;
        let next = hansen_fixed(j, n, m, e, n_points);
        if (next - prev).abs() <= tol * next.abs() + 1e-13 {
            return next;
        }
        prev = next;
    }
    debug_assert!(
        false,
        "Hansen quadrature did not converge: j={j}, n={n}, m={m}, e={e}"
    );
    prev
}

/// Fixed-node evaluation on a uniform eccentric-anomaly midpoint grid.
fn hansen_fixed(j: i64, n: i32, m: i32, e: f64, n_points: usize) -> f64 {
    let de = TAU / n_points as f64;
    let mut sum = 0.0;
    for k in 0..n_points {
        let ea = (k as f64 + 0.5) * de;
        let r_over_a = 1.0 - e * ea.cos();
        let mean_anom = ea - e * ea.sin();
        // True anomaly from eccentric anomaly
        let true_anom = 2.0 * (((1.0 + e) / (1.0 - e)).sqrt() * (ea / 2.0).tan()).atan();
        let integrand =
            r_over_a.powi(n) * (m as f64 * true_anom - j as f64 * mean_anom).cos() * r_over_a;
        sum += integrand;
    }
    sum * de / TAU
}

/// Closed form X_0^{-3,0}(e) = (1 − e²)^{−3/2} (the secular orbit average of
/// (a/r)³).
///
/// # Example
///
/// ```
/// use starfield::secularlib::{hansen_coefficient, hansen_x0_neg3_0};
///
/// let e = 0.6;
/// assert!((hansen_coefficient(0, -3, 0, e, 1e-10) - hansen_x0_neg3_0(e)).abs() < 1e-8);
/// ```
pub fn hansen_x0_neg3_0(e: f64) -> f64 {
    (1.0 - e * e).powf(-1.5)
}

/// Asymptotic X_j^{-3,2} for a particle exterior to a planet at
/// `a_planet_au` (AU) in the j:2 mean-motion resonance chain:
///
///   X_j^{-3,2} ≈ (2j/5) · exp(−(q/a_planet)²)
///
/// where q = a(1 − e) is the particle's perihelion distance in AU. The length
/// scale in the exponent is the *planet's* semi-major axis, not the
/// particle's (using q/a, i.e. 1 − e, is dimensionally inconsistent and off
/// ~4x at e = 0.9 with the opposite e-trend).
///
/// Calibration regime: the fit was made for the Neptune 2:j chain, i.e.
/// j ≳ 10 and q/a_planet ∈ (1.0, 1.7) (30–50 AU at Neptune); for other
/// perturbers it assumes the same *scaled* regime. See
/// [`hansen_x_neg3_2_neptune_chain`] for the Neptune-specific form.
///
/// Reference: Mardling (2013) Appendix B; Batygin & Brown (2021)
pub fn hansen_x_neg3_2_chain(j: i64, q_au: f64, a_planet_au: f64) -> f64 {
    let jf = j.abs() as f64;
    (2.0 * jf / 5.0) * (-(q_au / a_planet_au).powi(2)).exp()
}

/// Asymptotic X_j^{-3,2} for a particle in the j:2 mean-motion resonance with
/// Neptune (a_N = 30.07 AU). Valid for j ≳ 10 and q ∈ (30, 50) AU.
///
/// Convenience wrapper over [`hansen_x_neg3_2_chain`].
///
/// # Example
///
/// ```
/// use starfield::secularlib::hansen_x_neg3_2_neptune_chain;
///
/// // Decreasing in perihelion distance q.
/// assert!(hansen_x_neg3_2_neptune_chain(20, 30.0) > hansen_x_neg3_2_neptune_chain(20, 40.0));
/// ```
pub fn hansen_x_neg3_2_neptune_chain(j: i64, q_au: f64) -> f64 {
    hansen_x_neg3_2_chain(j, q_au, A_NEPTUNE_AU)
}

/// Mean anomaly → true anomaly for an elliptic orbit (e < 1), via a
/// safeguarded Newton/bisection Kepler solve.
///
/// # Example
///
/// ```
/// use starfield::secularlib::mean_to_true_anomaly;
///
/// // Circular orbit: f = M. Perihelion: f = 0 for any e.
/// assert!((mean_to_true_anomaly(1.0, 0.0) - 1.0).abs() < 1e-12);
/// assert!(mean_to_true_anomaly(0.0, 0.7).abs() < 1e-12);
/// ```
pub fn mean_to_true_anomaly(mean_anomaly: f64, e: f64) -> f64 {
    let ea = solve_kepler_elliptic(e, mean_anomaly);
    2.0 * (((1.0 + e) / (1.0 - e)).sqrt() * (ea / 2.0).tan()).atan()
}

/// Solve Kepler's equation M = E − e sin E for the eccentric anomaly E
/// (elliptic case, 0 ≤ e < 1) with a bisection-safeguarded Newton iteration.
///
/// (`keplerlib`'s internal solver targets the e > 0 elliptic/hyperbolic cases
/// and divides by e in its starting guess; this one is well-defined down to
/// e = 0.)
fn solve_kepler_elliptic(e: f64, mean_anomaly: f64) -> f64 {
    debug_assert!((0.0..1.0).contains(&e), "eccentricity out of range: {e}");

    // Reduce M to (−π, π] and exploit the odd symmetry E(−M) = −E(M).
    let mut m = mean_anomaly % TAU;
    if m > PI {
        m -= TAU;
    } else if m < -PI {
        m += TAU;
    }
    let sign = if m < 0.0 { -1.0 } else { 1.0 };
    let m = m.abs();

    // For M in [0, π] the root lies in [M, M + e] (since 0 ≤ e sin E ≤ e
    // there: f(M) ≤ 0 and f(M + e) ≥ 0).
    let mut lo = m;
    let mut hi = m + e;
    let mut ea = (if e > 0.8 { PI } else { m + 0.85 * e }).clamp(lo, hi);

    for _ in 0..100 {
        let f = ea - e * ea.sin() - m;
        if f > 0.0 {
            hi = ea;
        } else {
            lo = ea;
        }
        let fp = 1.0 - e * ea.cos();
        let step = f / fp;
        // A tiny Newton step means we are converged; take it even when it
        // grazes the bracket boundary (the bisection fallback below would
        // stop short at the bracket midpoint).
        if step.abs() < 1e-14 * ea.abs().max(1.0) {
            ea -= step;
            break;
        }
        let next = ea - step;
        // Fall back to bisection whenever Newton leaves the bracket.
        ea = if next > lo && next < hi {
            next
        } else {
            0.5 * (lo + hi)
        };
    }
    sign * ea
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_hansen_matches_closed_form_x0() {
        for &e in &[0.0, 0.3, 0.7, 0.9] {
            let num = hansen_coefficient(0, -3, 0, e, 1e-10);
            let exact = hansen_x0_neg3_0(e);
            assert_relative_eq!(num, exact, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_hansen_x0_neg2_0_closed_form() {
        // X_0^{-2,0}(e) = (1 - e²)^{-1/2}
        for &e in &[0.0, 0.4, 0.8] {
            let num = hansen_coefficient(0, -2, 0, e, 1e-10);
            let exact = (1.0 - e * e).powf(-0.5);
            assert_relative_eq!(num, exact, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_hansen_x0_1_0_closed_form() {
        // X_0^{1,0}(e) = 1 + e²/2 (orbit average of r/a)
        let num = hansen_coefficient(0, 1, 0, 0.6, 1e-10);
        assert_relative_eq!(num, 1.0 + 0.18, max_relative = 1e-8);
    }

    #[test]
    fn test_hansen_circular_kronecker() {
        // At e = 0, X_j^{n,m} = δ_{jm}.
        assert_relative_eq!(
            hansen_coefficient(2, -3, 2, 0.0, 1e-10),
            1.0,
            epsilon = 1e-10
        );
        assert!(hansen_coefficient(3, -3, 2, 0.0, 1e-10).abs() < 1e-10);
    }

    #[test]
    fn test_hansen_high_e_converges() {
        // High-e, large-j case that aliases on a fixed 256-point grid.
        let v = hansen_coefficient(40, -3, 2, 0.95, 1e-8);
        assert!(v.is_finite());
    }

    #[test]
    fn test_neptune_chain_asymptotic_trends() {
        // Decreasing in q, increasing in j, positive.
        let x = hansen_x_neg3_2_neptune_chain(20, 35.0);
        assert!(x > 0.0);
        assert!(hansen_x_neg3_2_neptune_chain(20, 30.0) > hansen_x_neg3_2_neptune_chain(20, 40.0));
        assert!(hansen_x_neg3_2_neptune_chain(30, 35.0) > x);
    }

    #[test]
    fn test_neptune_chain_matches_generic_form() {
        // The Neptune wrapper is exactly the generic form at a_N = 30.07 AU.
        for &(j, q) in &[(10, 30.0), (20, 35.0), (40, 50.0)] {
            assert_eq!(
                hansen_x_neg3_2_neptune_chain(j, q),
                hansen_x_neg3_2_chain(j, q, A_NEPTUNE_AU)
            );
        }
    }

    #[test]
    fn test_mean_to_true_anomaly() {
        // Circular: f = M. Perihelion: f = 0 for any e.
        assert_relative_eq!(mean_to_true_anomaly(1.0, 0.0), 1.0, epsilon = 1e-10);
        assert!(mean_to_true_anomaly(0.0, 0.7).abs() < 1e-10);
        // High-e quarter orbit stays finite and ahead of M.
        let f = mean_to_true_anomaly(0.5, 0.9);
        assert!(f > 0.5 && f < PI);
    }

    #[test]
    fn test_kepler_solver_residual() {
        // E − e sin E must reproduce M across e and M.
        for &e in &[0.0, 0.1, 0.5, 0.9, 0.99] {
            for k in 0..20 {
                let m = -PI + (k as f64 + 0.5) * (TAU / 20.0);
                let ea = solve_kepler_elliptic(e, m);
                let residual = (ea - e * ea.sin() - m).abs();
                assert!(residual < 1e-12, "e={e}, M={m}: residual {residual:.3e}");
            }
        }
    }
}
