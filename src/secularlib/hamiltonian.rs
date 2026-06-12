//! Doubly-averaged secular Hamiltonians.
//!
//! Two levels of fidelity:
//!
//! 1. **Analytic quadrupole** ([`quadrupole_hamiltonian`]): the doubly-averaged
//!    quadrupole for an inner test particle and an outer eccentric perturber.
//!    This is *axisymmetric in the apsidal angle*: it depends on (e, i, ω)
//!    only — at quadrupole order there is no Δϖ dependence and therefore no
//!    aligned/anti-aligned structure. Valid only for α = a/a_p ≪ 1.
//!
//! 2. **Numerical Gauss-ring double averaging**
//!    ([`numerical_secular_hamiltonian`]): the exact 1/Δ averaged over both
//!    mean anomalies. The multipole expansion diverges for α ≈ 0.3–0.9
//!    (the regime relevant to distant-perturber problems such as Planet Nine);
//!    Batygin & Brown (2016) used numerical ring averaging for exactly this
//!    reason. This captures the octupole-and-higher Δϖ dependence (the
//!    anti-aligned libration islands) automatically. For orbit-crossing
//!    geometries the doubly averaged integral is singular and a softening
//!    length must be supplied.
//!
//! Units: AU, radians, GM in AU³/day² (see the [`crate::secularlib`] docs).
//!
//! Reference: Kozai (1962); Naoz (2016) §3; Batygin & Brown (2016)

use std::f64::consts::PI;

/// Quadrupole coupling constant C = gm_p α² / (4 a_p (1 − e_p²)^{3/2}).
fn coupling(a: f64, a_p: f64, e_p: f64, gm_p: f64) -> f64 {
    let alpha = a / a_p;
    gm_p * alpha * alpha / (4.0 * a_p * (1.0 - e_p * e_p).powf(1.5))
}

/// Doubly-averaged quadrupole Hamiltonian for an inner test particle
/// perturbed by an outer body (Kozai-Lidov form):
///
///   H_quad = −(C/4) · \[ (2 + 3e²)(3cos²i − 1) + 15 e² sin²i cos 2ω \]
///
/// with C = gm_p α²/(4 a_p (1−e_p²)^{3/2}). Angles are measured in the
/// perturber's orbital plane: `i` is the mutual inclination and `omega` the
/// particle's argument of perihelion from the common node.
///
/// Note the Hamiltonian is independent of the nodal and apsidal longitudes —
/// at quadrupole order the averaged perturber is an axisymmetric ring, so no
/// Δϖ-dependent (aligned/anti-aligned) dynamics exists at this order. Use
/// [`numerical_secular_hamiltonian`] for apsidal structure.
///
/// Reference: Kozai (1962); Naoz (2016) §3, eq. (20)
///
/// * `a`: test particle semi-major axis (AU)
/// * `e`: test particle eccentricity
/// * `i`: mutual inclination (radians)
/// * `omega`: argument of perihelion from the common node (radians)
/// * `a_p`: perturber semi-major axis (AU)
/// * `e_p`: perturber eccentricity
/// * `gm_p`: perturber GM (AU³/day²)
///
/// # Example
///
/// ```
/// use starfield::secularlib::{quadrupole_hamiltonian, GM_NEPTUNE_AU3_DAY2};
///
/// // A test particle well inside a Neptune-mass ring perturber.
/// let h = quadrupole_hamiltonian(5.0, 0.3, 0.5, 1.0, 30.07, 0.01, GM_NEPTUNE_AU3_DAY2);
/// assert!(h.is_finite());
/// ```
pub fn quadrupole_hamiltonian(
    a: f64,
    e: f64,
    i: f64,
    omega: f64,
    a_p: f64,
    e_p: f64,
    gm_p: f64,
) -> f64 {
    let c = coupling(a, a_p, e_p, gm_p);
    let e2 = e * e;
    let cos_i2 = i.cos().powi(2);
    let sin_i2 = 1.0 - cos_i2;

    -(c / 4.0)
        * ((2.0 + 3.0 * e2) * (3.0 * cos_i2 - 1.0) + 15.0 * e2 * sin_i2 * (2.0 * omega).cos())
}

/// Coplanar quadrupole Hamiltonian (i = 0):
///   H = −C (2 + 3e²) / 2.
///
/// A function of e alone — the coplanar quadrupole has *no* libration islands;
/// those appear only at octupole order (∝ e e_p α³) and beyond, which the
/// numerical averaging includes.
///
/// # Example
///
/// ```
/// use starfield::secularlib::{coplanar_quadrupole, GM_NEPTUNE_AU3_DAY2};
///
/// // The coplanar quadrupole energy decreases (more negative) with e.
/// let h1 = coplanar_quadrupole(100.0, 0.1, 700.0, 0.6, GM_NEPTUNE_AU3_DAY2);
/// let h2 = coplanar_quadrupole(100.0, 0.7, 700.0, 0.6, GM_NEPTUNE_AU3_DAY2);
/// assert!(h2 < h1);
/// ```
pub fn coplanar_quadrupole(a: f64, e: f64, a_p: f64, e_p: f64, gm_p: f64) -> f64 {
    quadrupole_hamiltonian(a, e, 0.0, 0.0, a_p, e_p, gm_p)
}

/// Position on an orbit at eccentric anomaly `ea`, rotated into the reference
/// frame by (omega, i, omega_big). Returns the 3D position (units of `a`).
fn orbit_position(a: f64, e: f64, i: f64, omega: f64, omega_big: f64, ea: f64) -> [f64; 3] {
    let x_orb = a * (ea.cos() - e);
    let y_orb = a * (1.0 - e * e).sqrt() * ea.sin();

    let (sw, cw) = omega.sin_cos();
    let (si, ci) = i.sin_cos();
    let (so, co) = omega_big.sin_cos();

    let x = (co * cw - so * sw * ci) * x_orb + (-co * sw - so * cw * ci) * y_orb;
    let y = (so * cw + co * sw * ci) * x_orb + (-so * sw + co * cw * ci) * y_orb;
    let z = sw * si * x_orb + cw * si * y_orb;

    [x, y, z]
}

/// Doubly-averaged secular Hamiltonian from numerical Gauss-ring averaging of
/// the exact interaction:
///
///   H = −gm_p ⟨⟨ 1/Δ ⟩⟩
///     = −gm_p/(4π²) ∮∮ dM dM_p / sqrt(Δ² + b²)
///
/// (The indirect term −gm_p r·r_p/r_p³ double-averages to zero because the
/// orbit average of the Keplerian acceleration vanishes.)
///
/// The integral is evaluated on uniform eccentric-anomaly grids with the
/// Jacobians dM = (1 − e cos E) dE, which is spectrally accurate for smooth
/// (non-crossing) geometries. For crossing orbits the unsoftened average
/// diverges; pass a softening length `softening_au` > 0 (a fraction of a_p)
/// to regularize — the resulting portraits follow Batygin & Brown (2016).
///
/// The perturber's apsidal line defines the x-axis and its orbit plane the
/// reference plane, so the particle's `omega`/`omega_big` are relative to the
/// perturber (coplanar: Δϖ = omega + omega_big).
///
/// `n_quad` is the number of quadrature nodes per anomaly (n_quad² total
/// evaluations); 64–128 suffices for portrait work.
///
/// Reference: Batygin & Brown (2016), AJ 151, 22
///
/// # Example
///
/// ```
/// use starfield::secularlib::{numerical_secular_hamiltonian, GM_NEPTUNE_AU3_DAY2};
///
/// // Non-crossing coplanar geometry: the averaged energy is finite and < 0.
/// let h = numerical_secular_hamiltonian(
///     250.0, 0.3, 0.0, 0.0, 0.0, 700.0, 0.6, GM_NEPTUNE_AU3_DAY2, 64, 0.0,
/// );
/// assert!(h.is_finite() && h < 0.0);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn numerical_secular_hamiltonian(
    a: f64,
    e: f64,
    i: f64,
    omega: f64,
    omega_big: f64,
    a_p: f64,
    e_p: f64,
    gm_p: f64,
    n_quad: usize,
    softening_au: f64,
) -> f64 {
    let b2 = softening_au * softening_au;
    let de = 2.0 * PI / n_quad as f64;

    // Precompute perturber ring samples (perturber defines the frame:
    // i_p = 0, omega_p = 0, Omega_p = 0).
    let mut p_pos = Vec::with_capacity(n_quad);
    let mut p_weight = Vec::with_capacity(n_quad);
    for kp in 0..n_quad {
        let ea_p = (kp as f64 + 0.5) * de;
        p_pos.push(orbit_position(a_p, e_p, 0.0, 0.0, 0.0, ea_p));
        p_weight.push(1.0 - e_p * ea_p.cos());
    }

    let mut sum = 0.0;
    for k in 0..n_quad {
        let ea = (k as f64 + 0.5) * de;
        let w = 1.0 - e * ea.cos();
        let r = orbit_position(a, e, i, omega, omega_big, ea);

        for (rp, wp) in p_pos.iter().zip(&p_weight) {
            let dx = r[0] - rp[0];
            let dy = r[1] - rp[1];
            let dz = r[2] - rp[2];
            let delta = (dx * dx + dy * dy + dz * dz + b2).sqrt();
            sum += w * wp / delta;
        }
    }

    // (1/2π)² ∮∮ ... dE dE_p with the dM Jacobians folded into the weights.
    -gm_p * sum * (de * de) / (4.0 * PI * PI)
}

/// Generate a phase-space portrait grid: evaluate the numerically averaged
/// Hamiltonian at a grid of (e, Δϖ) values for a fixed semi-major axis,
/// coplanar with the perturber.
///
/// Uses Gauss-ring averaging of the exact 1/Δ (the multipole expansion does
/// not converge for the relevant α), with a softening of 0.01·a_p to
/// regularize crossing geometries.
///
/// Returns (e_vals, dvarpi_vals, portrait) where portrait\[i\]\[j\] = H(e_i, Δϖ_j).
///
/// # Example
///
/// ```
/// use starfield::secularlib::{phase_portrait, GM_NEPTUNE_AU3_DAY2};
///
/// let (e_vals, dv_vals, portrait) =
///     phase_portrait(400.0, 700.0, 0.6, GM_NEPTUNE_AU3_DAY2, 4, 6);
/// assert_eq!((e_vals.len(), dv_vals.len()), (4, 6));
/// assert_eq!((portrait.len(), portrait[0].len()), (4, 6));
/// ```
pub fn phase_portrait(
    a: f64,
    a_p: f64,
    e_p: f64,
    gm_p: f64,
    n_e: usize,
    n_dvarpi: usize,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let softening = 0.01 * a_p;
    let n_quad = 64;

    let e_vals: Vec<f64> = (0..n_e)
        .map(|i| (i as f64 + 0.5) / n_e as f64 * 0.95)
        .collect();

    let dvarpi_vals: Vec<f64> = (0..n_dvarpi)
        .map(|j| (j as f64) / n_dvarpi as f64 * 2.0 * PI - PI)
        .collect();

    let mut portrait = vec![vec![0.0; n_dvarpi]; n_e];

    for (i, &e) in e_vals.iter().enumerate() {
        for (j, &dv) in dvarpi_vals.iter().enumerate() {
            portrait[i][j] = numerical_secular_hamiltonian(
                a, e, 0.0, dv, 0.0, a_p, e_p, gm_p, n_quad, softening,
            );
        }
    }

    (e_vals, dvarpi_vals, portrait)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::secularlib::GM_SUN_AU3_DAY2;

    const GM_P: f64 = 10.0 * 3.003489e-6 * GM_SUN_AU3_DAY2; // 10 Earth masses

    #[test]
    fn test_quadrupole_has_no_apsidal_dependence() {
        // The coplanar quadrupole must be a function of e only.
        let h1 = coplanar_quadrupole(100.0, 0.5, 700.0, 0.6, GM_P);
        let h2 = coplanar_quadrupole(100.0, 0.5, 700.0, 0.6, GM_P);
        assert_eq!(h1, h2);
        // And independent of omega at i = 0:
        let ha = quadrupole_hamiltonian(100.0, 0.5, 0.0, 0.3, 700.0, 0.6, GM_P);
        let hb = quadrupole_hamiltonian(100.0, 0.5, 0.0, 2.1, 700.0, 0.6, GM_P);
        assert!((ha - hb).abs() < 1e-30);
    }

    #[test]
    fn test_numerical_matches_quadrupole_at_small_alpha() {
        // Small α, circular perturber (octupole vanishes): the e-dependence
        // of the numerical average must match the quadrupole formula. The
        // constant monopole term −gm_p/a_p·(...) cancels in the difference.
        let a = 30.0;
        let a_p = 700.0;
        let e_p = 0.0;
        let n = 128;

        let num =
            |e: f64| numerical_secular_hamiltonian(a, e, 0.0, 0.0, 0.0, a_p, e_p, GM_P, n, 0.0);
        let quad = |e: f64| coplanar_quadrupole(a, e, a_p, e_p, GM_P);

        let d_num = num(0.6) - num(0.1);
        let d_quad = quad(0.6) - quad(0.1);
        let rel = ((d_num - d_quad) / d_quad).abs();
        // Hexadecapole correction is O(α²) ≈ 0.2%
        assert!(rel < 0.02, "numerical vs quadrupole ΔH mismatch: {rel:.3e}");
    }

    #[test]
    fn test_numerical_axisymmetric_for_circular_perturber() {
        // e_p = 0: the averaged perturber is an axisymmetric ring, so H must
        // not depend on Δϖ even at large α.
        let h0 =
            numerical_secular_hamiltonian(300.0, 0.4, 0.0, 0.0, 0.0, 700.0, 0.0, GM_P, 96, 0.0);
        let h1 = numerical_secular_hamiltonian(
            300.0,
            0.4,
            0.0,
            PI / 2.0,
            0.0,
            700.0,
            0.0,
            GM_P,
            96,
            0.0,
        );
        let rel = ((h0 - h1) / h0).abs();
        assert!(
            rel < 1e-10,
            "circular perturber should be axisymmetric: {rel:.3e}"
        );
    }

    #[test]
    fn test_numerical_apsidal_structure_with_eccentric_perturber() {
        // Eccentric perturber at moderate α: the exact average distinguishes
        // aligned from anti-aligned apsides (octupole+), which the quadrupole
        // cannot.
        let h_aligned =
            numerical_secular_hamiltonian(250.0, 0.3, 0.0, 0.0, 0.0, 700.0, 0.6, GM_P, 128, 0.0);
        let h_anti =
            numerical_secular_hamiltonian(250.0, 0.3, 0.0, PI, 0.0, 700.0, 0.6, GM_P, 128, 0.0);
        let rel = ((h_aligned - h_anti) / h_aligned).abs();
        assert!(rel > 1e-6, "expected Δϖ asymmetry, got {rel:.3e}");
    }

    #[test]
    fn test_numerical_quadrature_converged() {
        // n vs 2n agreement on a non-crossing geometry.
        let h64 =
            numerical_secular_hamiltonian(200.0, 0.5, 0.2, 1.0, 0.4, 700.0, 0.5, GM_P, 64, 0.0);
        let h128 =
            numerical_secular_hamiltonian(200.0, 0.5, 0.2, 1.0, 0.4, 700.0, 0.5, GM_P, 128, 0.0);
        let rel = ((h64 - h128) / h128).abs();
        assert!(rel < 1e-8, "quadrature not converged: {rel:.3e}");
    }

    #[test]
    fn test_phase_portrait_shape_and_finite() {
        let (e_vals, dv_vals, portrait) = phase_portrait(400.0, 700.0, 0.6, GM_P, 8, 12);
        assert_eq!(e_vals.len(), 8);
        assert_eq!(dv_vals.len(), 12);
        assert_eq!(portrait.len(), 8);
        assert_eq!(portrait[0].len(), 12);
        for row in &portrait {
            for &h in row {
                assert!(h.is_finite() && h < 0.0);
            }
        }
    }
}
