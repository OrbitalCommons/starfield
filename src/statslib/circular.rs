//! Circular (directional) statistics.
//!
//! Single source for the circular mean, mean resultant length, Rayleigh test
//! (with the small-n correction — the bare exp(−nR̄²) approximation is poor at
//! sample sizes of n ≈ 6–11), the Kuiper test, and seeded Monte Carlo
//! joint-significance helpers for testing two angle families at once (e.g.
//! position angles and orientation angles of a catalog field).
//!
//! Reference: Mardia & Jupp (2000), "Directional Statistics"; Fisher (1993)

use crate::constants::TAU;
use crate::StarfieldError;
use rand::Rng;

/// Circular mean of a set of angles (radians). Returns a value in [0, 2π).
///
/// Returns `None` for an empty slice or when the resultant vector vanishes
/// (perfectly balanced angles have no defined mean direction).
///
/// # Example
///
/// ```
/// use starfield::statslib::circular_mean;
///
/// // Angles straddling 0 average to ~0, not π.
/// let mean = circular_mean(&[0.1, std::f64::consts::TAU - 0.1]).unwrap();
/// assert!(mean < 0.01 || mean > std::f64::consts::TAU - 0.01);
/// ```
pub fn circular_mean(angles: &[f64]) -> Option<f64> {
    if angles.is_empty() {
        return None;
    }
    let s: f64 = angles.iter().map(|a| a.sin()).sum();
    let c: f64 = angles.iter().map(|a| a.cos()).sum();
    if s.hypot(c) < 1e-12 * angles.len() as f64 {
        return None;
    }
    Some(s.atan2(c).rem_euclid(TAU))
}

/// Mean resultant length R̄ ∈ [0, 1]: 1 = perfectly aligned, 0 = balanced.
///
/// # Example
///
/// ```
/// use starfield::statslib::mean_resultant_length;
///
/// assert!((mean_resultant_length(&[1.0, 1.0, 1.0]) - 1.0).abs() < 1e-12);
/// assert!(mean_resultant_length(&[0.0, std::f64::consts::PI]) < 1e-12);
/// ```
pub fn mean_resultant_length(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return 0.0;
    }
    let n = angles.len() as f64;
    let s: f64 = angles.iter().map(|a| a.sin()).sum();
    let c: f64 = angles.iter().map(|a| a.cos()).sum();
    s.hypot(c) / n
}

/// Circular standard deviation: sqrt(−2 ln R̄) (radians).
///
/// # Example
///
/// ```
/// use starfield::statslib::circular_std;
///
/// // Identical angles have zero circular spread.
/// assert_eq!(circular_std(&[0.7, 0.7, 0.7]), 0.0);
/// ```
pub fn circular_std(angles: &[f64]) -> f64 {
    let r = mean_resultant_length(angles);
    if r <= 0.0 {
        f64::INFINITY
    } else if r >= 1.0 {
        0.0
    } else {
        (-2.0 * r.ln()).sqrt()
    }
}

/// Rayleigh z-statistic z = n R̄² for a set of angles. Higher z indicates
/// stronger clustering around a single direction.
///
/// # Example
///
/// ```
/// use starfield::statslib::rayleigh_z;
///
/// // All angles identical: z = n.
/// assert!((rayleigh_z(&[1.0; 10]) - 10.0).abs() < 1e-9);
/// ```
pub fn rayleigh_z(angles: &[f64]) -> f64 {
    let n = angles.len() as f64;
    let r_bar = mean_resultant_length(angles);
    n * r_bar * r_bar
}

/// Rayleigh test for non-uniformity of circular data.
///
/// Returns the p-value for the null hypothesis that the angles are uniform on
/// the circle, using the series correction of Mardia & Jupp (2000), Eq. 6.3.5:
///
///   p ≈ exp(−Z)·[1 + (2Z − Z²)/(4n) − (24Z − 132Z² + 76Z³ − 9Z⁴)/(288n²)]
///
/// with Z = n R̄². The bare exp(−Z) first term is biased at small n; the
/// correction is accurate to O(n⁻³).
///
/// # Example
///
/// ```
/// use starfield::statslib::rayleigh_p_value;
///
/// // Tightly clustered angles: significant non-uniformity.
/// let clustered: Vec<f64> = (0..10).map(|k| 1.0 + 0.05 * k as f64).collect();
/// assert!(rayleigh_p_value(&clustered) < 1e-3);
/// ```
pub fn rayleigh_p_value(angles: &[f64]) -> f64 {
    let n = angles.len() as f64;
    if n == 0.0 {
        return 1.0;
    }
    let r_bar = mean_resultant_length(angles);
    let z = n * r_bar * r_bar;

    let p = (-z).exp()
        * (1.0 + (2.0 * z - z * z) / (4.0 * n)
            - (24.0 * z - 132.0 * z * z + 76.0 * z.powi(3) - 9.0 * z.powi(4)) / (288.0 * n * n));
    p.clamp(0.0, 1.0)
}

/// Kuiper statistic V_n for circular uniformity.
///
/// V_n = D⁺ + D⁻ against the uniform CDF on [0, 2π); rotation-invariant,
/// unlike Kolmogorov-Smirnov.
///
/// # Example
///
/// ```
/// use starfield::statslib::kuiper_statistic;
///
/// let angles = [0.2_f64, 1.1, 2.0, 4.5, 5.9];
/// let rotated: Vec<f64> = angles
///     .iter()
///     .map(|a| (a + 2.7).rem_euclid(std::f64::consts::TAU))
///     .collect();
/// let diff = (kuiper_statistic(&angles) - kuiper_statistic(&rotated)).abs();
/// assert!(diff < 1e-12);
/// ```
pub fn kuiper_statistic(angles: &[f64]) -> f64 {
    let n = angles.len();
    if n == 0 {
        return 0.0;
    }
    let mut u: Vec<f64> = angles.iter().map(|a| a.rem_euclid(TAU) / TAU).collect();
    u.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let nf = n as f64;
    let mut d_plus: f64 = 0.0;
    let mut d_minus: f64 = 0.0;
    for (i, &ui) in u.iter().enumerate() {
        d_plus = d_plus.max((i as f64 + 1.0) / nf - ui);
        d_minus = d_minus.max(ui - i as f64 / nf);
    }
    d_plus + d_minus
}

/// Approximate p-value for the Kuiper statistic (Stephens 1970 asymptotic
/// series with the finite-n stabilization factor).
///
/// # Example
///
/// ```
/// use starfield::statslib::kuiper_p_value;
///
/// let uniform: Vec<f64> = (0..20)
///     .map(|k| k as f64 * std::f64::consts::TAU / 20.0)
///     .collect();
/// assert!(kuiper_p_value(&uniform) > 0.5);
/// ```
pub fn kuiper_p_value(angles: &[f64]) -> f64 {
    let n = angles.len() as f64;
    if n < 2.0 {
        return 1.0;
    }
    let v = kuiper_statistic(angles);
    // Stabilized statistic
    let lambda = (n.sqrt() + 0.155 + 0.24 / n.sqrt()) * v;
    if lambda < 0.4 {
        return 1.0;
    }
    let mut p = 0.0;
    for j in 1..=100 {
        let jf = j as f64;
        let a = 4.0 * jf * jf * lambda * lambda;
        let term = (2.0 * a - 2.0) * (-0.5 * a).exp();
        p += term;
        if term.abs() < 1e-12 {
            break;
        }
    }
    p.clamp(0.0, 1.0)
}

/// Wrap an angle difference into (−π, π].
///
/// # Example
///
/// ```
/// use starfield::statslib::wrap_to_pi;
///
/// assert!((wrap_to_pi(std::f64::consts::TAU + 0.5) - 0.5).abs() < 1e-12);
/// assert!((wrap_to_pi(-0.5) + 0.5).abs() < 1e-12);
/// ```
pub fn wrap_to_pi(angle: f64) -> f64 {
    let mut a = angle.rem_euclid(TAU);
    if a > std::f64::consts::PI {
        a -= TAU;
    }
    a
}

/// Result of a Monte Carlo joint-significance test of two angle families.
#[derive(Debug, Clone)]
pub struct JointSignificance {
    /// Observed Rayleigh z-statistic of the first angle set
    pub observed_z_a: f64,
    /// Observed Rayleigh z-statistic of the second angle set
    pub observed_z_b: f64,
    /// Fraction of null draws with z_a at least as large as observed
    pub p_a: f64,
    /// Fraction of null draws with z_b at least as large as observed
    pub p_b: f64,
    /// Joint exceedance probability: fraction of null draws at least as
    /// clustered as observed in BOTH angle sets simultaneously (≈ p_a · p_b
    /// when the two are independent)
    pub p_joint: f64,
    /// Number of Monte Carlo iterations
    pub n_iterations: usize,
}

/// Monte Carlo joint-significance test of two angle families against the
/// isotropic (uniform) null.
///
/// For each iteration, draws `angles_a.len()` and `angles_b.len()` uniform
/// angles and compares their Rayleigh z-statistics to the observed ones.
/// P-values are exceedance fractions; `p_joint` is the probability that a
/// random sample clusters at least as strongly as observed in both families
/// at once. Deterministic for a seeded `rng`.
///
/// # Example
///
/// ```
/// use rand::SeedableRng;
/// use starfield::statslib::monte_carlo_joint_significance;
///
/// // Both families tightly clustered: joint p-value is small.
/// let a: Vec<f64> = (0..10).map(|k| 1.0 + 0.05 * k as f64).collect();
/// let b: Vec<f64> = (0..10).map(|k| 4.0 + 0.05 * k as f64).collect();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(42);
/// let result = monte_carlo_joint_significance(&a, &b, 2000, &mut rng);
/// assert!(result.p_joint < 0.01);
/// ```
pub fn monte_carlo_joint_significance(
    angles_a: &[f64],
    angles_b: &[f64],
    n_iterations: usize,
    rng: &mut impl Rng,
) -> JointSignificance {
    let observed_z_a = rayleigh_z(angles_a);
    let observed_z_b = rayleigh_z(angles_b);

    let mut count_a = 0usize;
    let mut count_b = 0usize;
    let mut count_both = 0usize;

    let mut sample_a = vec![0.0; angles_a.len()];
    let mut sample_b = vec![0.0; angles_b.len()];

    for _ in 0..n_iterations {
        for a in sample_a.iter_mut() {
            *a = rng.random::<f64>() * TAU;
        }
        for b in sample_b.iter_mut() {
            *b = rng.random::<f64>() * TAU;
        }

        let z_a = rayleigh_z(&sample_a);
        let z_b = rayleigh_z(&sample_b);
        let a_exceeds = z_a >= observed_z_a;
        let b_exceeds = z_b >= observed_z_b;
        if a_exceeds {
            count_a += 1;
        }
        if b_exceeds {
            count_b += 1;
        }
        if a_exceeds && b_exceeds {
            count_both += 1;
        }
    }

    JointSignificance {
        observed_z_a,
        observed_z_b,
        p_a: count_a as f64 / n_iterations as f64,
        p_b: count_b as f64 / n_iterations as f64,
        p_joint: count_both as f64 / n_iterations as f64,
        n_iterations,
    }
}

/// As [`monte_carlo_joint_significance`], with a caller-supplied null model.
///
/// `draw` produces one (a-angle, b-angle) pair per object under the null
/// hypothesis — this allows the two families to be coupled per object (e.g.
/// selection-bias nulls where both angles depend on a shared property). The
/// two observed sets must therefore have the same length (one pair per
/// object); a [`StarfieldError::DataError`] is returned otherwise.
///
/// # Example
///
/// ```
/// use rand::{Rng, SeedableRng};
/// use starfield::statslib::monte_carlo_joint_significance_with;
///
/// let a: Vec<f64> = (0..8).map(|k| 1.0 + 0.1 * k as f64).collect();
/// let b: Vec<f64> = (0..8).map(|k| 4.0 + 0.1 * k as f64).collect();
/// let mut rng = rand::rngs::StdRng::seed_from_u64(7);
/// let tau = std::f64::consts::TAU;
/// let result = monte_carlo_joint_significance_with(&a, &b, 1000, &mut rng, |rng| {
///     (rng.random::<f64>() * tau, rng.random::<f64>() * tau)
/// })
/// .unwrap();
/// assert!(result.p_joint <= result.p_a.min(result.p_b));
/// ```
pub fn monte_carlo_joint_significance_with<R: Rng, F: FnMut(&mut R) -> (f64, f64)>(
    angles_a: &[f64],
    angles_b: &[f64],
    n_iterations: usize,
    rng: &mut R,
    mut draw: F,
) -> Result<JointSignificance, StarfieldError> {
    if angles_a.len() != angles_b.len() {
        return Err(StarfieldError::DataError(format!(
            "joint significance with a paired null requires equal-length angle sets ({} vs {})",
            angles_a.len(),
            angles_b.len()
        )));
    }

    let observed_z_a = rayleigh_z(angles_a);
    let observed_z_b = rayleigh_z(angles_b);

    let n = angles_a.len();
    let mut count_a = 0usize;
    let mut count_b = 0usize;
    let mut count_both = 0usize;

    let mut sample_a = vec![0.0; n];
    let mut sample_b = vec![0.0; n];

    for _ in 0..n_iterations {
        for k in 0..n {
            let (a, b) = draw(rng);
            sample_a[k] = a;
            sample_b[k] = b;
        }

        let z_a = rayleigh_z(&sample_a);
        let z_b = rayleigh_z(&sample_b);
        let a_exceeds = z_a >= observed_z_a;
        let b_exceeds = z_b >= observed_z_b;
        if a_exceeds {
            count_a += 1;
        }
        if b_exceeds {
            count_b += 1;
        }
        if a_exceeds && b_exceeds {
            count_both += 1;
        }
    }

    Ok(JointSignificance {
        observed_z_a,
        observed_z_b,
        p_a: count_a as f64 / n_iterations as f64,
        p_b: count_b as f64 / n_iterations as f64,
        p_joint: count_both as f64 / n_iterations as f64,
        n_iterations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    #[test]
    fn test_circular_mean_wraps_correctly() {
        // Angles straddling 0: arithmetic mean would give π, circular gives ~0.
        let angles = [0.1, TAU - 0.1];
        let mean = circular_mean(&angles).unwrap();
        assert!(!(0.01..=TAU - 0.01).contains(&mean), "mean = {mean}");
    }

    #[test]
    fn test_circular_mean_balanced_is_none() {
        let angles = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        assert!(circular_mean(&angles).is_none());
        assert!(circular_mean(&[]).is_none());
    }

    #[test]
    fn test_resultant_length_limits() {
        assert_relative_eq!(
            mean_resultant_length(&[1.0, 1.0, 1.0]),
            1.0,
            epsilon = 1e-12
        );
        assert!(mean_resultant_length(&[0.0, PI]) < 1e-12);
    }

    #[test]
    fn test_rayleigh_clustered_vs_uniform() {
        // Tightly clustered: small p.
        let clustered: Vec<f64> = (0..10).map(|k| 1.0 + 0.05 * k as f64).collect();
        assert!(rayleigh_p_value(&clustered) < 1e-3);

        // Evenly spread: p near 1.
        let uniform: Vec<f64> = (0..10).map(|k| k as f64 * TAU / 10.0).collect();
        assert!(rayleigh_p_value(&uniform) > 0.9);
    }

    #[test]
    fn test_rayleigh_small_n_correction_matters() {
        // n = 6, R̄ moderate: the corrected p differs measurably from exp(-Z).
        let angles = [0.0, 0.4, 0.9, 1.5, 2.2, 3.0];
        let n = angles.len() as f64;
        let r = mean_resultant_length(&angles);
        let z = n * r * r;
        let naive = (-z).exp();
        let corrected = rayleigh_p_value(&angles);
        assert!(
            (naive - corrected).abs() / naive > 0.01,
            "correction too small: naive {naive:.4}, corrected {corrected:.4}"
        );
    }

    #[test]
    fn test_rayleigh_z_clustered() {
        // All angles the same → z = n
        let angles = vec![1.0; 10];
        let z = rayleigh_z(&angles);
        assert!(
            (z - 10.0).abs() < 0.01,
            "z = {z:.2}, expected 10.0 for identical angles"
        );
    }

    #[test]
    fn test_rayleigh_z_uniform() {
        // Evenly spaced angles → z ≈ 0
        let n = 100;
        let angles: Vec<f64> = (0..n).map(|i| TAU * i as f64 / n as f64).collect();
        let z = rayleigh_z(&angles);
        assert!(z < 0.1, "z = {z:.4}, expected near 0 for uniform");
    }

    #[test]
    fn test_kuiper_uniform_vs_clustered() {
        let uniform: Vec<f64> = (0..20).map(|k| k as f64 * TAU / 20.0).collect();
        let clustered: Vec<f64> = (0..20).map(|k| 1.0 + 0.02 * k as f64).collect();
        assert!(kuiper_statistic(&clustered) > kuiper_statistic(&uniform));
        assert!(kuiper_p_value(&clustered) < 0.01);
        assert!(kuiper_p_value(&uniform) > 0.5);
    }

    #[test]
    fn test_kuiper_rotation_invariant() {
        let angles: Vec<f64> = vec![0.2, 1.1, 2.0, 4.5, 5.9];
        let rotated: Vec<f64> = angles.iter().map(|a| (a + 2.7).rem_euclid(TAU)).collect();
        assert_relative_eq!(
            kuiper_statistic(&angles),
            kuiper_statistic(&rotated),
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_wrap_to_pi() {
        assert_relative_eq!(wrap_to_pi(3.0 * PI), PI, epsilon = 1e-12);
        assert_relative_eq!(wrap_to_pi(-0.5), -0.5, epsilon = 1e-12);
        assert_relative_eq!(wrap_to_pi(TAU + 0.5), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_joint_significance_clustered() {
        let a: Vec<f64> = (0..10).map(|k| 1.0 + 0.05 * k as f64).collect();
        let b: Vec<f64> = (0..10).map(|k| 4.0 + 0.05 * k as f64).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = monte_carlo_joint_significance(&a, &b, 5000, &mut rng);
        assert!(result.p_a < 0.01, "p_a = {}", result.p_a);
        assert!(result.p_b < 0.01, "p_b = {}", result.p_b);
        assert!(result.p_joint <= result.p_a.min(result.p_b));
        assert_eq!(result.n_iterations, 5000);
    }

    #[test]
    fn test_joint_significance_uniform_not_significant() {
        let a: Vec<f64> = (0..10).map(|k| k as f64 * TAU / 10.0).collect();
        let b: Vec<f64> = (0..10).map(|k| (k as f64 + 0.5) * TAU / 10.0).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = monte_carlo_joint_significance(&a, &b, 2000, &mut rng);
        assert!(result.p_a > 0.5, "p_a = {}", result.p_a);
        assert!(result.p_b > 0.5, "p_b = {}", result.p_b);
    }

    /// Seeded determinism: identical seeds give identical p-values.
    #[test]
    fn test_joint_significance_seeded() {
        let a: Vec<f64> = (0..8).map(|k| 1.0 + 0.2 * k as f64).collect();
        let b: Vec<f64> = (0..8).map(|k| 2.0 + 0.3 * k as f64).collect();
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(7);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(7);
        let r1 = monte_carlo_joint_significance(&a, &b, 1000, &mut rng1);
        let r2 = monte_carlo_joint_significance(&a, &b, 1000, &mut rng2);
        assert_eq!(r1.p_joint, r2.p_joint);
    }

    #[test]
    fn test_joint_significance_with_uniform_null_matches() {
        // The paired variant with a uniform draw agrees with the uniform
        // version statistically (same null, different draw order).
        let a: Vec<f64> = (0..10).map(|k| 1.0 + 0.05 * k as f64).collect();
        let b: Vec<f64> = (0..10).map(|k| 4.0 + 0.05 * k as f64).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let result = monte_carlo_joint_significance_with(&a, &b, 5000, &mut rng, |rng| {
            (rng.random::<f64>() * TAU, rng.random::<f64>() * TAU)
        })
        .unwrap();
        assert!(result.p_joint < 0.01, "p_joint = {}", result.p_joint);
    }

    #[test]
    fn test_joint_significance_with_rejects_mismatched_lengths() {
        let a = [0.0, 1.0, 2.0];
        let b = [0.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let result = monte_carlo_joint_significance_with(&a, &b, 10, &mut rng, |rng| {
            (rng.random::<f64>(), rng.random::<f64>())
        });
        assert!(result.is_err());
    }
}
