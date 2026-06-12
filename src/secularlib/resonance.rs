//! Mean-motion resonance machinery: resonant angles, libration detection,
//! and the Chirikov resonance-overlap (chaos) criterion for the 2:j chain of
//! an arbitrary planet (with Neptune-flavored convenience wrappers).
//!
//! Conventions: a particle *exterior* to the planet in the p:q resonance
//! (p > q, both positive, p orbits of the planet per q orbits of the
//! particle) sits at a_res = a_planet (p/q)^{2/3} and has the stationary
//! angle
//!
//!   φ = p λ − q λ_planet − (p − q) ϖ
//!
//! which satisfies the d'Alembert rule (coefficients sum to zero) and is
//! stationary when n/n_planet = q/p. (A p/q-swapped convention circulates
//! even for an exactly resonant orbit; the overlap exponent carries
//! q²/(2 a_pl²), the form consistent with the pendulum half-width of the
//! resonance potential and with [`critical_perihelion`] — a q²/(4 a_pl²)
//! variant is off by ≈ 1.4 at q = 35 AU for Neptune.)
//!
//! Units: AU, radians (see the [`crate::secularlib`] docs).
//!
//! Reference: Murray & Dermott (1999) Ch. 8; Batygin & Brown (2021)

use crate::constants::TAU;
use crate::secularlib::{A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR};

/// Semi-major axis of the exterior p:q resonance with a planet at
/// `a_planet` (AU): a_res = a_planet (p/q)^{2/3}.
///
/// # Example
///
/// ```
/// use starfield::secularlib::{resonance_semi_major_axis, A_NEPTUNE_AU};
///
/// // Neptune 3:2 (Plutinos): a ≈ 39.4 AU
/// let a = resonance_semi_major_axis(3, 2, A_NEPTUNE_AU);
/// assert!((a - 39.4).abs() < 0.1);
/// ```
pub fn resonance_semi_major_axis(p: u32, q: u32, a_planet: f64) -> f64 {
    a_planet * (p as f64 / q as f64).powf(2.0 / 3.0)
}

/// Resonant angle φ = p λ − q λ_planet − (p − q) ϖ for the exterior p:q
/// resonance, wrapped to [0, 2π).
///
/// `lambda`/`varpi` are the particle's mean longitude and longitude of
/// perihelion; `lambda_planet` the planet's mean longitude (all radians).
///
/// # Example
///
/// ```
/// use starfield::secularlib::resonant_angle;
///
/// // At t = 0 with λ = ϖ the 5:2 angle is (5 − 2) ϖ − (5 − 2) ϖ + ... = 2ϖ.
/// let phi = resonant_angle(5, 2, 1.3, 0.0, 1.3);
/// assert!((phi - 2.0 * 1.3).abs() < 1e-12);
/// ```
pub fn resonant_angle(p: u32, q: u32, lambda: f64, lambda_planet: f64, varpi: f64) -> f64 {
    let (pf, qf) = (p as f64, q as f64);
    (pf * lambda - qf * lambda_planet - (pf - qf) * varpi).rem_euclid(TAU)
}

/// Libration amplitude of a resonant-angle time series, or `None` if the
/// angle circulates.
///
/// Detection: sort the angles on the circle and find the largest angular gap.
/// A librating angle occupies an arc, leaving an empty gap > `min_gap`; a
/// circulating angle covers the circle (max gap → 2π·ln N/N for N uniform
/// samples). The returned amplitude is the half-width of the occupied arc.
///
/// `min_gap` of ~1 rad is a robust default for ≳ 50 samples spanning several
/// libration periods.
///
/// # Example
///
/// ```
/// use starfield::secularlib::libration_amplitude;
///
/// // φ oscillating about π with amplitude 0.8 librates.
/// let angles: Vec<f64> = (0..200)
///     .map(|k| std::f64::consts::PI + 0.8 * (0.1 * k as f64).sin())
///     .collect();
/// let amp = libration_amplitude(&angles, 1.0).unwrap();
/// assert!((amp - 0.8).abs() < 0.05);
/// ```
pub fn libration_amplitude(angles: &[f64], min_gap: f64) -> Option<f64> {
    if angles.len() < 2 {
        return None;
    }
    let mut sorted: Vec<f64> = angles.iter().map(|a| a.rem_euclid(TAU)).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut max_gap = TAU - (sorted[sorted.len() - 1] - sorted[0]);
    for w in sorted.windows(2) {
        max_gap = max_gap.max(w[1] - w[0]);
    }

    if max_gap > min_gap {
        Some(0.5 * (TAU - max_gap))
    } else {
        None
    }
}

/// Whether a resonant-angle time series librates (see [`libration_amplitude`]).
pub fn is_librating(angles: &[f64], min_gap: f64) -> bool {
    libration_amplitude(angles, min_gap).is_some()
}

/// Chirikov overlap parameter for the 2:j resonance chain of a planet at
/// `a_planet_au` (AU) with planet/star mass ratio `mass_ratio`:
///
///   K = Δa/δa = (24/√5) (a/a_pl)^{5/4} √(m_pl/M*) · exp(−q²/(2 a_pl²))
///
/// Adjacent resonances overlap (K > 1) ⇒ chaotic transport. The exponent
/// carries q²/(2 a_pl²) — this is the form consistent with the pendulum
/// half-width of the resonance potential and with [`critical_perihelion`].
/// The exponential inherits the calibration regime of the
/// [`crate::secularlib::hansen::hansen_x_neg3_2_chain`] asymptotic
/// (q/a_pl ∈ ~(1.0, 1.7), fitted on the Neptune chain).
///
/// Reference: Batygin & Brown (2021); Murray & Dermott (1999) Ch. 9
///
/// # Example
///
/// ```
/// use starfield::secularlib::{
///     chirikov_overlap_parameter, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR,
/// };
///
/// // Overlap strengthens with semi-major axis at fixed perihelion.
/// let k1 = chirikov_overlap_parameter(100.0, 35.0, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR);
/// let k2 = chirikov_overlap_parameter(500.0, 35.0, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR);
/// assert!(k2 > k1);
/// ```
pub fn chirikov_overlap_parameter(a_au: f64, q_au: f64, a_planet_au: f64, mass_ratio: f64) -> f64 {
    let alpha = a_au / a_planet_au;
    (24.0 / 5.0_f64.sqrt())
        * alpha.powf(1.25)
        * mass_ratio.sqrt()
        * (-q_au * q_au / (2.0 * a_planet_au * a_planet_au)).exp()
}

/// Chirikov overlap parameter for the Neptune 2:j chain (convenience wrapper
/// over [`chirikov_overlap_parameter`] with a_N = 30.07 AU and the Neptune/Sun
/// mass ratio).
pub fn chirikov_overlap_parameter_neptune(a_au: f64, q_au: f64) -> f64 {
    chirikov_overlap_parameter(a_au, q_au, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
}

/// Whether orbits at (a, q) are in the chaotic resonance-overlap regime of
/// the planet's 2:j chain (K > 1; see [`chirikov_overlap_parameter`]).
pub fn is_chaotic(a_au: f64, q_au: f64, a_planet_au: f64, mass_ratio: f64) -> bool {
    chirikov_overlap_parameter(a_au, q_au, a_planet_au, mass_ratio) > 1.0
}

/// Whether orbits at (a, q) are in Neptune's chaotic resonance-overlap regime.
pub fn is_chaotic_neptune(a_au: f64, q_au: f64) -> bool {
    is_chaotic(a_au, q_au, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
}

/// Critical perihelion below which the planet's 2:j chain overlaps (the
/// K = 1 root of [`chirikov_overlap_parameter`]):
///
///   q_crit = a_pl √( ln\[ (576/5) (m_pl/M*) (a/a_pl)^{5/2} \] )
///
/// Returns 0 when the argument of the log is ≤ 1 (no chaos at any q).
///
/// # Example
///
/// ```
/// use starfield::secularlib::{
///     chirikov_overlap_parameter, critical_perihelion, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR,
/// };
///
/// // K(a, q_crit(a)) = 1 by construction.
/// let q = critical_perihelion(500.0, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR);
/// let k = chirikov_overlap_parameter(500.0, q, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR);
/// assert!((k - 1.0).abs() < 1e-10);
/// ```
pub fn critical_perihelion(a_au: f64, a_planet_au: f64, mass_ratio: f64) -> f64 {
    let alpha = a_au / a_planet_au;
    let argument = (576.0 / 5.0) * mass_ratio * alpha.powf(2.5);
    if argument <= 1.0 {
        return 0.0;
    }
    a_planet_au * argument.ln().sqrt()
}

/// Critical perihelion of the Neptune 2:j chain (convenience wrapper over
/// [`critical_perihelion`]).
///
/// # Example
///
/// ```
/// use starfield::secularlib::critical_perihelion_neptune;
///
/// // q_crit(500 AU) ≈ 41.4 AU for Neptune.
/// let q = critical_perihelion_neptune(500.0);
/// assert!((q - 41.4).abs() < 0.1);
/// ```
pub fn critical_perihelion_neptune(a_au: f64) -> f64 {
    critical_perihelion(a_au, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_resonant_angle_stationary_at_exact_resonance() {
        // A perfectly resonant orbit: n/n_N = q/p. The angle must be
        // constant in time — the p/q-swapped convention circulates at
        // n_N (p² − q²)/q and fails this test.
        let (p, q) = (5u32, 2u32);
        let n_planet = 1.0e-2; // rad/day
        let n = n_planet * (q as f64) / (p as f64);
        let varpi = 1.3;

        let phi0 = resonant_angle(p, q, varpi, 0.0, varpi);
        for step in 1..200 {
            let t = step as f64 * 50.0;
            let phi = resonant_angle(p, q, varpi + n * t, n_planet * t, varpi);
            assert_relative_eq!(phi, phi0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_resonant_angle_circulates_off_resonance() {
        let (p, q) = (3u32, 1u32);
        let n_planet = 1.0e-2;
        let n = n_planet * (q as f64) / (p as f64) * 1.05; // 5% off
        let angles: Vec<f64> = (0..500)
            .map(|s| {
                let t = s as f64 * 200.0;
                resonant_angle(p, q, n * t, n_planet * t, 0.0)
            })
            .collect();
        assert!(!is_librating(&angles, 1.0));
    }

    #[test]
    fn test_libration_detection() {
        // Librating: φ oscillates about π with amplitude 0.8.
        let librating: Vec<f64> = (0..200)
            .map(|k| std::f64::consts::PI + 0.8 * (0.1 * k as f64).sin())
            .collect();
        let amp = libration_amplitude(&librating, 1.0).expect("should librate");
        assert!((amp - 0.8).abs() < 0.05, "amplitude {amp}");

        // Circulating: uniform coverage.
        let circulating: Vec<f64> = (0..200).map(|k| k as f64 * TAU / 200.0).collect();
        assert!(libration_amplitude(&circulating, 1.0).is_none());
    }

    #[test]
    fn test_resonance_location() {
        // Neptune 3:2 (Plutinos): a ≈ 39.4 AU
        let a = resonance_semi_major_axis(3, 2, A_NEPTUNE_AU);
        assert!((a - 39.4).abs() < 0.1, "a = {a}");
    }

    #[test]
    fn test_chirikov_critical_perihelion_consistent_with_overlap() {
        // The two public APIs must agree: K(a, q_crit(a)) = 1 exactly —
        // for Neptune and for a heavier, more distant perturber alike.
        for &(a_pl, mu) in &[(A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR), (19.19, 4.366_244_0e-5)] {
            for &a in &[300.0, 500.0, 1000.0] {
                let q_crit = critical_perihelion(a, a_pl, mu);
                assert!(q_crit > 0.0);
                let k = chirikov_overlap_parameter(a, q_crit, a_pl, mu);
                assert_relative_eq!(k, 1.0, epsilon = 1e-10);
                // And the classification flips across the boundary.
                assert!(is_chaotic(a, q_crit - 1.0, a_pl, mu));
                assert!(!is_chaotic(a, q_crit + 1.0, a_pl, mu));
            }
        }
    }

    #[test]
    fn test_chirikov_trends() {
        assert!(
            chirikov_overlap_parameter_neptune(500.0, 35.0)
                > chirikov_overlap_parameter_neptune(100.0, 35.0)
        );
        assert!(
            chirikov_overlap_parameter_neptune(200.0, 30.0)
                > chirikov_overlap_parameter_neptune(200.0, 45.0)
        );
        // q_crit at a = 500 AU is in the trans-Neptunian range
        let q = critical_perihelion_neptune(500.0);
        assert!(q > 25.0 && q < 60.0, "q_crit = {q}");
    }

    #[test]
    fn test_neptune_critical_perihelion_pin() {
        // Pin against the planet9 reference value: q_crit(500 AU) = 41.4 AU.
        let q = critical_perihelion_neptune(500.0);
        assert!((q - 41.4).abs() < 0.1, "q_crit(500) = {q}");
    }

    #[test]
    fn test_neptune_wrappers_match_generic_form() {
        for &(a, q) in &[(300.0, 35.0), (500.0, 41.0), (1000.0, 50.0)] {
            assert_eq!(
                chirikov_overlap_parameter_neptune(a, q),
                chirikov_overlap_parameter(a, q, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
            );
            assert_eq!(
                is_chaotic_neptune(a, q),
                is_chaotic(a, q, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
            );
        }
        assert_eq!(
            critical_perihelion_neptune(500.0),
            critical_perihelion(500.0, A_NEPTUNE_AU, MASS_NEPTUNE_SOLAR)
        );
    }
}
