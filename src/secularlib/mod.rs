//! Secular and resonance dynamics for long-term small-body evolution.
//!
//! Three submodules:
//!
//! * [`hamiltonian`] — doubly-averaged secular Hamiltonians: the analytic
//!   Kozai–Lidov quadrupole (valid in the hierarchical limit α = a/a_p ≪ 1)
//!   and numerical Gauss-ring double averaging of the exact interaction,
//!   which remains valid in the non-hierarchical regime (α ≳ 0.3) where the
//!   multipole series diverges and which captures the octupole-and-higher
//!   apsidal (Δϖ) structure automatically.
//! * [`hansen`] — Hansen coefficients X_j^{n,m}(e) of the disturbing
//!   function, via convergence-controlled quadrature plus standard closed
//!   forms and asymptotics.
//! * [`resonance`] — mean-motion resonance machinery: resonance locations,
//!   the canonical resonant angle, libration detection on time series, and
//!   the Chirikov resonance-overlap (chaos) criterion with its critical
//!   perihelion.
//!
//! The most common items are re-exported at this level.
//!
//! # Units
//!
//! This module uses the conventions of the dynamics literature throughout:
//! lengths in **AU**, angles in **radians**, time in **days**, and
//! gravitational parameters (GM) in **AU³/day²**. This matches starfield's
//! `keplerlib` internals; `elementslib` and `constants::GM_SUN` use km³/s²
//! instead — convert with
//!
//! ```
//! use starfield::constants::{AU_KM, DAY_S, GM_SUN};
//! use starfield::secularlib::GM_SUN_AU3_DAY2;
//!
//! let gm_sun_au3_day2 = GM_SUN * DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);
//! assert!((gm_sun_au3_day2 - GM_SUN_AU3_DAY2).abs() / GM_SUN_AU3_DAY2 < 1e-9);
//! ```

pub mod hamiltonian;
pub mod hansen;
pub mod resonance;

pub use hamiltonian::{
    coplanar_quadrupole, numerical_secular_hamiltonian, phase_portrait, quadrupole_hamiltonian,
};
pub use hansen::{
    hansen_coefficient, hansen_x0_neg3_0, hansen_x_neg3_2_chain, hansen_x_neg3_2_neptune_chain,
    mean_to_true_anomaly,
};
pub use resonance::{
    chirikov_overlap_parameter, chirikov_overlap_parameter_neptune, critical_perihelion,
    critical_perihelion_neptune, is_chaotic, is_chaotic_neptune, is_librating, libration_amplitude,
    resonance_semi_major_axis, resonant_angle,
};

/// GM of the Sun in AU³/day² (the square of the Gaussian gravitational
/// constant, k²).
///
/// Derived from the DE440 GM_SUN = 1.32712440041279419e+20 m³/s²
/// (Park et al. 2021) with AU = 1.495978707e+11 m and day = 86 400 s.
/// Agrees with `starfield::constants::GM_SUN` (km³/s², from a slightly older
/// determination) to ~5e-12 relative; the agreement is pinned by a test.
pub const GM_SUN_AU3_DAY2: f64 = 2.959122082855911e-4;

/// GM of Neptune in AU³/day². Source: JPL DE440 / Park et al. (2021).
pub const GM_NEPTUNE_AU3_DAY2: f64 = 1.524_358_9e-8;

/// Neptune's semi-major axis in AU.
pub const A_NEPTUNE_AU: f64 = 30.07;

/// Neptune/Sun mass ratio. Source: JPL DE440 / Park et al. (2021).
pub const MASS_NEPTUNE_SOLAR: f64 = 5.151_389_0e-5;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AU_KM, DAY_S, GM_SUN};

    #[test]
    fn test_gm_sun_conventions_agree() {
        // starfield's GM_SUN (km³/s²) converted to AU³/day² must match the
        // DE440-derived constant used here.
        let converted = GM_SUN * DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);
        let rel = ((converted - GM_SUN_AU3_DAY2) / GM_SUN_AU3_DAY2).abs();
        assert!(rel < 1e-9, "GM_SUN conventions disagree: rel = {rel:.3e}");
    }

    #[test]
    fn test_neptune_gm_consistent_with_mass_ratio() {
        let rel = ((GM_NEPTUNE_AU3_DAY2 / GM_SUN_AU3_DAY2 - MASS_NEPTUNE_SOLAR)
            / MASS_NEPTUNE_SOLAR)
            .abs();
        assert!(rel < 1e-6, "GM_NEPTUNE vs mass ratio: rel = {rel:.3e}");
    }
}
