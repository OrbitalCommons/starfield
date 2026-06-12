//! N-body integrators for solar-system dynamics.
//!
//! This module provides perturbed, multi-body orbit propagation on top of
//! starfield's two-body machinery:
//!
//! - [`whm`] — symplectic Wisdom-Holman mapping in democratic-heliocentric
//!   coordinates (Wisdom & Holman 1991; Duncan, Levison & Lee 1998), with
//!   bounded energy error over arbitrarily long integrations.
//! - [`bulirsch_stoer`] — Bulirsch-Stoer with recursive step-size control,
//!   for high-accuracy work and close encounters.
//! - [`hybrid`] — Mercury6-style hybrid switching (Chambers 1999): WHM far
//!   from planets, Bulirsch-Stoer through encounters, with the smooth
//!   changeover K(r) partitioning the planet force so nothing is
//!   double-counted.
//! - [`kepler`] — the universal-variable Kepler drift shared by all of the
//!   above (bisection-safeguarded Newton, robust for elliptic, parabolic,
//!   and hyperbolic states).
//! - [`forces`] — composable extra accelerations applied in the kick stage
//!   (orbit-averaged giant-planet quadrupole, galactic disk tide, custom
//!   closures).
//! - [`types`] — state vectors, orbital elements, massive bodies, and the
//!   bridges to starfield's `planetlib` (ephemeris initial conditions) and
//!   `keplerlib` (test particles from `KeplerOrbit`).
//! - [`democratic`] — heliocentric/barycentric coordinate transforms used by
//!   the WHM splitting.
//!
//! # Units and frame
//!
//! Everything in this module uses **AU** for length, **days** for time, and
//! **AU³/day²** for gravitational parameters (GM). State vectors are
//! heliocentric ecliptic-J2000 Cartesian coordinates, matching
//! `positions::ecliptic::EclipticStateVector`.
//!
//! # Example
//!
//! ```
//! use nalgebra::Vector3;
//! use starfield::nbodylib::{MassiveBody, SimConfig, StateVector, WhmIntegrator, GM_SUN};
//!
//! // A Jupiter-mass planet on a circular 5.2 AU orbit.
//! let gm = 1e-3 * GM_SUN;
//! let v = (GM_SUN / 5.2_f64).sqrt();
//! let mut bodies = vec![MassiveBody::new(
//!     "Jupiter",
//!     gm,
//!     StateVector::new(Vector3::new(5.2, 0.0, 0.0), Vector3::new(0.0, v, 0.0)),
//! )];
//! let mut particles: Vec<StateVector> = Vec::new();
//! let mut active: Vec<bool> = Vec::new();
//!
//! let config = SimConfig::default();
//! let integrator = WhmIntegrator::new();
//! for _ in 0..100 {
//!     integrator.step(&mut bodies, &mut particles, &mut active, 100.0, &config);
//! }
//!
//! // Still on a circular orbit after ~27 years.
//! assert!((bodies[0].state.distance() - 5.2).abs() < 0.01);
//! ```

pub mod bulirsch_stoer;
pub mod democratic;
pub mod forces;
pub mod hybrid;
pub mod kepler;
pub mod types;
pub mod whm;

pub use bulirsch_stoer::{bs_step, changeover_k};
pub use forces::{total_extra_acceleration, ExtraForce};
pub use hybrid::{changeover_radii, find_close_encounters, hybrid_step};
pub use kepler::kepler_drift;
pub use types::{
    cartesian_to_elements, elements_to_cartesian, giant_planets_from_ephemeris, solve_kepler,
    MassiveBody, OrbitalElements, SimConfig, StateVector,
};
pub use whm::{
    barycentric_energy, total_angular_momentum, validate_timestep, WhmIntegrator,
    MIN_STEPS_PER_ORBIT,
};

/// Heliocentric gravitational parameter in AU³/day².
///
/// JPL DE440 (Park et al. 2021): GM☉ = 1.32712440041279419e20 m³/s².
/// Note this differs from `crate::constants::GM_SUN` (km³/s², Pitjeva 2005,
/// 1.32712440042e20 m³/s²) by ~5.4e-12 relative — beyond every tolerance in
/// this module, but pinned by a test below so the two can never silently
/// diverge. The crate-level constant is left untouched.
pub const GM_SUN: f64 = 2.959122082855911e-4;

/// Giant-planet *system* gravitational parameters in AU³/day² (planet plus
/// satellites, JPL DE440). These pair with the planet-system *barycenter*
/// states that `planetlib` resolves for the outer planets, so barycenter
/// states and system masses describe the same dynamical body.
pub const GM_JUPITER: f64 = 2.825_345_818e-7;
/// Saturn system GM in AU³/day² (DE440).
pub const GM_SATURN: f64 = 8.459_715_6e-8;
/// Uranus system GM in AU³/day² (DE440).
pub const GM_URANUS: f64 = 1.292_024_9e-8;
/// Neptune system GM in AU³/day² (DE440).
pub const GM_NEPTUNE: f64 = 1.524_358_9e-8;

/// Giant-planet system masses as a fraction of the solar mass (DE440).
pub const MASS_JUPITER_SOLAR: f64 = 9.547_919_384e-4;
/// Saturn system mass in solar masses (DE440).
pub const MASS_SATURN_SOLAR: f64 = 2.858_859_81e-4;
/// Uranus system mass in solar masses (DE440).
pub const MASS_URANUS_SOLAR: f64 = 4.366_244_0e-5;
/// Neptune system mass in solar masses (DE440).
pub const MASS_NEPTUNE_SOLAR: f64 = 5.151_389_0e-5;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AU_KM, DAY_S};

    #[test]
    fn test_gm_sun_matches_crate_constant() {
        // crate::constants::GM_SUN (km³/s²) converted to AU³/day². The
        // underlying determinations differ by ~5.4e-12 relative (DE440 vs
        // Pitjeva 2005), so agreement is pinned at 1e-11 rather than full
        // f64 precision.
        let crate_au3_day2 = crate::constants::GM_SUN * DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);
        let rel = ((crate_au3_day2 - GM_SUN) / GM_SUN).abs();
        assert!(rel < 1e-11, "relative difference = {rel:e}");
    }

    #[test]
    fn test_planet_gm_mass_consistency() {
        // GM_planet / GM_sun must equal the tabulated mass ratios.
        for (gm, mass) in [
            (GM_JUPITER, MASS_JUPITER_SOLAR),
            (GM_SATURN, MASS_SATURN_SOLAR),
            (GM_URANUS, MASS_URANUS_SOLAR),
            (GM_NEPTUNE, MASS_NEPTUNE_SOLAR),
        ] {
            // The GM and mass-ratio constants are independently rounded
            // DE440 values, so they agree to ~1e-7, not full precision.
            let rel = (gm / GM_SUN - mass).abs() / mass;
            assert!(rel < 1e-6, "GM/mass inconsistency: {rel:e}");
        }
    }
}
