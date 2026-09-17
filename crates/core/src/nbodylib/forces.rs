//! Accelerations for the kick stage of the integrators.
//!
//! Two layers live here:
//!
//! 1. The Newtonian building blocks used by the Wisdom-Holman and hybrid
//!    drivers: direct and direct+indirect point-mass kicks, and the J2/J4
//!    oblateness field.
//! 2. [`ExtraForce`] — composable, position-dependent extra accelerations
//!    (potential kicks, so they preserve the symplectic structure of the
//!    Wisdom-Holman splitting): an orbit-averaged giant-planet quadrupole,
//!    the Milky Way vertical disk tide, and user closures.

use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};
use once_cell::sync::Lazy;

use crate::constants::{AU_KM, J2000};
use crate::framelib::{Frame, ECLIPJ2000_MATRIX_INV, GALACTIC};
use crate::time::Timescale;

use super::types::{MassiveBody, StateVector};
use super::{
    GM_JUPITER, GM_SATURN, GM_SUN, GM_URANUS, MASS_JUPITER_SOLAR, MASS_SATURN_SOLAR,
    MASS_URANUS_SOLAR,
};

// ===================================================================
// Point-mass kicks
// ===================================================================

/// Compute the gravitational acceleration on a body at position `r` due to
/// a perturbing body with GM `gm_pert` at position `r_pert`.
///
/// This is the direct + indirect term:
///   a = -gm_pert * [ (r - r_pert)/|r - r_pert|^3 + r_pert/|r_pert|^3 ]
///
/// The second term is the indirect (Sun-centered frame) acceleration.
#[inline]
pub fn perturbation_acceleration(
    r: &Vector3<f64>,
    r_pert: &Vector3<f64>,
    gm_pert: f64,
) -> Vector3<f64> {
    let dr = r - r_pert;
    let dr_mag = dr.norm();
    let r_pert_mag = r_pert.norm();

    if dr_mag < 1e-30 || r_pert_mag < 1e-30 {
        return Vector3::zeros();
    }

    let dr3 = dr_mag.powi(3);
    let rp3 = r_pert_mag.powi(3);

    // Direct term + indirect (Sun recoil) term
    -gm_pert * (dr / dr3 + r_pert / rp3)
}

/// Direct gravitational acceleration only (no indirect term).
///
/// In the democratic heliocentric splitting (Duncan, Levison & Lee 1998) the
/// Sun-recoil physics is carried by the separate solar-drift substep, so the
/// kick must apply only the direct term.
#[inline]
pub fn direct_acceleration(r: &Vector3<f64>, r_pert: &Vector3<f64>, gm_pert: f64) -> Vector3<f64> {
    let dr = r - r_pert;
    let dr_mag = dr.norm();
    if dr_mag < 1e-30 {
        return Vector3::zeros();
    }
    -gm_pert * dr / dr_mag.powi(3)
}

/// Direct-only mutual kick for massive bodies (democratic heliocentric scheme).
pub fn kick_bodies_direct(bodies: &mut [MassiveBody], dt: f64) {
    let n = bodies.len();
    if n < 2 {
        return;
    }

    let mut accels = vec![Vector3::<f64>::zeros(); n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            accels[i] +=
                direct_acceleration(&bodies[i].state.pos, &bodies[j].state.pos, bodies[j].gm);
        }
    }

    for (body, accel) in bodies.iter_mut().zip(accels.iter()) {
        body.state.vel += dt * accel;
    }
}

/// Direct-only kick on test particles from massive bodies (democratic
/// heliocentric).
pub fn kick_particles_direct(
    particles: &mut [StateVector],
    active: &[bool],
    bodies: &[MassiveBody],
    dt: f64,
) {
    for (i, particle) in particles.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }
        let mut accel = Vector3::zeros();
        for body in bodies {
            accel += direct_acceleration(&particle.pos, &body.state.pos, body.gm);
        }
        particle.vel += dt * accel;
    }
}

/// Apply a kick (velocity impulse) to all massive bodies due to mutual
/// interactions. Each body receives acceleration from all other bodies.
///
/// Direct + indirect form, for heliocentric-velocity schemes. The
/// democratic-heliocentric integrator uses `kick_bodies_direct` instead.
pub fn kick_bodies(bodies: &mut [MassiveBody], dt: f64) {
    let n = bodies.len();
    if n < 2 {
        return;
    }

    // Compute accelerations first, then apply (to avoid order dependence)
    let mut accels = vec![Vector3::<f64>::zeros(); n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            accels[i] +=
                perturbation_acceleration(&bodies[i].state.pos, &bodies[j].state.pos, bodies[j].gm);
        }
    }

    for (body, accel) in bodies.iter_mut().zip(accels.iter()) {
        body.state.vel += dt * accel;
    }
}

/// Apply a direct + indirect kick to all test particles due to massive
/// bodies. Test particles do not affect each other or the massive bodies.
pub fn kick_particles(
    particles: &mut [StateVector],
    active: &[bool],
    bodies: &[MassiveBody],
    dt: f64,
) {
    for (i, particle) in particles.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }

        let mut accel = Vector3::zeros();
        for body in bodies {
            accel += perturbation_acceleration(&particle.pos, &body.state.pos, body.gm);
        }

        particle.vel += dt * accel;
    }
}

// ===================================================================
// J2/J4 oblateness
// ===================================================================

/// J2 (and optionally J4) oblateness acceleration of a body centered at the
/// origin with equatorial radius `radius` (AU), symmetry axis along z.
///
/// The J2 potential is:
///   Φ_J2 = -GM/r * (R/r)^2 * J2 * P2(sin(φ))
/// where φ is the latitude.
pub fn j2_acceleration(
    r: &Vector3<f64>,
    gm: f64,
    radius: f64,
    j2: f64,
    j4: Option<f64>,
) -> Vector3<f64> {
    let r_mag = r.norm();
    if r_mag < 1e-30 {
        return Vector3::zeros();
    }

    let r2 = r_mag * r_mag;
    let r5 = r2 * r2 * r_mag;
    let r7 = r5 * r2;
    let rr = radius * radius;

    let z2 = r.z * r.z;
    let z2_r2 = z2 / r2;

    // J2 acceleration components
    let factor_j2 = 1.5 * gm * j2 * rr;
    let ax_j2 = -factor_j2 * r.x / r5 * (1.0 - 5.0 * z2_r2);
    let ay_j2 = -factor_j2 * r.y / r5 * (1.0 - 5.0 * z2_r2);
    let az_j2 = -factor_j2 * r.z / r5 * (3.0 - 5.0 * z2_r2);

    let mut accel = Vector3::new(ax_j2, ay_j2, az_j2);

    // J4 acceleration if provided
    if let Some(j4_val) = j4 {
        let r4 = rr * rr;
        let r9 = r7 * r2;
        let z4_r4 = z2_r2 * z2_r2;

        let factor_j4 = 0.625 * gm * j4_val * r4;
        let bracket = 3.0 - 42.0 * z2_r2 + 63.0 * z4_r4;
        let bracket_z = 15.0 - 70.0 * z2_r2 + 63.0 * z4_r4;

        accel.x += factor_j4 * r.x / r9 * bracket;
        accel.y += factor_j4 * r.y / r9 * bracket;
        accel.z += factor_j4 * r.z / r9 * bracket_z;
    }

    accel
}

// ===================================================================
// Orbit-averaged giant-planet quadrupole (secular J2)
// ===================================================================

/// Orbit-averaged J2 coefficient from a planet.
///
/// When a giant planet is not integrated directly (to save CPU), its
/// gravitational effect can be approximated as an enhanced J2 moment on the
/// central body, valid for particles with perihelia well outside the
/// planet's orbit:
///   J2_eff = (1/2) * (m_planet / M_sun) * (a_planet / R_ref)^2
///
/// `mass_ratio`: m_planet / M_sun;
/// `a_planet_au`: semi-major axis of the planet (AU);
/// `r_ref_au`: reference radius for the J2 expansion (typically 1 AU).
///
/// Reference: Murray & Dermott (1999), Section 6.11; Batygin & Brown (2016)
pub fn effective_j2(mass_ratio: f64, a_planet_au: f64, r_ref_au: f64) -> f64 {
    0.5 * mass_ratio * (a_planet_au / r_ref_au).powi(2)
}

/// Combined effective field from orbit-averaging Jupiter, Saturn, and
/// Uranus. Uses a reference radius of 1 AU.
///
/// Returns `(j2_eff, j4_eff, gm_boost)` where `gm_boost = Σ GM_JSU`
/// (AU³/day²) is the monopole that must be added to the central GM.
/// Absorbing a planet into the central body requires this monopole boost;
/// without it mean motions and secular frequencies are systematically off
/// by ~6e-4.
///
/// Validity: the J2+J4 ring expansion is in powers of (a_planet/r)². With
/// a_Uranus = 19.2 AU it converges slowly for perihelia q ≲ 30 AU — at
/// q ≈ 30 AU the J6 term is still at the percent level of J4. Do not use
/// this approximation for Neptune-crossing perihelia; integrate Neptune
/// directly instead.
pub fn combined_j2_giants() -> (f64, f64, f64) {
    let r_ref = 1.0; // AU

    let j2_jup = effective_j2(MASS_JUPITER_SOLAR, 5.2026, r_ref);
    let j2_sat = effective_j2(MASS_SATURN_SOLAR, 9.5549, r_ref);
    let j2_ura = effective_j2(MASS_URANUS_SOLAR, 19.2184, r_ref);

    let j2_total = j2_jup + j2_sat + j2_ura;

    // Effective J4 from the same averaging (second order)
    // J4_eff = -(3/8) * (m/M) * (a/R)^4
    let j4_jup = -0.375 * MASS_JUPITER_SOLAR * (5.2026 / r_ref).powi(4);
    let j4_sat = -0.375 * MASS_SATURN_SOLAR * (9.5549 / r_ref).powi(4);
    let j4_ura = -0.375 * MASS_URANUS_SOLAR * (19.2184 / r_ref).powi(4);

    let j4_total = j4_jup + j4_sat + j4_ura;

    // Monopole: the averaged planets' mass joins the central body.
    let gm_boost = GM_JUPITER + GM_SATURN + GM_URANUS;

    (j2_total, j4_total, gm_boost)
}

/// Perturbation acceleration of the J2/J4-averaged planets, relative to bare
/// Keplerian motion about `gm_central`.
///
/// Includes both the ring field (J2/J4 of the boosted central mass) and the
/// extra monopole `-gm_boost * r / r³`. Apply this in the kick stage of an
/// integrator whose Kepler drift uses `gm_central` alone.
pub fn secular_j2_acceleration(
    pos: &Vector3<f64>,
    gm_central: f64,
    j2: f64,
    j4: f64,
    gm_boost: f64,
    r_ref: f64,
) -> Vector3<f64> {
    let r_mag = pos.norm();
    if r_mag < 1e-30 {
        return Vector3::zeros();
    }

    let gm_eff = gm_central + gm_boost;
    let mut accel = j2_acceleration(pos, gm_eff, r_ref, j2, Some(j4));
    accel -= gm_boost * pos / r_mag.powi(3);
    accel
}

// ===================================================================
// Galactic tide
// ===================================================================

/// Local Milky Way disk density in M_sun/AU³ (0.1 M_sun/pc³ converted).
pub const RHO_MW_MSUN_AU3: f64 = 0.1 / {
    let pc_km = 3.085_677_6e13_f64;
    let ratio = pc_km / AU_KM;
    ratio * ratio * ratio
};

/// Ecliptic-J2000 → galactic rotation, assembled from framelib's SPICE
/// frame table: (ICRF → GALACTIC) · (ecliptic → ICRF). Both factors are
/// static (the GALACTIC frame ignores the epoch argument).
static ECL_TO_GAL: Lazy<Matrix3<f64>> = Lazy::new(|| {
    let eq_to_gal = GALACTIC.rotation_at(&Timescale::default().tt_jd(J2000, None));
    eq_to_gal * *ECLIPJ2000_MATRIX_INV
});

/// Unit vector toward the north galactic pole in ecliptic J2000 coordinates
/// (the galactic z-axis row of the ecliptic→galactic rotation).
///
/// λ ≈ 180.023°, β ≈ +29.812°; the complement of β gives the ≈60.19°
/// ecliptic–galactic obliquity.
pub fn galactic_pole_ecliptic() -> Vector3<f64> {
    ECL_TO_GAL.row(2).transpose()
}

/// Galactic tidal acceleration on a particle at position `pos`
/// (AU, heliocentric ecliptic J2000), for local disk density `rho_msun_au3`.
///
/// The dominant component is the vertical tide from the Milky Way disk,
///   a_z̃ = -4π G ρ z̃,
/// where z̃ is the height above the *galactic* mid-plane. The galactic plane
/// is inclined ~60.2° to the ecliptic, so the position is rotated into the
/// galactic frame (via framelib's SPICE `GALACTIC` frame), the vertical disk
/// tide is applied along galactic z, and the result is rotated back —
/// applying it along ecliptic z would misdirect the torque by the full
/// obliquity.
///
/// The radial tide terms (Heisler & Tremaine's G1, G2, from the galactic
/// rotation curve) are ~10x weaker than the disk term at the Sun's location
/// and are omitted; only the G3 (vertical) term is applied. The tide is
/// important for a ≳ 1000 AU (inner Oort cloud) and dominant beyond
/// ~10,000 AU.
///
/// Reference: Heisler & Tremaine (1986), Nesvorny et al. (2017)
pub fn galactic_tide_acceleration(pos: &Vector3<f64>, rho_msun_au3: f64) -> Vector3<f64> {
    // 4*pi*G*rho in day^-2:
    // G = GM_SUN / M_sun (AU^3/(M_sun*day^2)), rho in M_sun/AU^3.
    let four_pi_g_rho = 4.0 * std::f64::consts::PI * GM_SUN * rho_msun_au3;

    let pos_gal = *ECL_TO_GAL * pos;
    let acc_gal = Vector3::new(0.0, 0.0, -four_pi_g_rho * pos_gal.z);

    ECL_TO_GAL.transpose() * acc_gal
}

// ===================================================================
// ExtraForce
// ===================================================================

/// A user-supplied acceleration field a(r) in AU/day², heliocentric.
pub type CustomForce = Arc<dyn Fn(&Vector3<f64>) -> Vector3<f64> + Send + Sync>;

/// Extra acceleration applied during the kick stage of an integrator.
///
/// These are position-dependent perturbations (potential kicks), so they
/// preserve the symplectic structure of the Wisdom-Holman splitting. They
/// are applied to massive bodies and test particles alike, in heliocentric
/// ecliptic-J2000 coordinates.
#[derive(Clone)]
pub enum ExtraForce {
    /// Orbit-averaged planets as an effective J2/J4 ring field plus the
    /// monopole boost `gm_boost` on the central body (see
    /// [`secular_j2_acceleration`]). Use [`ExtraForce::j2_giants`] for the
    /// standard Jupiter+Saturn+Uranus field.
    J2Averaged {
        /// Effective J2 at the reference radius.
        j2: f64,
        /// Effective J4 at the reference radius.
        j4: f64,
        /// Monopole added to the central GM (AU³/day²).
        gm_boost: f64,
        /// Reference radius of the J2/J4 expansion (AU).
        r_ref_au: f64,
    },
    /// Milky Way vertical disk tide ([`galactic_tide_acceleration`] with the
    /// local density [`RHO_MW_MSUN_AU3`]), evaluated in the galactic frame.
    GalacticTide,
    /// User-supplied acceleration field a(r) in AU/day², heliocentric.
    Custom(CustomForce),
}

impl ExtraForce {
    /// Orbit-averaged Jupiter+Saturn+Uranus quadrupole with the ΣGM_JSU
    /// monopole boost ([`combined_j2_giants`]). Valid for perihelia
    /// q ≳ 30 AU; integrate Neptune directly.
    pub fn j2_giants() -> Self {
        let (j2, j4, gm_boost) = combined_j2_giants();
        ExtraForce::J2Averaged {
            j2,
            j4,
            gm_boost,
            r_ref_au: 1.0,
        }
    }

    /// Acceleration at heliocentric position `pos` (AU), in AU/day².
    pub fn acceleration(&self, pos: &Vector3<f64>) -> Vector3<f64> {
        match self {
            ExtraForce::J2Averaged {
                j2,
                j4,
                gm_boost,
                r_ref_au,
            } => secular_j2_acceleration(pos, GM_SUN, *j2, *j4, *gm_boost, *r_ref_au),
            ExtraForce::GalacticTide => galactic_tide_acceleration(pos, RHO_MW_MSUN_AU3),
            ExtraForce::Custom(f) => f(pos),
        }
    }
}

impl std::fmt::Debug for ExtraForce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtraForce::J2Averaged {
                j2,
                j4,
                gm_boost,
                r_ref_au,
            } => write!(
                f,
                "ExtraForce::J2Averaged {{ j2: {j2:e}, j4: {j4:e}, gm_boost: {gm_boost:e}, \
                 r_ref_au: {r_ref_au} }}"
            ),
            ExtraForce::GalacticTide => write!(f, "ExtraForce::GalacticTide"),
            ExtraForce::Custom(_) => write!(f, "ExtraForce::Custom(..)"),
        }
    }
}

/// Total extra acceleration from a list of extra forces.
pub fn total_extra_acceleration(forces: &[ExtraForce], pos: &Vector3<f64>) -> Vector3<f64> {
    let mut accel = Vector3::zeros();
    for force in forces {
        accel += force.acceleration(pos);
    }
    accel
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::RAD2DEG;
    use approx::assert_relative_eq;

    #[test]
    fn test_perturbation_self_consistency() {
        // Two bodies on opposite sides of the Sun.
        let r = Vector3::new(5.0, 0.0, 0.0);
        let r_pert = Vector3::new(-5.0, 0.0, 0.0);
        let gm = 1e-4;

        let accel = perturbation_acceleration(&r, &r_pert, gm);

        // direct: -gm * (r-r_pert)/|r-r_pert|^3 = -gm * (10,0,0)/1000
        // indirect: -gm * r_pert/|r_pert|^3 = -gm * (-5,0,0)/125
        // ax = -gm*(10/1000 - 5/125) = -gm*(0.01 - 0.04) = +0.03*gm
        // Positive x (away from the perturber): the indirect term dominates
        // here, accounting for the Sun being accelerated toward the
        // perturber.
        assert!(accel.x > 0.0);
        assert_relative_eq!(accel.x, gm * 0.03, epsilon = 1e-10);
    }

    #[test]
    fn test_j2_equatorial_vs_polar() {
        // J2 acceleration should differ at the pole vs the equator.
        let gm = GM_SUN;
        let r_eq = Vector3::new(1.0, 0.0, 0.0);
        let r_pole = Vector3::new(0.0, 0.0, 1.0);

        let a_eq = j2_acceleration(&r_eq, gm, 0.00465, 0.01, None);
        let a_pole = j2_acceleration(&r_pole, gm, 0.00465, 0.01, None);

        // At the equator (z=0), J2 makes gravity stronger (pulls inward);
        // at the pole (z=r), J2 makes gravity weaker (pushes outward).
        assert!(a_eq.x < 0.0);
        assert!(a_pole.z > 0.0);
    }

    #[test]
    fn test_gm_boost_magnitude() {
        // ΣGM_JSU ≈ 1.28e-3 GM_sun
        let (_, _, gm_boost) = combined_j2_giants();
        let ratio = gm_boost / GM_SUN;
        assert!(
            (ratio - 1.28e-3).abs() < 5e-5,
            "GM boost ratio {ratio:.3e} should be ~1.28e-3"
        );
    }

    #[test]
    fn test_secular_acceleration_dominated_by_monopole_at_large_r() {
        // Far from the rings the perturbation tends to the extra monopole.
        let (j2, j4, gm_boost) = combined_j2_giants();
        let pos = Vector3::new(500.0, 0.0, 0.0);
        let a = secular_j2_acceleration(&pos, GM_SUN, j2, j4, gm_boost, 1.0);
        let monopole = gm_boost / (500.0_f64 * 500.0);
        assert!((a.norm() - monopole).abs() / monopole < 1e-3);
        // Pointing inward (extra attraction).
        assert!(a.x < 0.0);
    }

    #[test]
    fn test_galactic_pole_is_unit_and_inclined_60deg() {
        let n = galactic_pole_ecliptic();
        assert_relative_eq!(n.norm(), 1.0, epsilon = 1e-12);
        // Obliquity between galactic plane and ecliptic: acos(n.z) ~ 60.2 deg
        let obliquity = n.z.acos() * RAD2DEG;
        assert!((obliquity - 60.19).abs() < 0.1, "obliquity {obliquity}");
    }

    #[test]
    fn test_tide_restoring_toward_galactic_plane() {
        // A point displaced along the galactic pole feels a pure restoring
        // force.
        let n = galactic_pole_ecliptic();
        let pos = 10_000.0 * n;
        let a = galactic_tide_acceleration(&pos, RHO_MW_MSUN_AU3);
        // Anti-parallel to displacement
        assert!(a.dot(&pos) < 0.0);
        assert_relative_eq!(a.cross(&pos).norm(), 0.0, epsilon = 1e-20);
    }

    #[test]
    fn test_tide_vanishes_in_galactic_plane() {
        // Any vector orthogonal to the pole sits in the galactic mid-plane.
        let n = galactic_pole_ecliptic();
        let in_plane = n.cross(&Vector3::new(0.0, 0.0, 1.0)).normalize() * 10_000.0;
        let a = galactic_tide_acceleration(&in_plane, RHO_MW_MSUN_AU3);
        assert!(a.norm() < 1e-25);
    }

    #[test]
    fn test_tide_matches_pole_projection_form() {
        // The rotate-apply-rotate-back implementation must equal the
        // algebraically collapsed form a = -4πGρ (r·n̂) n̂.
        let four_pi_g_rho = 4.0 * std::f64::consts::PI * GM_SUN * RHO_MW_MSUN_AU3;
        let n = galactic_pole_ecliptic();
        let pos = Vector3::new(3_000.0, -7_000.0, 5_000.0);
        let expected = -four_pi_g_rho * pos.dot(&n) * n;
        let a = galactic_tide_acceleration(&pos, RHO_MW_MSUN_AU3);
        assert_relative_eq!((a - expected).norm(), 0.0, epsilon = 1e-30);
    }

    #[test]
    fn test_custom_force_composes() {
        let drag = ExtraForce::Custom(Arc::new(|pos: &Vector3<f64>| -1e-12 * pos));
        let forces = vec![drag, ExtraForce::GalacticTide];
        let pos = Vector3::new(100.0, 0.0, 0.0);
        let total = total_extra_acceleration(&forces, &pos);
        let parts = forces[0].acceleration(&pos) + forces[1].acceleration(&pos);
        assert_relative_eq!((total - parts).norm(), 0.0, epsilon = 1e-30);
    }
}
