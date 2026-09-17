//! Democratic heliocentric coordinate system for the Wisdom-Holman integrator.
//!
//! In this splitting, the Hamiltonian is decomposed as:
//!   H = H_Kepler + H_interaction + H_sun
//!
//! H_Kepler: each body (planet or test particle) orbits the Sun in a
//!           Keplerian ellipse
//! H_interaction: planet-planet and planet-particle gravitational
//!           perturbations
//! H_sun:    |Σpᵢ|²/(2 m_sun), the solar-drift substep
//!
//! Coordinates are heliocentric positions with barycentric momenta
//! (velocities). This is the coordinate system used by mercury6
//! (Chambers 1999); see Duncan, Levison & Lee (1998).
//!
//! Masses are in solar units throughout (`m_sun = 1.0` for a solar-mass
//! central body).

use nalgebra::Vector3;

use super::types::{MassiveBody, StateVector};

/// Convert from heliocentric positions + velocities to democratic heliocentric.
/// In DH coords:
///   - Positions remain heliocentric
///   - Momenta (velocities) become barycentric:
///     v_DH_i = v_helio_i - v_bary
///
/// where v_bary = sum(m_i * v_i) / M_total (including the Sun at rest in the
/// heliocentric frame).
///
/// Returns the barycentric velocity of the system.
pub fn helio_to_democratic(
    bodies: &mut [MassiveBody],
    particles: &mut [StateVector],
    m_sun: f64,
) -> Vector3<f64> {
    // Compute total momentum in heliocentric frame
    let mut total_momentum = Vector3::zeros();
    let mut total_mass = m_sun; // Sun is at rest in heliocentric frame

    for b in bodies.iter() {
        total_momentum += b.mass * b.state.vel;
        total_mass += b.mass;
    }
    // Test particles are massless, don't contribute to momentum

    let v_bary = total_momentum / total_mass;

    // Transform velocities
    for b in bodies.iter_mut() {
        b.state.vel -= v_bary;
    }
    for p in particles.iter_mut() {
        p.vel -= v_bary;
    }

    v_bary
}

/// Convert from democratic heliocentric back to heliocentric.
pub fn democratic_to_helio(
    bodies: &mut [MassiveBody],
    particles: &mut [StateVector],
    v_bary: &Vector3<f64>,
) {
    for b in bodies.iter_mut() {
        b.state.vel += v_bary;
    }
    for p in particles.iter_mut() {
        p.vel += v_bary;
    }
}

/// Compute the barycentric correction velocity from current body states.
pub fn compute_v_bary(bodies: &[MassiveBody], m_sun: f64) -> Vector3<f64> {
    let mut total_momentum = Vector3::zeros();
    let mut total_mass = m_sun;

    for b in bodies {
        total_momentum += b.mass * b.state.vel;
        total_mass += b.mass;
    }

    total_momentum / total_mass
}

/// Inverse-transform correction for states currently in DH coordinates.
///
/// With barycentric (DH) velocities, momentum conservation fixes the Sun's
/// barycentric velocity to −Σmᵢvᵢ/m_sun, so heliocentric velocities are
/// recovered as v_helio = v_DH + Σmᵢvᵢ/m_sun. (At the instant of the forward
/// transform this equals `compute_v_bary` of the heliocentric velocities, but
/// it must be re-evaluated after the velocities have evolved.)
pub fn dh_velocity_correction(bodies: &[MassiveBody], m_sun: f64) -> Vector3<f64> {
    let mut total_momentum = Vector3::zeros();
    for b in bodies {
        total_momentum += b.mass * b.state.vel;
    }
    total_momentum / m_sun
}

/// Solar-drift substep of the democratic heliocentric scheme (DLL98).
///
/// H_sun = |Σpᵢ|²/(2 m_sun) advances every heliocentric position (bodies and
/// test particles alike) by dt·Σmᵢvᵢ/m_sun, with vᵢ the DH (barycentric)
/// velocities. Omitting this substep is what makes a naive heliocentric
/// "WHM" non-symplectic.
pub fn solar_drift(
    bodies: &mut [MassiveBody],
    particles: &mut [StateVector],
    active: &[bool],
    dt: f64,
    m_sun: f64,
) {
    let shift = dt * dh_velocity_correction(bodies, m_sun);
    for b in bodies.iter_mut() {
        b.state.pos += shift;
    }
    for (i, p) in particles.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }
        p.pos += shift;
    }
}
