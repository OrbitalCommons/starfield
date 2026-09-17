//! Mercury6-style hybrid integrator (Chambers 1999).
//!
//! Uses the Wisdom-Holman splitting for the majority of particles, switching
//! to Bulirsch-Stoer for any particle predicted to pass near a planet during
//! the step.
//!
//! The planet force on each particle is partitioned with the Chambers
//! changeover function K(d) so that no force is double-counted:
//!
//! - the **kick** stage applies `K(d) * F_direct` plus the full indirect
//!   (heliocentric-frame) term,
//! - the **drift** stage of an encountering particle integrates
//!   Sun + `(1 - K(d)) * F_direct` with Bulirsch-Stoer, Kepler-drifting the
//!   planets to each substep epoch.
//!
//! Far from planets K = 1 and the scheme reduces to plain WHM; deep inside an
//! encounter K = 0 and the BS step carries the full planet force.
//!
//! Encounter detection uses the predicted minimum separation over the whole
//! step [t, t+dt] (linear relative motion), not just the step start, so a
//! particle cannot cross a Hill sphere entirely between checks.
//!
//! The changeover radius is `changeover * r_Hill` with
//! r_Hill = r_planet * (m_planet / 3 M_sun)^(1/3), changeover typically 3.0.
//!
//! Note: this driver uses the heliocentric-velocity splitting (kick includes
//! the indirect term), not the democratic-heliocentric scheme of
//! [`super::whm::WhmIntegrator`]; for encounter work the BS accuracy
//! dominates the error budget.
//!
//! Reference: Chambers (1999), "A hybrid symplectic integrator that permits
//! close encounters between massive bodies"

use nalgebra::Vector3;

use super::bulirsch_stoer::{bs_step, changeover_k};
use super::forces::kick_bodies;
use super::kepler::kepler_drift;
use super::types::{MassiveBody, SimConfig, StateVector};
use super::GM_SUN;

/// Changeover radius (AU) for each body: `changeover * r_Hill`.
pub fn changeover_radii(bodies: &[MassiveBody], changeover_hill: f64) -> Vec<f64> {
    bodies
        .iter()
        .map(|body| {
            let r_planet = body.state.pos.norm();
            changeover_hill * r_planet * (body.mass / 3.0).cbrt()
        })
        .collect()
}

/// Predicted minimum particle-body separation over [0, dt] assuming linear
/// relative motion: min over s of |dr0 + s*dv0|, s clamped to [0, dt].
fn min_separation(particle: &StateVector, body: &MassiveBody, dt: f64) -> f64 {
    let dr = particle.pos - body.state.pos;
    let dv = particle.vel - body.state.vel;
    let dv2 = dv.norm_squared();
    let s = if dv2 > 1e-30 {
        (-dr.dot(&dv) / dv2).clamp(0.0, dt)
    } else {
        0.0
    };
    (dr + s * dv).norm()
}

/// Determine which particles need Bulirsch-Stoer integration this step.
///
/// A particle is flagged when its predicted minimum separation from any body
/// over [t, t+dt] (linear relative motion) falls inside that body's
/// changeover radius. Sampling only the step start would let particles cross
/// the encounter sphere entirely between checks.
///
/// Returns a boolean mask (true = needs BS).
pub fn find_close_encounters(
    particles: &[StateVector],
    active: &[bool],
    bodies: &[MassiveBody],
    r_crit: &[f64],
    dt: f64,
) -> Vec<bool> {
    let mut needs_bs = vec![false; particles.len()];

    for (i, particle) in particles.iter().enumerate() {
        if !active[i] {
            continue;
        }
        for (body, &rc) in bodies.iter().zip(r_crit) {
            if min_separation(particle, body, dt) < rc {
                needs_bs[i] = true;
                break;
            }
        }
    }

    needs_bs
}

/// Kick stage for particles with the Chambers force partition: each body
/// contributes `K(d) * direct + indirect`. Far from every body this is the
/// standard full perturbation kick.
fn kick_particles_changeover(
    particles: &mut [StateVector],
    active: &[bool],
    bodies: &[MassiveBody],
    r_crit: &[f64],
    dt: f64,
) {
    for (i, particle) in particles.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }
        let mut accel = Vector3::zeros();
        for (body, &rc) in bodies.iter().zip(r_crit) {
            let dr = particle.pos - body.state.pos;
            let dr_mag = dr.norm();
            let rp = body.state.pos.norm();
            if dr_mag > 1e-30 {
                accel -= changeover_k(dr_mag, rc) * body.gm * dr / dr_mag.powi(3);
            }
            if rp > 1e-30 {
                // Indirect term: always full weight (frame correction, not a
                // particle-planet interaction).
                accel -= body.gm * body.state.pos / rp.powi(3);
            }
        }
        particle.vel += dt * accel;
    }
}

/// Perform one hybrid step: WHM for most particles, BS for close encounters.
///
/// Returns the number of particles whose Bulirsch-Stoer step failed to
/// converge even after the maximum number of halvings (0 = all clean).
pub fn hybrid_step(
    bodies: &mut [MassiveBody],
    particles: &mut [StateVector],
    active: &mut [bool],
    dt: f64,
    config: &SimConfig,
) -> usize {
    let half_dt = 0.5 * dt;

    // Changeover radii and encounter mask, evaluated at the step start with
    // predicted minimum separations over [t, t+dt].
    let r_crit = changeover_radii(bodies, config.hybrid_changeover_hill);
    let needs_bs = find_close_encounters(particles, active, bodies, &r_crit, dt);

    // === KICK dt/2 ===
    kick_bodies(bodies, half_dt);
    kick_particles_changeover(particles, active, bodies, &r_crit, half_dt);

    // Snapshot body states for the BS force evaluation (BS drifts them to the
    // substep epochs internally).
    let bodies_at_start: Vec<MassiveBody> = bodies.to_vec();

    // === DRIFT dt ===
    // Bodies always use Kepler drift
    for body in bodies.iter_mut() {
        let gm_total = GM_SUN + body.gm;
        body.state = kepler_drift(&body.state, dt, gm_total);
    }

    // Particles: Kepler for most, BS (with the complementary force share and
    // moving planets) for close encounters.
    let mut failed = 0;
    for (i, particle) in particles.iter_mut().enumerate() {
        if !active[i] {
            continue;
        }

        if needs_bs[i] {
            let (new_state, converged) =
                bs_step(particle, &bodies_at_start, &r_crit, dt, config.bs_epsilon);
            if !converged {
                failed += 1;
            }
            *particle = new_state;
        } else {
            *particle = kepler_drift(particle, dt, GM_SUN);
        }
    }

    // === KICK dt/2 ===
    kick_bodies(bodies, half_dt);
    kick_particles_changeover(particles, active, bodies, &r_crit, half_dt);

    // === BOUNDARY CHECK ===
    for (i, particle) in particles.iter().enumerate() {
        if !active[i] {
            continue;
        }
        let r = particle.pos.norm();
        if r < config.removal_inner_au || r > config.removal_outer_au {
            active[i] = false;
        }
    }

    failed
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_config(dt: f64) -> SimConfig {
        SimConfig {
            dt,
            removal_inner_au: 0.01,
            removal_outer_au: 1e6,
            hybrid_changeover_hill: 3.0,
            bs_epsilon: 1e-12,
        }
    }

    fn jupiter() -> MassiveBody {
        let gm = 1e-3 * GM_SUN;
        let a = 5.2;
        let v = (GM_SUN / a).sqrt();
        MassiveBody::new(
            "Jupiter",
            gm,
            StateVector::new(Vector3::new(a, 0.0, 0.0), Vector3::new(0.0, v, 0.0)),
        )
    }

    #[test]
    fn test_encounter_detected_mid_step() {
        // Particle currently outside the changeover sphere but crossing it
        // during the step: the start-only check would miss this.
        let jup = jupiter();
        let r_crit = changeover_radii(std::slice::from_ref(&jup), 3.0);
        assert!(r_crit[0] > 1.0 && r_crit[0] < 1.2, "r_crit = {}", r_crit[0]);

        let dt = 300.0;
        // 3 AU "below" Jupiter, closing at 0.02 AU/day relative: passes
        // within ~0 AU mid-step.
        let particle = StateVector::new(
            Vector3::new(5.2, -3.0, 0.0),
            jup.state.vel + Vector3::new(0.0, 0.02, 0.0),
        );

        let flagged = find_close_encounters(
            &[particle],
            &[true],
            std::slice::from_ref(&jup),
            &r_crit,
            dt,
        );
        assert!(flagged[0], "mid-step crossing must be flagged");

        // Start-distance check alone would say "no encounter":
        let d_start = (particle.pos - jup.state.pos).norm();
        assert!(d_start > r_crit[0]);
    }

    #[test]
    fn test_hybrid_matches_kepler_far_from_planets() {
        // A distant particle (no encounter) must follow its Kepler orbit.
        let mut bodies = vec![jupiter()];
        let a = 60.0;
        let v = (GM_SUN / a).sqrt();
        let p0 = StateVector::new(Vector3::new(a, 0.0, 0.0), Vector3::new(0.0, v, 0.0));
        let mut particles = vec![p0];
        let mut active = vec![true];
        let config = test_config(100.0);

        for _ in 0..100 {
            let failed = hybrid_step(&mut bodies, &mut particles, &mut active, config.dt, &config);
            assert_eq!(failed, 0);
        }

        // Compare against pure Kepler + Jupiter perturbation: at 60 AU the
        // perturbation is tiny, so the orbit radius is preserved well.
        let r = particles[0].pos.norm();
        assert_relative_eq!(r, a, epsilon = 0.05);
    }

    #[test]
    fn test_hybrid_energy_through_encounter() {
        // Particle on an orbit crossing Jupiter's: total (Sun + Jupiter)
        // specific energy in the rotating-free frame is not conserved, but
        // the sanity check is that the integration neither blows up nor
        // silently fails.
        let mut bodies = vec![jupiter()];
        // Eccentric orbit with perihelion near Jupiter's orbit
        let q = 5.0;
        let a = 20.0;
        let v_peri = (GM_SUN * (2.0 / q - 1.0 / a)).sqrt();
        let p0 = StateVector::new(Vector3::new(q, 0.0, 0.0), Vector3::new(0.0, v_peri, 0.0));
        let mut particles = vec![p0];
        let mut active = vec![true];
        let config = test_config(150.0);

        let mut total_failed = 0;
        for _ in 0..500 {
            total_failed +=
                hybrid_step(&mut bodies, &mut particles, &mut active, config.dt, &config);
        }
        assert_eq!(total_failed, 0, "BS should converge through encounters");
        assert!(active[0], "particle should survive");
        let r = particles[0].pos.norm();
        assert!(r.is_finite() && r > 0.1 && r < 1e4, "r = {r}");
    }
}
