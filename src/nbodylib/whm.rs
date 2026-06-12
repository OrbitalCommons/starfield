//! Wisdom-Holman Mapping (WHM) symplectic integrator in democratic
//! heliocentric coordinates (Duncan, Levison & Lee 1998).
//!
//! The Hamiltonian splits into three commuting-flow pieces:
//!   H_Kepler: each body on a Keplerian orbit about GM_sun
//!             (heliocentric positions, barycentric momenta)
//!   H_int:    direct planet-planet / planet-particle interactions
//!   H_sun:    |Σpᵢ|²/(2 m_sun) — the solar-drift substep
//!
//! and the second-order step is
//!   kick(dt/2) · solar_drift(dt/2) · Kepler(dt) · solar_drift(dt/2) · kick(dt/2).
//!
//! Heliocentric positions with *barycentric* momenta are canonical, so this
//! splitting preserves a shadow Hamiltonian and gives bounded energy error
//! over arbitrarily long integrations. (Heliocentric positions with
//! heliocentric velocities are not canonical and drift secularly at
//! O((m/M)²) per step.)
//!
//! Public state remains heliocentric positions + heliocentric velocities; the
//! barycentric transform is applied internally per step (it is exact, so this
//! does not break symplecticity).
//!
//! Reference: Wisdom & Holman (1991), Duncan, Levison & Lee (1998),
//! Chambers (1999)

use nalgebra::Vector3;

use crate::constants::TAU as TWO_PI;

use super::democratic::{
    democratic_to_helio, dh_velocity_correction, helio_to_democratic, solar_drift,
};
use super::forces::{kick_bodies_direct, kick_particles_direct, total_extra_acceleration};
use super::kepler::kepler_drift;
use super::types::{MassiveBody, SimConfig, StateVector};
use super::{ExtraForce, GM_SUN};

/// Minimum number of steps per orbit of the innermost massive body needed for
/// the WHM kick to resolve the interaction Hamiltonian (Wisdom & Holman 1991).
pub const MIN_STEPS_PER_ORBIT: f64 = 20.0;

/// Validate the timestep against the innermost massive body's orbital period.
///
/// WHM requires dt ≲ P_inner / 20; e.g. with Jupiter integrated directly
/// (P ≈ 4333 d) the timestep must be ≲ 215 d. Coarser steps (e.g. 300 d)
/// are only valid when the inner giants are absorbed into the J2-averaged
/// field ([`ExtraForce::j2_giants`]) and Neptune is the innermost direct
/// body.
///
/// Returns Err with a description when the timestep is too coarse.
pub fn validate_timestep(bodies: &[MassiveBody], dt: f64) -> Result<(), String> {
    for body in bodies {
        let r = body.state.pos.norm();
        let v2 = body.state.vel.norm_squared();
        let energy = 0.5 * v2 - GM_SUN / r.max(1e-30);
        if energy >= 0.0 {
            continue; // unbound body: no period to resolve
        }
        let a = -GM_SUN / (2.0 * energy);
        let period = TWO_PI * (a.powi(3) / GM_SUN).sqrt();
        let max_dt = period / MIN_STEPS_PER_ORBIT;
        if dt > max_dt {
            return Err(format!(
                "dt = {:.1} d under-resolves {} (P = {:.1} d; need dt <= P/{} = {:.1} d)",
                dt, body.name, period, MIN_STEPS_PER_ORBIT, max_dt
            ));
        }
    }
    Ok(())
}

/// The WHM integrator. Stateless apart from its extra-force list; the system
/// being integrated is passed to [`WhmIntegrator::step`].
#[derive(Debug, Default, Clone)]
pub struct WhmIntegrator {
    /// Extra accelerations applied in the kick stage (J2-averaged giants,
    /// galactic tide, custom fields). Empty by default.
    pub extra_forces: Vec<ExtraForce>,
}

impl WhmIntegrator {
    /// Plain WHM with no extra kick-stage forces.
    pub fn new() -> Self {
        Self {
            extra_forces: Vec::new(),
        }
    }

    /// Convenience constructor with extra kick-stage forces.
    pub fn with_extra_forces(extra_forces: Vec<ExtraForce>) -> Self {
        Self { extra_forces }
    }

    /// Interaction kick: direct-only gravity plus any extra forces, applied
    /// to bodies and particles for time dt.
    fn kick(
        &self,
        bodies: &mut [MassiveBody],
        particles: &mut [StateVector],
        active: &[bool],
        dt: f64,
    ) {
        kick_bodies_direct(bodies, dt);
        kick_particles_direct(particles, active, bodies, dt);

        if !self.extra_forces.is_empty() {
            for body in bodies.iter_mut() {
                body.state.vel +=
                    dt * total_extra_acceleration(&self.extra_forces, &body.state.pos);
            }
            for (i, particle) in particles.iter_mut().enumerate() {
                if !active[i] {
                    continue;
                }
                particle.vel += dt * total_extra_acceleration(&self.extra_forces, &particle.pos);
            }
        }
    }

    /// Perform a single WHM step (democratic heliocentric scheme).
    ///
    /// `bodies`: massive bodies (planets, hypothetical perturbers),
    ///           heliocentric pos/vel
    /// `particles`: test particle state vectors, heliocentric pos/vel
    /// `active`: which particles are still active
    /// `dt`: timestep in days (must resolve the innermost body, see
    ///       [`validate_timestep`]; checked via debug_assert)
    pub fn step(
        &self,
        bodies: &mut [MassiveBody],
        particles: &mut [StateVector],
        active: &mut [bool],
        dt: f64,
        config: &SimConfig,
    ) {
        debug_assert!(
            validate_timestep(bodies, dt).is_ok(),
            "{}",
            validate_timestep(bodies, dt).err().unwrap_or_default()
        );

        let half_dt = 0.5 * dt;

        // Heliocentric velocities -> barycentric (canonical DH momenta).
        helio_to_democratic(bodies, particles, 1.0);

        // === KICK dt/2 (direct interactions + extra forces) ===
        self.kick(bodies, particles, active, half_dt);

        // === SOLAR DRIFT dt/2 ===
        solar_drift(bodies, particles, active, half_dt, 1.0);

        // === KEPLER DRIFT dt about GM_sun ===
        // In DH variables the Kepler Hamiltonian is p²/2m − GM_sun·m/r for
        // every body, so the drift central mass is GM_sun (not GM_sun + gm).
        for body in bodies.iter_mut() {
            body.state = kepler_drift(&body.state, dt, GM_SUN);
        }
        for (i, particle) in particles.iter_mut().enumerate() {
            if !active[i] {
                continue;
            }
            *particle = kepler_drift(particle, dt, GM_SUN);
        }

        // === SOLAR DRIFT dt/2 ===
        solar_drift(bodies, particles, active, half_dt, 1.0);

        // === KICK dt/2 ===
        self.kick(bodies, particles, active, half_dt);

        // Barycentric velocities -> heliocentric (re-evaluated: the momentum
        // sum changes during the Kepler drift).
        let v_back = dh_velocity_correction(bodies, 1.0);
        democratic_to_helio(bodies, particles, &v_back);

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
    }
}

/// Compute total energy of the system in *heliocentric* coordinates
/// (for diagnostics). Returns (kinetic, potential, total).
///
/// Note: this is not the conserved quantity — heliocentric velocities omit
/// the barycentric correction and active test particles pollute the budget.
/// Use [`barycentric_energy`] / [`total_angular_momentum`] for conservation
/// checks and [`particle_energies`] for per-particle diagnostics.
pub fn system_energy(
    bodies: &[MassiveBody],
    particles: &[StateVector],
    active: &[bool],
) -> (f64, f64, f64) {
    let mut kinetic = 0.0;
    let mut potential = 0.0;

    // Body kinetic energy (relative to Sun)
    for body in bodies {
        kinetic += 0.5 * body.mass * body.state.vel.norm_squared();
    }

    // Body-Sun potential
    for body in bodies {
        let r = body.state.pos.norm();
        if r > 0.0 {
            potential -= GM_SUN * body.mass / r;
        }
    }

    // Body-body potential
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let dr = (bodies[i].state.pos - bodies[j].state.pos).norm();
            if dr > 0.0 {
                potential -= bodies[i].gm * bodies[j].mass / dr;
            }
        }
    }

    // Particle kinetic energy (massless, but track for diagnostics using unit mass)
    for (i, p) in particles.iter().enumerate() {
        if !active[i] {
            continue;
        }
        kinetic += 0.5 * p.vel.norm_squared(); // unit mass
        let r = p.pos.norm();
        if r > 0.0 {
            potential -= GM_SUN / r;
            for body in bodies {
                let dr = (p.pos - body.state.pos).norm();
                if dr > 0.0 {
                    potential -= body.gm / dr;
                }
            }
        }
    }

    (kinetic, potential, kinetic + potential)
}

/// Barycentric total energy of the Sun + massive bodies, in
/// M_sun·AU²/day². Test particles are massless and excluded; see
/// [`particle_energies`] for their diagnostics.
///
/// This is the quantity a symplectic integrator conserves (up to bounded
/// oscillation); the heliocentric [`system_energy`] is not.
pub fn barycentric_energy(bodies: &[MassiveBody]) -> f64 {
    // Heliocentric Sun state: origin, at rest.
    let mut m_total = 1.0;
    let mut p_total = Vector3::zeros();
    for b in bodies {
        m_total += b.mass;
        p_total += b.mass * b.state.vel;
    }
    let v_bary = p_total / m_total;

    // Kinetic: bodies + Sun, in the barycentric frame.
    let mut kinetic = 0.5 * v_bary.norm_squared(); // Sun, mass 1, v = -v_bary
    for b in bodies {
        kinetic += 0.5 * b.mass * (b.state.vel - v_bary).norm_squared();
    }

    // Potential: Sun-body + body-body (gm = G * mass in these units).
    let mut potential = 0.0;
    for b in bodies {
        potential -= GM_SUN * b.mass / b.state.pos.norm();
    }
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let dr = (bodies[i].state.pos - bodies[j].state.pos).norm();
            potential -= bodies[i].gm * bodies[j].mass / dr;
        }
    }

    kinetic + potential
}

/// Barycentric total angular momentum of the Sun + massive bodies,
/// in M_sun·AU²/day. Exactly conserved by the DH splitting (each substep
/// Hamiltonian is rotation-invariant).
pub fn total_angular_momentum(bodies: &[MassiveBody]) -> Vector3<f64> {
    let mut m_total = 1.0;
    let mut p_total = Vector3::zeros();
    let mut mr_total = Vector3::zeros();
    for b in bodies {
        m_total += b.mass;
        p_total += b.mass * b.state.vel;
        mr_total += b.mass * b.state.pos;
    }
    let v_bary = p_total / m_total;
    let r_bary = mr_total / m_total;

    // Sun: heliocentric origin/rest -> barycentric (-r_bary, -v_bary).
    let mut l = (-r_bary).cross(&(-v_bary)); // mass 1
    for b in bodies {
        l += b.mass * (b.state.pos - r_bary).cross(&(b.state.vel - v_bary));
    }
    l
}

/// Specific orbital energies (AU²/day²) of the test particles in the
/// heliocentric frame, including the body potentials. Kept separate from the
/// planetary energy budget: particle "energy" is a per-particle diagnostic
/// (escape/capture flagging), not a conserved system quantity.
pub fn particle_energies(
    particles: &[StateVector],
    active: &[bool],
    bodies: &[MassiveBody],
) -> Vec<f64> {
    particles
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if !active[i] {
                return f64::NAN;
            }
            let mut e = 0.5 * p.vel.norm_squared() - GM_SUN / p.pos.norm();
            for body in bodies {
                let dr = (p.pos - body.state.pos).norm();
                if dr > 0.0 {
                    e -= body.gm / dr;
                }
            }
            e
        })
        .collect()
}

/// Compute the Jacobi constant for the circular restricted three-body problem.
/// Useful for testing. `mu` is the mass ratio m2/(m1+m2).
pub fn jacobi_constant(state: &StateVector, mu: f64, omega: f64) -> f64 {
    let x = state.pos.x;
    let y = state.pos.y;

    let r1 = ((x + mu).powi(2) + y.powi(2) + state.pos.z.powi(2)).sqrt();
    let r2 = ((x - 1.0 + mu).powi(2) + y.powi(2) + state.pos.z.powi(2)).sqrt();

    let v2 = state.vel.norm_squared();

    // Jacobi constant: C_J = -2E - 2 * omega * L_z in rotating frame
    // Or equivalently: C_J = (x^2 + y^2)*omega^2 + 2*(1-mu)/r1 + 2*mu/r2 - v^2
    omega * omega * (x * x + y * y) + 2.0 * (1.0 - mu) / r1 + 2.0 * mu / r2 - v2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::J2000;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::nbodylib::types::giant_planets_from_ephemeris;
    use crate::planetlib::Ephemeris;
    use crate::time::Timescale;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;

    fn test_config(dt: f64) -> SimConfig {
        SimConfig {
            dt,
            removal_inner_au: 0.1,
            removal_outer_au: 1e6,
            hybrid_changeover_hill: 3.0,
            bs_epsilon: 1e-12,
        }
    }

    fn jupiter_like() -> MassiveBody {
        let a = 5.2;
        let gm_planet = 1e-3 * GM_SUN;
        let v = (GM_SUN / a).sqrt();
        MassiveBody::new(
            "Jupiter",
            gm_planet,
            StateVector::new(Vector3::new(a, 0.0, 0.0), Vector3::new(0.0, v, 0.0)),
        )
    }

    fn saturn_like() -> MassiveBody {
        let a = 9.55;
        let gm_planet = 2.86e-4 * GM_SUN;
        let v = (GM_SUN / a).sqrt();
        MassiveBody::new(
            "Saturn",
            gm_planet,
            StateVector::new(Vector3::new(0.0, a, 0.0), Vector3::new(-v, 0.0, 0.0)),
        )
    }

    #[test]
    fn test_whm_two_body_energy_conservation() {
        let mut bodies = vec![jupiter_like()];
        let mut particles = vec![];
        let mut active = vec![];

        let config = test_config(100.0);
        let integrator = WhmIntegrator::new();
        let e0 = barycentric_energy(&bodies);

        let mut max_de: f64 = 0.0;
        for step in 0..1000 {
            integrator.step(&mut bodies, &mut particles, &mut active, config.dt, &config);
            if step % 50 == 0 {
                let e1 = barycentric_energy(&bodies);
                max_de = max_de.max(((e1 - e0) / e0).abs());
            }
        }

        // Symplectic integrator: bounded (not zero) energy oscillation.
        assert!(max_de < 1e-5, "Energy error too large: {:.2e}", max_de);
    }

    #[test]
    fn test_whm_two_planet_long_run_energy_bounded() {
        // 1e5 steps (~41 kyr): a non-symplectic splitting drifts secularly,
        // the DH scheme oscillates within a bounded band.
        let mut bodies = vec![jupiter_like(), saturn_like()];
        let mut particles = vec![];
        let mut active = vec![];

        let config = test_config(150.0);
        let integrator = WhmIntegrator::new();
        let e0 = barycentric_energy(&bodies);

        let mut max_de: f64 = 0.0;
        for step in 0..100_000 {
            integrator.step(&mut bodies, &mut particles, &mut active, config.dt, &config);
            if step % 1000 == 999 {
                let e1 = barycentric_energy(&bodies);
                max_de = max_de.max(((e1 - e0) / e0).abs());
            }
        }

        assert!(
            max_de < 1e-5,
            "Long-run energy error not bounded: {:.2e}",
            max_de
        );
    }

    #[test]
    fn test_whm_angular_momentum_conservation() {
        // Every substep Hamiltonian is rotation invariant, so the total
        // barycentric L is conserved to roundoff.
        let mut bodies = vec![jupiter_like(), saturn_like()];
        // Tilt Saturn a little so L has all three components.
        bodies[1].state.vel.z = 0.1 * bodies[1].state.vel.norm();

        let mut particles = vec![];
        let mut active = vec![];

        let config = test_config(150.0);
        let integrator = WhmIntegrator::new();
        let l0 = total_angular_momentum(&bodies);

        for _ in 0..10_000 {
            integrator.step(&mut bodies, &mut particles, &mut active, config.dt, &config);
        }

        let l1 = total_angular_momentum(&bodies);
        let dl = (l1 - l0).norm() / l0.norm();
        // Conserved up to accumulated roundoff (~1e-13/step random walk).
        assert!(dl < 1e-8, "Angular momentum drift: {:.2e}", dl);
    }

    #[test]
    fn test_whm_second_order_dt_scaling() {
        // The energy-error amplitude of a second-order symplectic map scales
        // as dt²: halving dt should reduce it by ~4x.
        let amplitude = |dt: f64| -> f64 {
            let mut bodies = vec![jupiter_like(), saturn_like()];
            let mut particles = vec![];
            let mut active = vec![];
            let config = test_config(dt);
            let integrator = WhmIntegrator::new();
            let e0 = barycentric_energy(&bodies);

            let n_steps = (40_000.0 / dt).round() as usize;
            let mut max_de: f64 = 0.0;
            for _ in 0..n_steps {
                integrator.step(&mut bodies, &mut particles, &mut active, dt, &config);
                let e1 = barycentric_energy(&bodies);
                max_de = max_de.max(((e1 - e0) / e0).abs());
            }
            max_de
        };

        let err_coarse = amplitude(200.0);
        let err_fine = amplitude(100.0);
        let ratio = err_coarse / err_fine;
        assert!(
            (2.5..6.0).contains(&ratio),
            "dt-scaling ratio {ratio:.2} not consistent with 2nd order (expect ~4)"
        );
    }

    #[test]
    fn test_whm_particle_on_circular_orbit() {
        // No planets, just a particle on a circular orbit
        let a = 30.0; // Neptune distance
        let v = (GM_SUN / a).sqrt();

        let mut bodies = vec![];
        let mut particles = vec![StateVector::new(
            Vector3::new(a, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
        )];
        let mut active = vec![true];

        let config = test_config(300.0);
        let integrator = WhmIntegrator::new();

        // One orbital period of Neptune
        let period = 2.0 * std::f64::consts::PI * (a * a * a / GM_SUN).sqrt();
        let n_steps = (period / config.dt).round() as usize;

        for _ in 0..n_steps {
            integrator.step(&mut bodies, &mut particles, &mut active, config.dt, &config);
        }

        // Should return close to starting position
        assert_relative_eq!(particles[0].pos.x, a, epsilon = 0.1);
        assert_relative_eq!(particles[0].pos.y, 0.0, epsilon = 0.1);
    }

    #[test]
    fn test_extra_force_j2_giants_induces_apsidal_precession() {
        // A q = 35 AU, a = 100 AU particle under the J2-averaged giants
        // precesses; without the extra force it stays Keplerian.
        use crate::nbodylib::types::{cartesian_to_elements, OrbitalElements};

        let elements = OrbitalElements {
            a: 100.0,
            e: 0.65,
            i: 0.1,
            omega_big: 0.3,
            omega: 0.5,
            mean_anomaly: 1.0,
        };
        let state = elements.to_state_vector(GM_SUN);

        let mut bodies = vec![];
        let mut particles = vec![state];
        let mut active = vec![true];
        let config = test_config(3000.0);
        let integrator = WhmIntegrator::with_extra_forces(vec![ExtraForce::j2_giants()]);

        // ~8 kyr per 1000 steps at dt = 3000 d
        for _ in 0..1000 {
            integrator.step(&mut bodies, &mut particles, &mut active, config.dt, &config);
        }

        let elem1 = cartesian_to_elements(&particles[0], GM_SUN);
        let varpi0 = elements.omega + elements.omega_big;
        let varpi1 = elem1.omega + elem1.omega_big;
        let dvarpi = (varpi1 - varpi0).rem_euclid(TWO_PI);
        let dvarpi = if dvarpi > std::f64::consts::PI {
            dvarpi - TWO_PI
        } else {
            dvarpi
        };
        assert!(
            dvarpi.abs() > 1e-4,
            "J2 giants extra force produced no apsidal precession (dvarpi = {dvarpi:.2e})"
        );
        // Semi-major axis is secularly preserved by the axisymmetric field.
        assert!((elem1.a - 100.0).abs() < 1.0, "a drifted: {}", elem1.a);
    }

    #[test]
    fn test_validate_timestep() {
        let bodies = vec![jupiter_like()];
        // P_jup ~ 4333 d -> max dt ~ 217 d
        assert!(validate_timestep(&bodies, 200.0).is_ok());
        assert!(validate_timestep(&bodies, 300.0).is_err());
    }

    #[test]
    fn test_whm_giants_from_ephemeris_energy_bounded() {
        // Real DE421 initial conditions at J2000, integrated 1000 years:
        // barycentric energy and angular momentum stay bounded.
        let kernel = SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421");
        let mut eph = Ephemeris::from_kernel(kernel);
        let ts = Timescale::default();
        let t0 = ts.tdb_jd(J2000);
        let mut bodies = giant_planets_from_ephemeris(&mut eph, &t0).expect("kernel lookup");

        let dt = 150.0; // resolves Jupiter (P/20 ≈ 217 d)
        validate_timestep(&bodies, dt).expect("timestep must resolve Jupiter");

        let mut particles = vec![];
        let mut active = vec![];
        let config = test_config(dt);
        let integrator = WhmIntegrator::new();

        let e0 = barycentric_energy(&bodies);
        let l0 = total_angular_momentum(&bodies);

        let n_steps = (1000.0 * 365.25 / dt).round() as usize;
        let mut max_de: f64 = 0.0;
        for step in 0..n_steps {
            integrator.step(&mut bodies, &mut particles, &mut active, dt, &config);
            if step % 100 == 99 {
                let e1 = barycentric_energy(&bodies);
                max_de = max_de.max(((e1 - e0) / e0).abs());
            }
        }

        // Bounded symplectic oscillation: ~2e-6 for the real (eccentric,
        // inclined) giants at dt = 150 d.
        assert!(max_de < 1e-5, "energy drift over 1 kyr: {max_de:.2e}");
        let dl = (total_angular_momentum(&bodies) - l0).norm() / l0.norm();
        assert!(dl < 1e-8, "angular momentum drift: {dl:.2e}");

        // The giants stay near their catalog semi-major axes.
        for (body, a) in bodies.iter().zip([5.2, 9.5, 19.2, 30.1]) {
            let r = body.state.distance();
            assert!(
                (r - a).abs() / a < 0.15,
                "{} wandered to r = {r:.2} AU",
                body.name
            );
        }
    }
}
