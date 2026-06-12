//! Integrate the four giant planets for 1000 years with the Wisdom-Holman
//! symplectic integrator, starting from real DE421 ephemeris states.
//!
//! Demonstrates the planetlib → nbodylib bridge: kernel states in,
//! long-term dynamics out, with energy and angular momentum diagnostics.
//!
//! Usage: cargo run --example nbody_giants

use starfield::constants::J2000;
use starfield::jplephem::SpiceKernel;
use starfield::nbodylib::{
    barycentric_energy, cartesian_to_elements, giant_planets_from_ephemeris,
    total_angular_momentum, validate_timestep, SimConfig, StateVector, WhmIntegrator, GM_SUN,
};
use starfield::planetlib::Ephemeris;
use starfield::Timescale;

fn main() -> starfield::Result<()> {
    let ts = Timescale::default();
    let bsp_path = "test_data/de421.bsp";

    println!("Wisdom-Holman integration of the giant planets");
    println!("==============================================\n");

    // Initial conditions: heliocentric ecliptic-J2000 barycenter states from
    // DE421 at J2000, paired with DE440 system GMs.
    let mut ephemeris = Ephemeris::from_kernel(SpiceKernel::open(bsp_path)?);
    let t0 = ts.tdb_jd(J2000);
    let mut bodies = giant_planets_from_ephemeris(&mut ephemeris, &t0)
        .map_err(|e| starfield::StarfieldError::DataError(e.to_string()))?;

    println!("Initial states (J2000, from {bsp_path}):");
    for body in &bodies {
        let elem = cartesian_to_elements(&body.state, GM_SUN);
        println!(
            "  {:<8} r = {:6.3} AU   a = {:6.3} AU   e = {:.4}",
            body.name,
            body.state.distance(),
            elem.a,
            elem.e
        );
    }

    // 1000 years at dt = 150 d (P_Jupiter / 20 ≈ 217 d, so Jupiter is
    // resolved; validate_timestep enforces this).
    let dt = 150.0;
    validate_timestep(&bodies, dt).expect("timestep under-resolves a planet");
    let years = 1000.0;
    let n_steps = (years * 365.25 / dt).round() as usize;

    let mut particles: Vec<StateVector> = Vec::new();
    let mut active: Vec<bool> = Vec::new();
    let config = SimConfig {
        dt,
        ..SimConfig::default()
    };
    let integrator = WhmIntegrator::new();

    let e0 = barycentric_energy(&bodies);
    let l0 = total_angular_momentum(&bodies);

    let mut max_de: f64 = 0.0;
    for step in 0..n_steps {
        integrator.step(&mut bodies, &mut particles, &mut active, dt, &config);
        if step % 100 == 99 {
            let e1 = barycentric_energy(&bodies);
            max_de = max_de.max(((e1 - e0) / e0).abs());
        }
    }

    let dl = (total_angular_momentum(&bodies) - l0).norm() / l0.norm();

    println!("\nAfter {years:.0} years ({n_steps} steps of {dt} d):");
    for body in &bodies {
        let elem = cartesian_to_elements(&body.state, GM_SUN);
        println!(
            "  {:<8} r = {:6.3} AU   a = {:6.3} AU   e = {:.4}",
            body.name,
            body.state.distance(),
            elem.a,
            elem.e
        );
    }
    println!("\nDiagnostics (symplectic: bounded, not growing):");
    println!("  max |dE/E| = {max_de:.3e}");
    println!("  |dL|/|L|   = {dl:.3e}");

    Ok(())
}
