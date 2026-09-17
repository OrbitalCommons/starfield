//! Heliocentric ecliptic J2000 state vectors for solar system bodies
//!
//! Demonstrates computing ecliptic state vectors (position + velocity)
//! relative to the Sun in the ecliptic J2000 coordinate system.
//!
//! Usage: cargo run --example ecliptic_vectors

use starfield::jplephem::SpiceKernel;
use starfield::planetlib::{Body, Ephemeris};
use starfield::positions::ecliptic::EclipticStateVector;
use starfield::Timescale;

fn main() -> starfield::Result<()> {
    let ts = Timescale::default();
    let bsp_path = "src/jplephem/test_data/de421.bsp";

    println!("Heliocentric Ecliptic J2000 State Vectors");
    println!("=========================================\n");

    // --- Using EclipticStateVector::compute directly ---
    let mut kernel = SpiceKernel::open(bsp_path)?;
    let t = ts.tdb_jd(2451545.0);

    println!("At J2000 (2000-01-01 12:00 TDB):\n");

    let bodies = ["mercury", "venus", "earth", "mars"];
    for name in bodies {
        let sv = EclipticStateVector::compute(&mut kernel, name, "sun", &t)
            .expect("Failed to compute state vector");

        println!("  {name}:");
        println!(
            "    Position: ({:>12.6}, {:>12.6}, {:>12.6}) AU",
            sv.position.x, sv.position.y, sv.position.z
        );
        println!(
            "    Velocity: ({:>12.8}, {:>12.8}, {:>12.8}) AU/day",
            sv.velocity.x, sv.velocity.y, sv.velocity.z
        );
        println!("    Distance:  {:.6} AU", sv.distance());
        println!("    Speed:     {:.8} AU/day", sv.speed());
        println!("    Longitude: {:.4} deg", sv.longitude().to_degrees());
        println!("    Latitude:  {:.4} deg", sv.latitude().to_degrees());
        println!();
    }

    // --- Using the Ephemeris convenience method ---
    println!("Using Ephemeris.ecliptic_state():\n");
    let mut ephemeris = Ephemeris::from_kernel(SpiceKernel::open(bsp_path)?);

    let mars_ecl = ephemeris
        .ecliptic_state(Body::Mars, &t)
        .expect("Failed to compute Mars ecliptic state");
    println!("  Mars (via Ephemeris): {mars_ecl}");

    // --- Roundtrip to equatorial ---
    let (eq_pos, eq_vel) = mars_ecl.to_equatorial();
    println!("\n  Mars equatorial (roundtrip):");
    println!(
        "    Position: ({:.6}, {:.6}, {:.6}) AU",
        eq_pos.x, eq_pos.y, eq_pos.z
    );
    println!(
        "    Velocity: ({:.8}, {:.8}, {:.8}) AU/day",
        eq_vel.x, eq_vel.y, eq_vel.z
    );

    Ok(())
}
