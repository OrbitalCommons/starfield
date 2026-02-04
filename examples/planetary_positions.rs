//! Planetary position computation using JPL DE421 ephemeris
//!
//! Demonstrates loading a BSP file and computing planetary positions.
//!
//! Usage: cargo run --example planetary_positions

use starfield::jplephem::SpiceKernel;
use starfield::planetlib::{Body, Ephemeris};
use starfield::Timescale;

fn main() -> starfield::Result<()> {
    let ts = Timescale::default();

    // Load the DE421 ephemeris
    let bsp_path = "src/jplephem/test_data/de421.bsp";
    println!("Loading ephemeris from {bsp_path}...");

    let mut kernel = SpiceKernel::open(bsp_path)?;

    // Print available segments
    println!("\nSegments in DE421:");
    for seg in &kernel.spk().segments {
        println!("  {seg}");
    }

    // Compute positions at J2000 (2000-01-01 12:00 TDB)
    let t = ts.tdb_jd(2451545.0);
    println!("\nPlanetary positions at J2000 (2000-01-01 12:00 TDB):");

    let bodies = ["sun", "mercury", "venus", "earth", "moon", "mars"];
    for name in bodies {
        let state = kernel.compute_at(name, &t)?;
        let dist =
            (state.position.x.powi(2) + state.position.y.powi(2) + state.position.z.powi(2)).sqrt();
        println!(
            "  {:<10} pos=({:>12.6}, {:>12.6}, {:>12.6}) AU  |r|={:.6} AU",
            name, state.position.x, state.position.y, state.position.z, dist
        );
    }

    // Compute Earth-Mars distance over a year
    println!("\nEarth-Mars distance over 2024:");
    let mut ephemeris = Ephemeris::from_kernel(SpiceKernel::open(bsp_path)?);

    for month in 1..=12 {
        let t = ts.tdb_jd(starfield::jplephem::calendar::compute_julian_date(
            2024, month, 1.0,
        ));
        let earth = ephemeris.get_state(Body::Earth, &t).unwrap();
        let mars = ephemeris.get_state(Body::Mars, &t).unwrap();

        let dx = mars.position.x - earth.position.x;
        let dy = mars.position.y - earth.position.y;
        let dz = mars.position.z - earth.position.z;
        let dist_au = (dx * dx + dy * dy + dz * dz).sqrt();
        let dist_km = dist_au * 149_597_870.7;

        println!(
            "  2024-{:02}-01: {:.4} AU ({:.0} million km)",
            month,
            dist_au,
            dist_km / 1e6
        );
    }

    Ok(())
}
