//! Satellite tracking example using SGP4
//!
//! Demonstrates loading a TLE and computing satellite position.
//!
//! Run with: cargo run --example satellite_tracking

use starfield::sgp4lib::EarthSatellite;
use starfield::time::Timescale;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a timescale
    let ts = Timescale::default();

    // ISS TLE (from September 2008)
    // In a real application, you would fetch this from Celestrak or Space-Track
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // Parse the TLE
    let iss = EarthSatellite::from_tle(line1, line2, Some("ISS (ZARYA)"), &ts)?;

    println!("=== Satellite Tracking Example ===\n");
    println!("Satellite: {}", iss.name.as_deref().unwrap_or("Unknown"));
    println!("NORAD ID:  {}", iss.norad_id);
    println!("Orbits/day: {:.2}", iss.revs_per_day);
    println!(
        "Orbital period: {:.1} minutes",
        24.0 * 60.0 / iss.revs_per_day
    );
    println!();

    // Compute position at TLE epoch
    let epoch = &iss.epoch;
    let pos_epoch = iss.at(epoch)?;

    println!("Position at TLE epoch:");
    println!(
        "  Date: {}",
        epoch
            .utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            .unwrap_or_else(|_| "unknown".to_string())
    );
    println!(
        "  GCRS X: {:.6} AU ({:.1} km)",
        pos_epoch.position.x,
        pos_epoch.position.x * 149_597_870.7
    );
    println!(
        "  GCRS Y: {:.6} AU ({:.1} km)",
        pos_epoch.position.y,
        pos_epoch.position.y * 149_597_870.7
    );
    println!(
        "  GCRS Z: {:.6} AU ({:.1} km)",
        pos_epoch.position.z,
        pos_epoch.position.z * 149_597_870.7
    );

    let r_km = pos_epoch.position.norm() * 149_597_870.7;
    let altitude_km = r_km - 6371.0; // Approximate Earth radius
    println!("  Distance from Earth center: {:.1} km", r_km);
    println!("  Approximate altitude: {:.1} km", altitude_km);
    println!();

    // Compute velocity
    let v_km_s = pos_epoch.velocity.norm() * 149_597_870.7 / 86400.0;
    println!("Velocity: {:.3} km/s", v_km_s);
    println!();

    // Get TEME coordinates (raw SGP4 output)
    let (teme_pos, teme_vel) = iss.position_and_velocity_teme_km(epoch)?;
    println!("Raw TEME coordinates:");
    println!(
        "  Position: [{:.3}, {:.3}, {:.3}] km",
        teme_pos.x, teme_pos.y, teme_pos.z
    );
    println!(
        "  Velocity: [{:.6}, {:.6}, {:.6}] km/s",
        teme_vel.x, teme_vel.y, teme_vel.z
    );
    println!();

    // Propagate 1 hour into the future
    let t_plus_1h = ts.tt_jd(iss.epoch_jd() + 1.0 / 24.0, None);
    let pos_1h = iss.at(&t_plus_1h)?;

    println!("Position 1 hour after epoch:");
    println!(
        "  Date: {}",
        t_plus_1h
            .utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            .unwrap_or_else(|_| "unknown".to_string())
    );
    let r_km_1h = pos_1h.position.norm() * 149_597_870.7;
    let altitude_km_1h = r_km_1h - 6371.0;
    println!("  Distance from Earth center: {:.1} km", r_km_1h);
    println!("  Approximate altitude: {:.1} km", altitude_km_1h);

    Ok(())
}
