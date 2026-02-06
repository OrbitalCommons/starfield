//! Satellite tracking example using SGP4
//!
//! Demonstrates loading a TLE, computing satellite position, and finding
//! rise/set/culmination events for a ground observer.
//!
//! Run with: cargo run --example satellite_tracking

use starfield::sgp4lib::{EarthSatellite, SatelliteEvent};
use starfield::time::Timescale;
use starfield::toposlib::WGS84;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ts = Timescale::default();

    // ISS TLE (from September 2008)
    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    let iss = EarthSatellite::from_tle(line1, line2, Some("ISS (ZARYA)"), &ts)?;

    println!("=== Satellite Tracking Example ===\n");
    println!("{}", iss);
    println!("Orbits/day: {:.2}", iss.revs_per_day);
    println!(
        "Orbital period: {:.1} minutes",
        24.0 * 60.0 / iss.revs_per_day
    );
    println!();

    // Compute position at TLE epoch
    let epoch = &iss.epoch;
    let pos_epoch = iss.at(epoch)?;

    let r_km = pos_epoch.position.norm() * 149_597_870.7;
    let altitude_km = r_km - 6371.0;
    println!("Position at TLE epoch:");
    println!(
        "  Date: {}",
        epoch
            .utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            .unwrap_or_else(|_| "unknown".to_string())
    );
    println!("  Distance from Earth center: {:.1} km", r_km);
    println!("  Approximate altitude: {:.1} km", altitude_km);

    let v_km_s = pos_epoch.velocity.norm() * 149_597_870.7 / 86400.0;
    println!("  Velocity: {:.3} km/s", v_km_s);
    println!();

    // Find satellite passes over Bluffton, Ohio
    let observer = WGS84.latlon(40.8939, -83.8917, 244.0);
    let t0 = ts.tt_jd(iss.epoch_jd(), None);
    let t1 = ts.tt_jd(iss.epoch_jd() + 1.0, None);

    println!("Satellite passes over Bluffton, Ohio (next 24 hours):");
    println!("  Observer: {}", observer);
    println!();

    let events = iss.find_events(&observer, &t0, &t1, &ts, 0.0)?;

    if events.is_empty() {
        println!("  No passes found.");
    } else {
        for (time, event) in &events {
            let time_str = time
                .utc_strftime("%Y-%m-%d %H:%M:%S UTC")
                .unwrap_or_else(|_| "unknown".to_string());
            let event_str = match event {
                SatelliteEvent::Rise => "Rise     ",
                SatelliteEvent::Culminate => "Culminate",
                SatelliteEvent::Set => "Set      ",
            };
            println!("  {} {}", event_str, time_str);
        }
    }

    Ok(())
}
