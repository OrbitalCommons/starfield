//! Example: Query upcoming asteroid close approaches to Earth
//!
//! Uses the JPL Close Approach Data (CAD) API to find asteroids
//! passing near Earth in the near future.
//!
//! Usage: cargo run --example sbdb_close_approach

use starfield::sbdb::{CadParams, SbdbClient};

fn main() {
    let client = SbdbClient::new().expect("Failed to create SBDB client");

    // Query close approaches from now, within 0.05 AU of Earth
    let params = CadParams {
        date_min: Some("now".into()),
        dist_max: Some("0.05".into()),
        limit: Some(10),
        sort: Some("dist".into()),
        ..Default::default()
    };

    println!("Querying upcoming close approaches within 0.05 AU of Earth...\n");
    let response = client
        .close_approaches(&params)
        .expect("Failed to query close approaches");

    println!("Found {} close approaches:\n", response.count);

    for record in &response.records {
        println!(
            "  {} — {} at {:.6} AU",
            record.date,
            record.fullname.as_deref().unwrap_or(&record.designation),
            record.dist_au,
        );
        if let Some(v) = record.v_rel_km_s {
            print!("    Relative velocity: {:.2} km/s", v);
        }
        if let Some(h) = record.h_mag {
            print!("    H={:.1}", h);
        }
        println!();
    }
}
