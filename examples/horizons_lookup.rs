//! Look up solar system objects by name using the HORIZONS lookup API.
//!
//! Demonstrates searching for asteroids, comets, and other objects
//! by name or designation.
//!
//! Run with: `cargo run --example horizons_lookup`

use starfield::horizons::{HorizonsClient, ObjectGroup};

fn main() -> starfield::Result<()> {
    let client = HorizonsClient::new()?;

    // Look up Apophis (a well-known near-Earth asteroid)
    println!("Looking up 'Apophis'...");
    let response = client.lookup("Apophis", Some(ObjectGroup::Asteroid))?;
    println!("  Found {} match(es)", response.count());
    if let Some(results) = &response.result {
        for r in results {
            println!(
                "  - {} (designation: {}, SPK-ID: {})",
                r.name.as_deref().unwrap_or("?"),
                r.pdes.as_deref().unwrap_or("?"),
                r.spkid.as_deref().unwrap_or("?"),
            );
        }
    }

    // Look up Halley's Comet
    println!("\nLooking up 'Halley'...");
    let response = client.lookup("Halley", Some(ObjectGroup::Comet))?;
    println!("  Found {} match(es)", response.count());
    if let Some(results) = &response.result {
        for r in results {
            println!(
                "  - {} (designation: {})",
                r.name.as_deref().unwrap_or("?"),
                r.pdes.as_deref().unwrap_or("?"),
            );
        }
    }

    // Look up a major body
    println!("\nLooking up 'Europa'...");
    let response = client.lookup("Europa", None)?;
    println!("  Found {} match(es)", response.count());
    if let Some(results) = &response.result {
        for r in results {
            println!(
                "  - {} (SPK-ID: {})",
                r.name.as_deref().unwrap_or("?"),
                r.spkid.as_deref().unwrap_or("?"),
            );
        }
    }

    Ok(())
}
