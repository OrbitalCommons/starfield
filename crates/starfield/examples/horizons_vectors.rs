//! Fetch Mars state vectors from the HORIZONS API.
//!
//! Demonstrates querying Cartesian position and velocity vectors for
//! a planet relative to the Solar System Barycenter.
//!
//! Run with: `cargo run --example horizons_vectors`

use starfield::horizons::parser;
use starfield::horizons::{Center, Command, EphemerisRequest, HorizonsClient, TimeSpec};

fn main() -> starfield::Result<()> {
    let client = HorizonsClient::new()?;

    // Request Mars state vectors for the first week of 2024
    let request = EphemerisRequest::vectors(
        Command::MajorBody(499), // Mars
        Center::SolarSystemBarycenter,
        TimeSpec::Range {
            start: "2024-01-01".into(),
            stop: "2024-01-08".into(),
            step: "1 d".into(),
        },
    );

    println!("Querying HORIZONS for Mars state vectors...");
    let response = client.query(&request)?;

    let result = response
        .result
        .as_ref()
        .expect("No result in HORIZONS response");

    let block = parser::extract_ephemeris_block(result)?;
    let rows = parser::parse_vector_rows(block)?;

    println!("\nMars state vectors (AU, AU/day) relative to SSB:");
    println!(
        "{:<14} {:>15} {:>15} {:>15} {:>12}",
        "JD (TDB)", "X", "Y", "Z", "Range (AU)"
    );
    println!("{}", "-".repeat(75));

    for row in &rows {
        println!(
            "{:<14.1} {:>15.10} {:>15.10} {:>15.10} {:>12.8}",
            row.jd_tdb,
            row.x,
            row.y,
            row.z,
            row.range.unwrap_or(0.0),
        );
    }

    println!("\n{} data points retrieved.", rows.len());
    Ok(())
}
