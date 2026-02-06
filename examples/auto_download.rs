//! Zero-config planetary positions with auto-downloading
//!
//! Demonstrates the Loader's automatic data file downloading.
//! No manual file download or path management needed.
//!
//! Run with: cargo run --example auto_download

use starfield::Loader;

fn main() -> starfield::Result<()> {
    let loader = Loader::new();
    let ts = loader.timescale();

    // This automatically downloads de421.bsp to ~/.cache/starfield/
    let mut kernel = loader.open("de421.bsp")?;

    let t = ts.tdb_jd(2451545.0); // J2000
    println!("Planetary positions at J2000 (2000-01-01 12:00 TDB):\n");

    for name in [
        "sun",
        "mercury",
        "venus",
        "earth",
        "mars",
        "jupiter barycenter",
    ] {
        let state = kernel.compute_at(name, &t)?;
        println!(
            "  {:<20} ({:>12.6}, {:>12.6}, {:>12.6}) AU",
            name, state.position.x, state.position.y, state.position.z
        );
    }

    Ok(())
}
