//! Example showing how to use the Hipparcos star catalog
//!
//! Run with: cargo run --example hipparcos

use starfield::catalogs::hipparcos::HipparcosCatalog;
use starfield::catalogs::StarCatalog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // For this example, we'll use a synthetic catalog
    // In a real application, you would use:
    // let loader = starfield::Loader::new();
    // let catalog = loader.load_hipparcos_catalog(6.0)?;

    println!("Creating synthetic star catalog...");
    let catalog = HipparcosCatalog::create_synthetic();

    println!("Loaded {} stars", catalog.len());

    // Display the 10 brightest stars
    println!("\nTop 10 brightest stars:");
    let mut bright_stars: Vec<_> = catalog.stars().collect();
    bright_stars.sort_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap());

    for (i, star) in bright_stars.iter().take(10).enumerate() {
        println!(
            "{}. HIP {} - Magnitude: {:.2}, RA: {:.2}°, Dec: {:.2}°",
            i + 1,
            star.hip,
            star.mag,
            star.ra,
            star.dec
        );
    }

    // Find stars in a specific region of the sky (around Orion's Belt)
    let orion_belt = catalog.filter(|star| {
        // Approximate coordinates for Orion's Belt region
        star.ra >= 80.0 && star.ra <= 85.0 && star.dec >= -2.0 && star.dec <= 2.0
    });

    println!("\nStars in Orion's Belt region:");
    for star in orion_belt.iter().take(5) {
        println!(
            "HIP {} - Magnitude: {:.2}, RA: {:.2}°, Dec: {:.2}°",
            star.hip, star.mag, star.ra, star.dec
        );
    }

    Ok(())
}
