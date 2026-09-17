//! Script to create a minimal catalog from Hipparcos data

use starfield::catalogs::{MinimalCatalog, StarCatalog, StarData};
use starfield::Loader;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a loader
    let loader = Loader::new();

    // Load the Hipparcos catalog with magnitude limit up to 9.0
    // This will include all stars visible with moderate amateur telescopes
    println!("Loading Hipparcos catalog...");
    let hip_catalog = loader.load_hipparcos_catalog(9.0)?;

    println!("Loaded {} stars from Hipparcos catalog", hip_catalog.len());

    // Convert to StarData for minimal catalog
    let star_data: Vec<StarData> = hip_catalog.star_data().collect();

    // Create output path in the test_output directory
    let output_dir = PathBuf::from("test_output");
    std::fs::create_dir_all(&output_dir)?;
    let output_path = output_dir.join("hipparcos_mag9.bin");

    // Get the count before we move the data
    let star_count = star_data.len();

    // Create the minimal catalog
    println!("Creating minimal catalog at {}...", output_path.display());
    MinimalCatalog::write_from_star_data(
        &output_path,
        star_data.into_iter(),
        "Hipparcos Catalog (magnitude < 9.0)",
        Some(star_count as u64),
    )?;

    println!(
        "Minimal catalog created successfully with {} stars",
        star_count
    );
    println!("Use with --catalog {}", output_path.display());

    Ok(())
}
