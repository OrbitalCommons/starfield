//! Example demonstrating the synthetic star catalog generator
//!
//! This example shows how to create and use synthetic star catalogs
//! with realistic magnitude distributions.

use starfield::catalogs::{
    create_fov_catalog, create_synthetic_catalog, MinimalCatalog, SpatialDistribution,
    SyntheticCatalogConfig,
};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Synthetic Star Catalog Example");
    println!("=============================\n");

    // Example 1: Simple catalog with default settings
    println!("Example 1: Creating a basic synthetic catalog");
    let catalog = create_synthetic_catalog(
        100, // 100 stars
        3.0, // Minimum magnitude (brightest)
        8.0, // Maximum magnitude (dimmest)
        42,  // Random seed
    )?;

    print_catalog_info(&catalog, "Basic Synthetic Catalog");

    // Example 2: Creating a catalog limited to a field of view
    println!("\nExample 2: Creating a field-of-view limited catalog");
    let fov_catalog = create_fov_catalog(
        100,   // 100 stars
        3.0,   // Minimum magnitude
        8.0,   // Maximum magnitude
        100.0, // Center RA (degrees)
        45.0,  // Center Dec (degrees)
        5.0,   // Field of view (degrees)
        42,    // Random seed
    )?;

    print_catalog_info(&fov_catalog, "FOV-Limited Catalog");

    // Example 3: Creating a catalog with a custom configuration
    println!("\nExample 3: Creating a catalog with custom configuration");
    let custom_catalog = SyntheticCatalogConfig::new()
        .with_count(500)
        .with_magnitude_range(1.0, 10.0)
        .with_seed(123456)
        .with_spatial_distribution(SpatialDistribution::GalacticPlane { concentration: 2.0 })
        .with_description("Custom Galactic Plane Catalog")
        .generate()?;

    print_catalog_info(&custom_catalog, "Custom Galactic Plane Catalog");

    // Example 4: Creating a star cluster
    println!("\nExample 4: Creating a star cluster");
    let cluster_catalog = SyntheticCatalogConfig::new()
        .with_count(200)
        .with_magnitude_range(4.0, 9.0)
        .with_spatial_distribution(SpatialDistribution::Cluster {
            center_ra: 150.0,
            center_dec: 30.0,
            radius: 2.0,
        })
        .with_description("Synthetic Star Cluster")
        .generate()?;

    print_catalog_info(&cluster_catalog, "Star Cluster Catalog");

    // Save the cluster catalog to a file
    let output_path = Path::new("test_output/synthetic_cluster.bin");
    cluster_catalog.save(&output_path)?;
    println!("\nSaved cluster catalog to: {}", output_path.display());

    Ok(())
}

/// Print information about a catalog
fn print_catalog_info(catalog: &MinimalCatalog, title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(title.len()));
    println!("Description: {}", catalog.description());
    println!("Star count: {}", catalog.len());

    // Calculate magnitude statistics
    let stars = catalog.stars();
    if !stars.is_empty() {
        let min_mag = stars
            .iter()
            .map(|s| s.magnitude)
            .fold(f64::INFINITY, f64::min);
        let max_mag = stars
            .iter()
            .map(|s| s.magnitude)
            .fold(f64::NEG_INFINITY, f64::max);

        println!("Magnitude range: {:.2} to {:.2}", min_mag, max_mag);

        // Count by magnitude bins
        let mut mag_bins = vec![0; 10];
        for star in stars {
            let bin = (star.magnitude.floor() as usize).min(9);
            mag_bins[bin] += 1;
        }

        println!("\nMagnitude distribution:");
        for i in 0..10 {
            if mag_bins[i] > 0 {
                let percentage = (mag_bins[i] as f64 / catalog.len() as f64) * 100.0;
                println!(
                    "  Magnitude {}-{}: {} stars ({:.1}%)",
                    i,
                    i + 1,
                    mag_bins[i],
                    percentage
                );
            }
        }
    }
}
