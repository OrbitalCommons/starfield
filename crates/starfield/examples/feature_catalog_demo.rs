//! Feature catalog demonstration
//!
//! This example shows how to use the astronomical feature catalog
//! to find interesting targets for simulation.

use starfield::catalogs::{FeatureCatalog, FeatureType};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Parse arguments to filter by feature type if provided
    let filter_type = if args.len() > 1 {
        match args[1].to_lowercase().as_str() {
            "constellation" => Some(FeatureType::Constellation),
            "open" | "opencluster" => Some(FeatureType::OpenCluster),
            "globular" | "globularcluster" => Some(FeatureType::GlobularCluster),
            "nebula" => Some(FeatureType::Nebula),
            "galaxy" => Some(FeatureType::Galaxy),
            "star" => Some(FeatureType::Star),
            _ => None,
        }
    } else {
        None
    };

    // Load the default feature catalog
    let catalog = FeatureCatalog::default();

    println!("Astronomical Feature Catalog");
    println!("============================");
    println!("Total features: {}", catalog.len());

    // Display features by type or all features
    if let Some(feature_type) = filter_type {
        display_features_by_type(&catalog, &feature_type);
    } else {
        display_all_feature_counts(&catalog);

        // Ask user if they want to see all features
        println!("\nTo see details for a specific feature type, run with argument:");
        println!("  cargo run --example feature_catalog_demo [constellation|open|globular|nebula|galaxy|star]");
    }

    Ok(())
}

/// Display all feature type counts
fn display_all_feature_counts(catalog: &FeatureCatalog) {
    println!("\nFeatures by type:");
    println!("----------------");

    let constellations = catalog.get_features_by_type(&FeatureType::Constellation);
    let open_clusters = catalog.get_features_by_type(&FeatureType::OpenCluster);
    let globular_clusters = catalog.get_features_by_type(&FeatureType::GlobularCluster);
    let nebulae = catalog.get_features_by_type(&FeatureType::Nebula);
    let galaxies = catalog.get_features_by_type(&FeatureType::Galaxy);
    let stars = catalog.get_features_by_type(&FeatureType::Star);
    let other = catalog.get_features_by_type(&FeatureType::Other);

    println!("Constellations:    {}", constellations.len());
    println!("Open Clusters:     {}", open_clusters.len());
    println!("Globular Clusters: {}", globular_clusters.len());
    println!("Nebulae:           {}", nebulae.len());
    println!("Galaxies:          {}", galaxies.len());
    println!("Stars:             {}", stars.len());
    println!("Other:             {}", other.len());
}

/// Display features of a specific type
fn display_features_by_type(catalog: &FeatureCatalog, feature_type: &FeatureType) {
    let features = catalog.get_features_by_type(feature_type);

    let type_name = match feature_type {
        FeatureType::Constellation => "Constellations",
        FeatureType::OpenCluster => "Open Clusters",
        FeatureType::GlobularCluster => "Globular Clusters",
        FeatureType::Nebula => "Nebulae",
        FeatureType::Galaxy => "Galaxies",
        FeatureType::Star => "Stars",
        FeatureType::Other => "Other Features",
    };

    println!("\n{} ({})", type_name, features.len());
    println!("{}", "=".repeat(type_name.len() + 4));

    println!(
        "{:<20} {:<10} {:<10} {:<8} {}",
        "Name", "RA (°)", "Dec (°)", "Size (°)", "Description"
    );
    println!("{:-<80}", "");

    for feature in &features {
        println!(
            "{:<20} {:<10.2} {:<10.2} {:<8.2} {}",
            feature.name,
            feature.ra_deg,
            feature.dec_deg,
            feature.diameter_deg,
            feature.description
        );
    }

    // Print telescope targeting instructions
    println!("\nTo target these features in the star simulator:");
    println!("----------------------------------------------");
    println!("cargo run --example star_simulation -- --catalog test_output/hipparcos_mag9.bin --ra [RA] --dec [DEC]");
    println!("\nExample:");
    if !features.is_empty() {
        let example = &features[0];
        println!("cargo run --example star_simulation -- --catalog test_output/hipparcos_mag9.bin --ra {} --dec {}", 
                 example.ra_deg, example.dec_deg);
    }
}
