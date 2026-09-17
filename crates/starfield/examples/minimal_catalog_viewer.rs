//! Minimal Star Catalog Viewer
//!
//! This utility displays information about minimal star catalog files and can
//! convert them to CSV format if needed.
//!
//! Usage:
//!   cargo run --example minimal_catalog_viewer -- [options]
//!
//! Options:
//!   --input PATH       Minimal catalog file to view
//!   --convert PATH     Convert minimal catalog to CSV at specified path
//!   --magnitude FLOAT  Only include stars brighter than this magnitude when converting

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use starfield::catalogs::{MinimalCatalog, StarPosition};

/// Print information about a minimal catalog file
fn view_catalog<P: AsRef<Path>>(catalog_path: P) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Loading minimal catalog: {}",
        catalog_path.as_ref().display()
    );

    // Load the catalog
    let catalog = MinimalCatalog::load(&catalog_path)?;

    // Print basic information
    println!("\nCatalog Information:");
    println!("  Total stars: {}", catalog.len());
    println!("  Maximum magnitude: {:.2}", catalog.max_magnitude());

    // Calculate magnitude distribution
    let mag_ranges = [
        (-10.0, 0.0, "Very bright stars (mag -10 to 0)"),
        (0.0, 5.0, "Bright stars (mag 0 to 5)"),
        (5.0, 10.0, "Medium bright stars (mag 5 to 10)"),
        (10.0, 15.0, "Dim stars (mag 10 to 15)"),
        (15.0, 20.0, "Very dim stars (mag 15 to 20)"),
        (20.0, 30.0, "Extremely dim stars (mag 20+)"),
    ];

    println!("\nMagnitude Distribution:");
    for (min, max, desc) in &mag_ranges {
        let count = catalog
            .filter(|star| star.magnitude >= *min && star.magnitude < *max)
            .len();

        let percentage = if catalog.len() > 0 {
            (count as f64 / catalog.len() as f64) * 100.0
        } else {
            0.0
        };

        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Print the first few stars
    if !catalog.is_empty() {
        println!("\nFirst 5 stars:");
        for (i, star) in catalog.stars().iter().take(5).enumerate() {
            println!(
                "  {}. ID: {}, RA: {:.4}°, Dec: {:.4}°, Mag: {:.2}",
                i + 1,
                star.id,
                star.ra(),
                star.dec(),
                star.magnitude
            );
        }
    }

    // Find brightest star
    if !catalog.is_empty() {
        let brightest = catalog
            .stars()
            .iter()
            .min_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
            .unwrap();

        println!("\nBrightest star:");
        println!(
            "  ID: {}, RA: {:.4}°, Dec: {:.4}°, Mag: {:.2}",
            brightest.id,
            brightest.ra(),
            brightest.dec(),
            brightest.magnitude
        );
    }

    // Calculate approximate file size savings
    let binary_size = std::fs::metadata(&catalog_path)?.len();
    let csv_estimate = catalog.len() as u64 * 40; // Rough estimate: ~40 bytes per star in CSV

    println!("\nStorage Information:");
    println!("  Binary file size: {} bytes", binary_size);
    println!("  Estimated CSV size: {} bytes", csv_estimate);

    if csv_estimate > binary_size {
        let savings = csv_estimate - binary_size;
        let savings_percent = (savings as f64 / csv_estimate as f64) * 100.0;
        println!(
            "  Space savings: {} bytes ({:.1}%)",
            savings, savings_percent
        );
    }

    Ok(())
}

/// Convert a minimal catalog to CSV format
fn convert_to_csv<P: AsRef<Path>, Q: AsRef<Path>>(
    input_path: P,
    output_path: Q,
    magnitude_limit: Option<f64>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Converting minimal catalog to CSV: {} -> {}",
        input_path.as_ref().display(),
        output_path.as_ref().display()
    );

    // Load the catalog
    let catalog = MinimalCatalog::load(input_path)?;

    // Create output CSV file
    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    // Write CSV header
    writeln!(writer, "source_id,ra,dec,magnitude")?;

    // Write each star as CSV
    let mut stars_written = 0;

    for star in catalog.stars() {
        // Apply magnitude filter if specified
        if let Some(limit) = magnitude_limit {
            if star.magnitude > limit {
                continue;
            }
        }

        writeln!(
            writer,
            "{},{},{},{}",
            star.id,
            star.ra(),
            star.dec(),
            star.magnitude
        )?;
        stars_written += 1;
    }

    println!(
        "Conversion complete. Wrote {} stars to CSV file.",
        stars_written
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Default values
    let mut input_path = None;
    let mut convert_path = None;
    let mut magnitude_limit = None;

    // Parse command-line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                if i + 1 < args.len() {
                    input_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --input".into());
                }
            }
            "--convert" => {
                if i + 1 < args.len() {
                    convert_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --convert".into());
                }
            }
            "--magnitude" => {
                if i + 1 < args.len() {
                    magnitude_limit = Some(args[i + 1].parse()?);
                    i += 2;
                } else {
                    return Err("Missing value for --magnitude".into());
                }
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    println!("Minimal Star Catalog Viewer");
    println!("===========================");

    // Check if input path is provided
    if input_path.is_none() {
        println!("Usage:");
        println!(
            "  cargo run --example minimal_catalog_viewer -- --input <catalog_file> [options]"
        );
        println!("");
        println!("Options:");
        println!("  --input PATH       Minimal catalog file to view");
        println!("  --convert PATH     Convert minimal catalog to CSV at specified path");
        println!(
            "  --magnitude FLOAT  Only include stars brighter than this magnitude when converting"
        );

        return Err("Missing required input path".into());
    }

    // View the catalog
    view_catalog(input_path.as_ref().unwrap())?;

    // Convert to CSV if requested
    if let Some(path) = convert_path {
        convert_to_csv(input_path.unwrap(), path, magnitude_limit)?;
    }

    Ok(())
}
