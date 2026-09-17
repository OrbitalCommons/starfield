//! Utility to show information about minimal star catalogs
//!
//! This tool reads minimal catalogs and displays metadata and statistics
//! about the catalog contents.

use std::env;
use std::path::Path;

use starfield::catalogs::{MinimalCatalog, StarPosition};

fn print_catalog_info<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
    // Get path as string for later use
    let path_string = path.as_ref().to_string_lossy().to_string();

    // Load the catalog
    println!("Loading minimal catalog: {}", path.as_ref().display());
    let catalog = MinimalCatalog::load(&path)?;

    // Print basic information
    println!("\nCatalog Information:");
    println!("-------------------");
    println!("Description: {}", catalog.description());
    println!("Total stars: {}", catalog.len());
    println!("Maximum magnitude: {:.2}", catalog.max_magnitude());

    // Calculate magnitude statistics
    let mut magnitude_counts = vec![0; 10];
    let magnitude_ranges = [
        (-2.0, 0.0, "Very bright (-2 to 0)"),
        (0.0, 2.0, "Bright (0 to 2)"),
        (2.0, 4.0, "Medium (2 to 4)"),
        (4.0, 6.0, "Dim (4 to 6)"),
        (6.0, 8.0, "Very dim (6 to 8)"),
        (8.0, 10.0, "Faint (8 to 10)"),
        (10.0, 12.0, "Very faint (10 to 12)"),
        (12.0, 14.0, "Extremely faint (12 to 14)"),
        (14.0, 16.0, "Barely visible (14 to 16)"),
        (16.0, 30.0, "Telescope only (16+)"),
    ];

    for star in catalog.stars() {
        for (i, (min, max, _)) in magnitude_ranges.iter().enumerate() {
            if star.magnitude >= *min && star.magnitude < *max {
                magnitude_counts[i] += 1;
                break;
            }
        }
    }

    // Print magnitude distribution
    println!("\nMagnitude Distribution:");
    for (i, (_, _, desc)) in magnitude_ranges.iter().enumerate() {
        let count = magnitude_counts[i];
        let percentage = if catalog.len() > 0 {
            (count as f64 / catalog.len() as f64) * 100.0
        } else {
            0.0
        };

        if count > 0 {
            println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
        }
    }

    // Calculate declination statistics
    let mut declination_counts = vec![0; 6];
    let declination_bands = [
        (-90.0, -60.0, "South polar region (-90° to -60°)"),
        (-60.0, -30.0, "South temperate (-60° to -30°)"),
        (-30.0, 0.0, "South tropical (-30° to 0°)"),
        (0.0, 30.0, "North tropical (0° to 30°)"),
        (30.0, 60.0, "North temperate (30° to 60°)"),
        (60.0, 90.0, "North polar region (60° to 90°)"),
    ];

    for star in catalog.stars() {
        for (i, (min, max, _)) in declination_bands.iter().enumerate() {
            if star.dec() >= *min && star.dec() < *max {
                declination_counts[i] += 1;
                break;
            }
        }
    }

    // Print declination distribution
    println!("\nSpatial Distribution (by declination):");
    for (i, (_, _, desc)) in declination_bands.iter().enumerate() {
        let count = declination_counts[i];
        let percentage = if catalog.len() > 0 {
            (count as f64 / catalog.len() as f64) * 100.0
        } else {
            0.0
        };

        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Calculate RA statistics
    let mut ra_counts = vec![0; 6];
    let ra_bands = [
        (0.0, 60.0, "RA 0° to 60°"),
        (60.0, 120.0, "RA 60° to 120°"),
        (120.0, 180.0, "RA 120° to 180°"),
        (180.0, 240.0, "RA 180° to 240°"),
        (240.0, 300.0, "RA 240° to 300°"),
        (300.0, 360.0, "RA 300° to 360°"),
    ];

    for star in catalog.stars() {
        for (i, (min, max, _)) in ra_bands.iter().enumerate() {
            if star.ra() >= *min && star.ra() < *max {
                ra_counts[i] += 1;
                break;
            }
        }
    }

    // Print RA distribution
    println!("\nSpatial Distribution (by right ascension):");
    for (i, (_, _, desc)) in ra_bands.iter().enumerate() {
        let count = ra_counts[i];
        let percentage = if catalog.len() > 0 {
            (count as f64 / catalog.len() as f64) * 100.0
        } else {
            0.0
        };

        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Print catalog binary size
    let file_size = std::fs::metadata(&path_string)?.len();
    println!("\nFile size: {:.2} KB", file_size as f64 / 1024.0);
    println!(
        "Bytes per star: {:.2}",
        if catalog.len() > 0 {
            file_size as f64 / catalog.len() as f64
        } else {
            0.0
        }
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: cargo run --example minimal_catalog_info -- <catalog_file_path>");
        return Ok(());
    }

    let file_path = &args[1];
    print_catalog_info(file_path)?;

    Ok(())
}
