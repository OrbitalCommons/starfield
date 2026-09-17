//! Example showing how to load and use the Gaia star catalog
//!
//! Run with: cargo run --example gaia_catalog -- /path/to/gaia/catalog.csv
//! Or without arguments to use synthetic data: cargo run --example gaia_catalog

use std::env;
use std::io::{self, Write};

use starfield::catalogs::StarCatalog;
use starfield::Loader;

/// Print a simple progress bar
fn print_progress(progress: f64, width: usize) {
    let filled_width = (progress * width as f64).round() as usize;
    let empty_width = width - filled_width;

    print!("\r[");
    for _ in 0..filled_width {
        print!("#");
    }
    for _ in 0..empty_width {
        print!(" ");
    }
    print!("] {:.1}%", progress * 100.0);
    io::stdout().flush().unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if a catalog path is provided as an argument
    let args: Vec<String> = env::args().collect();
    let catalog_path = args.get(1);

    println!("Gaia Catalog Analysis Tool");
    println!("=========================");

    // Create a loader
    let loader = Loader::new();

    // Load the catalog - either from specific file, all cached files, or synthetic
    let catalog = if let Some(path) = catalog_path {
        println!("Loading Gaia catalog from specific file: {}...", path);
        loader.load_gaia_catalog_from_file(path, 20.0)?
    } else {
        // Try to load from cached files, fall back to synthetic if none available
        match loader.load_gaia_catalog(20.0) {
            Ok(catalog) => catalog,
            Err(e) => {
                println!("Error loading cached Gaia files: {}", e);
                println!("Falling back to synthetic data...");
                loader.load_synthetic_gaia_catalog()
            }
        }
    };

    println!("\nCatalog loaded successfully!");
    println!("Total stars: {}", catalog.len());

    // Calculate magnitude distribution
    println!("\nCalculating magnitude distribution...");
    let mag_ranges = [
        (-5.0, 0.0, "Very bright stars (mag -5 to 0)"),
        (0.0, 5.0, "Bright stars (mag 0 to 5)"),
        (5.0, 10.0, "Medium bright stars (mag 5 to 10)"),
        (10.0, 15.0, "Dim stars (mag 10 to 15)"),
        (15.0, 20.0, "Very dim stars (mag 15 to 20)"),
    ];

    let mut magnitude_counts = vec![0; mag_ranges.len()];

    for (i, (min, max, _)) in mag_ranges.iter().enumerate() {
        print_progress(i as f64 / mag_ranges.len() as f64, 50);
        let stars_in_range = catalog
            .filter(|star| star.phot_g_mean_mag >= *min && star.phot_g_mean_mag < *max)
            .len();
        magnitude_counts[i] = stars_in_range;
    }
    print_progress(1.0, 50);
    println!();

    // Print magnitude distribution
    println!("\nMagnitude Distribution:");
    for (i, (_, _, desc)) in mag_ranges.iter().enumerate() {
        let count = magnitude_counts[i];
        let percentage = count as f64 / catalog.len() as f64 * 100.0;
        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Calculate spatial distribution by galactic latitude
    println!("\nCalculating spatial distribution by galactic latitude bands...");
    let lat_bands = [
        (-90.0, -60.0, "South galactic polar region (b -90° to -60°)"),
        (-60.0, -30.0, "South galactic temperate (b -60° to -30°)"),
        (-30.0, 0.0, "South galactic tropical (b -30° to 0°)"),
        (0.0, 30.0, "North galactic tropical (b 0° to 30°)"),
        (30.0, 60.0, "North galactic temperate (b 30° to 60°)"),
        (60.0, 90.0, "North galactic polar region (b 60° to 90°)"),
    ];

    let mut lat_counts = vec![0; lat_bands.len()];
    for (i, (min, max, _)) in lat_bands.iter().enumerate() {
        print_progress(i as f64 / lat_bands.len() as f64, 50);
        let stars_in_band = catalog.filter(|star| star.b >= *min && star.b < *max).len();
        lat_counts[i] = stars_in_band;
    }
    print_progress(1.0, 50);
    println!();

    // Print galactic latitude distribution
    println!("\nSpatial Distribution (by galactic latitude):");
    for (i, (_, _, desc)) in lat_bands.iter().enumerate() {
        let count = lat_counts[i];
        let percentage = count as f64 / catalog.len() as f64 * 100.0;
        println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
    }

    // Count variable stars
    let variable_stars = catalog
        .filter(|star| star.phot_variable_flag == "VARIABLE")
        .len();
    println!(
        "\nVariable stars: {} ({:.2}% of catalog)",
        variable_stars,
        variable_stars as f64 / catalog.len() as f64 * 100.0
    );

    // Find and list some notable stars by source ID
    println!("\nNotable Stars:");

    // Try to find some known bright stars (will work with real data or our synthetic data)
    let notable_stars = [
        (6752096595359340032u64, "Sirius (Alpha Canis Majoris)"),
        (5530942935258330368u64, "Canopus (Alpha Carinae)"),
        (5853498713190525696u64, "Alpha Centauri"),
        (2095947430657671296u64, "Vega (Alpha Lyrae)"),
        (3428908132419580672u64, "Betelgeuse (Alpha Orionis)"),
    ];

    for (source_id, name) in notable_stars.iter() {
        if let Some(star) = catalog.get_star(*source_id as usize) {
            println!("  {} (Gaia Source ID: {}):", name, source_id);
            println!("    Magnitude (G band): {:.2}", star.phot_g_mean_mag);
            println!("    Position: RA {:.4}°, Dec {:.4}°", star.ra, star.dec);
            println!("    Galactic: l {:.4}°, b {:.4}°", star.l, star.b);

            if let Some(parallax) = star.parallax {
                let distance_pc = 1000.0 / parallax;
                println!(
                    "    Distance: {:.2} parsecs ({:.2} light years)",
                    distance_pc,
                    distance_pc * 3.26156
                );
            }

            if let (Some(pmra), Some(pmdec)) = (star.pmra, star.pmdec) {
                let pm_total = (pmra * pmra + pmdec * pmdec).sqrt();
                println!("    Proper Motion: {:.2} mas/year", pm_total);
            }

            println!();
        } else {
            println!("  {} not found in catalog", name);
        }
    }

    // Find stars with highest proper motion
    println!("Stars with Highest Proper Motion:");
    let mut stars_vec: Vec<_> = catalog
        .stars()
        .filter(|star| star.pmra.is_some() && star.pmdec.is_some())
        .collect();

    stars_vec.sort_by(|a, b| {
        let a_pm = a.pmra.unwrap_or(0.0).hypot(a.pmdec.unwrap_or(0.0));
        let b_pm = b.pmra.unwrap_or(0.0).hypot(b.pmdec.unwrap_or(0.0));
        b_pm.partial_cmp(&a_pm).unwrap()
    });

    for (i, star) in stars_vec.iter().take(5).enumerate() {
        let pm = star.pmra.unwrap_or(0.0).hypot(star.pmdec.unwrap_or(0.0));
        println!(
            "  {}. Source ID {} - Proper Motion: {:.2} mas/year",
            i + 1,
            star.source_id,
            pm
        );
    }

    // Calculate statistics on magnitude error
    println!("\nPosition Precision:");
    let total_ra_error: f64 = catalog.stars().map(|star| star.ra_error).sum();
    let total_dec_error: f64 = catalog.stars().map(|star| star.dec_error).sum();
    let avg_ra_error = total_ra_error / catalog.len() as f64;
    let avg_dec_error = total_dec_error / catalog.len() as f64;

    println!("  Average RA Error: {:.4} mas", avg_ra_error);
    println!("  Average Dec Error: {:.4} mas", avg_dec_error);

    Ok(())
}
