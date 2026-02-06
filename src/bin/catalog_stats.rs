//! Tool to download, analyze, and filter star catalogs
//!
//! This binary can analyze or filter Hipparcos and Gaia star catalogs with various options.

use std::env;
use std::io::{self, Write};
use std::path::Path;

use starfield::catalogs::hipparcos::HipparcosEntry;
use starfield::catalogs::{MinimalCatalog, MinimalStar, StarCatalog};
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

/// Filter a Hipparcos catalog and save to binary format
fn filter_and_save<P: AsRef<Path>>(
    catalog: &impl StarCatalog<Star = HipparcosEntry>,
    output_path: P,
    magnitude_limit: f64,
) -> Result<usize, Box<dyn std::error::Error>> {
    println!(
        "Filtering catalog to magnitude {} and saving to {}...",
        magnitude_limit,
        output_path.as_ref().display()
    );

    // Create description for the minimal catalog
    let desc = format!(
        "Hipparcos filtered catalog: magnitude <= {}, created on {}",
        magnitude_limit,
        chrono::Local::now().format("%Y-%m-%d")
    );

    // Filter stars and add to minimal catalog
    let mut count = 0;
    // We'll collect the stars first, then build the catalog
    let mut filtered_stars = Vec::new();

    for star in catalog.stars() {
        if star.mag <= magnitude_limit {
            // Create minimal star entry
            let minimal_star = MinimalStar::new(star.hip as u64, star.ra, star.dec, star.mag);
            filtered_stars.push(minimal_star);
            count += 1;
        }
    }

    // Create the catalog from collected stars
    let catalog = MinimalCatalog::from_stars(filtered_stars, &desc);

    // Save the catalog
    catalog.save(output_path)?;

    println!("Saved {} stars to minimal catalog", count);
    Ok(count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Default values
    let mut catalog_type = "hipparcos";
    let mut operation = "stats";
    let mut magnitude_limit = 10.0;
    let mut output_path = None;

    // Parse command-line arguments
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--catalog" | "-c" => {
                if i + 1 < args.len() {
                    catalog_type = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --catalog".into());
                }
            }
            "--operation" | "-o" => {
                if i + 1 < args.len() {
                    operation = &args[i + 1];
                    i += 2;
                } else {
                    return Err("Missing value for --operation".into());
                }
            }
            "--magnitude" | "-m" => {
                if i + 1 < args.len() {
                    magnitude_limit = args[i + 1].parse()?;
                    i += 2;
                } else {
                    return Err("Missing value for --magnitude".into());
                }
            }
            "--output" | "-f" => {
                if i + 1 < args.len() {
                    output_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --output".into());
                }
            }
            "--help" | "-h" => {
                println!("Star Catalog Tool");
                println!("================");
                println!("Usage: cargo run --bin catalog_stats -- [OPTIONS]");
                println!();
                println!("Options:");
                println!(
                    "  -c, --catalog TYPE    Catalog type: hipparcos or gaia (default: hipparcos)"
                );
                println!("  -o, --operation OP    Operation: stats or filter (default: stats)");
                println!("  -m, --magnitude MAG   Magnitude limit (default: 10.0)");
                println!("  -f, --output PATH     Output file path for filter operation");
                println!("  -h, --help            Show this help message");
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    println!("Star Catalog Tool");
    println!("================");

    // Create a loader
    let loader = Loader::new();

    // Load the appropriate catalog
    println!(
        "Loading {} catalog (magnitude <= {})...",
        catalog_type, magnitude_limit
    );

    let catalog = if catalog_type == "hipparcos" {
        loader.load_hipparcos_catalog(magnitude_limit)?
    } else if catalog_type == "gaia" {
        return Err("Direct Gaia catalog loading not implemented in stats tool. Use gaia_filter example instead.".into());
    } else {
        return Err(format!("Unknown catalog type: {}", catalog_type).into());
    };

    println!("\nCatalog loaded successfully!");
    println!("Total stars: {}", catalog.len());

    // Choose operation based on command-line arguments
    match operation {
        "stats" => {
            // Calculate magnitude distribution
            println!("\nCalculating magnitude distribution...");
            let mag_ranges = [
                (-2.0, 0.0, "Very bright stars (mag -2 to 0)"),
                (0.0, 2.0, "Bright stars (mag 0 to 2)"),
                (2.0, 4.0, "Medium bright stars (mag 2 to 4)"),
                (4.0, 6.0, "Naked eye visible stars (mag 4 to 6)"),
                (6.0, 8.0, "Binocular visible stars (mag 6 to 8)"),
                (8.0, 10.0, "Telescope visible stars (mag 8 to 10)"),
            ];

            let mut magnitude_counts = vec![0; mag_ranges.len()];

            for (i, (min, max, _)) in mag_ranges.iter().enumerate() {
                print_progress(i as f64 / mag_ranges.len() as f64, 50);
                let stars_in_range = catalog
                    .filter(|star| star.mag >= *min && star.mag < *max)
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

            // Calculate spatial distribution
            println!("\nCalculating spatial distribution by declination bands...");
            let dec_bands = [
                (-90.0, -60.0, "South polar region (dec -90° to -60°)"),
                (-60.0, -30.0, "South temperate (dec -60° to -30°)"),
                (-30.0, 0.0, "South tropical (dec -30° to 0°)"),
                (0.0, 30.0, "North tropical (dec 0° to 30°)"),
                (30.0, 60.0, "North temperate (dec 30° to 60°)"),
                (60.0, 90.0, "North polar region (dec 60° to 90°)"),
            ];

            let mut dec_counts = vec![0; dec_bands.len()];
            for (i, (min, max, _)) in dec_bands.iter().enumerate() {
                print_progress(i as f64 / dec_bands.len() as f64, 50);
                let stars_in_band = catalog
                    .filter(|star| star.dec >= *min && star.dec < *max)
                    .len();
                dec_counts[i] = stars_in_band;
            }
            print_progress(1.0, 50);
            println!();

            // Print declination distribution
            println!("\nSpatial Distribution (by declination):");
            for (i, (_, _, desc)) in dec_bands.iter().enumerate() {
                let count = dec_counts[i];
                let percentage = count as f64 / catalog.len() as f64 * 100.0;
                println!("  {}: {} stars ({:.1}%)", desc, count, percentage);
            }

            // Find and list some notable stars
            println!("\nNotable Stars:");

            // Sirius (Alpha Canis Majoris) - brightest star
            if let Some(sirius) = catalog.get_star(32349) {
                println!("  Sirius (Alpha Canis Majoris):");
                println!("    HIP: {}", sirius.hip);
                println!("    Magnitude: {:.2}", sirius.mag);
                println!("    Position: RA {:.4}°, Dec {:.4}°", sirius.ra, sirius.dec);
                if let Some(parallax) = sirius.parallax {
                    let distance_pc = 1000.0 / parallax;
                    println!(
                        "    Distance: {:.2} parsecs ({:.2} light years)",
                        distance_pc,
                        distance_pc * 3.26156
                    );
                }
            } else {
                println!("  Sirius not found in catalog");
            }

            // Betelgeuse (Alpha Orionis) - red supergiant
            if let Some(betelgeuse) = catalog.get_star(27989) {
                println!("\n  Betelgeuse (Alpha Orionis):");
                println!("    HIP: {}", betelgeuse.hip);
                println!("    Magnitude: {:.2}", betelgeuse.mag);
                println!(
                    "    Position: RA {:.4}°, Dec {:.4}°",
                    betelgeuse.ra, betelgeuse.dec
                );
                if let Some(b_v) = betelgeuse.b_v {
                    println!("    B-V Color Index: {:.2} (reddish)", b_v);
                }
            } else {
                println!("  Betelgeuse not found in catalog");
            }

            // Vega (Alpha Lyrae) - blue-white star
            if let Some(vega) = catalog.get_star(91262) {
                println!("\n  Vega (Alpha Lyrae):");
                println!("    HIP: {}", vega.hip);
                println!("    Magnitude: {:.2}", vega.mag);
                println!("    Position: RA {:.4}°, Dec {:.4}°", vega.ra, vega.dec);
                if let Some(b_v) = vega.b_v {
                    println!("    B-V Color Index: {:.2} (blue-white)", b_v);
                }
            } else {
                println!("  Vega not found in catalog");
            }

            // Find fastest moving stars (highest proper motion)
            println!("\nStars with Highest Proper Motion:");
            let mut stars_vec: Vec<_> = catalog.stars().collect();
            stars_vec.sort_by(|a, b| {
                let a_pm = a
                    .pm_ra
                    .unwrap_or(0.0)
                    .abs()
                    .hypot(a.pm_dec.unwrap_or(0.0).abs());
                let b_pm = b
                    .pm_ra
                    .unwrap_or(0.0)
                    .abs()
                    .hypot(b.pm_dec.unwrap_or(0.0).abs());
                b_pm.partial_cmp(&a_pm).unwrap()
            });

            for (i, star) in stars_vec.iter().take(5).enumerate() {
                let pm = star.pm_ra.unwrap_or(0.0).hypot(star.pm_dec.unwrap_or(0.0));
                println!(
                    "  {}. HIP {} - Proper Motion: {:.2} mas/year",
                    i + 1,
                    star.hip,
                    pm
                );
            }

            // Create a simple ASCII sky map
            println!("\nSimple Sky Map (RA vs Dec, brightest stars only):");
            println!("  Legend: * (mag < 0), + (mag < 1), . (mag < 2)");

            // Get terminal dimensions and set grid size
            let height: usize = 20; // Fixed height

            // Use a fixed width for the grid
            let width: usize = 100;

            // Create a grid for the map
            let mut grid = vec![vec![' '; width]; height];

            // Place stars on the grid (only bright ones)
            let bright_stars = catalog.brighter_than(2.0);
            for star in bright_stars {
                // Map RA (0-360) to x (0-width)
                let x = ((star.ra / 360.0) * width as f64) as usize % width;

                // Map Dec (-90 to +90) to y (height-1 to 0)
                let y = ((90.0 - star.dec) / 180.0 * height as f64) as usize;
                let y = y.min(height - 1);

                // Choose a character based on magnitude
                let char = if star.mag < 0.0 {
                    '*' // Very bright stars
                } else if star.mag < 1.0 {
                    '+' // Bright stars
                } else {
                    '.' // Visible stars
                };

                // Place the star on the grid if in bounds
                if y < height && x < width {
                    grid[y][x] = char;
                }
            }

            // Draw the grid with borders
            println!("  {}", "-".repeat(width + 2));
            for row in grid {
                print!("  |");
                for cell in row {
                    print!("{}", cell);
                }
                println!("|");
            }
            println!("  {}", "-".repeat(width + 2));
            println!("  South Pole");
            println!("  (RA increases from left to right, Dec increases from bottom to top)");
            println!(
                "  Map width: {} characters (auto-adjusted to terminal size)",
                width
            );
        }
        "filter" => {
            // Check if output path is provided
            if let Some(path) = output_path {
                // Filter catalog and save it
                filter_and_save(&catalog, path, magnitude_limit)?;
            } else {
                return Err(
                    "Output path required for filter operation. Use --output option.".into(),
                );
            }
        }
        _ => {
            return Err(format!("Unknown operation: {}", operation).into());
        }
    }

    Ok(())
}
