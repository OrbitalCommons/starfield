//! Tool for analyzing magnitude distributions in Gaia catalog files
//!
//! This example processes downloaded Gaia catalog files and generates
//! a simple ASCII histogram of apparent magnitudes.
//!
//! Usage:
//!   cargo run --example magnitude_histogram

use std::time::Instant;

use starfield::catalogs::{GaiaCatalog, StarCatalog};
use starfield::data::list_cached_gaia_files;

/// Simple ASCII histogram generator
struct AsciiHistogram {
    min: f64,
    max: f64,
    bins: Vec<usize>,
}

impl AsciiHistogram {
    fn new(min: f64, max: f64, num_bins: usize) -> Self {
        Self {
            min,
            max,
            bins: vec![0; num_bins],
        }
    }

    fn add(&mut self, value: f64) {
        if value < self.min || value >= self.max {
            return;
        }

        let bin_width = (self.max - self.min) / self.bins.len() as f64;
        let bin_idx = ((value - self.min) / bin_width) as usize;

        if bin_idx < self.bins.len() {
            self.bins[bin_idx] += 1;
        }
    }

    fn add_all(&mut self, values: impl IntoIterator<Item = f64>) {
        for value in values {
            self.add(value);
        }
    }

    fn format(&self) -> String {
        let max_count = *self.bins.iter().max().unwrap_or(&1);
        let max_bar_width = 40;
        let bin_width = (self.max - self.min) / self.bins.len() as f64;

        let mut result = String::new();
        result.push_str("Magnitude Histogram\n");
        result.push_str("==================\n\n");

        for (i, &count) in self.bins.iter().enumerate() {
            let lower = self.min + i as f64 * bin_width;
            let upper = lower + bin_width;

            let bar_length = if count > 0 {
                (count as f64 / max_count as f64 * max_bar_width as f64).round() as usize
            } else {
                0
            };

            result.push_str(&format!(
                "{:5.1} - {:5.1} [{:8}] {}\n",
                lower,
                upper,
                count,
                "#".repeat(bar_length)
            ));
        }

        result
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Gaia Magnitude Histogram Generator");
    println!("==================================");

    // Load cached Gaia files
    println!("Finding cached Gaia files...");
    let gaia_files = list_cached_gaia_files()?;

    if gaia_files.is_empty() {
        return Err("No Gaia files found. Run gaia_downloader example first.".into());
    }

    println!("Found {} Gaia files", gaia_files.len());

    // Define magnitude range
    let min_mag = -1.0;
    let max_mag = 30.0;
    let num_bins = 31; // 1 magnitude per bin from -1 to 30

    // Create the histogram
    let mut hist = AsciiHistogram::new(min_mag, max_mag, num_bins);

    // Process each file
    let start_time = Instant::now();
    let mut total_stars: u64 = 0;

    for (i, file_path) in gaia_files.iter().enumerate() {
        println!(
            "[{}/{}] Processing {}...",
            i + 1,
            gaia_files.len(),
            file_path.display()
        );

        // Load catalog with no magnitude limit
        match GaiaCatalog::from_file(file_path, f64::MAX) {
            Ok(catalog) => {
                // Extract magnitudes and add to histogram
                let magnitudes: Vec<f64> = catalog
                    .stars()
                    .map(|star| star.phot_g_mean_mag)
                    .filter(|&mag| mag >= min_mag && mag < max_mag)
                    .collect();

                // Add all values to histogram
                hist.add_all(magnitudes);
                total_stars += catalog.len() as u64;

                println!("  Processed {} stars", catalog.len());
            }
            Err(e) => {
                println!("  Failed to process file: {}", e);
                continue;
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "\nProcessing completed in {:.2} seconds",
        elapsed.as_secs_f64()
    );
    println!("Total stars analyzed: {}", total_stars);

    // Print histogram
    println!("\n{}", hist.format());

    // Calculate stars visible to naked eye (magnitude < 6.0)
    let naked_eye_count: usize = hist
        .bins
        .iter()
        .take(7) // Bins for magnitudes -1 to 6
        .sum();

    let total_count: usize = hist.bins.iter().sum();
    let percentage = if total_count > 0 {
        (naked_eye_count as f64 / total_count as f64) * 100.0
    } else {
        0.0
    };

    println!("\nStars brighter than magnitude 6 (visible to naked eye):");
    println!("  {} stars ({:.4}% of total)", naked_eye_count, percentage);

    Ok(())
}
