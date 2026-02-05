//! Gaia catalog downloader
//!
//! This module provides functionality for downloading and caching Gaia catalog files.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
// No need for sync primitives yet

use crate::Result;
use crate::StarfieldError;
use regex::Regex;

// Base URL for Gaia DR1 catalog
const GAIA_DR1_BASE_URL: &str = "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/";
// URL to the MD5SUMS file
const GAIA_MD5SUMS_URL: &str = "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/MD5SUM.txt";

/// Get the Gaia cache directory path
pub fn get_gaia_cache_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("starfield")
        .join("gaia")
}

/// Ensure that the Gaia cache directory exists
pub fn ensure_gaia_cache_dir() -> io::Result<PathBuf> {
    let cache_dir = get_gaia_cache_dir();
    fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

/// Check if a file exists and is not empty
fn file_exists_and_not_empty<P: AsRef<Path>>(path: P) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}

/// Download a file from URL to a local path
fn download_file<P: AsRef<Path>>(url: &str, path: P) -> Result<()> {
    // Create parent directories if they don't exist
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent).map_err(StarfieldError::IoError)?;
    }

    // Create a temporary file first to avoid partial downloads
    let temp_path = path.as_ref().with_extension("tmp");
    let mut file = BufWriter::new(File::create(&temp_path).map_err(StarfieldError::IoError)?);

    // Create HTTP client with timeout
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600)) // 10 minute timeout for large files
        .build()
        .map_err(|e| StarfieldError::DataError(format!("Failed to create HTTP client: {}", e)))?;

    println!("Downloading: {}", url);

    // Make the request
    let mut response = client
        .get(url)
        .send()
        .map_err(|e| StarfieldError::DataError(format!("Failed to download file: {}", e)))?;

    // Check if the request was successful
    if !response.status().is_success() {
        return Err(StarfieldError::DataError(format!(
            "Failed to download file, status: {}",
            response.status()
        )));
    }

    // Get file size for progress tracking
    let total_size = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let start_time = std::time::Instant::now();

    // Copy the response body to the file with progress reporting
    let mut buffer = [0; 8192];

    loop {
        match response.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => {
                file.write_all(&buffer[..n])
                    .map_err(StarfieldError::IoError)?;
                downloaded += n as u64;

                // Print progress every 5MB
                if downloaded.is_multiple_of(5 * 1024 * 1024) {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let speed = if elapsed > 0.0 {
                        downloaded as f64 / elapsed / 1024.0 / 1024.0
                    } else {
                        0.0
                    };

                    if total_size > 0 {
                        let percentage = (downloaded as f64 / total_size as f64) * 100.0;
                        print!(
                            "\rDownloaded: {:.1}% ({:.1}MB/{:.1}MB) at {:.1} MB/s",
                            percentage,
                            downloaded as f64 / 1024.0 / 1024.0,
                            total_size as f64 / 1024.0 / 1024.0,
                            speed
                        );
                    } else {
                        print!(
                            "\rDownloaded: {:.1}MB at {:.1} MB/s",
                            downloaded as f64 / 1024.0 / 1024.0,
                            speed
                        );
                    }
                    io::stdout().flush().unwrap();
                }
            }
            Err(e) => {
                return Err(StarfieldError::DataError(format!(
                    "Error downloading file: {}",
                    e
                )))
            }
        }
    }

    // Final progress update
    let elapsed = start_time.elapsed().as_secs_f64();
    let speed = if elapsed > 0.0 {
        downloaded as f64 / elapsed / 1024.0 / 1024.0
    } else {
        0.0
    };
    println!(
        "\rDownload complete: {:.1}MB at {:.1} MB/s in {:.1}s",
        downloaded as f64 / 1024.0 / 1024.0,
        speed,
        elapsed
    );

    // Flush and sync the file
    file.flush().map_err(StarfieldError::IoError)?;
    drop(file);

    // Rename the temporary file to the final path
    fs::rename(temp_path, path).map_err(StarfieldError::IoError)?;

    Ok(())
}

/// Calculate MD5 checksum of a file
fn calculate_md5<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut file = File::open(path).map_err(StarfieldError::IoError)?;
    let mut buffer = [0; 1024 * 1024]; // 1MB buffer
    let mut context = md5::Context::new();

    loop {
        match file.read(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(n) => context.consume(&buffer[..n]),
            Err(e) => return Err(StarfieldError::IoError(e)),
        }
    }

    let digest = context.compute();
    Ok(format!("{:x}", digest))
}

/// Download the MD5SUMS file and parse it
fn download_md5sums() -> Result<HashMap<String, String>> {
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;
    let md5sums_path = cache_dir.join("MD5SUM.txt");

    // Download MD5SUMS file if it doesn't exist or is empty
    if !file_exists_and_not_empty(&md5sums_path) {
        download_file(GAIA_MD5SUMS_URL, &md5sums_path)?;
    }

    // Parse MD5SUMS file
    let file = File::open(md5sums_path).map_err(StarfieldError::IoError)?;
    let reader = BufReader::new(file);
    let mut checksums = HashMap::new();

    for line in reader.lines() {
        let line = line.map_err(StarfieldError::IoError)?;
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() >= 2 {
            let checksum = parts[0].to_string();
            let filename = parts[1].trim_start_matches("*").to_string();
            checksums.insert(filename, checksum);
        }
    }

    Ok(checksums)
}

/// List all files in the Gaia DR1 catalog index
fn list_gaia_files() -> Result<Vec<String>> {
    // Get the index page containing the file list
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| StarfieldError::DataError(format!("Failed to create HTTP client: {}", e)))?;

    println!("Fetching Gaia catalog index...");
    let response = client
        .get(GAIA_DR1_BASE_URL)
        .send()
        .map_err(|e| StarfieldError::DataError(format!("Failed to fetch Gaia index: {}", e)))?;

    if !response.status().is_success() {
        return Err(StarfieldError::DataError(format!(
            "Failed to fetch Gaia index, status: {}",
            response.status()
        )));
    }

    let html = response
        .text()
        .map_err(|e| StarfieldError::DataError(format!("Failed to read index: {}", e)))?;

    // Extract file names using regex
    // The pattern is likely GaiaSource_XXX-YYY-ZZZ.csv.gz where XXX, YYY, ZZZ are numbers
    let re = Regex::new(r#"href="(GaiaSource_\d{3}-\d{3}-\d{3}\.csv\.gz)""#)
        .map_err(|e| StarfieldError::DataError(format!("Failed to compile regex: {}", e)))?;

    let files = re
        .captures_iter(&html)
        .map(|cap| cap[1].to_string())
        .collect::<Vec<_>>();

    if files.is_empty() {
        println!("Warning: No Gaia files found in index. Using fallback enumeration.");
        // Fallback to the previous implementation if no files are found
        let fallback_files = (0..=999)
            .map(|i| format!("GaiaSource_000-000-{:03}.csv.gz", i))
            .collect::<Vec<_>>();
        return Ok(fallback_files);
    }

    println!("Found {} Gaia catalog files", files.len());
    Ok(files)
}

/// List all Gaia files that have been cached locally
pub fn list_cached_gaia_files() -> Result<Vec<PathBuf>> {
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;

    let entries = fs::read_dir(cache_dir).map_err(StarfieldError::IoError)?;
    let mut files = Vec::new();

    for entry in entries {
        let entry = entry.map_err(StarfieldError::IoError)?;
        let path = entry.path();

        if path.is_file() {
            // Check if file is either .csv or .csv.gz
            let is_csv = path.extension().is_some_and(|ext| ext == "csv");
            let is_gz = path.extension().is_some_and(|ext| ext == "gz")
                && path.to_string_lossy().ends_with(".csv.gz");

            if is_csv || is_gz {
                files.push(path);
            }
        }
    }

    Ok(files)
}

/// Verify a file against its MD5 checksum
fn verify_file<P: AsRef<Path>>(path: P, expected_md5: &str) -> Result<bool> {
    println!("Verifying {}...", path.as_ref().display());
    let actual_md5 = calculate_md5(&path)?;

    let valid = actual_md5 == expected_md5;
    if !valid {
        println!("Checksum mismatch for {}", path.as_ref().display());
        println!("  Expected: {}", expected_md5);
        println!("  Actual:   {}", actual_md5);
    } else {
        println!("Checksum verified for {}", path.as_ref().display());
    }

    Ok(valid)
}

/// Download and verify a specific Gaia file
pub fn download_gaia_file(filename: &str) -> Result<PathBuf> {
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;

    // Check if the file is a *.csv.gz and extract base name
    let base_name = if filename.ends_with(".csv.gz") {
        filename.trim_end_matches(".gz").to_string()
    } else {
        filename.to_string()
    };

    let csv_path = cache_dir.join(&base_name);
    let gz_path = cache_dir.join(filename);

    // If the CSV exists, we've already processed this file
    if file_exists_and_not_empty(&csv_path) {
        return Ok(csv_path);
    }

    // Download checksums
    let checksums = download_md5sums()?;

    // Download the gzipped file if it doesn't exist
    if !file_exists_and_not_empty(&gz_path) {
        let file_url = format!("{}{}", GAIA_DR1_BASE_URL, filename);
        download_file(&file_url, &gz_path)?;
    }

    // Verify the file
    if let Some(expected_md5) = checksums.get(filename) {
        if !verify_file(&gz_path, expected_md5)? {
            return Err(StarfieldError::DataError(format!(
                "MD5 checksum verification failed for {}",
                filename
            )));
        }
    } else {
        println!("Warning: No MD5 checksum found for {}", filename);
    }

    // Return the gz file path directly, without decompressing
    println!("File verified and ready for streaming decompression.");

    Ok(gz_path)
}

/// Download the entire Gaia catalog (all files)
pub fn download_gaia_catalog(max_files: Option<usize>) -> Result<Vec<PathBuf>> {
    let files = list_gaia_files()?;
    let max_files = max_files.unwrap_or(files.len());
    let files_to_download = files.into_iter().take(max_files).collect::<Vec<_>>();

    println!("Downloading {} Gaia catalog files", files_to_download.len());

    // Process files
    let mut downloaded_files = Vec::new();

    for (i, filename) in files_to_download.iter().enumerate() {
        println!(
            "[{}/{}] Processing {}",
            i + 1,
            files_to_download.len(),
            filename
        );
        match download_gaia_file(filename) {
            Ok(path) => {
                downloaded_files.push(path);
            }
            Err(e) => {
                println!("Error downloading {}: {}", filename, e);
                // Continue with other files
            }
        }
    }

    println!(
        "Downloaded and verified {} Gaia catalog files",
        downloaded_files.len()
    );
    Ok(downloaded_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_gaia_cache_dir();
        assert!(cache_dir
            .to_str()
            .unwrap()
            .contains(".cache/starfield/gaia"));
    }
}
