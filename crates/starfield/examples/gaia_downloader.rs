//! Tool for downloading and managing Gaia catalog files
//!
//! This example downloads Gaia catalog files, verifies them against MD5 checksums,
//! and decompresses them for use.
//!
//! Usage:
//!   cargo run --example gaia_downloader -- [options]
//!
//! Options:
//!   --download N     Download N files (default: 1)
//!   --list           List cached files
//!   --file FILENAME  Download a specific file

use std::env;
// No need for PathBuf here

use starfield::data::{download_gaia_catalog, download_gaia_file, list_cached_gaia_files};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    // Parse command-line arguments
    let mut download_count: Option<usize> = None;
    let mut list_files = false;
    let mut specific_file: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--download" => {
                if i + 1 < args.len() {
                    download_count = Some(args[i + 1].parse()?);
                    i += 2;
                } else {
                    return Err("Missing value for --download".into());
                }
            }
            "--list" => {
                list_files = true;
                i += 1;
            }
            "--file" => {
                if i + 1 < args.len() {
                    specific_file = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    return Err("Missing value for --file".into());
                }
            }
            _ => {
                println!("Unknown argument: {}", args[i]);
                i += 1;
            }
        }
    }

    // Handle command options
    if list_files {
        println!("Listing cached Gaia catalog files:");
        let files = list_cached_gaia_files()?;

        if files.is_empty() {
            println!("No files found in cache.");
        } else {
            for (i, path) in files.iter().enumerate() {
                println!("  {}. {}", i + 1, path.display());
            }
            println!("Total: {} files", files.len());
        }
    } else if let Some(filename) = specific_file {
        println!("Downloading specific file: {}", filename);
        let path = download_gaia_file(&filename)?;
        println!("File downloaded and verified: {}", path.display());
    } else {
        println!("Downloading Gaia catalog files");
        let downloaded = download_gaia_catalog(download_count)?;
        println!("Successfully downloaded {} files", downloaded.len());

        // Print the list of downloaded files
        for (i, path) in downloaded.iter().enumerate() {
            println!("  {}. {}", i + 1, path.display());
        }
    }

    Ok(())
}
