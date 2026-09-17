//! Gaia catalog downloader
//!
//! This module provides functionality for downloading and caching Gaia catalog files.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read};
#[cfg(not(feature = "datastore"))]
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
#[cfg(not(feature = "datastore"))]
use std::time::Duration;
// No need for sync primitives yet

use crate::Result;
use crate::StarfieldError;

// Base URL for Gaia DR1 catalog
#[cfg(not(feature = "datastore"))]
const GAIA_DR1_BASE_URL: &str = "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/";
// URL to the MD5SUMS file
#[cfg(not(feature = "datastore"))]
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
#[cfg(not(feature = "datastore"))]
fn file_exists_and_not_empty<P: AsRef<Path>>(path: P) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}

/// Download a file from URL to a local path
#[cfg(not(feature = "datastore"))]
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

/// The release's MD5 manifest, resolved through the environment-configured
/// store; see [`md5sums_path_with`].
#[cfg(feature = "datastore")]
fn md5sums_path() -> Result<PathBuf> {
    md5sums_path_with(&super::artifacts::store_for(None)?, &get_gaia_cache_dir())
}

/// The release's MD5 manifest through `store`, adopting a copy the previous
/// downloader left in `gaia_dir`. Hermetic: reads no environment.
#[cfg(feature = "datastore")]
fn md5sums_path_with(store: &starfield_datastore::Datastore, gaia_dir: &Path) -> Result<PathBuf> {
    use super::artifacts::{adopt_legacy_cache_from, gaia_md5sums_artifact, GaiaRelease};

    let artifact = gaia_md5sums_artifact(GaiaRelease::Dr1);
    adopt_legacy_cache_from(store, &artifact, &gaia_dir.join("MD5SUM.txt"));
    Ok(store.get(&artifact)?)
}

#[cfg(not(feature = "datastore"))]
fn md5sums_path() -> Result<PathBuf> {
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;
    let md5sums_path = cache_dir.join("MD5SUM.txt");

    // Download MD5SUMS file if it doesn't exist or is empty
    if !file_exists_and_not_empty(&md5sums_path) {
        download_file(GAIA_MD5SUMS_URL, &md5sums_path)?;
    }
    Ok(md5sums_path)
}

/// Download the MD5SUMS file and parse it
fn download_md5sums() -> Result<HashMap<String, String>> {
    parse_md5sums(&md5sums_path()?)
}

/// Parse an archive MD5 manifest: `<md5> [*]<filename>` per line.
fn parse_md5sums(md5sums_path: &Path) -> Result<HashMap<String, String>> {
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

/// Every `gaia_source` shard in the release, from its MD5 manifest.
///
/// The archive's directory page is a JavaScript shell with no anchors in
/// it, so it cannot be scraped; the MD5 manifest is the authoritative,
/// immutable list of shards and is cached like any artifact. An empty list
/// is an error, never a guess.
fn list_gaia_files() -> Result<Vec<String>> {
    let checksums = download_md5sums()?;
    let mut files: Vec<String> = checksums
        .into_keys()
        .filter(|name| name.starts_with("GaiaSource_") && name.ends_with(".csv.gz"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(StarfieldError::DataError(
            "the Gaia MD5 manifest lists no GaiaSource_*.csv.gz shards".to_string(),
        ));
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

/// A shard name as the archive lists it: a bare file name, no path.
fn validate_shard_name(filename: &str) -> Result<()> {
    if filename.is_empty() || filename.contains(['/', '\\']) || filename == "." || filename == ".."
    {
        return Err(StarfieldError::DataError(format!(
            "Gaia shard name must be a bare filename, got {filename:?}"
        )));
    }
    Ok(())
}

/// Download and verify a specific Gaia file.
///
/// Returns the gzipped shard under its archive name in the Gaia cache
/// directory (`~/.cache/starfield/gaia/<file>`), a hard link (or copy) of
/// the validated blob in the pull-through cache, so readers that detect gzip
/// from the `.gz` suffix keep working. A shard absent from the release's MD5
/// manifest is refused before any fetch; one whose MD5 does not match is
/// dropped from the cache and reported.
#[cfg(feature = "datastore")]
pub fn download_gaia_file(filename: &str) -> Result<PathBuf> {
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;
    download_gaia_file_with(&super::artifacts::store_for(None)?, &cache_dir, filename)
}

/// [`download_gaia_file`] against `store`, exposing the shard in `gaia_dir`.
/// Hermetic: reads no environment.
#[cfg(feature = "datastore")]
pub(crate) fn download_gaia_file_with(
    store: &starfield_datastore::Datastore,
    gaia_dir: &Path,
    filename: &str,
) -> Result<PathBuf> {
    validate_shard_name(filename)?;
    let checksums = parse_md5sums(&md5sums_path_with(store, gaia_dir)?)?;
    let expected = checksums.get(filename).ok_or_else(|| {
        StarfieldError::DataError(format!(
            "{filename} is absent from the Gaia release MD5 manifest; refusing to fetch an unverifiable shard"
        ))
    })?;
    let alias = gaia_dir.join(filename);
    let path = fetch_shard_with(store, filename, &alias)?;
    if !verify_file(&path, expected)? {
        discard_shard_with(store, filename, &alias)?;
        return Err(StarfieldError::DataError(format!(
            "MD5 checksum verification failed for {filename}"
        )));
    }
    println!("File verified and ready for streaming decompression.");
    Ok(path)
}

/// Download and verify a specific Gaia file into the flat Gaia cache
/// directory.
#[cfg(not(feature = "datastore"))]
pub fn download_gaia_file(filename: &str) -> Result<PathBuf> {
    validate_shard_name(filename)?;
    let cache_dir = ensure_gaia_cache_dir().map_err(StarfieldError::IoError)?;
    let gz_path = cache_dir.join(filename);

    // If the decompressed CSV exists, we've already processed this file
    let csv_path = cache_dir.join(filename.trim_end_matches(".gz"));
    if file_exists_and_not_empty(&csv_path) {
        return Ok(csv_path);
    }

    let checksums = download_md5sums()?;
    if !file_exists_and_not_empty(&gz_path) {
        let file_url = format!("{}{}", GAIA_DR1_BASE_URL, filename);
        download_file(&file_url, &gz_path)?;
    }
    if let Some(expected_md5) = checksums.get(filename) {
        if !verify_file(&gz_path, expected_md5)? {
            fs::remove_file(&gz_path).map_err(StarfieldError::IoError)?;
            return Err(StarfieldError::DataError(format!(
                "MD5 checksum verification failed for {}",
                filename
            )));
        }
    } else {
        println!("Warning: No MD5 checksum found for {}", filename);
    }
    println!("File verified and ready for streaming decompression.");
    Ok(gz_path)
}

/// Resolve one shard through `store` and expose it at `alias`, its archive
/// name in the flat Gaia cache directory.
///
/// A shard the previous downloader left at `alias` is adopted (validated,
/// then copied into the store) rather than re-fetched; a flat file that
/// fails validation is ignored and the store resolves the shard normally.
/// Whatever was at `alias` is then replaced, atomically, by a hard link to
/// the validated blob (a copy when the cache directory is on another
/// filesystem), so the path always holds the bytes the store vouches for
/// and the `.gz` suffix readers key on survives the content-addressed
/// layout. Hermetic: reads no environment.
#[cfg(feature = "datastore")]
pub(crate) fn fetch_shard_with(
    store: &starfield_datastore::Datastore,
    filename: &str,
    alias: &Path,
) -> Result<PathBuf> {
    use super::artifacts::{adopt_legacy_cache_from, gaia_artifact, GaiaRelease};

    let artifact = gaia_artifact(GaiaRelease::Dr1, filename)?;
    adopt_legacy_cache_from(store, &artifact, alias);
    let blob = store.get(&artifact)?;
    publish_alias(&blob, alias)?;
    Ok(alias.to_path_buf())
}

/// Atomically point `alias` at `blob`: link or copy into a uniquely named
/// temp file beside it, then rename over whatever was there. Concurrent
/// publishers of the same alias each work on their own temp file, and a
/// failed link or copy leaves nothing behind.
#[cfg(feature = "datastore")]
fn publish_alias(blob: &Path, alias: &Path) -> Result<()> {
    if same_file(blob, alias) {
        return Ok(());
    }
    let parent = alias
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent).map_err(StarfieldError::IoError)?;
    let staging = tempfile::NamedTempFile::new_in(parent)
        .map_err(StarfieldError::IoError)?
        .into_temp_path();
    let staging_path = staging.to_path_buf();
    // The placeholder reserved a unique name; the link (or copy) takes it over.
    fs::remove_file(&staging).map_err(StarfieldError::IoError)?;
    if fs::hard_link(blob, &staging).is_err() {
        fs::copy(blob, &staging).map_err(StarfieldError::IoError)?;
    }
    let persisted = staging
        .persist(alias)
        .map_err(|error| StarfieldError::IoError(error.error));
    // rename() of two links to one inode is a no-op that leaves the source:
    // another publisher may have installed the same blob first.
    let _ = fs::remove_file(&staging_path);
    persisted
}

/// Whether `alias` already is the blob (a hard link to the same inode).
#[cfg(all(feature = "datastore", unix))]
fn same_file(blob: &Path, alias: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    match (fs::metadata(blob), fs::metadata(alias)) {
        (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
        _ => false,
    }
}

#[cfg(all(feature = "datastore", not(unix)))]
fn same_file(_blob: &Path, _alias: &Path) -> bool {
    false
}

/// Forget a shard whose archive MD5 did not match: the alias goes, and the
/// store forgets the key (a blob shared with another key is left alone by
/// the store's own reference counting).
#[cfg(feature = "datastore")]
fn discard_shard_with(
    store: &starfield_datastore::Datastore,
    filename: &str,
    alias: &Path,
) -> Result<()> {
    use super::artifacts::{gaia_artifact, GaiaRelease};

    let _ = fs::remove_file(alias);
    store.remove(&gaia_artifact(GaiaRelease::Dr1, filename)?.key)?;
    Ok(())
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
    fn shard_names_must_be_bare_filenames() {
        for bad in ["", "../x.csv.gz", "a/b.csv.gz", "..", "."] {
            assert!(validate_shard_name(bad).is_err(), "{bad:?}");
        }
        assert!(validate_shard_name("GaiaSource_000-000-000.csv.gz").is_ok());
    }

    /// A gzipped `gaia_source` shard with enough entropy to stay above the
    /// cache's 1 KiB minimum, plus its archive MD5.
    #[cfg(feature = "datastore")]
    fn synthetic_shard(rows: u64) -> Vec<u8> {
        use std::io::Write as _;
        let mut csv = String::from(
            "source_id,solution_id,ra,dec,ra_error,dec_error,parallax,parallax_error,pmra,pmdec,phot_g_mean_mag,phot_g_mean_flux,phot_variable_flag,l,b,ecl_lon,ecl_lat\n",
        );
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 1_000_000) as f64 / 1_000_000.0
        };
        for i in 0..rows {
            csv.push_str(&format!(
                "{},1635378410,{:.6},{:.6},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{:.3},{:.1},NOT_AVAILABLE,{:.5},{:.5},{:.5},{:.5}\n",
                1000 + i,
                next() * 360.0,
                next() * 180.0 - 90.0,
                next(),
                next(),
                next() * 10.0,
                next(),
                next() * 20.0 - 10.0,
                next() * 20.0 - 10.0,
                6.0 + next() * 12.0,
                next() * 10_000.0,
                next() * 360.0,
                next() * 180.0 - 90.0,
                next() * 360.0,
                next() * 180.0 - 90.0
            ));
        }
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(csv.as_bytes()).unwrap();
        encoder.finish().unwrap()
    }

    /// A loopback stand-in for the ephemeris server: canned bodies by key,
    /// and a log of the keys requested. Never the network.
    #[cfg(feature = "datastore")]
    fn stub_mirror(
        routes: HashMap<String, Vec<u8>>,
    ) -> (
        starfield_datastore::Mirror,
        std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    ) {
        use std::io::Write as _;
        use std::net::TcpListener;
        use std::sync::{Arc, Mutex};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let requested = Arc::new(Mutex::new(Vec::new()));
        let log = requested.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    continue;
                }
                let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let key = path.trim_start_matches("/artifact/").to_string();
                log.lock().unwrap().push(key.clone());
                let (status, payload) = match routes.get(&key) {
                    Some(body) => (200, body.as_slice()),
                    None => (404, &[][..]),
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} Stub\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = stream.write_all(payload);
            }
        });
        (
            starfield_datastore::Mirror::Http {
                base_url: base,
                writable: false,
            },
            requested,
        )
    }

    #[cfg(feature = "datastore")]
    fn store_with(
        root: &Path,
        mirror: starfield_datastore::Mirror,
    ) -> starfield_datastore::Datastore {
        starfield_datastore::Datastore::builder()
            .cache_root(root.to_path_buf())
            .mirror(mirror)
            .progress(false)
            .build()
            .unwrap()
    }

    #[cfg(feature = "datastore")]
    const SHARD: &str = "GaiaSource_000-000-000.csv.gz";
    #[cfg(feature = "datastore")]
    const SHARD_KEY: &str = "gaia/dr1/gaia_source/GaiaSource_000-000-000.csv.gz";
    #[cfg(feature = "datastore")]
    const MD5_KEY: &str = "gaia/dr1/gaia_source/MD5SUM.txt";

    #[cfg(feature = "datastore")]
    #[test]
    fn a_resolved_shard_keeps_its_gz_name_and_parses_through_the_catalog_reader() {
        use crate::catalogs::{GaiaCatalog, StarCatalog};

        let shard = synthetic_shard(400);
        assert!(shard.len() > 1024);
        let md5 = format!("{:x}", md5::compute(&shard));
        let (mirror, requested) = stub_mirror(HashMap::from([
            (SHARD_KEY.to_string(), shard.clone()),
            (
                MD5_KEY.to_string(),
                format!("{md5}  {SHARD}\n").repeat(40).into_bytes(),
            ),
        ]));
        let root = tempfile::tempdir().unwrap();
        let store = store_with(&root.path().join("store"), mirror);
        let gaia_dir = root.path().join("gaia");

        let path = download_gaia_file_with(&store, &gaia_dir, SHARD).unwrap();
        assert_eq!(path, gaia_dir.join(SHARD), "exposed under its archive name");
        assert!(path.to_string_lossy().ends_with(".csv.gz"));
        assert_eq!(
            fs::read(&path).unwrap(),
            shard,
            "alias is the validated blob"
        );
        assert_eq!(calculate_md5(&path).unwrap(), md5);

        let catalog = GaiaCatalog::from_file(&path, 20.0).unwrap();
        assert_eq!(catalog.len(), 400, "the suffix-detecting reader parses it");

        // A second resolve is served from disk: the mirror saw each key once.
        let again = download_gaia_file_with(&store, &gaia_dir, SHARD).unwrap();
        assert_eq!(again, path);
        let log = requested.lock().unwrap();
        assert_eq!(log.iter().filter(|k| k.as_str() == SHARD_KEY).count(), 1);
        assert_eq!(log.iter().filter(|k| k.as_str() == MD5_KEY).count(), 1);
        assert_eq!(list_cached_in(&gaia_dir), vec![path]);
    }

    #[cfg(feature = "datastore")]
    #[test]
    fn a_corrupt_flat_alias_is_replaced_by_the_validated_blob() {
        let shard = synthetic_shard(400);
        let (mirror, _) = stub_mirror(HashMap::from([(SHARD_KEY.to_string(), shard.clone())]));
        let root = tempfile::tempdir().unwrap();
        let store = store_with(&root.path().join("store"), mirror);
        let alias = root.path().join("gaia").join(SHARD);
        fs::create_dir_all(alias.parent().unwrap()).unwrap();
        fs::write(&alias, "<html>a login page under a .csv.gz name</html>").unwrap();

        let path = fetch_shard_with(&store, SHARD, &alias).unwrap();
        assert_eq!(path, alias);
        assert_eq!(
            fs::read(&path).unwrap(),
            shard,
            "the corrupt alias was replaced, not returned"
        );
        assert!(!alias
            .parent()
            .unwrap()
            .join(format!(".{SHARD}.staging"))
            .exists());
    }

    #[cfg(feature = "datastore")]
    #[test]
    fn concurrent_resolves_of_one_alias_all_end_with_the_validated_blob() {
        let shard = synthetic_shard(400);
        let (mirror, requested) =
            stub_mirror(HashMap::from([(SHARD_KEY.to_string(), shard.clone())]));
        let root = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(store_with(&root.path().join("store"), mirror));
        let alias = root.path().join("gaia").join(SHARD);
        fs::create_dir_all(alias.parent().unwrap()).unwrap();
        fs::write(&alias, "<html>stale</html>").unwrap();

        let workers: Vec<_> = (0..8)
            .map(|_| {
                let store = store.clone();
                let alias = alias.clone();
                std::thread::spawn(move || fetch_shard_with(&store, SHARD, &alias).unwrap())
            })
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), alias);
        }
        assert_eq!(fs::read(&alias).unwrap(), shard);
        assert_eq!(list_cached_in(alias.parent().unwrap()), vec![alias.clone()]);
        let leftovers: Vec<_> = fs::read_dir(alias.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no staging files survive: {leftovers:?}"
        );
        assert_eq!(
            requested
                .lock()
                .unwrap()
                .iter()
                .filter(|k| k.as_str() == SHARD_KEY)
                .count(),
            1,
            "the key lock let one fetch through"
        );
    }

    #[cfg(feature = "datastore")]
    #[test]
    fn a_shard_missing_from_the_md5_manifest_is_refused_before_any_fetch() {
        let shard = synthetic_shard(400);
        let (mirror, requested) = stub_mirror(HashMap::from([
            (SHARD_KEY.to_string(), shard),
            (
                MD5_KEY.to_string(),
                format!("{:032x}  GaiaSource_999-999-999.csv.gz\n", 0u128)
                    .repeat(40)
                    .into_bytes(),
            ),
        ]));
        let root = tempfile::tempdir().unwrap();
        let store = store_with(&root.path().join("store"), mirror);
        let err = download_gaia_file_with(&store, &root.path().join("gaia"), SHARD).unwrap_err();
        assert!(
            err.to_string()
                .contains("absent from the Gaia release MD5 manifest"),
            "{err}"
        );
        assert!(
            !requested.lock().unwrap().iter().any(|k| k == SHARD_KEY),
            "no shard request was made"
        );
    }

    #[cfg(feature = "datastore")]
    #[test]
    fn an_md5_mismatch_evicts_the_shard_and_its_alias() {
        let shard = synthetic_shard(400);
        let (mirror, _) = stub_mirror(HashMap::from([
            (SHARD_KEY.to_string(), shard),
            (
                MD5_KEY.to_string(),
                format!("{:032x}  {SHARD}\n", 0u128).repeat(40).into_bytes(),
            ),
        ]));
        let root = tempfile::tempdir().unwrap();
        let store = store_with(&root.path().join("store"), mirror);
        let gaia_dir = root.path().join("gaia");
        let err = download_gaia_file_with(&store, &gaia_dir, SHARD).unwrap_err();
        assert!(
            err.to_string().contains("MD5 checksum verification failed"),
            "{err}"
        );
        assert!(!gaia_dir.join(SHARD).exists(), "alias removed");
        assert!(
            !store.contains(
                &super::super::artifacts::gaia_artifact(
                    super::super::artifacts::GaiaRelease::Dr1,
                    SHARD
                )
                .unwrap()
                .key
            ),
            "key forgotten"
        );
    }

    #[cfg(feature = "datastore")]
    fn list_cached_in(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".csv.gz"))
            .collect();
        files.sort();
        files
    }

    #[test]
    fn test_cache_dir() {
        let cache_dir = get_gaia_cache_dir();
        assert!(cache_dir
            .to_str()
            .unwrap()
            .contains(".cache/starfield/gaia"));
    }
}
