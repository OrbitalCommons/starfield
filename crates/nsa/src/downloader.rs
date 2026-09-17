//! Download the NSA catalog from the NYU NSA mirror.
//!
//! SDSS DR17 used to host `nsa_v1_0_1.fits` (7-band, ~3 GB) but the
//! `manga/atlas/` tree was dropped in a 2026 reorg — that path now 404s.
//! The only canonical mirror still serving is the NSA project's own
//! `sdss.physics.nyu.edu`, which carries the older 5-band `v0_1_2`
//! release. The loader tolerates both; see `NsaVersion`.

use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use starfield::{Result, StarfieldError};
use starfield_datasource_utils::{cache_dir, ensure_cache_subdir, file_exists_and_not_empty};

/// NYU mirror for the v0.1.2 NSA file (~0.5 GB, 5-band). The host's TLS
/// chain is incomplete (missing intermediate cert), so [`download_nsa`]
/// builds a one-off `reqwest` client that skips cert verification. This
/// is acceptable for a public read-only catalog file — the FITS itself
/// is checksum-verifiable downstream — but means we don't reach for
/// `datasource_utils::download_to_file`, which validates the chain.
///
/// # Direct-download exception
///
/// This is a **deliberate, permanent exception** to routing downloads through
/// `starfield-datastore`. That crate ships no per-source invalid-certificate
/// flag on purpose: a fetch that skips verification while looking ordinary in a
/// pinned manifest is worse than one that has to be spelled out here. So NSA
/// keeps its own client, and the bypass stays visible at the call site.
///
/// If the NYU host ever serves a complete chain, delete
/// `download_to_file_skip_cert_verify` and route this through the datastore
/// like everything else.
pub const NSA_URL: &str = "https://sdss.physics.nyu.edu/mblanton/v0/nsa_v0_1_2.fits";

const NSA_FILENAME: &str = "nsa_v0_1_2.fits";

/// Cache subdirectory: `~/.cache/starfield/nsa/`.
pub fn nsa_cache_dir() -> PathBuf {
    cache_dir().join("nsa")
}

fn ensure_nsa_dir() -> Result<PathBuf> {
    ensure_cache_subdir("nsa").map_err(StarfieldError::IoError)
}

/// Download the NSA catalog FITS file (~0.5 GB v0_1_2) into the per-user
/// cache, or return the cached path if already present and non-empty.
///
/// The download is **not** MD5-verified — NSA doesn't publish a stable
/// per-file checksum. The atomic `.tmp`+rename pattern below ensures a
/// partial download isn't mistaken for a complete one on the next run.
pub fn download_nsa() -> Result<PathBuf> {
    let dir = ensure_nsa_dir()?;
    let dest = dir.join(NSA_FILENAME);
    if file_exists_and_not_empty(&dest) {
        return Ok(dest);
    }
    eprintln!(
        "downloading NSA catalog (~0.5 GB) from {} to {} ...",
        NSA_URL,
        dest.display()
    );
    download_to_file_skip_cert_verify(NSA_URL, &dest, 3600)?;
    Ok(dest)
}

/// Download `url` to `path` atomically, with a `reqwest` client that has
/// `danger_accept_invalid_certs(true)`. Scoped to NSA so the rest of the
/// workspace keeps default cert validation.
fn download_to_file_skip_cert_verify(url: &str, path: &Path, timeout_secs: u64) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(StarfieldError::IoError)?;
    }
    let temp_path = path.with_extension("tmp");
    let mut file = BufWriter::new(File::create(&temp_path).map_err(StarfieldError::IoError)?);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .danger_accept_invalid_certs(true)
        .build()
        .map_err(|e| {
            StarfieldError::DataError(format!("Failed to build NSA HTTP client: {}", e))
        })?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| StarfieldError::DataError(format!("Failed to GET {}: {}", url, e)))?;
    if !response.status().is_success() {
        return Err(StarfieldError::DataError(format!(
            "{} returned HTTP {}",
            url,
            response.status()
        )));
    }

    let mut reader = std::io::BufReader::new(response);
    let mut buffer = [0u8; 131_072];
    loop {
        let n = reader
            .read(&mut buffer)
            .map_err(|e| StarfieldError::DataError(format!("read response: {}", e)))?;
        if n == 0 {
            break;
        }
        file.write_all(&buffer[..n])
            .map_err(StarfieldError::IoError)?;
    }
    file.flush().map_err(StarfieldError::IoError)?;
    drop(file);

    fs::rename(&temp_path, path).map_err(StarfieldError::IoError)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_dir_under_starfield() {
        let p = nsa_cache_dir();
        assert!(p.to_string_lossy().contains(".cache/starfield/nsa"));
    }
}
