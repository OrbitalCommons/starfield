//! SHA-256 verification for downloaded archive products.
//!
//! Map mosaics are gigabytes fetched over plain HTTP redirects from mirrors we
//! do not control. A truncated or substituted file still decodes as an image,
//! so corruption shows up as a subtly wrong render rather than an error. Pinning
//! a digest turns that into an immediate, legible failure.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::{Result, StarfieldError};
use sha2::{Digest, Sha256};

/// Bytes read per chunk. Large products must not be held in memory to be hashed.
const CHUNK_BYTES: usize = 1 << 20;

/// SHA-256 of a file, as lowercase hex.
///
/// Streams the file, so it is safe on multi-gigabyte products.
pub fn sha256_file<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; CHUNK_BYTES];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// SHA-256 of a byte slice, as lowercase hex.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex(&hasher.finalize())
}

/// Check a file against an expected lowercase-hex SHA-256.
///
/// Comparison is case-insensitive. The error names the file and both digests,
/// because the usual cause is a truncated download and the length of the
/// mismatch is the first thing anyone wants to see.
pub fn verify_sha256<P: AsRef<Path>>(path: P, expected: &str) -> Result<()> {
    let path = path.as_ref();
    let actual = sha256_file(path)?;
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }
    Err(StarfieldError::DataError(format!(
        "checksum mismatch for {}: expected {expected}, got {actual}",
        path.display()
    )))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// The SHA-256 of the empty input, from FIPS 180-4.
    const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    /// The SHA-256 of "abc", from FIPS 180-4.
    const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

    #[test]
    fn matches_the_fips_180_4_vectors() {
        assert_eq!(sha256_bytes(b""), EMPTY);
        assert_eq!(sha256_bytes(b"abc"), ABC);
    }

    #[test]
    fn hashes_a_file_identically_to_its_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.bin");
        let payload: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        File::create(&path).unwrap().write_all(&payload).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(&payload));
    }

    #[test]
    fn streams_across_chunk_boundaries() {
        // A payload larger than one read chunk exercises the incremental update.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.bin");
        let payload = vec![0x5au8; CHUNK_BYTES * 2 + 12345];
        File::create(&path).unwrap().write_all(&payload).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(&payload));
    }

    #[test]
    fn verification_accepts_a_match_in_either_case() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc.txt");
        File::create(&path).unwrap().write_all(b"abc").unwrap();
        assert!(verify_sha256(&path, ABC).is_ok());
        assert!(verify_sha256(&path, &ABC.to_uppercase()).is_ok());
    }

    #[test]
    fn verification_reports_both_digests_on_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc.txt");
        File::create(&path).unwrap().write_all(b"abc").unwrap();
        let err = verify_sha256(&path, EMPTY).unwrap_err().to_string();
        assert!(err.contains(EMPTY), "{err}");
        assert!(err.contains(ABC), "{err}");
    }

    #[test]
    fn a_truncated_file_fails_verification() {
        // The failure this exists to catch.
        let dir = tempfile::tempdir().unwrap();
        let full = dir.path().join("full.bin");
        let cut = dir.path().join("cut.bin");
        let payload = vec![7u8; 5000];
        File::create(&full).unwrap().write_all(&payload).unwrap();
        File::create(&cut)
            .unwrap()
            .write_all(&payload[..4999])
            .unwrap();
        let expected = sha256_file(&full).unwrap();
        assert!(verify_sha256(&cut, &expected).is_err());
    }

    #[test]
    fn missing_file_is_an_io_error() {
        assert!(sha256_file("/nonexistent/path/to/nothing").is_err());
    }
}
