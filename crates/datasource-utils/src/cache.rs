//! Shared cache directory and file-existence helpers

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Return the base starfield cache directory (`~/.cache/starfield`).
pub fn cache_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache").join("starfield")
}

/// Return a cache subdirectory (e.g. `~/.cache/starfield/gaia`),
/// creating it if it does not exist.
pub fn ensure_cache_subdir(subdir: &str) -> io::Result<PathBuf> {
    let dir = cache_dir().join(subdir);
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Ensure the base cache directory exists and return its path.
pub fn ensure_cache_dir() -> io::Result<PathBuf> {
    let dir = cache_dir();
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Check if a file exists and is not empty.
pub fn file_exists_and_not_empty<P: AsRef<Path>>(path: P) -> bool {
    match fs::metadata(path) {
        Ok(metadata) => metadata.is_file() && metadata.len() > 0,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_dir_contains_starfield() {
        let dir = cache_dir();
        assert!(dir.to_str().unwrap().contains(".cache/starfield"));
    }

    #[test]
    fn test_file_exists_nonexistent() {
        assert!(!file_exists_and_not_empty("/nonexistent/path/xyz"));
    }

    #[test]
    fn test_file_exists_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, b"data").unwrap();
        assert!(file_exists_and_not_empty(&path));
    }

    #[test]
    fn test_file_exists_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, b"").unwrap();
        assert!(!file_exists_and_not_empty(&path));
    }
}
