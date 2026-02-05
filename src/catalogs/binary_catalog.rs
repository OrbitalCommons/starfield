//! Binary star catalog format for efficient storage and loading
//!
//! This module provides a compact binary format for storing star catalogs with
//! minimal fields (ID, position, magnitude), optimized for size and loading speed.

use crate::coordinates::Equatorial;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use super::{StarCatalog, StarData, StarPosition};
use crate::StarfieldError;

/// Magic bytes for identification of binary catalog format files
pub const MAGIC_BYTES: &[u8; 6] = b"BINCAT";

/// Current version of the binary format
pub const FORMAT_VERSION: u8 = 3;

/// Fixed length of the catalog description
pub const DESCRIPTION_LENGTH: usize = 128;

/// Minimal star entry with only essential fields
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MinimalStar {
    /// Star identifier (usually from source catalog)
    pub id: u64,
    /// Position in right ascension and declination
    pub position: Equatorial,
    /// Apparent magnitude
    pub magnitude: f64,
}

impl MinimalStar {
    /// Create a new minimal star entry with RA/Dec in degrees
    #[inline]
    pub fn new(id: u64, ra_deg: f64, dec_deg: f64, magnitude: f64) -> Self {
        Self {
            id,
            position: Equatorial::from_degrees(ra_deg, dec_deg),
            magnitude,
        }
    }

    /// Create from an existing Equatorial position
    pub fn with_position(id: u64, position: Equatorial, magnitude: f64) -> Self {
        Self {
            id,
            position,
            magnitude,
        }
    }

    /// Size of a single star entry in bytes
    pub const fn size_bytes() -> usize {
        // u64 + f64 + f64 + f64 = 8 + 8 + 8 + 8 = 32 bytes
        32
    }

    /// Write star data in binary format
    #[inline]
    pub fn write_binary<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u64::<LittleEndian>(self.id)?;
        writer.write_f64::<LittleEndian>(self.position.ra_degrees())?;
        writer.write_f64::<LittleEndian>(self.position.dec_degrees())?;
        writer.write_f64::<LittleEndian>(self.magnitude)?;
        Ok(())
    }

    /// Read star data from binary format
    #[inline]
    pub fn read_binary<R: Read>(reader: &mut R) -> io::Result<Self> {
        let id = reader.read_u64::<LittleEndian>()?;
        let ra_deg = reader.read_f64::<LittleEndian>()?;
        let dec_deg = reader.read_f64::<LittleEndian>()?;
        let magnitude = reader.read_f64::<LittleEndian>()?;

        Ok(MinimalStar {
            id,
            position: Equatorial::from_degrees(ra_deg, dec_deg),
            magnitude,
        })
    }
}

impl StarPosition for MinimalStar {
    fn ra(&self) -> f64 {
        self.position.ra_degrees()
    }

    fn dec(&self) -> f64 {
        self.position.dec_degrees()
    }
}

/// Binary star catalog container
#[derive(Debug, Clone)]
pub struct BinaryCatalog {
    /// Vector of minimal star entries
    stars: Vec<MinimalStar>,
    /// Catalog description
    description: String,
}

impl BinaryCatalog {
    /// Create a new empty binary catalog
    pub fn new() -> Self {
        Self {
            stars: Vec::new(),
            description: String::new(),
        }
    }

    /// Create a catalog with a description
    pub fn with_description(description: &str) -> Self {
        Self {
            stars: Vec::new(),
            description: description.to_string(),
        }
    }

    /// Create a catalog from a vector of stars
    pub fn from_stars(stars: Vec<MinimalStar>, description: &str) -> Self {
        Self {
            stars,
            description: description.to_string(),
        }
    }

    /// Get the catalog description
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the number of stars in the catalog
    pub fn len(&self) -> usize {
        self.stars.len()
    }

    /// Check if the catalog is empty
    pub fn is_empty(&self) -> bool {
        self.stars.is_empty()
    }

    /// Calculate the maximum magnitude in the catalog
    pub fn max_magnitude(&self) -> f64 {
        self.stars
            .iter()
            .map(|star| star.magnitude)
            .fold(f64::MIN, f64::max)
    }

    /// Get a reference to all stars
    pub fn stars(&self) -> &[MinimalStar] {
        &self.stars
    }

    /// Get a mutable reference to all stars
    pub fn stars_mut(&mut self) -> &mut Vec<MinimalStar> {
        &mut self.stars
    }

    /// Builder method to add a star and return a new catalog
    pub fn add_star(self, star: MinimalStar) -> Self {
        let mut new_stars = self.stars;
        new_stars.push(star);

        Self {
            stars: new_stars,
            description: self.description,
        }
    }

    /// Get stars brighter than a given magnitude
    pub fn brighter_than(&self, magnitude: f64) -> Vec<&MinimalStar> {
        self.stars
            .iter()
            .filter(|star| star.magnitude <= magnitude)
            .collect()
    }

    /// Get a filtered view of the catalog based on a predicate
    pub fn filter<F>(&self, predicate: F) -> Vec<&MinimalStar>
    where
        F: Fn(&MinimalStar) -> bool,
    {
        self.stars.iter().filter(|star| predicate(star)).collect()
    }

    /// Save catalog to a binary file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), StarfieldError> {
        // Open file for writing
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header: magic bytes
        writer.write_all(MAGIC_BYTES)?;

        // Write format version
        writer.write_u8(FORMAT_VERSION)?;

        // Write number of stars as u64
        writer.write_u64::<LittleEndian>(self.stars.len() as u64)?;

        // Write description with fixed length
        let mut description_bytes = [0u8; DESCRIPTION_LENGTH];
        let desc_bytes = self.description.as_bytes();
        let copy_len = desc_bytes.len().min(DESCRIPTION_LENGTH);
        description_bytes[..copy_len].copy_from_slice(&desc_bytes[..copy_len]);
        writer.write_all(&description_bytes)?;

        // Write all stars
        for star in &self.stars {
            star.write_binary(&mut writer)?;
        }

        // Ensure all data is flushed to disk
        writer.flush()?;

        Ok(())
    }

    /// Load catalog from a binary file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, StarfieldError> {
        // Open file for reading
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);

        // Read and verify magic bytes
        let mut magic = [0u8; 6];
        reader.read_exact(&mut magic)?;

        if &magic != MAGIC_BYTES {
            return Err(StarfieldError::DataError(
                "Invalid binary catalog format: incorrect magic bytes".to_string(),
            ));
        }

        // Read and verify version
        let version = reader.read_u8()?;
        if version != FORMAT_VERSION {
            return Err(StarfieldError::DataError(format!(
                "Unsupported binary catalog version: {}. Expected version {}",
                version, FORMAT_VERSION
            )));
        }

        // Read number of stars
        let star_count = reader.read_u64::<LittleEndian>()?;

        // Read description
        let mut description_bytes = [0u8; DESCRIPTION_LENGTH];
        reader.read_exact(&mut description_bytes)?;

        // Convert to string, trimming null bytes
        let null_pos = description_bytes
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(DESCRIPTION_LENGTH);

        let description = String::from_utf8_lossy(&description_bytes[..null_pos]).to_string();

        // Pre-allocate stars vector
        let mut stars = Vec::with_capacity(star_count as usize);

        // Read all stars
        for _ in 0..star_count {
            match MinimalStar::read_binary(&mut reader) {
                Ok(star) => stars.push(star),
                Err(e) => {
                    if e.kind() == io::ErrorKind::UnexpectedEof {
                        return Err(StarfieldError::DataError(
                            "Truncated binary catalog file".to_string(),
                        ));
                    } else {
                        return Err(StarfieldError::IoError(e));
                    }
                }
            }
        }

        // Verify we've read the expected number of stars
        if stars.len() != star_count as usize {
            return Err(StarfieldError::DataError(format!(
                "Expected {} stars but read {}",
                star_count,
                stars.len()
            )));
        }

        Ok(Self { stars, description })
    }
}

impl BinaryCatalog {
    /// Write a star catalog directly from an iterator of StarData
    ///
    /// This method is designed for large datasets where you don't want to
    /// load all the data into memory at once. It streams the data directly
    /// to disk as it's being processed.
    ///
    /// # Arguments
    /// * `path` - The file path to save the catalog to
    /// * `stars` - An iterator that yields StarData objects
    /// * `description` - A description string for the catalog
    /// * `star_count` - Optional pre-known count of stars (if None, will count during processing)
    ///
    /// # Returns
    /// * The number of stars written to the file
    pub fn write_from_star_data<P, I>(
        path: P,
        stars: I,
        description: &str,
        star_count: Option<u64>,
    ) -> Result<u64, StarfieldError>
    where
        P: AsRef<Path>,
        I: Iterator<Item = StarData>,
    {
        // Create the output file
        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);

        // Write magic bytes and version
        writer.write_all(MAGIC_BYTES)?;
        writer.write_u8(FORMAT_VERSION)?;

        // Remember position to write star count later if not provided
        let count_position = writer.stream_position()?;

        // Write placeholder count (we'll update this at the end if not provided)
        writer.write_u64::<LittleEndian>(star_count.unwrap_or(0))?;

        // Write description with fixed length
        let mut description_bytes = [0u8; DESCRIPTION_LENGTH];
        let desc_bytes = description.as_bytes();
        let copy_len = desc_bytes.len().min(DESCRIPTION_LENGTH);
        description_bytes[..copy_len].copy_from_slice(&desc_bytes[..copy_len]);
        writer.write_all(&description_bytes)?;

        // Process stars and write them
        let mut actual_count: u64 = 0;
        for star in stars {
            // Convert StarData to MinimalStar and write directly
            let minimal_star = MinimalStar::with_position(star.id, star.position, star.magnitude);

            minimal_star.write_binary(&mut writer)?;
            actual_count += 1;
        }

        // If star count wasn't provided, go back and update it
        if star_count.is_none() {
            let current_position = writer.stream_position()?;
            writer.seek(SeekFrom::Start(count_position))?;
            writer.write_u64::<LittleEndian>(actual_count)?;
            writer.seek(SeekFrom::Start(current_position))?;
        }

        // Make sure everything is written to disk
        writer.flush()?;

        Ok(actual_count)
    }
}

impl Default for BinaryCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl StarCatalog for BinaryCatalog {
    type Star = MinimalStar;

    fn get_star(&self, id: usize) -> Option<&Self::Star> {
        self.stars.iter().find(|star| star.id == id as u64)
    }

    fn stars(&self) -> impl Iterator<Item = &Self::Star> {
        self.stars.iter()
    }

    fn len(&self) -> usize {
        self.stars.len()
    }

    fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
    where
        F: Fn(&Self::Star) -> bool,
    {
        self.stars.iter().filter(|star| predicate(star)).collect()
    }

    fn star_data(&self) -> impl Iterator<Item = StarData> + '_ {
        self.stars.iter().map(|star| {
            StarData::with_position(
                star.id,
                star.position,
                star.magnitude,
                None, // Binary catalog doesn't store B-V color
            )
        })
    }

    fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
    where
        F: Fn(&StarData) -> bool,
    {
        self.star_data()
            .filter(|star_data| predicate(star_data))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use tempfile::tempdir;

    // Helper to create a test catalog
    fn create_test_catalog() -> BinaryCatalog {
        // Create stars vector directly
        let stars = vec![
            MinimalStar::new(1, 100.0, 10.0, -1.5), // Sirius-like
            MinimalStar::new(2, 50.0, -20.0, 0.5),  // Canopus-like
            MinimalStar::new(3, 150.0, 30.0, 1.2),  // Bright
            MinimalStar::new(4, 200.0, -45.0, 3.7), // Medium
            MinimalStar::new(5, 250.0, 60.0, 5.9),  // Dim
        ];

        // Create catalog from stars
        BinaryCatalog::from_stars(stars, "Test star catalog with bright stars")
    }

    #[test]
    fn test_minimal_star_binary_roundtrip() {
        let star = MinimalStar::new(42, 123.456, -45.678, 3.21);

        // Write to memory buffer
        let mut buffer = Vec::new();
        star.write_binary(&mut buffer).unwrap();

        // Read back
        let mut cursor = Cursor::new(buffer);
        let read_star = MinimalStar::read_binary(&mut cursor).unwrap();

        // Compare
        assert_eq!(star.id, read_star.id);
        assert_eq!(star.ra(), read_star.ra());
        assert_eq!(star.dec(), read_star.dec());
        assert_eq!(star.magnitude, read_star.magnitude);
    }

    #[test]
    fn test_catalog_save_load_roundtrip() {
        // Create temporary directory for test
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("test_catalog.bin");

        // Create and save catalog
        let catalog = create_test_catalog();
        catalog.save(&file_path).unwrap();

        // Load catalog
        let loaded_catalog = BinaryCatalog::load(&file_path).unwrap();

        // Verify contents
        assert_eq!(catalog.len(), loaded_catalog.len());
        assert_eq!(catalog.max_magnitude(), loaded_catalog.max_magnitude());

        // Compare individual stars
        for (orig, loaded) in catalog.stars().iter().zip(loaded_catalog.stars().iter()) {
            assert_eq!(orig.id, loaded.id);
            assert_eq!(orig.ra(), loaded.ra());
            assert_eq!(orig.dec(), loaded.dec());
            assert_eq!(orig.magnitude, loaded.magnitude);
        }
    }

    #[test]
    fn test_brighter_than_filter() {
        let catalog = create_test_catalog();

        // Test filtering
        let bright_stars = catalog.brighter_than(1.0);
        assert_eq!(bright_stars.len(), 2); // Should include Sirius-like and Canopus-like

        let visible_stars = catalog.brighter_than(6.0);
        assert_eq!(visible_stars.len(), 5); // Should include all test stars

        let very_bright = catalog.brighter_than(-1.0);
        assert_eq!(very_bright.len(), 1); // Should only include Sirius-like
    }

    #[test]
    fn test_custom_filter() {
        let catalog = create_test_catalog();

        // Filter stars in northern hemisphere
        let northern_stars = catalog.filter(|star| star.dec() > 0.0);
        assert_eq!(northern_stars.len(), 3); // Should be 3 stars with positive declination

        for star in northern_stars {
            assert!(
                star.dec() > 0.0,
                "Expected star to have positive declination"
            );
        }

        // Filter stars in a specific RA range
        let ra_range_stars = catalog.filter(|star| star.ra() >= 100.0 && star.ra() <= 200.0);
        // Count how many stars are in the 100-200 RA range in our test data
        let expected_count = catalog
            .stars()
            .iter()
            .filter(|star| star.ra() >= 100.0 && star.ra() <= 200.0)
            .count();
        assert_eq!(ra_range_stars.len(), expected_count);
    }

    #[test]
    fn test_invalid_magic_bytes() {
        // Create a temporary file with invalid magic bytes
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("invalid_catalog.bin");

        let file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(file);

        // Write invalid magic bytes
        writer.write_all(b"BADCAT").unwrap();
        writer.write_u8(FORMAT_VERSION).unwrap();
        writer.write_u64::<LittleEndian>(0).unwrap();

        // Write empty description (version 3 format)
        let empty_desc = [0u8; DESCRIPTION_LENGTH];
        writer.write_all(&empty_desc).unwrap();
        writer.flush().unwrap();

        // Try to load the catalog
        let result = BinaryCatalog::load(&file_path);
        assert!(result.is_err());

        if let Err(StarfieldError::DataError(msg)) = result {
            assert!(msg.contains("incorrect magic bytes"));
        } else {
            panic!("Expected DataError with 'incorrect magic bytes' message");
        }
    }

    #[test]
    fn test_invalid_version() {
        // Create a temporary file with invalid version
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("invalid_version.bin");

        let file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(file);

        // Write valid magic bytes but invalid version
        writer.write_all(MAGIC_BYTES).unwrap();
        writer.write_u8(2).unwrap(); // Invalid version (older version)
        writer.write_u64::<LittleEndian>(0).unwrap();

        // Empty description (version 2 had format field here)
        let empty_desc = [0u8; DESCRIPTION_LENGTH];
        writer.write_all(&empty_desc).unwrap();
        writer.flush().unwrap();

        // Try to load the catalog
        let result = BinaryCatalog::load(&file_path);
        assert!(result.is_err());

        if let Err(StarfieldError::DataError(msg)) = result {
            assert!(msg.contains("Unsupported binary catalog version"));
        } else {
            panic!("Expected DataError with 'Unsupported binary catalog version' message");
        }
    }

    #[test]
    fn test_truncated_file() {
        // Create a temporary file that's truncated
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("truncated_catalog.bin");

        let file = File::create(&file_path).unwrap();
        let mut writer = BufWriter::new(file);

        // Write header with 5 stars
        writer.write_all(MAGIC_BYTES).unwrap();
        writer.write_u8(FORMAT_VERSION).unwrap();
        writer.write_u64::<LittleEndian>(5).unwrap();

        // Write empty description (version 3 format)
        let empty_desc = [0u8; DESCRIPTION_LENGTH];
        writer.write_all(&empty_desc).unwrap();

        // But only write 2 stars
        MinimalStar::new(1, 100.0, 10.0, 1.5)
            .write_binary(&mut writer)
            .unwrap();
        MinimalStar::new(2, 50.0, -20.0, 2.5)
            .write_binary(&mut writer)
            .unwrap();
        writer.flush().unwrap();

        // Try to load the catalog
        let result = BinaryCatalog::load(&file_path);
        assert!(result.is_err());

        if let Err(StarfieldError::DataError(msg)) = result {
            assert!(msg.contains("Truncated binary catalog") || msg.contains("Expected"));
        } else {
            panic!("Expected DataError with truncated file message");
        }
    }

    #[test]
    fn test_write_from_star_data() {
        // Create a temporary directory for test
        let temp_dir = tempdir().unwrap();
        let file_path = temp_dir.path().join("streamed_catalog.bin");

        // Create some test star data
        let star_data = vec![
            StarData::new(1, 100.0, 10.0, -1.5, Some(-0.4)),
            StarData::new(2, 50.0, -20.0, 0.5, Some(0.1)),
            StarData::new(3, 150.0, 30.0, 1.2, Some(0.5)),
            StarData::new(4, 200.0, -45.0, 3.7, Some(1.1)),
            StarData::new(5, 250.0, 60.0, 5.9, Some(1.5)),
        ];

        // Test with pre-known count
        let count = BinaryCatalog::write_from_star_data(
            &file_path,
            star_data.iter().copied(),
            "Test streaming catalog",
            Some(star_data.len() as u64),
        )
        .unwrap();

        assert_eq!(count, 5);

        // Load the catalog and verify
        let loaded_catalog = BinaryCatalog::load(&file_path).unwrap();

        // Check basic properties
        assert_eq!(loaded_catalog.len(), 5);
        assert_eq!(loaded_catalog.description(), "Test streaming catalog");

        // Test without pre-known count (requires seeking)
        let file_path2 = temp_dir.path().join("streamed_catalog2.bin");
        let count2 = BinaryCatalog::write_from_star_data(
            &file_path2,
            star_data.iter().copied(),
            "Test streaming catalog 2",
            None,
        )
        .unwrap();

        assert_eq!(count2, 5);

        // Load the second catalog and verify
        let loaded_catalog2 = BinaryCatalog::load(&file_path2).unwrap();
        assert_eq!(loaded_catalog2.len(), 5);

        // Verify star data was preserved correctly in both cases
        for (i, star) in loaded_catalog.stars().iter().enumerate() {
            let original = &star_data[i];
            assert_eq!(star.id, original.id);
            assert_eq!(star.ra(), original.ra());
            assert_eq!(star.dec(), original.dec());
            assert_eq!(star.magnitude, original.magnitude);
        }
    }
}
