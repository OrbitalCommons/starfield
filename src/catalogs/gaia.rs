//! Gaia star catalog implementation
//!
//! This module provides functionality for loading and using the Gaia star catalog.

use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{StarCatalog, StarData};
use crate::Result;
use crate::StarfieldError;

/// Struct representing an entry in the Gaia catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaiaEntry {
    /// Gaia source_id (unique identifier)
    pub source_id: u64,
    /// Solution ID
    pub solution_id: u64,
    /// Right ascension in degrees (epoch J2000)
    pub ra: f64,
    /// Declination in degrees (epoch J2000)
    pub dec: f64,
    /// Error in RA (mas)
    pub ra_error: f64,
    /// Error in Dec (mas)
    pub dec_error: f64,
    /// Parallax (mas), if available
    pub parallax: Option<f64>,
    /// Parallax error (mas), if available
    pub parallax_error: Option<f64>,
    /// Proper motion in RA (mas/year), if available
    pub pmra: Option<f64>,
    /// Proper motion in Dec (mas/year), if available
    pub pmdec: Option<f64>,
    /// G-band mean magnitude
    pub phot_g_mean_mag: f64,
    /// G-band mean flux (electron/s)
    pub phot_g_mean_flux: f64,
    /// Flag indicating if star is variable
    pub phot_variable_flag: String,
    /// Galactic longitude (degrees)
    pub l: f64,
    /// Galactic latitude (degrees)
    pub b: f64,
    /// Ecliptic longitude (degrees)
    pub ecl_lon: f64,
    /// Ecliptic latitude (degrees)
    pub ecl_lat: f64,
}

impl GaiaEntry {
    /// Convert RA/Dec to unit vector in ICRS coordinates
    pub fn unit_vector(&self) -> na::Vector3<f64> {
        // Convert degrees to radians
        let ra_rad = self.ra.to_radians();
        let dec_rad = self.dec.to_radians();

        // Create unit vector
        na::Vector3::new(
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        )
    }

    /// Calculate cartesian position in parsecs (if parallax is available)
    pub fn cartesian_position(&self) -> Option<na::Vector3<f64>> {
        self.parallax.filter(|&p| p > 0.0).map(|parallax| {
            // Distance in parsecs (1000/parallax_in_mas)
            let distance = 1000.0 / parallax;
            self.unit_vector() * distance
        })
    }

    /// Convert G magnitude to approximate V magnitude
    /// This is a rough approximation - for precise values, color information is needed
    pub fn approx_v_magnitude(&self) -> f64 {
        // A simple approximation (within ~0.3 mag for most stars)
        self.phot_g_mean_mag
    }
}

/// Gaia catalog
#[derive(Debug, Clone)]
pub struct GaiaCatalog {
    /// Stars by source_id
    stars: HashMap<u64, GaiaEntry>,
    /// Magnitude limit used when loading
    mag_limit: f64,
}

impl GaiaCatalog {
    /// Create a new empty Gaia catalog
    pub fn new() -> Self {
        Self {
            stars: HashMap::new(),
            mag_limit: f64::MAX,
        }
    }

    /// Load from a file (either CSV or gzipped CSV)
    pub fn from_file<P: AsRef<Path>>(path: P, mag_limit: f64) -> Result<Self> {
        let file = File::open(&path).map_err(StarfieldError::IoError)?;

        // Check if the file is empty
        let metadata = file.metadata().map_err(StarfieldError::IoError)?;
        if metadata.len() == 0 {
            return Err(StarfieldError::DataError(
                "Gaia data file is empty".to_string(),
            ));
        }

        // Determine if the file is gzipped or not
        let path_str = path.as_ref().to_string_lossy().to_string();
        let is_gzipped = path_str.ends_with(".gz");

        let reader: Box<dyn BufRead> = if is_gzipped {
            println!("Loading gzipped file: {}", path.as_ref().display());
            // Create a gzip decoder
            let gz_reader = BufReader::new(file);
            let decoder = flate2::read::GzDecoder::new(gz_reader);
            Box::new(BufReader::new(decoder))
        } else {
            println!("Loading CSV file: {}", path.as_ref().display());
            Box::new(BufReader::new(file))
        };

        let mut catalog = Self {
            stars: HashMap::new(),
            mag_limit,
        };

        let mut line_count = 0;
        let mut valid_stars = 0;

        // Read lines from the reader, which might be wrapped with a gzip decoder
        let mut lines_iter = reader.lines();
        let header = match lines_iter.next() {
            Some(Ok(line)) => line,
            _ => {
                return Err(StarfieldError::DataError(
                    "Failed to read header from Gaia file".to_string(),
                ))
            }
        };

        // Parse header to find column indices
        let headers: Vec<&str> = header.split(',').collect();
        let find_column = |name: &str| -> Result<usize> {
            headers
                .iter()
                .position(|&h| h == name)
                .ok_or_else(|| StarfieldError::DataError(format!("Missing column: {}", name)))
        };

        // Find required column indices
        let source_id_idx = find_column("source_id")?;
        let solution_id_idx = find_column("solution_id")?;
        let ra_idx = find_column("ra")?;
        let dec_idx = find_column("dec")?;
        let ra_error_idx = find_column("ra_error")?;
        let dec_error_idx = find_column("dec_error")?;
        let parallax_idx = find_column("parallax")?;
        let parallax_error_idx = find_column("parallax_error")?;
        let pmra_idx = find_column("pmra")?;
        let pmdec_idx = find_column("pmdec")?;
        let g_mag_idx = find_column("phot_g_mean_mag")?;
        let g_flux_idx = find_column("phot_g_mean_flux")?;
        let var_flag_idx = find_column("phot_variable_flag")?;
        let l_idx = find_column("l")?;
        let b_idx = find_column("b")?;
        let ecl_lon_idx = find_column("ecl_lon")?;
        let ecl_lat_idx = find_column("ecl_lat")?;

        // Process data lines
        for line_result in lines_iter {
            let line = match line_result {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("Error reading line: {}", e);
                    continue;
                }
            };

            line_count += 1;

            if line.trim().is_empty() {
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < headers.len() {
                continue; // Skip lines with insufficient columns
            }

            // Parse required fields
            let source_id = match fields[source_id_idx].parse::<u64>() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let solution_id = match fields[solution_id_idx].parse::<u64>() {
                Ok(id) => id,
                Err(_) => continue,
            };

            let ra = match fields[ra_idx].parse::<f64>() {
                Ok(ra) => ra,
                Err(_) => continue,
            };

            let dec = match fields[dec_idx].parse::<f64>() {
                Ok(dec) => dec,
                Err(_) => continue,
            };

            let ra_error = match fields[ra_error_idx].parse::<f64>() {
                Ok(err) => err,
                Err(_) => continue,
            };

            let dec_error = match fields[dec_error_idx].parse::<f64>() {
                Ok(err) => err,
                Err(_) => continue,
            };

            let g_mag = match fields[g_mag_idx].parse::<f64>() {
                Ok(mag) => mag,
                Err(_) => continue,
            };

            // Skip stars fainter than magnitude limit
            if g_mag > mag_limit {
                continue;
            }

            let g_flux = match fields[g_flux_idx].parse::<f64>() {
                Ok(flux) => flux,
                Err(_) => continue,
            };

            // Parse optional fields
            let parallax = if !fields[parallax_idx].is_empty() {
                fields[parallax_idx].parse::<f64>().ok()
            } else {
                None
            };

            let parallax_error = if !fields[parallax_error_idx].is_empty() {
                fields[parallax_error_idx].parse::<f64>().ok()
            } else {
                None
            };

            let pmra = if !fields[pmra_idx].is_empty() {
                fields[pmra_idx].parse::<f64>().ok()
            } else {
                None
            };

            let pmdec = if !fields[pmdec_idx].is_empty() {
                fields[pmdec_idx].parse::<f64>().ok()
            } else {
                None
            };

            let var_flag = fields[var_flag_idx].to_string();

            let l = match fields[l_idx].parse::<f64>() {
                Ok(l) => l,
                Err(_) => continue,
            };

            let b = match fields[b_idx].parse::<f64>() {
                Ok(b) => b,
                Err(_) => continue,
            };

            let ecl_lon = match fields[ecl_lon_idx].parse::<f64>() {
                Ok(lon) => lon,
                Err(_) => continue,
            };

            let ecl_lat = match fields[ecl_lat_idx].parse::<f64>() {
                Ok(lat) => lat,
                Err(_) => continue,
            };

            let entry = GaiaEntry {
                source_id,
                solution_id,
                ra,
                dec,
                ra_error,
                dec_error,
                parallax,
                parallax_error,
                pmra,
                pmdec,
                phot_g_mean_mag: g_mag,
                phot_g_mean_flux: g_flux,
                phot_variable_flag: var_flag,
                l,
                b,
                ecl_lon,
                ecl_lat,
            };

            catalog.stars.insert(source_id, entry);
            valid_stars += 1;
        }

        if catalog.stars.is_empty() {
            return Err(StarfieldError::DataError(format!(
                "No valid stars found in Gaia catalog. Read {} lines.",
                line_count
            )));
        }

        println!(
            "Loaded {} stars from Gaia catalog (processed {} lines).",
            valid_stars, line_count
        );
        Ok(catalog)
    }

    /// Alias for from_file with default magnitude limit (for backward compatibility)
    pub fn from_csv<P: AsRef<Path>>(path: P, mag_limit: f64) -> Result<Self> {
        Self::from_file(path, mag_limit)
    }

    /// Get stars brighter than a given magnitude
    pub fn brighter_than(&self, magnitude: f64) -> Vec<&GaiaEntry> {
        self.stars
            .values()
            .filter(|star| star.phot_g_mean_mag <= magnitude)
            .collect()
    }

    /// Get the magnitude limit used when loading this catalog
    pub fn mag_limit(&self) -> f64 {
        self.mag_limit
    }

    /// Merge another catalog into this one
    pub fn merge(&mut self, other: GaiaCatalog) -> Result<()> {
        // Merge stars, using our catalog's entries if there are duplicates
        for (id, star) in other.stars {
            self.stars.entry(id).or_insert(star);
        }

        // Keep the lower magnitude limit of the two catalogs
        self.mag_limit = self.mag_limit.min(other.mag_limit);

        Ok(())
    }

    /// Create a synthetic Gaia catalog for testing
    pub fn create_synthetic() -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut catalog = Self {
            stars: HashMap::new(),
            mag_limit: 20.0,
        };

        // Use a fixed seed for reproducibility
        let mut rng = StdRng::seed_from_u64(42);

        // Add some well-known bright stars
        let bright_stars = [
            // Sirius
            (
                6752096595359340032,
                1635721458409799680,
                101.2874,
                -16.7161,
                0.3,
                0.2,
                Some(379.21),
                Some(1.58),
                Some(-546.05),
                Some(-1223.14),
                -1.46,
                16842868.0,
                "NOT_AVAILABLE",
                227.2,
                -8.89,
                173.10,
                -5.86,
            ),
            // Canopus
            (
                5530942935258330368,
                1635721458409799681,
                95.99,
                -52.6954,
                0.2,
                0.2,
                Some(10.43),
                Some(0.56),
                Some(19.93),
                Some(23.24),
                -0.74,
                14255350.0,
                "NOT_AVAILABLE",
                261.21,
                -25.29,
                -76.25,
                -15.90,
            ),
            // Alpha Centauri
            (
                5853498713190525696,
                1635721458409799682,
                219.9,
                -60.8,
                0.1,
                0.1,
                Some(747.1),
                Some(1.33),
                Some(-3678.19),
                Some(481.84),
                -0.01,
                12567990.0,
                "NOT_AVAILABLE",
                315.73,
                -0.68,
                312.31,
                -0.3,
            ),
            // Vega
            (
                2095947430657671296,
                1635721458409799683,
                279.2,
                38.78,
                0.1,
                0.1,
                Some(130.23),
                Some(0.36),
                Some(200.94),
                Some(286.23),
                0.03,
                11986543.0,
                "NOT_AVAILABLE",
                67.45,
                19.24,
                37.95,
                61.32,
            ),
            // Betelgeuse
            (
                3428908132419580672,
                1635721458409799684,
                88.79,
                7.41,
                0.8,
                0.7,
                Some(5.95),
                Some(0.85),
                Some(26.4),
                Some(9.56),
                0.42,
                10854320.0,
                "VARIABLE",
                199.79,
                -9.02,
                204.03,
                -16.04,
            ),
        ];

        // Add known bright stars
        for star in bright_stars.iter() {
            let entry = GaiaEntry {
                source_id: star.0,
                solution_id: star.1,
                ra: star.2,
                dec: star.3,
                ra_error: star.4,
                dec_error: star.5,
                parallax: star.6,
                parallax_error: star.7,
                pmra: star.8,
                pmdec: star.9,
                phot_g_mean_mag: star.10,
                phot_g_mean_flux: star.11,
                phot_variable_flag: star.12.to_string(),
                l: star.13,
                b: star.14,
                ecl_lon: star.15,
                ecl_lat: star.16,
            };

            catalog.stars.insert(star.0, entry);
        }

        // Generate random stars
        for i in 0..5000 {
            let source_id = 5900000000000000000u64.wrapping_add(i);
            let solution_id = 1635721458409799680u64.wrapping_add(i + 5);

            let ra = rng.random_range(0.0..360.0);
            let dec = rng.random_range(-90.0..90.0);

            // Create a magnitude distribution weighted toward fainter stars
            let g_mag = (rng.random_range(0.0..20.0_f64).powf(1.5) - 0.75).max(0.0);

            // Galactic coordinates (approximation)
            let l = rng.random_range(0.0..360.0);
            let b = rng.random_range(-90.0..90.0);

            // Ecliptic coordinates (approximation)
            let ecl_lon = rng.random_range(0.0..360.0);
            let ecl_lat = rng.random_range(-90.0..90.0);

            // Create a realistic flux based on magnitude
            let g_flux = 10.0_f64.powf(10.0 - 0.4 * g_mag);

            let entry = GaiaEntry {
                source_id,
                solution_id,
                ra,
                dec,
                ra_error: rng.random_range(0.1..1.0),
                dec_error: rng.random_range(0.1..1.0),
                parallax: if rng.random_bool(0.8) {
                    Some(rng.random_range(0.1..100.0))
                } else {
                    None
                },
                parallax_error: if rng.random_bool(0.7) {
                    Some(rng.random_range(0.1..5.0))
                } else {
                    None
                },
                pmra: if rng.random_bool(0.7) {
                    Some(rng.random_range(-100.0..100.0))
                } else {
                    None
                },
                pmdec: if rng.random_bool(0.7) {
                    Some(rng.random_range(-100.0..100.0))
                } else {
                    None
                },
                phot_g_mean_mag: g_mag,
                phot_g_mean_flux: g_flux,
                phot_variable_flag: if rng.random_bool(0.05) {
                    "VARIABLE".to_string()
                } else {
                    "NOT_AVAILABLE".to_string()
                },
                l,
                b,
                ecl_lon,
                ecl_lat,
            };

            catalog.stars.insert(source_id, entry);
        }

        println!(
            "Created synthetic Gaia catalog with {} stars",
            catalog.stars.len()
        );
        catalog
    }
}

impl Default for GaiaCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl StarCatalog for GaiaCatalog {
    type Star = GaiaEntry;

    fn get_star(&self, id: usize) -> Option<&Self::Star> {
        self.stars.get(&(id as u64))
    }

    fn stars(&self) -> impl Iterator<Item = &Self::Star> {
        self.stars.values()
    }

    fn len(&self) -> usize {
        self.stars.len()
    }

    fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
    where
        F: Fn(&Self::Star) -> bool,
    {
        self.stars.values().filter(|star| predicate(star)).collect()
    }

    fn star_data(&self) -> impl Iterator<Item = StarData> + '_ {
        self.stars.values().map(|star| {
            // Gaia doesn't have B-V color index, so we'll leave it as None
            StarData::new(
                star.source_id,
                star.ra,
                star.dec,
                star.phot_g_mean_mag,
                None,
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

    #[test]
    fn test_synthetic_catalog() {
        let catalog = GaiaCatalog::create_synthetic();
        assert!(catalog.len() > 1000);

        // Test getting a specific star (Sirius)
        let sirius = catalog.get_star(6752096595359340032);
        assert!(sirius.is_some());
        if let Some(star) = sirius {
            assert!(star.phot_g_mean_mag < 0.0); // Very bright
        }

        // Test magnitude filtering
        let bright_stars = catalog.brighter_than(1.0);
        assert!(!bright_stars.is_empty());

        // Test unit vector calculation
        if let Some(vega) = catalog.get_star(2095947430657671296) {
            let vec = vega.unit_vector();
            assert!(vec.norm() > 0.99 && vec.norm() < 1.01);
        }
    }
}
