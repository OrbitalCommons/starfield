//! Hipparcos star catalog implementation
//!
//! This module provides functionality for loading and using the Hipparcos star catalog.

use nalgebra as na;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{StarCatalog, StarData, StarPosition};
use crate::Result;
use crate::StarfieldError;

/// Struct representing an entry in the Hipparcos catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HipparcosEntry {
    /// Hipparcos identifier
    pub hip: usize,
    /// Right ascension in degrees (epoch J2000)
    pub ra: f64,
    /// Declination in degrees (epoch J2000)
    pub dec: f64,
    /// Magnitude (brightness)
    pub mag: f64,
    /// B-V color index
    pub b_v: Option<f64>,
    /// Proper motion in RA (mas/year)
    pub pm_ra: Option<f64>,
    /// Proper motion in declination (mas/year)
    pub pm_dec: Option<f64>,
    /// Parallax (mas)
    pub parallax: Option<f64>,
}

impl HipparcosEntry {
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

    /// Calculate cartesian position in parsecs
    pub fn cartesian_position(&self) -> Option<na::Vector3<f64>> {
        self.parallax.filter(|&p| p > 0.0).map(|parallax| {
            // Distance in parsecs (1000/parallax_in_mas)
            let distance = 1000.0 / parallax;
            self.unit_vector() * distance
        })
    }
}

/// Implement StarPosition for HipparcosEntry
impl StarPosition for HipparcosEntry {
    fn ra(&self) -> f64 {
        self.ra
    }

    fn dec(&self) -> f64 {
        self.dec
    }
}

/// Hipparcos catalog
#[derive(Debug, Clone)]
pub struct HipparcosCatalog {
    /// Stars by HIP number
    stars: HashMap<usize, HipparcosEntry>,
    /// Magnitude limit used when loading
    mag_limit: f64,
}

impl HipparcosCatalog {
    /// Create a new empty Hipparcos catalog
    pub fn new() -> Self {
        Self {
            stars: HashMap::new(),
            mag_limit: f64::MAX,
        }
    }

    /// Load from the Hipparcos .dat file
    pub fn from_dat_file<P: AsRef<Path>>(path: P, mag_limit: f64) -> Result<Self> {
        let file = File::open(&path).map_err(StarfieldError::IoError)?;

        // Check if the file is empty
        let metadata = file.metadata().map_err(StarfieldError::IoError)?;
        if metadata.len() == 0 {
            return Err(StarfieldError::DataError(
                "Hipparcos data file is empty".to_string(),
            ));
        }

        let reader = BufReader::new(file);
        let mut catalog = Self {
            stars: HashMap::new(),
            mag_limit,
        };

        let mut line_count = 0;
        let mut skipped_lines = 0;
        let mut accepted_stars = 0;

        // Process each line in the Hipparcos .dat file
        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    println!("Error reading line: {}", e);
                    skipped_lines += 1;
                    continue;
                }
            };

            line_count += 1;

            // Skip empty lines
            if line.trim().is_empty() {
                skipped_lines += 1;
                continue;
            }

            // Hipparcos fixed width format parsing
            // Check if line is long enough to contain the required fields
            if line.len() < 110 {
                // We only need up to column 103 as minimum
                skipped_lines += 1;
                continue; // Skip short lines
            }

            // The file format has pipe-separated fields, let's adapt to this
            let fields: Vec<&str> = line.split('|').collect();
            if fields.len() < 10 {
                skipped_lines += 1;
                continue; // Not enough fields
            }

            // Extract HIP number (field 1)
            let hip = match fields[1].trim().parse::<usize>() {
                Ok(hip) => hip,
                Err(_) => {
                    skipped_lines += 1;
                    continue; // Skip if HIP number is invalid
                }
            };

            // Extract magnitude (field 5)
            let mag = match fields[5].trim().parse::<f64>() {
                Ok(mag) => mag,
                Err(_) => {
                    skipped_lines += 1;
                    continue; // Skip if magnitude is invalid
                }
            };

            // Skip stars fainter than magnitude limit
            if mag > mag_limit {
                continue; // Not counting these as skipped since they're filtered by design
            }

            // Extract RA and Dec
            // The RA and Dec are in fields 8 and 9, in decimal degrees
            let ra = match fields[8].trim().parse::<f64>() {
                Ok(ra) => ra,
                Err(_) => {
                    skipped_lines += 1;
                    continue; // Skip if RA is invalid
                }
            };

            let dec = match fields[9].trim().parse::<f64>() {
                Ok(dec) => dec,
                Err(_) => {
                    skipped_lines += 1;
                    continue; // Skip if Dec is invalid
                }
            };

            // Extract parallax from field 11
            let parallax = fields.get(11).and_then(|s| s.trim().parse::<f64>().ok());

            // Extract proper motion in RA and Dec from fields 12 and 13
            let pm_ra = fields.get(12).and_then(|s| s.trim().parse::<f64>().ok());
            let pm_dec = fields.get(13).and_then(|s| s.trim().parse::<f64>().ok());

            // B-V color index from field 37
            let b_v = fields.get(37).and_then(|s| s.trim().parse::<f64>().ok());

            let entry = HipparcosEntry {
                hip,
                ra,
                dec,
                mag,
                b_v,
                pm_ra,
                pm_dec,
                parallax,
            };

            catalog.stars.insert(hip, entry);
            accepted_stars += 1;
        }

        if catalog.stars.is_empty() {
            if line_count == 0 {
                return Err(StarfieldError::DataError(
                    "Hipparcos data file appears to be empty or corrupted".to_string(),
                ));
            } else {
                return Err(StarfieldError::DataError(
                    format!("No stars loaded. Read {} lines, skipped {} due to parsing errors, but none met the magnitude limit of {}",
                            line_count, skipped_lines, mag_limit)
                ));
            }
        }

        println!("Loaded {} stars from Hipparcos catalog (read {} lines, skipped {} lines, accepted {} stars within magnitude limit)",
                 catalog.stars.len(), line_count, skipped_lines, accepted_stars);
        Ok(catalog)
    }

    /// Get stars brighter than a given magnitude
    pub fn brighter_than(&self, magnitude: f64) -> Vec<&HipparcosEntry> {
        self.stars
            .values()
            .filter(|star| star.mag <= magnitude)
            .collect()
    }

    /// Get the magnitude limit used when loading this catalog
    pub fn mag_limit(&self) -> f64 {
        self.mag_limit
    }
}

impl Default for HipparcosCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl HipparcosCatalog {
    /// Create a synthetic star catalog for testing
    pub fn create_synthetic() -> Self {
        let mut catalog = Self {
            stars: HashMap::new(),
            mag_limit: 10.0,
        };

        // Add some well-known stars

        // Sirius (Alpha Canis Majoris)
        catalog.stars.insert(
            32349,
            HipparcosEntry {
                hip: 32349,
                ra: 101.2874,
                dec: -16.7161,
                mag: -1.46,
                b_v: Some(0.00),
                pm_ra: Some(-546.05),
                pm_dec: Some(-1223.14),
                parallax: Some(379.21),
            },
        );

        // Vega (Alpha Lyrae)
        catalog.stars.insert(
            91262,
            HipparcosEntry {
                hip: 91262,
                ra: 279.2347,
                dec: 38.7837,
                mag: 0.03,
                b_v: Some(0.00),
                pm_ra: Some(200.94),
                pm_dec: Some(286.23),
                parallax: Some(130.23),
            },
        );

        // Betelgeuse (Alpha Orionis)
        catalog.stars.insert(
            27989,
            HipparcosEntry {
                hip: 27989,
                ra: 88.7929,
                dec: 7.4070,
                mag: 0.42,
                b_v: Some(1.85),
                pm_ra: Some(26.40),
                pm_dec: Some(9.56),
                parallax: Some(5.95),
            },
        );

        // Add stars in Orion's belt
        // Alnitak (Zeta Orionis)
        catalog.stars.insert(
            26727,
            HipparcosEntry {
                hip: 26727,
                ra: 84.0533,
                dec: -1.9426,
                mag: 1.74,
                b_v: Some(-0.21),
                pm_ra: Some(3.19),
                pm_dec: Some(2.03),
                parallax: Some(3.99),
            },
        );

        // Alnilam (Epsilon Orionis)
        catalog.stars.insert(
            26311,
            HipparcosEntry {
                hip: 26311,
                ra: 84.0533,
                dec: -1.2019,
                mag: 1.69,
                b_v: Some(-0.18),
                pm_ra: Some(1.49),
                pm_dec: Some(-1.06),
                parallax: Some(2.43),
            },
        );

        // Mintaka (Delta Orionis)
        catalog.stars.insert(
            25930,
            HipparcosEntry {
                hip: 25930,
                ra: 82.0310,
                dec: -0.2993,
                mag: 2.25,
                b_v: Some(-0.22),
                pm_ra: Some(0.92),
                pm_dec: Some(-1.20),
                parallax: Some(3.56),
            },
        );

        // Add 2000 random stars to simulate a larger catalog
        use rand::Rng;
        let mut rng = rand::rng();

        for i in 1..2000 {
            let hip = 100000 + i;
            let ra = rng.random_range(0.0..360.0);
            let dec = rng.random_range(-90.0..90.0);
            let mag = rng.random_range(3.0..10.0);

            catalog.stars.insert(
                hip,
                HipparcosEntry {
                    hip,
                    ra,
                    dec,
                    mag,
                    b_v: Some(rng.random_range(-0.5..2.0)),
                    pm_ra: Some(rng.random_range(-100.0..100.0)),
                    pm_dec: Some(rng.random_range(-100.0..100.0)),
                    parallax: Some(rng.random_range(1.0..1000.0)),
                },
            );
        }

        catalog
    }
}

impl StarCatalog for HipparcosCatalog {
    type Star = HipparcosEntry;

    fn get_star(&self, id: usize) -> Option<&Self::Star> {
        self.stars.get(&id)
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
        self.stars
            .values()
            .map(|star| StarData::new(star.hip as u64, star.ra, star.dec, star.mag, star.b_v))
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
    fn test_unit_vector() {
        let star = HipparcosEntry {
            hip: 1,
            ra: 0.0,  // RA = 0 degrees
            dec: 0.0, // Dec = 0 degrees
            mag: 0.0,
            b_v: None,
            pm_ra: None,
            pm_dec: None,
            parallax: None,
        };

        let vec = star.unit_vector();
        assert!((vec.x - 1.0).abs() < 1e-10);
        assert!(vec.y.abs() < 1e-10);
        assert!(vec.z.abs() < 1e-10);
    }
}
