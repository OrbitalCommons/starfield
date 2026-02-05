//! Star catalogs module
//!
//! This module provides functionality for loading and using star catalogs,
//! including efficient binary formats for optimized storage and loading.
//! It also provides an index of interesting astronomical features for targeting simulations.

use crate::coordinates::Equatorial;

pub mod binary_catalog;
pub mod features;
mod gaia;
pub mod hipparcos;
pub mod synthetic;

pub use binary_catalog::{BinaryCatalog, MinimalStar};
pub use features::{FeatureCatalog, FeatureType, SkyFeature};
pub use gaia::{GaiaCatalog, GaiaEntry};
pub use hipparcos::{HipparcosCatalog, HipparcosEntry};
pub use synthetic::{
    create_fov_catalog, create_synthetic_catalog, MagnitudeDistribution, SpatialDistribution,
    SyntheticCatalogConfig,
};

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

/// Trait for accessing star position data
pub trait StarPosition {
    /// Get star right ascension in degrees
    fn ra(&self) -> f64;

    /// Get star declination in degrees
    fn dec(&self) -> f64;
}

/// Common star properties that all catalog entries must provide
/// This represents the minimal set of properties required for rendering and calculations
#[derive(Debug, Clone, Copy)]
pub struct StarData {
    /// Star identifier
    pub id: u64,
    /// Star position in right ascension and declination (in degrees)
    pub position: Equatorial,
    /// Apparent magnitude (lower is brighter)
    pub magnitude: f64,
    /// Optional B-V color index for rendering
    pub b_v: Option<f64>,
}

impl StarData {
    /// Create a new minimal star data structure with RA/Dec in degrees
    pub fn new(id: u64, ra_deg: f64, dec_deg: f64, magnitude: f64, b_v: Option<f64>) -> Self {
        Self {
            id,
            position: Equatorial::from_degrees(ra_deg, dec_deg),
            magnitude,
            b_v,
        }
    }

    /// Create a new star data structure with an existing Equatorial position
    pub fn with_position(id: u64, position: Equatorial, magnitude: f64, b_v: Option<f64>) -> Self {
        Self {
            id,
            position,
            magnitude,
            b_v,
        }
    }

    /// Get right ascension in degrees
    pub fn ra_deg(&self) -> f64 {
        self.position.ra_degrees()
    }

    /// Get declination in degrees
    pub fn dec_deg(&self) -> f64 {
        self.position.dec_degrees()
    }
}

impl StarPosition for StarData {
    fn ra(&self) -> f64 {
        self.ra_deg()
    }

    fn dec(&self) -> f64 {
        self.dec_deg()
    }
}

/// Generic trait for all star catalogs
pub trait StarCatalog {
    /// Star entry type for this catalog
    type Star;

    /// Get a star by its identifier
    fn get_star(&self, id: usize) -> Option<&Self::Star>;

    /// Get all stars in the catalog
    fn stars(&self) -> impl Iterator<Item = &Self::Star>;

    /// Get the number of stars in the catalog
    fn len(&self) -> usize;

    /// Check if the catalog is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Filter stars based on a predicate
    fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
    where
        F: Fn(&Self::Star) -> bool;

    /// Get stars as a unified StarData format
    /// This allows consistent handling of stars from different catalog types
    fn star_data(&self) -> impl Iterator<Item = StarData> + '_;

    /// Filter stars and return them in the standard format
    fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
    where
        F: Fn(&StarData) -> bool;

    /// Get stars brighter than a specified magnitude in the standard format
    fn brighter_than(&self, magnitude: f64) -> Vec<StarData> {
        self.filter_star_data(|star| star.magnitude <= magnitude)
    }

    /// Get stars within a circular field of view
    fn stars_in_field(&self, ra_deg: f64, dec_deg: f64, fov_deg: f64) -> Vec<StarData> {
        let center = Equatorial::from_degrees(ra_deg, dec_deg);
        let radius_rad = (fov_deg / 2.0).to_radians();

        // Get cosine of the radius for faster checks
        let cos_radius = radius_rad.cos();

        self.filter_star_data(|star| {
            // Calculate the angular distance between the center and the star
            let cos_dist = star.position.dec.sin() * center.dec.sin()
                + star.position.dec.cos() * center.dec.cos() * (star.position.ra - center.ra).cos();

            // Star is in the field if cosine of distance is greater than cosine of radius
            // (inverse relationship: cos(small angle) > cos(large angle))
            cos_dist > cos_radius
        })
    }
}

/// Options for star catalog sources
#[derive(Debug, Clone)]
pub enum CatalogSource {
    /// Hipparcos catalog (default path in cache)
    Hipparcos,
    /// Binary catalog with specified path (checks relative path and cache)
    Binary(PathBuf),
    /// Random synthetic stars with specified seed and count
    Random { seed: u64, count: usize },
}

/// Get stars from the specified catalog source
pub fn get_stars_in_window(
    source: CatalogSource,
    position: Equatorial,
    fov_deg: f64,
) -> crate::Result<Vec<StarData>> {
    let ra_deg = position.ra_degrees();
    let dec_deg = position.dec_degrees();

    match source {
        CatalogSource::Random { seed, count } => {
            println!("Using synthetic stars ({})", count);
            println!("Seed: {}", seed);
            Ok(generate_synthetic_stars(
                count, ra_deg, dec_deg, fov_deg, seed,
            ))
        }

        CatalogSource::Binary(path) => {
            println!("Loading binary catalog from: {}", path.display());
            let catalog = BinaryCatalog::load(&path)?;
            println!("Loaded catalog: {}", catalog.description());
            println!("Total stars in catalog: {}", catalog.len());

            // Get stars in the specified field
            let stars = catalog.stars_in_field(ra_deg, dec_deg, fov_deg);
            println!("Found {} stars in field of view", stars.len());

            Ok(stars)
        }

        CatalogSource::Hipparcos => {
            // Default path for Hipparcos catalog
            let path = PathBuf::from("hip_main.dat");
            if !path.exists() {
                return Err(crate::StarfieldError::DataError(format!(
                    "Hipparcos catalog not found at: {}",
                    path.display()
                )));
            }

            println!("Loading Hipparcos catalog from: {}", path.display());
            // Use the Hipparcos parser
            let catalog = HipparcosCatalog::from_dat_file(&path, 8.0)?;
            println!("Total stars in catalog: {}", catalog.len());

            // Get stars in the specified field
            let stars = catalog.stars_in_field(ra_deg, dec_deg, fov_deg);
            println!("Found {} stars in field of view", stars.len());

            Ok(stars)
        }
    }
}

/// Generate synthetic stars for testing without a catalog
fn generate_synthetic_stars(
    count: usize,
    center_ra: f64,
    center_dec: f64,
    fov_deg: f64,
    seed: u64,
) -> Vec<StarData> {
    // Create a seeded RNG for reproducible results
    let mut rng = StdRng::seed_from_u64(seed);
    let mut stars = Vec::with_capacity(count);

    // Half FOV in degrees
    let half_fov = fov_deg / 2.0;

    // Distributions for random star positions and magnitudes
    let ra_dist = Uniform::from(center_ra - half_fov..center_ra + half_fov);
    let dec_dist = Uniform::from(center_dec - half_fov..center_dec + half_fov);

    // For realistic magnitude distribution, use exponential distribution
    // For every step in magnitude, there are ~2.5x more stars
    let min_mag = 3.0; // Brightest stars (lower magnitude = brighter)
    let max_mag = 8.0; // Dimmest stars

    // We'll generate random values and transform them to follow stellar magnitude distribution
    let uniform = Uniform::from(0.0..1.0);

    for id in 1..=count {
        let ra = ra_dist.sample(&mut rng);
        let dec = dec_dist.sample(&mut rng);

        // Generate magnitude using the exponential distribution
        let u = uniform.sample(&mut rng);

        // Transform uniform distribution to exponential distribution
        // Using the fact that for every magnitude step, we have 2.5× more stars
        let log_base: f64 = 2.5; // Pogson ratio
        let exp_range = log_base.powf(max_mag - min_mag) - 1.0;
        let t: f64 = u * exp_range + 1.0; // Transform to [1, 2.5^range]

        // Convert back to magnitude scale
        let magnitude = min_mag + t.log(log_base).clamp(0.0, max_mag - min_mag);

        // Generate random B-V color index (simplified)
        let b_v = if uniform.sample(&mut rng) > 0.3 {
            Some(uniform.sample(&mut rng) * 2.0 - 0.3) // Range from -0.3 to 1.7
        } else {
            None
        };

        stars.push(StarData::new(id as u64, ra, dec, magnitude, b_v));
    }

    stars
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalogs::binary_catalog::{BinaryCatalog, MinimalStar};

    /// Test the StarCatalog trait with a simple binary catalog
    #[test]
    fn test_star_data() {
        // Create a binary catalog
        let stars = vec![
            MinimalStar::new(1, 100.0, 10.0, -1.5), // Sirius-like
            MinimalStar::new(2, 50.0, -20.0, 0.5),  // Canopus-like
            MinimalStar::new(3, 150.0, 30.0, 1.2),  // Bright
            MinimalStar::new(4, 200.0, -45.0, 3.7), // Medium
            MinimalStar::new(5, 250.0, 60.0, 5.9),  // Dim
        ];

        let catalog = BinaryCatalog::from_stars(stars, "Test catalog");

        // Test star_data iterator
        let star_data: Vec<StarData> = catalog.star_data().collect();
        assert_eq!(star_data.len(), 5);

        // Test brighter_than
        let bright_stars = catalog.brighter_than(1.0);
        assert_eq!(bright_stars.len(), 2);

        // Verify the brightest star
        let brightest = star_data
            .iter()
            .min_by(|a, b| a.magnitude.partial_cmp(&b.magnitude).unwrap())
            .unwrap();
        assert_eq!(brightest.magnitude, -1.5);
    }
}
