//! Synthetic star catalog generator with realistic stellar distributions
//!
//! This module provides functionality to generate realistic synthetic star catalogs
//! for testing and development purposes. It uses statistical models of actual
//! stellar magnitude and spatial distributions to create catalogs that
//! approximate real-world astronomical data.

use rand::distributions::{Distribution, Uniform};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::f64::consts::PI;

use super::{MinimalCatalog, MinimalStar};
use crate::StarfieldError;

/// Statistical star magnitude distribution parameters
pub struct MagnitudeDistribution {
    /// Minimum star magnitude (brightest stars)
    pub min_magnitude: f64,
    /// Maximum star magnitude (dimmest stars)
    pub max_magnitude: f64,
    /// Factor for logarithmic distribution (stars per magnitude)
    /// Typical value is ~2.5 (100^0.4), meaning a 2.5x increase in the number of stars per magnitude
    pub log_base: f64,
}

impl Default for MagnitudeDistribution {
    fn default() -> Self {
        Self {
            min_magnitude: 1.0,  // Very bright stars
            max_magnitude: 12.0, // Fairly dim stars
            log_base: 2.5,       // Standard astronomical distribution (100^0.4)
        }
    }
}

/// Spatial distribution models for stars
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpatialDistribution {
    /// Uniform distribution across the entire sphere
    Uniform,
    /// Stars clustered toward the galactic plane
    GalacticPlane { concentration: f64 },
    /// Stars clustered around a specific point (simulates a star cluster)
    Cluster {
        center_ra: f64,
        center_dec: f64,
        radius: f64,
    },
}

/// Configuration for synthetic star catalog generation
pub struct SyntheticCatalogConfig {
    /// Number of stars to generate
    pub count: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Magnitude distribution parameters
    pub magnitude_dist: MagnitudeDistribution,
    /// Spatial distribution model
    pub spatial_dist: SpatialDistribution,
    /// Optional center RA for field of view limited catalogs (degrees)
    pub center_ra: Option<f64>,
    /// Optional center Dec for field of view limited catalogs (degrees)
    pub center_dec: Option<f64>,
    /// Optional field of view diameter for limited catalogs (degrees)
    pub fov_deg: Option<f64>,
    /// Optional catalog description
    pub description: String,
}

impl Default for SyntheticCatalogConfig {
    fn default() -> Self {
        Self {
            count: 100,
            seed: 42,
            magnitude_dist: MagnitudeDistribution::default(),
            spatial_dist: SpatialDistribution::Uniform,
            center_ra: None,
            center_dec: None,
            fov_deg: None,
            description: "Synthetic star catalog".to_string(),
        }
    }
}

impl SyntheticCatalogConfig {
    /// Create a new synthetic catalog configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of stars to generate
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }

    /// Set the random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the magnitude distribution parameters
    pub fn with_magnitude_range(mut self, min: f64, max: f64) -> Self {
        self.magnitude_dist.min_magnitude = min;
        self.magnitude_dist.max_magnitude = max;
        self
    }

    /// Set the magnitude distribution log base (stars per magnitude)
    pub fn with_magnitude_base(mut self, log_base: f64) -> Self {
        self.magnitude_dist.log_base = log_base;
        self
    }

    /// Set the spatial distribution model
    pub fn with_spatial_distribution(mut self, dist: SpatialDistribution) -> Self {
        self.spatial_dist = dist;
        self
    }

    /// Limit catalog to a field of view around a center point
    pub fn with_field_of_view(mut self, ra: f64, dec: f64, fov_deg: f64) -> Self {
        self.center_ra = Some(ra);
        self.center_dec = Some(dec);
        self.fov_deg = Some(fov_deg);
        self
    }

    /// Set the catalog description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Generate a synthetic star catalog with the configured parameters
    pub fn generate(&self) -> Result<MinimalCatalog, StarfieldError> {
        // Create seeded RNG
        let mut rng = StdRng::seed_from_u64(self.seed);

        // Generate stars
        let mut stars = Vec::with_capacity(self.count);

        // Determine actual number of stars to generate
        // We might need to generate more if using a field of view filter
        let generation_count = match (self.center_ra, self.center_dec, self.fov_deg) {
            (Some(_), Some(_), Some(fov)) => {
                // When using a field of view filter, we need to generate more stars
                // to ensure we have enough within the desired field
                // This is a rough estimate; 4*pi/(fov area) * count
                let sphere_area = 4.0 * PI;
                let fov_area = 2.0 * PI * (1.0 - (fov / 2.0).to_radians().cos());
                let ratio = sphere_area / fov_area;
                // Ensure at least as many as requested, plus some buffer
                (self.count as f64 * ratio.min(100.0) * 1.5) as usize
            }
            _ => self.count,
        };

        for id in 1..=generation_count {
            // Generate spatial coordinates based on distribution model
            let (ra, dec) = self.generate_star_position(&mut rng);

            // Check if the star is within our field of view, if specified
            if !self.is_in_field_of_view(ra, dec) {
                continue;
            }

            // Generate magnitude based on distribution model
            let magnitude = self.generate_star_magnitude(&mut rng);

            // Create star
            let star = MinimalStar::new(id as u64, ra, dec, magnitude);
            stars.push(star);

            // Break if we've generated enough stars
            if stars.len() >= self.count {
                break;
            }
        }

        // If we couldn't generate enough stars, adjust the catalog size
        let final_count = stars.len();
        if final_count < self.count {
            // Print a warning if we couldn't generate the requested number
            println!(
                "Warning: Could only generate {} of {} requested stars within the specified field of view",
                final_count,
                self.count
            );
        }

        // Create and return the minimal catalog
        Ok(MinimalCatalog::from_stars(stars, &self.description))
    }

    /// Generate a star's position based on the spatial distribution model
    fn generate_star_position(&self, rng: &mut StdRng) -> (f64, f64) {
        match self.spatial_dist {
            SpatialDistribution::Uniform => {
                // Uniform distribution over a sphere
                // Using the rejection method for uniform spherical distribution
                let u = Uniform::from(-1.0..1.0);
                let v = Uniform::from(0.0..1.0);

                // Generate a random point on the unit sphere
                let z: f64 = u.sample(rng); // z-coordinate in [-1, 1]
                let phi: f64 = v.sample(rng) * 2.0 * PI; // azimuthal angle in [0, 2π]

                // Convert to equatorial coordinates
                let dec = z.asin().to_degrees(); // declination = arcsin(z)
                let ra = phi.to_degrees(); // right ascension

                (ra, dec)
            }
            SpatialDistribution::GalacticPlane { concentration } => {
                // Stars concentrated toward the galactic plane
                // Using a cosine distribution for declination
                let u = Uniform::from(0.0..1.0);

                // Generate RA uniformly
                let ra: f64 = u.sample(rng) * 360.0; // RA in [0, 360]

                // Generate DEC with concentration toward the galactic plane
                // Higher concentration = more stars toward the plane
                let beta = concentration.max(0.1); // Ensure positive concentration
                let x: f64 = u.sample(rng);

                // Transform to have more probability near the equator
                // using a beta distribution approximation
                let dec = 90.0 - 180.0 * (x.powf(1.0 / beta) + (1.0 - x).powf(1.0 / beta)) / 2.0;

                (ra, dec)
            }
            SpatialDistribution::Cluster {
                center_ra,
                center_dec,
                radius,
            } => {
                // Stars clustered around a specific point
                let u = Uniform::from(0.0..1.0);

                // Generate a random radius within the cluster
                // using a normal-like distribution (more stars near center)
                let sample: f64 = u.sample(rng);
                let r = radius * (1.0 - sample.sqrt()); // Inverse transform sampling

                // Generate a random angle
                let theta = u.sample(rng) * 2.0 * PI;

                // Convert to RA/Dec offset (approximation for small angles)
                let ra_offset = r * theta.cos();
                let dec_offset = r * theta.sin();

                // Apply offset from center (handling RA wrapping)
                let ra = (center_ra + ra_offset).rem_euclid(360.0);
                let dec = (center_dec + dec_offset).clamp(-90.0, 90.0);

                (ra, dec)
            }
        }
    }

    /// Generate a star's magnitude based on the magnitude distribution
    fn generate_star_magnitude(&self, rng: &mut StdRng) -> f64 {
        let u = Uniform::from(0.0..1.0);

        // Magnitude distribution parameters
        let min_mag = self.magnitude_dist.min_magnitude;
        let max_mag = self.magnitude_dist.max_magnitude;
        let log_base = self.magnitude_dist.log_base;

        // Transform uniform random variable to follow magnitude distribution
        // N(m) ~ log_base^m
        // More stars at higher magnitudes (dimmer stars)

        // Calculate distribution range
        let exp_range = log_base.powf(max_mag - min_mag) - 1.0;

        // Sample from distribution
        let uniform_sample = u.sample(rng);
        let t = uniform_sample * exp_range + 1.0; // Transform to [1, base^range]

        // Convert to magnitude scale
        min_mag + t.log(log_base).clamp(0.0, max_mag - min_mag)
    }

    /// Check if a star is within the specified field of view
    fn is_in_field_of_view(&self, ra: f64, dec: f64) -> bool {
        match (self.center_ra, self.center_dec, self.fov_deg) {
            (Some(center_ra), Some(center_dec), Some(fov_deg)) => {
                // Convert to radians
                let ra_rad = ra.to_radians();
                let dec_rad = dec.to_radians();
                let center_ra_rad = center_ra.to_radians();
                let center_dec_rad = center_dec.to_radians();
                let fov_rad = fov_deg.to_radians();

                // Calculate angular distance using the haversine formula
                let d_ra = ra_rad - center_ra_rad;
                let d_dec = dec_rad - center_dec_rad;

                let a = (d_dec / 2.0).sin().powi(2)
                    + center_dec_rad.cos() * dec_rad.cos() * (d_ra / 2.0).sin().powi(2);
                let angular_distance = 2.0 * a.sqrt().asin();

                // Check if within field of view radius
                angular_distance <= fov_rad / 2.0
            }
            _ => true, // If no FOV specified, include all stars
        }
    }
}

/// Convenience function to create a synthetic catalog with default settings
pub fn create_synthetic_catalog(
    count: usize,
    min_magnitude: f64,
    max_magnitude: f64,
    seed: u64,
) -> Result<MinimalCatalog, StarfieldError> {
    SyntheticCatalogConfig::new()
        .with_count(count)
        .with_magnitude_range(min_magnitude, max_magnitude)
        .with_seed(seed)
        .generate()
}

/// Convenience function to create a synthetic catalog within a field of view
pub fn create_fov_catalog(
    count: usize,
    min_magnitude: f64,
    max_magnitude: f64,
    ra: f64,
    dec: f64,
    fov_deg: f64,
    seed: u64,
) -> Result<MinimalCatalog, StarfieldError> {
    SyntheticCatalogConfig::new()
        .with_count(count)
        .with_magnitude_range(min_magnitude, max_magnitude)
        .with_field_of_view(ra, dec, fov_deg)
        .with_seed(seed)
        .generate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalogs::StarPosition;

    #[test]
    fn test_synthetic_catalog_generation() {
        // Test basic catalog generation
        let catalog = create_synthetic_catalog(100, 1.0, 6.0, 42).unwrap();

        // Check we have the right number of stars
        assert_eq!(catalog.len(), 100);

        // Check magnitude bounds
        let stars = catalog.stars();
        for star in stars {
            assert!(star.magnitude >= 1.0);
            assert!(star.magnitude <= 6.0);
        }
    }

    #[test]
    fn test_fov_catalog() {
        // Test FOV-limited catalog
        let ra = 100.0;
        let dec = 45.0;
        let fov = 10.0;

        let catalog = create_fov_catalog(50, 1.0, 6.0, ra, dec, fov, 42).unwrap();

        // Check that all stars are within the FOV
        let stars = catalog.stars();
        for star in stars {
            // Convert to radians
            let ra_rad = star.ra().to_radians();
            let dec_rad = star.dec().to_radians();
            let center_ra_rad = ra.to_radians();
            let center_dec_rad = dec.to_radians();
            let fov_rad = fov.to_radians();

            // Calculate angular distance
            let d_ra = ra_rad - center_ra_rad;
            let d_dec = dec_rad - center_dec_rad;

            let a = (d_dec / 2.0).sin().powi(2)
                + center_dec_rad.cos() * dec_rad.cos() * (d_ra / 2.0).sin().powi(2);
            let angular_distance = 2.0 * a.sqrt().asin();

            // Verify star is within FOV
            assert!(angular_distance <= fov_rad / 2.0);
        }
    }

    #[test]
    fn test_magnitude_distribution() {
        // Generate a large catalog to test magnitude distribution
        let catalog = create_synthetic_catalog(10000, 0.0, 10.0, 42).unwrap();

        // Count stars in magnitude bins
        let mut bins = vec![0; 10];

        for star in catalog.stars() {
            let bin = star.magnitude.floor() as usize;
            if bin < bins.len() {
                bins[bin] += 1;
            }
        }

        // Verify distribution roughly follows expected pattern
        // Each bin should have more stars than the previous (for standard distribution)
        for i in 1..bins.len() {
            // Allow some statistical variation
            assert!(
                bins[i] > (bins[i - 1] as f64 * 0.9) as usize,
                "Expected more stars in bin {} than {}: {} vs {}",
                i,
                i - 1,
                bins[i],
                bins[i - 1]
            );
        }
    }
}
