//! Star finding algorithms for astronomical image analysis
//!
//! This module contains implementations of various star finding algorithms
//! commonly used in astronomical data reduction.

pub mod dao;
pub mod iraf;
pub mod moments;

/// Common trait for stellar sources detected by star finding algorithms
///
/// This trait provides a unified interface for accessing fundamental properties
/// of detected stellar sources, regardless of the specific detection algorithm used.
pub trait StellarSource {
    /// Get the source's unique identifier
    fn id(&self) -> usize;

    /// Get the centroid coordinates as (x, y) in pixels
    fn get_centroid(&self) -> (f64, f64);

    /// Get the integrated flux of the source
    fn flux(&self) -> f64;

    /// Calculate the instrumental magnitude from the flux
    ///
    /// Returns the instrumental magnitude using the standard formula:
    /// mag = -2.5 × log₁₀(flux)
    ///
    /// # Returns
    ///
    /// * Finite magnitude for positive flux values
    /// * f64::INFINITY for zero or negative flux
    /// * f64::NAN for non-finite flux values
    fn mag(&self) -> f64 {
        if self.flux() > 0.0 && self.flux().is_finite() {
            -2.5 * self.flux().log10()
        } else if self.flux() <= 0.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

// Re-export commonly used types
pub use dao::{DAOStarFinder, DAOStarFinderConfig, DaoStar};
pub use iraf::{IRAFStar, IRAFStarFinder, IRAFStarFinderConfig};
pub use moments::ImageMoments;
