//! Image moment calculation for astronomical source analysis
//!
//! This module provides functionality for calculating image moments up to second order,
//! which are used for centroid determination and shape analysis of astronomical sources.

use ndarray::Array2;

/// Image moments structure for astronomical source analysis
///
/// Contains raw image moments up to second order, used for calculating
/// centroid positions and shape parameters of detected sources.
#[derive(Debug, Clone)]
pub struct ImageMoments {
    /// Total intensity (zeroth moment)
    pub m00: f64,
    /// First moment in x direction
    pub m10: f64,
    /// First moment in y direction
    pub m01: f64,
    /// Second moment xy (cross term)
    pub m11: f64,
    /// Second moment in x direction (x^2)
    pub m20: f64,
    /// Second moment in y direction (y^2)
    pub m02: f64,
}

impl ImageMoments {
    /// Calculate raw moments up to order 2 for the given image data
    ///
    /// # Arguments
    ///
    /// * `data` - 2D array containing image data
    ///
    /// # Returns
    ///
    /// ImageMoments struct containing all calculated moments
    pub fn calculate(data: &Array2<f64>) -> Self {
        let mut m00 = 0.0;
        let mut m10 = 0.0;
        let mut m01 = 0.0;
        let mut m11 = 0.0;
        let mut m20 = 0.0;
        let mut m02 = 0.0;

        for (y, row) in data.rows().into_iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if value > 0.0 {
                    let x = x as f64;
                    let y = y as f64;

                    m00 += value;
                    m10 += x * value;
                    m01 += y * value;
                    m11 += x * y * value;
                    m20 += x * x * value;
                    m02 += y * y * value;
                }
            }
        }

        ImageMoments {
            m00,
            m10,
            m01,
            m11,
            m20,
            m02,
        }
    }

    /// Calculate the centroid coordinates from the moments
    ///
    /// # Returns
    ///
    /// (x_centroid, y_centroid) or None if m00 <= 0
    pub fn centroid(&self) -> Option<(f64, f64)> {
        if self.m00 <= 0.0 {
            return None;
        }

        let x_centroid = self.m10 / self.m00;
        let y_centroid = self.m01 / self.m00;

        if x_centroid.is_finite() && y_centroid.is_finite() {
            Some((x_centroid, y_centroid))
        } else {
            None
        }
    }

    /// Calculate central moments around the given center coordinates
    ///
    /// # Arguments
    ///
    /// * `x_center` - x coordinate of the center
    /// * `y_center` - y coordinate of the center  
    /// * `data` - 2D array containing image data
    ///
    /// # Returns
    ///
    /// (mu11, mu20, mu02) - normalized central moments
    pub fn central_moments(
        &self,
        x_center: f64,
        y_center: f64,
        data: &Array2<f64>,
    ) -> (f64, f64, f64) {
        let mut mu11 = 0.0;
        let mut mu20 = 0.0;
        let mut mu02 = 0.0;

        for (y, row) in data.rows().into_iter().enumerate() {
            for (x, &value) in row.iter().enumerate() {
                if value > 0.0 {
                    let dx = x as f64 - x_center;
                    let dy = y as f64 - y_center;

                    mu11 += dx * dy * value;
                    mu20 += dx * dx * value;
                    mu02 += dy * dy * value;
                }
            }
        }

        // Normalize by m00
        if self.m00 > 0.0 {
            mu11 /= self.m00;
            mu20 /= self.m00;
            mu02 /= self.m00;
        }

        (mu11, mu20, mu02)
    }

    /// Calculate shape parameters from the moments
    ///
    /// # Returns
    ///
    /// (fwhm, roundness, position_angle_degrees) or None if calculation fails
    pub fn shape_parameters(
        &self,
        x_center: f64,
        y_center: f64,
        data: &Array2<f64>,
    ) -> Option<(f64, f64, f64)> {
        if self.m00 <= 0.0 {
            return None;
        }

        let (mu11, mu20, mu02) = self.central_moments(x_center, y_center, data);

        let mu_sum = mu02 + mu20;
        let mu_diff = mu02 - mu20;

        if mu_sum <= 0.0 {
            return None;
        }

        // FWHM from second moments
        let fwhm = 2.0 * (2.0_f64.ln() * mu_sum).sqrt();

        // Roundness parameter (0 = circular, 1 = maximally elongated)
        let roundness = ((mu_diff * mu_diff + 4.0 * mu11 * mu11).sqrt()) / mu_sum;

        // Position angle in degrees
        let pa = 0.5 * (2.0 * mu11).atan2(mu_diff).to_degrees();
        let pa = if pa < 0.0 { pa + 180.0 } else { pa };

        Some((fwhm, roundness, pa))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moments_calculation() {
        // Create a simple 3x3 array with single pixel
        let mut data = Array2::zeros((3, 3));
        data[[1, 1]] = 1.0; // center pixel

        let moments = ImageMoments::calculate(&data);
        assert_eq!(moments.m00, 1.0);
        assert_eq!(moments.m10, 1.0); // x=1
        assert_eq!(moments.m01, 1.0); // y=1
        assert_eq!(moments.m11, 1.0); // xy=1*1=1
        assert_eq!(moments.m20, 1.0); // x^2=1^2=1
        assert_eq!(moments.m02, 1.0); // y^2=1^2=1

        // Test with a 2x2 pattern
        let mut data2 = Array2::zeros((3, 3));
        data2[[0, 0]] = 1.0;
        data2[[0, 1]] = 1.0;
        data2[[1, 0]] = 1.0;
        data2[[1, 1]] = 1.0;

        let moments2 = ImageMoments::calculate(&data2);
        assert_eq!(moments2.m00, 4.0); // total intensity
        assert_eq!(moments2.m10, 2.0); // sum of x coordinates: 0+1+0+1 = 2
        assert_eq!(moments2.m01, 2.0); // sum of y coordinates: 0+0+1+1 = 2
        assert_eq!(moments2.m11, 1.0); // sum of xy: 0*0 + 1*0 + 0*1 + 1*1 = 1
        assert_eq!(moments2.m20, 2.0); // sum of x^2: 0^2 + 1^2 + 0^2 + 1^2 = 2
        assert_eq!(moments2.m02, 2.0); // sum of y^2: 0^2 + 0^2 + 1^2 + 1^2 = 2
    }

    #[test]
    fn test_centroid_calculation() {
        // Test with single pixel
        let mut data = Array2::zeros((5, 5));
        data[[2, 3]] = 10.0; // offset from center

        let moments = ImageMoments::calculate(&data);
        let centroid = moments.centroid().unwrap();
        assert_eq!(centroid, (3.0, 2.0)); // x=3, y=2

        // Test with zero intensity
        let empty_data = Array2::zeros((3, 3));
        let empty_moments = ImageMoments::calculate(&empty_data);
        assert!(empty_moments.centroid().is_none());
    }

    #[test]
    fn test_shape_parameters() {
        // Create a circular-ish source
        let mut data = Array2::zeros((7, 7));
        let center_x = 3.0;
        let center_y = 3.0;

        // Add a small circular pattern
        for y in 2..=4 {
            for x in 2..=4 {
                let dx = x as f64 - center_x;
                let dy = y as f64 - center_y;
                let r = (dx * dx + dy * dy).sqrt();
                if r <= 1.5 {
                    data[[y, x]] = 1.0;
                }
            }
        }

        let moments = ImageMoments::calculate(&data);
        let shape = moments.shape_parameters(center_x, center_y, &data);
        assert!(shape.is_some());

        let (fwhm, roundness, _pa) = shape.unwrap();
        assert!(fwhm > 0.0);
        assert!(roundness >= 0.0 && roundness <= 1.0);
    }

    #[test]
    fn test_central_moments() {
        // Create test data with known center
        let mut data = Array2::zeros((5, 5));
        data[[2, 2]] = 1.0; // center at (2,2)

        let moments = ImageMoments::calculate(&data);
        let (mu11, mu20, mu02) = moments.central_moments(2.0, 2.0, &data);

        // For a single pixel at the center, central moments should be zero
        assert!((mu11.abs()) < 1e-10);
        assert!((mu20.abs()) < 1e-10);
        assert!((mu02.abs()) < 1e-10);
    }
}
