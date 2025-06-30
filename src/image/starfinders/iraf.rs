use ndarray::{s, Array2};
use std::f64::consts::PI;

/// Result type for IRAF star detection
#[derive(Debug, Clone)]
pub struct IRAFStar {
    pub id: usize,
    pub x_centroid: f64,
    pub y_centroid: f64,
    pub fwhm: f64,
    pub sharpness: f64,
    pub roundness: f64,
    pub pa: f64, // position angle in degrees
    pub npix: usize,
    pub peak: f64,
    pub flux: f64,
    pub mag: f64,
    pub sky: f64,
}

/// Configuration for IRAFStarFinder
#[derive(Debug, Clone)]
pub struct IRAFStarFinderConfig {
    /// Absolute image value threshold for source detection
    pub threshold: f64,
    /// Full-width half-maximum of the circular Gaussian kernel (pixels)
    pub fwhm: f64,
    /// Truncation radius in units of sigma
    pub sigma_radius: f64,
    /// Minimum separation in units of FWHM
    pub minsep_fwhm: f64,
    /// Lower bound on sharpness
    pub sharplo: f64,
    /// Upper bound on sharpness
    pub sharphi: f64,
    /// Lower bound on roundness
    pub roundlo: f64,
    /// Upper bound on roundness
    pub roundhi: f64,
    /// Exclude sources near borders
    pub exclude_border: bool,
    /// Keep only N brightest sources
    pub brightest: Option<usize>,
    /// Maximum allowed peak value
    pub peakmax: Option<f64>,
    /// Override minimum separation (pixels)
    pub min_separation: Option<f64>,
}

impl Default for IRAFStarFinderConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            fwhm: 3.0,
            sigma_radius: 1.5,
            minsep_fwhm: 2.5,
            sharplo: 0.5,
            sharphi: 2.0,
            roundlo: 0.0,
            roundhi: 0.2,
            exclude_border: false,
            brightest: None,
            peakmax: None,
            min_separation: None,
        }
    }
}

/// Image moments structure
#[derive(Debug, Clone)]
struct ImageMoments {
    m00: f64, // total intensity
    m10: f64, // x moment
    m01: f64, // y moment
    m11: f64, // xy moment
    m20: f64, // x^2 moment
    m02: f64, // y^2 moment
}

impl ImageMoments {
    /// Calculate raw moments up to order 2
    fn calculate(data: &Array2<f64>) -> Self {
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

    /// Calculate central moments
    fn central_moments(&self, x_center: f64, y_center: f64, data: &Array2<f64>) -> (f64, f64, f64) {
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
}

/// Main IRAF star finder implementation
pub struct IRAFStarFinder {
    config: IRAFStarFinderConfig,
    kernel: StarFinderKernel,
    min_separation: f64,
}

impl IRAFStarFinder {
    pub fn new(config: IRAFStarFinderConfig) -> Result<Self, String> {
        // Validate configuration
        if config.threshold < 0.0 {
            return Err("Threshold must be non-negative".to_string());
        }
        if config.fwhm <= 0.0 {
            return Err("FWHM must be positive".to_string());
        }
        if config.sigma_radius <= 0.0 {
            return Err("Sigma radius must be positive".to_string());
        }

        // IRAF always uses circular kernel (ratio=1.0, theta=0.0)
        let kernel = StarFinderKernel::new(
            config.fwhm,
            1.0, // circular
            0.0, // no rotation
            config.sigma_radius,
        );

        // Calculate minimum separation
        let min_separation = if let Some(sep) = config.min_separation {
            if sep < 0.0 {
                return Err("Minimum separation must be non-negative".to_string());
            }
            sep
        } else {
            (config.fwhm * config.minsep_fwhm + 0.5).floor().max(2.0)
        };

        Ok(Self {
            config,
            kernel,
            min_separation,
        })
    }

    /// Find stars in the input image
    pub fn find_stars(&self, data: &Array2<f64>, mask: Option<&Array2<bool>>) -> Vec<IRAFStar> {
        // Convolve the image with the kernel
        let convolved = convolve2d(data, &self.kernel.data);

        // Find peak positions
        let peaks = self.find_peaks(&convolved, mask);

        if peaks.is_empty() {
            return Vec::new();
        }

        // Calculate properties for each peak
        let mut stars = Vec::new();

        for (i, &(x, y)) in peaks.iter().enumerate() {
            if let Some(star) = self.measure_star(data, x, y, i + 1) {
                stars.push(star);
            }
        }

        // Apply filters
        stars = self.apply_filters(stars);

        // Select brightest if configured
        if let Some(n) = self.config.brightest {
            stars.sort_by(|a, b| b.flux.partial_cmp(&a.flux).unwrap());
            stars.truncate(n);
        }

        // Reassign IDs
        for (i, star) in stars.iter_mut().enumerate() {
            star.id = i + 1;
        }

        stars
    }

    /// Find peaks in the convolved image
    fn find_peaks(
        &self,
        convolved: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> Vec<(usize, usize)> {
        let (ny, nx) = convolved.dim();
        let mut peaks = Vec::new();

        let footprint_radius = self.min_separation as usize;

        // Border exclusion
        let (y_border, x_border) = if self.config.exclude_border {
            (self.kernel.yradius, self.kernel.xradius)
        } else {
            (0, 0)
        };

        for y in y_border..ny.saturating_sub(y_border) {
            for x in x_border..nx.saturating_sub(x_border) {
                // Check mask
                if let Some(m) = mask {
                    if m[[y, x]] {
                        continue;
                    }
                }

                let value = convolved[[y, x]];

                // Check threshold
                if value <= self.config.threshold {
                    continue;
                }

                // Check if local maximum
                let mut is_peak = true;

                for dy in -(footprint_radius as i32)..=(footprint_radius as i32) {
                    for dx in -(footprint_radius as i32)..=(footprint_radius as i32) {
                        if dx == 0 && dy == 0 {
                            continue;
                        }

                        let ny = y as i32 + dy;
                        let nx = x as i32 + dx;

                        if ny >= 0
                            && ny < convolved.nrows() as i32
                            && nx >= 0
                            && nx < convolved.ncols() as i32
                        {
                            if convolved[[ny as usize, nx as usize]] >= value {
                                is_peak = false;
                                break;
                            }
                        }
                    }
                    if !is_peak {
                        break;
                    }
                }

                if is_peak {
                    peaks.push((x, y));
                }
            }
        }

        peaks
    }

    /// Measure properties of a detected star using image moments
    fn measure_star(&self, data: &Array2<f64>, x: usize, y: usize, id: usize) -> Option<IRAFStar> {
        let kernel_shape = self.kernel.data.dim();

        // Extract cutouts centered on the star
        let data_cutout = extract_array(data, x, y, kernel_shape)?;

        // Calculate sky background
        let sky = self.calculate_sky(&data_cutout);

        // Apply sky subtraction and mask
        let mut cutout_skysub = Array2::zeros(kernel_shape);
        for ((y, x), value) in data_cutout.indexed_iter() {
            if self.kernel.mask[[y, x]] {
                let sky_sub = value - sky;
                // IRAF discards negative pixels
                cutout_skysub[[y, x]] = sky_sub.max(0.0);
            }
        }

        // Calculate moments
        let moments = ImageMoments::calculate(&cutout_skysub);

        if moments.m00 <= 0.0 {
            return None;
        }

        // Calculate centroid in cutout coordinates
        let x_cutout = moments.m10 / moments.m00;
        let y_cutout = moments.m01 / moments.m00;

        if !x_cutout.is_finite() || !y_cutout.is_finite() {
            return None;
        }

        // Convert to image coordinates
        let x_centroid = x as f64 - self.kernel.xradius as f64 + x_cutout;
        let y_centroid = y as f64 - self.kernel.yradius as f64 + y_cutout;

        // Calculate central moments
        let (mu11, mu20, mu02) = moments.central_moments(x_cutout, y_cutout, &cutout_skysub);

        // Calculate shape parameters
        let mu_sum = mu02 + mu20;
        let mu_diff = mu02 - mu20;

        let fwhm = 2.0 * (2.0_f64.ln() * mu_sum).sqrt();
        let roundness = if mu_sum > 0.0 {
            ((mu_diff * mu_diff + 4.0 * mu11 * mu11).sqrt()) / mu_sum
        } else {
            0.0
        };

        let sharpness = fwhm / self.kernel.fwhm;

        // Calculate position angle
        let pa = 0.5 * (2.0 * mu11).atan2(mu_diff).to_degrees();
        let pa = if pa < 0.0 { pa + 180.0 } else { pa };

        // Calculate peak and flux
        let peak = cutout_skysub
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let flux = cutout_skysub.sum();
        let mag = -2.5 * flux.log10();

        // Count positive pixels
        let npix = cutout_skysub.iter().filter(|&&v| v > 0.0).count();

        if npix <= 1 {
            return None;
        }

        Some(IRAFStar {
            id,
            x_centroid,
            y_centroid,
            fwhm,
            sharpness,
            roundness,
            pa,
            npix,
            peak,
            flux,
            mag,
            sky,
        })
    }

    /// Calculate local sky background
    fn calculate_sky(&self, data_cutout: &Array2<f64>) -> f64 {
        let mut sky_values = Vec::new();

        // Collect values in regions not covered by the kernel mask
        for ((y, x), &value) in data_cutout.indexed_iter() {
            if !self.kernel.mask[[y, x]] {
                sky_values.push(value);
            }
        }

        if sky_values.is_empty() {
            // Fallback: use difference between max values
            let max_data = data_cutout
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let max_conv = self
                .kernel
                .data
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            max_data - max_conv
        } else {
            // Calculate mean of sky pixels
            sky_values.iter().sum::<f64>() / sky_values.len() as f64
        }
    }

    /// Apply sharpness, roundness, and peak filters
    fn apply_filters(&self, stars: Vec<IRAFStar>) -> Vec<IRAFStar> {
        stars
            .into_iter()
            .filter(|star| {
                star.sharpness >= self.config.sharplo
                    && star.sharpness <= self.config.sharphi
                    && star.roundness >= self.config.roundlo
                    && star.roundness <= self.config.roundhi
                    && star.x_centroid.is_finite()
                    && star.y_centroid.is_finite()
                    && star.flux.is_finite()
                    && star.npix > 1
                    && self.config.peakmax.map_or(true, |max| star.peak <= max)
            })
            .collect()
    }
}

// Reuse the kernel structure from DAOStarFinder
#[derive(Debug, Clone)]
struct StarFinderKernel {
    data: Array2<f64>,
    mask: Array2<bool>,
    fwhm: f64,
    xradius: usize,
    yradius: usize,
}

impl StarFinderKernel {
    fn new(fwhm: f64, ratio: f64, theta: f64, sigma_radius: f64) -> Self {
        // Convert FWHM to sigma
        let gaussian_fwhm_to_sigma = 1.0 / (2.0 * (2.0_f64.ln()).sqrt());
        let xsigma = fwhm * gaussian_fwhm_to_sigma;
        let ysigma = xsigma * ratio;

        // Convert theta to radians
        let theta_rad = theta * PI / 180.0;
        let cost = theta_rad.cos();
        let sint = theta_rad.sin();

        // Calculate ellipse parameters
        let xsigma2 = xsigma * xsigma;
        let ysigma2 = ysigma * ysigma;

        let a = (cost * cost) / (2.0 * xsigma2) + (sint * sint) / (2.0 * ysigma2);
        let b = 0.5 * cost * sint * (1.0 / xsigma2 - 1.0 / ysigma2);
        let c = (sint * sint) / (2.0 * xsigma2) + (cost * cost) / (2.0 * ysigma2);

        // Find kernel extent
        let f = sigma_radius * sigma_radius / 2.0;
        let denom = a * c - b * b;

        let nx = 2 * ((c * f / denom).sqrt() as usize).max(2) + 1;
        let ny = 2 * ((a * f / denom).sqrt() as usize).max(2) + 1;

        let xradius = nx / 2;
        let yradius = ny / 2;

        // Create kernel grid
        let mut mask = Array2::<bool>::from_elem((ny, nx), false);
        let mut gaussian_kernel = Array2::zeros((ny, nx));

        for y in 0..ny {
            for x in 0..nx {
                let dx = x as f64 - xradius as f64;
                let dy = y as f64 - yradius as f64;

                let circular_radius = (dx * dx + dy * dy).sqrt();
                let elliptical_radius = a * dx * dx + 2.0 * b * dx * dy + c * dy * dy;

                if elliptical_radius <= f || circular_radius <= 2.0 {
                    mask[[y, x]] = true;
                    gaussian_kernel[[y, x]] = (-elliptical_radius).exp();
                }
            }
        }

        let npixels = mask.iter().filter(|&&m| m).count();

        // Calculate relative error and normalize
        let gauss_sum: f64 = gaussian_kernel
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(g, _)| g)
            .sum();

        let gauss_sum2: f64 = gaussian_kernel
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(g, _)| g * g)
            .sum();

        let variance_npixels = gauss_sum2 - (gauss_sum * gauss_sum / npixels as f64);

        // Normalize to zero sum
        let mean = gauss_sum / npixels as f64;
        let mut data = Array2::zeros((ny, nx));

        for y in 0..ny {
            for x in 0..nx {
                if mask[[y, x]] {
                    data[[y, x]] = (gaussian_kernel[[y, x]] - mean) / variance_npixels;
                }
            }
        }

        StarFinderKernel {
            data,
            mask: mask.mapv(|v| v),
            fwhm,
            xradius,
            yradius,
        }
    }
}

/// Simple 2D convolution (reused from DAOStarFinder)
fn convolve2d(data: &Array2<f64>, kernel: &Array2<f64>) -> Array2<f64> {
    let (data_h, data_w) = data.dim();
    let (kernel_h, kernel_w) = kernel.dim();
    let mut result = Array2::zeros((data_h, data_w));

    let pad_h = kernel_h / 2;
    let pad_w = kernel_w / 2;

    for y in 0..data_h {
        for x in 0..data_w {
            let mut sum = 0.0;

            for ky in 0..kernel_h {
                for kx in 0..kernel_w {
                    let dy = y as i32 + ky as i32 - pad_h as i32;
                    let dx = x as i32 + kx as i32 - pad_w as i32;

                    if dy >= 0 && dy < data_h as i32 && dx >= 0 && dx < data_w as i32 {
                        sum += data[[dy as usize, dx as usize]] * kernel[[ky, kx]];
                    }
                }
            }

            result[[y, x]] = sum;
        }
    }

    result
}

/// Extract a subarray centered at (x, y)
fn extract_array(
    data: &Array2<f64>,
    x: usize,
    y: usize,
    shape: (usize, usize),
) -> Option<Array2<f64>> {
    let (h, w) = shape;
    let (data_h, data_w) = data.dim();

    let half_h = h / 2;
    let half_w = w / 2;

    // Check bounds
    if y < half_h || y + half_h >= data_h || x < half_w || x + half_w >= data_w {
        return None;
    }

    let y_start = y - half_h;
    let x_start = x - half_w;

    Some(
        data.slice(s![y_start..y_start + h, x_start..x_start + w])
            .to_owned(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iraf_finder_creation() {
        let config = IRAFStarFinderConfig {
            threshold: 10.0,
            fwhm: 3.0,
            ..Default::default()
        };

        let finder = IRAFStarFinder::new(config);
        assert!(finder.is_ok());
    }

    #[test]
    fn test_moments_calculation() {
        // Create a simple 3x3 array
        let mut data = Array2::zeros((3, 3));
        data[[1, 1]] = 1.0; // center pixel

        let moments = ImageMoments::calculate(&data);
        assert_eq!(moments.m00, 1.0);
        assert_eq!(moments.m10, 1.0); // x=1
        assert_eq!(moments.m01, 1.0); // y=1
    }

    #[test]
    fn test_synthetic_star_iraf() {
        // Create a simple image with a Gaussian-like star
        let mut data = Array2::zeros((50, 50));
        let cx = 25.0;
        let cy = 25.0;
        let sigma = 2.0;

        for y in 0..50 {
            for x in 0..50 {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let r2 = dx * dx + dy * dy;
                data[[y, x]] = 100.0 * (-r2 / (2.0 * sigma * sigma)).exp() + 5.0;
                // add background
            }
        }

        let config = IRAFStarFinderConfig {
            threshold: 10.0,
            fwhm: 4.0,
            ..Default::default()
        };

        let finder = IRAFStarFinder::new(config).unwrap();
        let stars = finder.find_stars(&data, None);

        assert_eq!(stars.len(), 1);
        assert!((stars[0].x_centroid - cx).abs() < 1.0);
        assert!((stars[0].y_centroid - cy).abs() < 1.0);
    }
}
