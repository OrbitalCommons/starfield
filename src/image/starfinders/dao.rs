use ndarray::{s, Array2, Axis};
use std::f64::consts::PI;

/// Result type for star detection
#[derive(Debug, Clone)]
pub struct DaoStar {
    pub id: usize,
    pub x_centroid: f64,
    pub y_centroid: f64,
    pub sharpness: f64,
    pub roundness1: f64,
    pub roundness2: f64,
    pub peak: f64,
    pub flux: f64,
    pub mag: f64,
}

/// Configuration for DAOStarFinder
#[derive(Debug, Clone)]
pub struct DAOStarFinderConfig {
    /// Absolute image value threshold for source detection
    pub threshold: f64,
    /// Full-width half-maximum of the Gaussian kernel (pixels)
    pub fwhm: f64,
    /// Ratio of minor to major axis (0 < ratio <= 1)
    pub ratio: f64,
    /// Position angle of major axis (degrees)
    pub theta: f64,
    /// Truncation radius in units of sigma
    pub sigma_radius: f64,
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
    /// Minimum separation between sources (pixels)
    pub min_separation: f64,
}

impl Default for DAOStarFinderConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            fwhm: 3.0,
            ratio: 1.0,
            theta: 0.0,
            sigma_radius: 1.5,
            sharplo: 0.2,
            sharphi: 1.0,
            roundlo: -1.0,
            roundhi: 1.0,
            exclude_border: false,
            brightest: None,
            peakmax: None,
            min_separation: 0.0,
        }
    }
}

/// 2D Gaussian kernel for star detection
#[derive(Debug, Clone)]
struct StarFinderKernel {
    data: Array2<f64>,
    mask: Array2<bool>,
    gaussian_kernel: Array2<f64>,
    xsigma: f64,
    ysigma: f64,
    xradius: usize,
    yradius: usize,
    npixels: usize,
    relerr: f64,
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
        let mut mask = Array2::from_elem((ny, nx), false);
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

        // Calculate relative error
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
        let relerr = 1.0 / variance_npixels.sqrt();

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
            gaussian_kernel,
            xsigma,
            ysigma,
            xradius,
            yradius,
            npixels,
            relerr,
        }
    }
}

/// Main star finder implementation
pub struct DAOStarFinder {
    config: DAOStarFinderConfig,
    kernel: StarFinderKernel,
    threshold_eff: f64,
}

impl DAOStarFinder {
    pub fn new(config: DAOStarFinderConfig) -> Result<Self, String> {
        // Validate configuration
        if config.threshold < 0.0 {
            return Err("Threshold must be non-negative".to_string());
        }
        if config.fwhm <= 0.0 {
            return Err("FWHM must be positive".to_string());
        }
        if config.ratio <= 0.0 || config.ratio > 1.0 {
            return Err("Ratio must be in (0, 1]".to_string());
        }
        if config.sigma_radius <= 0.0 {
            return Err("Sigma radius must be positive".to_string());
        }
        if config.min_separation < 0.0 {
            return Err("Minimum separation must be non-negative".to_string());
        }

        let kernel =
            StarFinderKernel::new(config.fwhm, config.ratio, config.theta, config.sigma_radius);

        let threshold_eff = config.threshold * kernel.relerr;

        Ok(Self {
            config,
            kernel,
            threshold_eff,
        })
    }

    /// Find stars in the input image
    pub fn find_stars(&self, data: &Array2<f64>, mask: Option<&Array2<bool>>) -> Vec<DaoStar> {
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
            if let Some(star) = self.measure_star(data, &convolved, x, y, i + 1) {
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

        // Define footprint for peak detection
        let footprint_radius = if self.config.min_separation > 0.0 {
            self.config.min_separation as usize
        } else {
            self.kernel.xradius.max(self.kernel.yradius)
        };

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
                if value <= self.threshold_eff {
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

    /// Measure properties of a detected star
    fn measure_star(
        &self,
        data: &Array2<f64>,
        convolved: &Array2<f64>,
        x: usize,
        y: usize,
        id: usize,
    ) -> Option<DaoStar> {
        let kernel_shape = self.kernel.data.dim();

        // Extract cutouts centered on the star
        let data_cutout = extract_array(data, x, y, kernel_shape);
        let conv_cutout = extract_array(convolved, x, y, kernel_shape);

        if data_cutout.is_none() || conv_cutout.is_none() {
            return None;
        }

        let data_cutout = data_cutout.unwrap();
        let conv_cutout = conv_cutout.unwrap();

        // Calculate basic properties
        let peak = data_cutout[[self.kernel.yradius, self.kernel.xradius]];
        let conv_peak = conv_cutout[[self.kernel.yradius, self.kernel.xradius]];

        // Calculate sharpness
        let masked_sum: f64 = data_cutout
            .iter()
            .zip(self.kernel.mask.iter())
            .filter(|(_, &m)| m)
            .map(|(d, _)| d)
            .sum();

        let data_mean = (masked_sum - peak) / (self.kernel.npixels - 1) as f64;
        let sharpness = (peak - data_mean) / conv_peak;

        // Calculate roundness1 (symmetry-based)
        let roundness1 = self.calculate_roundness1(&conv_cutout);

        // Calculate centroid and roundness2 using marginal fits
        let (dx, hx) = self.marginal_fit(&data_cutout, 0);
        let (dy, hy) = self.marginal_fit(&data_cutout, 1);

        if !dx.is_finite() || !dy.is_finite() || !hx.is_finite() || !hy.is_finite() {
            return None;
        }

        let x_centroid = x as f64 + dx;
        let y_centroid = y as f64 + dy;
        let roundness2 = 2.0 * (hx - hy) / (hx + hy);

        // Calculate flux
        let flux: f64 = data_cutout.sum();
        let mag = -2.5 * flux.log10();

        Some(DaoStar {
            id,
            x_centroid,
            y_centroid,
            sharpness,
            roundness1,
            roundness2,
            peak,
            flux,
            mag,
        })
    }

    /// Calculate symmetry-based roundness
    fn calculate_roundness1(&self, conv_cutout: &Array2<f64>) -> f64 {
        let mut cutout = conv_cutout.clone();

        // Zero out the central pixel
        cutout[[self.kernel.yradius, self.kernel.xradius]] = 0.0;

        let yc = self.kernel.yradius;
        let xc = self.kernel.xradius;

        // Calculate quadrant sums
        let quad1: f64 = cutout.slice(s![..=yc, xc + 1..]).sum();
        let quad2: f64 = cutout.slice(s![..yc, ..=xc]).sum();
        let quad3: f64 = cutout.slice(s![yc.., ..xc]).sum();
        let quad4: f64 = cutout.slice(s![yc + 1.., xc..]).sum();

        let sum2 = -quad1 + quad2 - quad3 + quad4;
        let sum4: f64 = cutout.iter().map(|v| v.abs()).sum();

        if sum4 == 0.0 {
            0.0
        } else {
            2.0 * sum2 / sum4
        }
    }

    /// Fit 1D Gaussian to marginal distribution
    fn marginal_fit(&self, data_cutout: &Array2<f64>, axis: usize) -> (f64, f64) {
        let (ny, nx) = data_cutout.dim();
        let yc = self.kernel.yradius;
        let xc = self.kernel.xradius;

        // Create triangular weighting functions
        let mut weights = Array2::zeros((ny, nx));
        for y in 0..ny {
            for x in 0..nx {
                let xwt = (xc as f64 - (x as f64 - xc as f64).abs() + 1.0).max(0.0);
                let ywt = (yc as f64 - (y as f64 - yc as f64).abs() + 1.0).max(0.0);
                weights[[y, x]] = if axis == 0 { ywt } else { xwt };
            }
        }

        // Compute marginal sums
        let marginal_data = if axis == 0 {
            data_cutout.sum_axis(Axis(0))
        } else {
            data_cutout.sum_axis(Axis(1))
        };

        let marginal_kernel = if axis == 0 {
            self.kernel.gaussian_kernel.sum_axis(Axis(0))
        } else {
            self.kernel.gaussian_kernel.sum_axis(Axis(1))
        };

        let wt_1d = if axis == 0 {
            weights.row(0).to_owned()
        } else {
            weights.column(0).to_owned()
        };

        // Weighted sums for linear least squares
        let wt_sum: f64 = wt_1d.sum();
        let kern_sum: f64 = marginal_kernel
            .iter()
            .zip(wt_1d.iter())
            .map(|(k, w)| k * w)
            .sum();
        let kern2_sum: f64 = marginal_kernel
            .iter()
            .zip(wt_1d.iter())
            .map(|(k, w)| k * k * w)
            .sum();
        let data_sum: f64 = marginal_data
            .iter()
            .zip(wt_1d.iter())
            .map(|(d, w)| d * w)
            .sum();
        let data_kern_sum: f64 = marginal_data
            .iter()
            .zip(marginal_kernel.iter())
            .zip(wt_1d.iter())
            .map(|((d, k), w)| d * k * w)
            .sum();

        // Fit amplitude
        let hx_numer = data_kern_sum - (data_sum * kern_sum) / wt_sum;
        let hx_denom = kern2_sum - (kern_sum * kern_sum / wt_sum);

        if hx_numer <= 0.0 || hx_denom <= 0.0 {
            return (0.0, f64::NAN);
        }

        let hx = hx_numer / hx_denom;

        // Calculate centroid shift (simplified)
        let center = if axis == 0 { xc } else { yc };
        let mut dx_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, (d, w)) in marginal_data.iter().zip(wt_1d.iter()).enumerate() {
            let offset = i as f64 - center as f64;
            dx_sum += d * offset * w;
            weight_sum += d * w;
        }

        let dx = if weight_sum > 0.0 {
            dx_sum / weight_sum
        } else {
            0.0
        };

        (dx, hx)
    }

    /// Apply sharpness, roundness, and peak filters
    fn apply_filters(&self, stars: Vec<DaoStar>) -> Vec<DaoStar> {
        stars
            .into_iter()
            .filter(|star| {
                star.sharpness >= self.config.sharplo
                    && star.sharpness <= self.config.sharphi
                    && star.roundness1 >= self.config.roundlo
                    && star.roundness1 <= self.config.roundhi
                    && star.roundness2 >= self.config.roundlo
                    && star.roundness2 <= self.config.roundhi
                    && star.x_centroid.is_finite()
                    && star.y_centroid.is_finite()
                    && star.flux.is_finite()
                    && self.config.peakmax.map_or(true, |max| star.peak <= max)
            })
            .collect()
    }
}

/// Simple 2D convolution (replace with FFT-based for performance)
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
    fn test_star_finder_creation() {
        let config = DAOStarFinderConfig {
            threshold: 10.0,
            fwhm: 3.0,
            ..Default::default()
        };

        let finder = DAOStarFinder::new(config);
        assert!(finder.is_ok());
    }

    #[test]
    fn test_synthetic_star() {
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
                data[[y, x]] = 100.0 * (-r2 / (2.0 * sigma * sigma)).exp();
            }
        }

        let config = DAOStarFinderConfig {
            threshold: 10.0,
            fwhm: 3.0,
            ..Default::default()
        };

        let finder = DAOStarFinder::new(config).unwrap();
        let stars = finder.find_stars(&data, None);

        assert_eq!(stars.len(), 1);
        assert!((stars[0].x_centroid - cx).abs() < 1.0);
        assert!((stars[0].y_centroid - cy).abs() < 1.0);
    }
}
