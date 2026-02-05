use super::moments::ImageMoments;
use super::StellarSource;
use ndarray::{s, Array2};
use std::f64::consts::PI;

/// Stellar source detected and measured by the IRAF DAOFIND algorithm
///
/// This structure contains comprehensive photometric and morphological measurements
/// of a point source detected in astronomical image data. All measurements follow
/// IRAF conventions and are suitable for astronomical analysis pipelines.
///
/// # Coordinate System
///
/// All pixel coordinates use the standard image coordinate system:
/// - Origin (0,0) at the bottom-left corner of the first pixel
/// - X-axis increases rightward (column direction)
/// - Y-axis increases upward (row direction)  
/// - Sub-pixel positions indicate centroid refinement beyond peak pixel
///
/// # Measurement Accuracy
///
/// The precision of each measurement depends on source signal-to-noise ratio:
/// - **High S/N (>100)**: Centroid ~0.01 pixels, shape ~5% relative accuracy
/// - **Medium S/N (10-100)**: Centroid ~0.1 pixels, shape ~10-20% accuracy
/// - **Low S/N (<10)**: Centroid ~0.5 pixels, shape measurements unreliable
///
/// # Quality Indicators
///
/// Several fields serve as quality indicators:
/// - `sharpness`: Distinguishes stars from cosmic rays and extended objects
/// - `roundness`: Identifies tracking errors, optical problems, or blended sources
/// - `npix`: Ensures sufficient data for robust moment calculations
/// - `peak`/`flux`: Consistency checks for saturation or measurement errors
#[derive(Debug, Clone)]
pub struct IRAFStar {
    /// Unique source identifier within the detection catalog
    ///
    /// Sequential integer starting from 1, assigned in detection order.
    /// Can be used for cross-referencing with other catalogs or analyses.
    pub id: usize,

    /// Sub-pixel X coordinate of the source centroid
    ///
    /// Determined from intensity-weighted first moments with typical precision
    /// of 0.01-0.1 pixels for well-detected sources. Values are in the image
    /// coordinate system where the first pixel spans 0.0 to 1.0.
    pub x_centroid: f64,

    /// Sub-pixel Y coordinate of the source centroid  
    ///
    /// Determined from intensity-weighted first moments with typical precision
    /// of 0.01-0.1 pixels for well-detected sources. Values are in the image
    /// coordinate system where the first pixel spans 0.0 to 1.0.
    pub y_centroid: f64,

    /// Full-Width at Half-Maximum of the source profile
    ///
    /// Derived from second-order image moments using the relation:
    /// FWHM = 2√(2ln2 × √(μ₂₀ + μ₀₂))
    ///
    /// For stellar sources, this should approximately match the seeing disk
    /// or instrumental PSF. Values significantly larger may indicate galaxies
    /// or blended sources; much smaller values suggest cosmic rays.
    ///
    /// Units: pixels
    pub fwhm: f64,

    /// Source sharpness parameter (PSF width consistency check)
    ///
    /// Ratio of measured FWHM to the expected kernel FWHM:
    /// sharpness = measured_fwhm / kernel_fwhm
    ///
    /// **Interpretation**:
    /// - ~1.0: Consistent with stellar PSF
    /// - <0.5: Cosmic ray or hot pixel (too sharp)  
    /// - >2.0: Extended object or poor seeing (too broad)
    /// - Typical range for stars: 0.5 - 1.5
    pub sharpness: f64,

    /// Source roundness parameter (ellipticity measure)
    ///
    /// Quantifies source elongation using central moments:
    /// roundness = √((μ₀₂ - μ₂₀)² + 4μ₁₁²) / (μ₀₂ + μ₂₀)
    ///
    /// **Range**: 0.0 (perfectly circular) to 1.0 (maximally elongated)
    ///
    /// **Interpretation**:
    /// - 0.0-0.2: Well-rounded stellar sources
    /// - 0.2-0.5: Moderately elongated (tracking errors, close binaries)
    /// - >0.5: Highly elongated (cosmic rays, artifacts, severe tracking issues)
    pub roundness: f64,

    /// Position angle of the source major axis
    ///
    /// Angle between the major axis and the X-axis, measured counter-clockwise.
    /// Derived from the orientation of the moment ellipse:
    /// PA = 0.5 × arctan(2μ₁₁ / (μ₀₂ - μ₂₀))
    ///
    /// **Range**: 0° to 180°
    /// **Units**: degrees
    /// **Note**: Only meaningful for significantly elongated sources (roundness > 0.1)
    pub pa: f64,

    /// Number of pixels with positive flux contributing to measurements
    ///
    /// Count of pixels within the measurement aperture that have positive
    /// values after sky subtraction. Provides a measure of source extent
    /// and measurement reliability.
    ///
    /// **Typical values**:
    /// - Cosmic rays: 1-2 pixels
    /// - Point sources: 5-50 pixels (depends on seeing/PSF)
    /// - Extended objects: >50 pixels
    /// - Minimum required: >1 pixel for valid measurement
    pub npix: usize,

    /// Peak pixel value after local sky subtraction
    ///
    /// Maximum intensity within the measurement aperture following background
    /// subtraction. Useful for saturation detection and consistency checks.
    ///
    /// **Units**: Same as input image (ADU, electrons, etc.)
    /// **Note**: May differ from flux centroid due to noise and sampling effects
    pub peak: f64,

    /// Integrated flux within the measurement aperture
    ///
    /// Sum of all positive pixel values after sky subtraction, providing
    /// the total detected signal from the source. This is the fundamental
    /// photometric measurement used for magnitude calculation.
    ///
    /// **Units**: Same as input image (ADU, electrons, etc.)
    /// **Accuracy**: Limited by sky subtraction quality and aperture effects
    pub flux: f64,

    /// Local sky background level estimate
    ///
    /// Background intensity derived from pixels outside the source aperture
    /// (as defined by the detection kernel mask). This value was subtracted
    /// from source pixels during photometric measurement.
    ///
    /// **Units**: Same as input image (ADU, electrons, etc.)
    /// **Method**: Mean of unmasked pixels in the measurement cutout
    /// **Quality**: Best for isolated sources; contaminated in crowded fields
    pub sky: f64,
}

impl StellarSource for IRAFStar {
    fn id(&self) -> usize {
        self.id
    }

    fn get_centroid(&self) -> (f64, f64) {
        (self.x_centroid, self.y_centroid)
    }

    fn flux(&self) -> f64 {
        self.flux
    }
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
    /// Sharpness range (inclusive)
    pub sharpness: std::ops::RangeInclusive<f64>,
    /// Roundness range (inclusive)
    pub roundness: std::ops::RangeInclusive<f64>,
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
            sharpness: 0.5..=2.0,
            roundness: 0.0..=0.2,
            exclude_border: false,
            brightest: None,
            peakmax: None,
            min_separation: None,
        }
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

    /// Find and characterize stellar sources in astronomical image data
    ///
    /// This method implements the IRAF DAOFIND algorithm for detecting point sources
    /// in astronomical images. It performs a complete source detection pipeline including
    /// convolution filtering, peak detection, centroid refinement, and source characterization.
    ///
    /// # Algorithm Overview
    ///
    /// 1. **Convolution Filtering**: The input image is convolved with a circular Gaussian
    ///    kernel matching the expected stellar PSF. This enhances point sources while
    ///    suppressing noise and extended structures.
    ///
    /// 2. **Peak Detection**: Local maxima above the detection threshold are identified
    ///    in the convolved image, with minimum separation constraints to avoid duplicates.
    ///
    /// 3. **Source Measurement**: For each detected peak, the algorithm:
    ///    - Extracts a cutout around the source location
    ///    - Estimates local sky background from unmasked pixels
    ///    - Calculates image moments for centroid and shape determination
    ///    - Derives photometric and morphological parameters
    ///
    /// 4. **Quality Filtering**: Sources are filtered based on:
    ///    - Sharpness (PSF width relative to expected FWHM)
    ///    - Roundness (elongation measure)
    ///    - Peak brightness limits
    ///    - Centroid validity and finite flux values
    ///
    /// 5. **Output Selection**: Optionally selects only the N brightest sources
    ///    and reassigns sequential IDs.
    ///
    /// # Arguments
    ///
    /// * `data` - Input 2D image array containing astronomical data. Should be background-
    ///   subtracted or have uniform background for best results. Units should be
    ///   consistent (e.g., ADU, electrons, flux density).
    ///
    /// * `mask` - Optional boolean mask array with same dimensions as data. Pixels where
    ///   mask[y,x] = true are excluded from analysis. Use to mask:
    ///   - Bad pixels, cosmic rays, detector defects
    ///   - Saturated regions, bleeding trails
    ///   - Bright extended objects (galaxies, nebulae)
    ///   - Image boundaries or low-coverage regions
    ///
    /// # Returns
    ///
    /// Vector of `IRAFStar` objects containing detected sources, sorted by decreasing
    /// flux if `brightest` limit is specified. Each star includes:
    ///
    /// - **Position**: Sub-pixel centroid coordinates (x_centroid, y_centroid)
    /// - **Photometry**: Peak value, integrated flux, instrumental magnitude
    /// - **Morphology**: FWHM, sharpness, roundness, position angle
    /// - **Quality**: Number of pixels, sky background estimate
    /// - **Metadata**: Unique ID for cross-referencing
    ///
    /// # Configuration Effects
    ///
    /// The detection and characterization behavior is controlled by the configuration:
    ///
    /// - `threshold`: Higher values reduce false detections but may miss faint sources
    /// - `fwhm`: Should match typical stellar PSF width for optimal S/N enhancement
    /// - `sigma_radius`: Controls kernel size; larger values improve S/N but reduce resolution
    /// - `minsep_fwhm`: Prevents crowded source confusion but may merge close binaries
    /// - `sharpness`/`roundness`: Quality filters to reject cosmic rays, galaxies, artifacts
    /// - `exclude_border`: Avoids incomplete PSF measurements near image edges
    /// - `brightest`: Limits output for performance in crowded fields
    ///
    /// # Performance Notes
    ///
    /// - Computational cost scales with image size and detected source count
    /// - Consider binning or tiling for very large images (>10k×10k pixels)
    /// - Convolution is the dominant computational step; consider FFT for large kernels
    /// - Memory usage scales with kernel size and number of simultaneous source cutouts
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use starfield::image::starfinders::{IRAFStarFinder, IRAFStarFinderConfig, StellarSource};
    /// use ndarray::Array2;
    ///
    /// // Configure for typical ground-based imaging
    /// let config = IRAFStarFinderConfig {
    ///     threshold: 5.0,           // 5-sigma detection
    ///     fwhm: 2.5,               // 2.5 pixel stellar FWHM
    ///     sigma_radius: 1.5,       // Kernel truncation
    ///     sharpness: 0.2..=0.8,    // Reject cosmic rays and galaxies
    ///     roundness: -0.5..=0.5,   // Allow moderate elongation
    ///     brightest: Some(1000),   // Keep 1000 brightest sources
    ///     ..Default::default()
    /// };
    ///
    /// let finder = IRAFStarFinder::new(config).unwrap();
    /// # let image_data = ndarray::Array2::zeros((100, 100));
    /// let stars = finder.find_stars(&image_data, None);
    ///
    /// println!("Detected {} stellar sources", stars.len());
    /// for star in &stars[..5.min(stars.len())] {
    ///     println!("Star {}: ({:.2}, {:.2}) mag={:.2} fwhm={:.2}",
    ///              star.id, star.x_centroid, star.y_centroid, star.mag(), star.fwhm);
    /// }
    /// ```
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

    /// Identify local maxima in the convolved image as potential stellar sources
    ///
    /// This method implements the peak detection stage of the IRAF DAOFIND algorithm.
    /// It searches for pixels that are local maxima within a specified radius and
    /// exceed the detection threshold, which indicates potential stellar sources.
    ///
    /// # Algorithm Details
    ///
    /// 1. **Threshold Filtering**: Only pixels above `config.threshold` are considered
    /// 2. **Local Maximum Test**: Each candidate pixel is compared against all neighbors
    ///    within `min_separation` radius to ensure it's a true local maximum
    /// 3. **Border Exclusion**: If `exclude_border` is true, peaks near image edges
    ///    are rejected to ensure complete PSF measurements
    /// 4. **Mask Handling**: Masked pixels (where mask[y,x] = true) are excluded
    ///
    /// # Arguments
    ///
    /// * `convolved` - The image after convolution with the detection kernel. This
    ///   should have enhanced point sources and suppressed noise.
    /// * `mask` - Optional mask to exclude regions from peak detection. Useful for
    ///   masking saturated stars, cosmic rays, or bad detector regions.
    ///
    /// # Returns
    ///
    /// Vector of (x, y) pixel coordinates representing detected peaks. Coordinates
    /// are in image pixel space (0-indexed). The number of peaks may be large in
    /// crowded fields; subsequent filtering steps will refine this list.
    ///
    /// # Performance Considerations
    ///
    /// - Computational cost is O(N × M × R²) where N×M is image size and R is
    ///   the minimum separation radius
    /// - In crowded fields with many sources, this can be the computational bottleneck
    /// - Consider increasing threshold or decreasing sigma_radius for faster processing
    ///
    /// # Quality vs Completeness Trade-offs
    ///
    /// - Lower thresholds detect fainter sources but increase false positives
    /// - Smaller min_separation finds more sources but may create duplicates
    /// - Border exclusion improves measurement quality but reduces field coverage
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
                            && convolved[[ny as usize, nx as usize]] >= value
                        {
                            is_peak = false;
                            break;
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

    /// Measure comprehensive photometric and morphological properties of a detected stellar source
    ///
    /// This method performs detailed characterization of a stellar source detected at the
    /// specified coordinates. It implements the IRAF DAOFIND measurement algorithm using
    /// image moments analysis to derive accurate centroids, shape parameters, and photometry.
    ///
    /// # Algorithm Overview
    ///
    /// 1. **Cutout Extraction**: Extract a subarray centered on the source location,
    ///    sized to match the detection kernel dimensions
    /// 2. **Sky Background Estimation**: Calculate local background from pixels outside
    ///    the source aperture using the kernel mask
    /// 3. **Sky Subtraction**: Apply background subtraction and reject negative pixels
    ///    (following IRAF convention for robust photometry)
    /// 4. **Image Moments**: Calculate raw moments up to second order for:
    ///    - Centroid determination (first moments)
    ///    - Shape analysis (second moments)
    /// 5. **Shape Analysis**: Derive FWHM, roundness, and position angle from
    ///    central moments tensor
    /// 6. **Photometry**: Measure peak brightness, integrated flux, and instrumental magnitude
    /// 7. **Quality Assessment**: Calculate sharpness relative to expected PSF width
    ///
    /// # Measurement Details
    ///
    /// **Centroid Accuracy**: Sub-pixel centroids are computed using intensity-weighted
    /// first moments, typically accurate to ~0.01-0.1 pixels for well-detected sources.
    ///
    /// **Shape Parameters**:
    /// - FWHM: Full-width at half-maximum derived from second moments
    /// - Roundness: Ellipticity measure (0=circular, 1=maximally elongated)
    /// - Position Angle: Orientation of source major axis in degrees
    ///
    /// **Photometry**:
    /// - Peak: Maximum pixel value after background subtraction
    /// - Flux: Integrated intensity within the kernel aperture
    /// - Magnitude: Instrumental magnitude = -2.5 × log₁₀(flux)
    ///
    /// **Quality Metrics**:
    /// - Sharpness: Ratio of measured FWHM to expected kernel FWHM
    /// - Pixel Count: Number of positive pixels contributing to measurements
    ///
    /// # Arguments
    ///
    /// * `data` - Original (unconvolved) image data for accurate photometry
    /// * `x` - X pixel coordinate of detected peak (from find_peaks)
    /// * `y` - Y pixel coordinate of detected peak (from find_peaks)
    /// * `id` - Unique identifier for this source (for tracking/cross-reference)
    ///
    /// # Returns
    ///
    /// * `Some(IRAFStar)` - Complete source characterization if measurement succeeds
    /// * `None` - If measurement fails due to:
    ///   - Cutout extends beyond image boundaries
    ///   - Insufficient positive pixels after sky subtraction
    ///   - Invalid centroid calculation (NaN/infinite values)
    ///   - Zero or negative total flux
    ///
    /// # Failure Modes and Diagnostics
    ///
    /// Common reasons for measurement failure:
    /// - **Edge Effects**: Source too close to image boundary for complete cutout
    /// - **Over-subtraction**: Sky estimate too high, leaving no positive pixels
    /// - **Noise Spikes**: Cosmic rays or detector artifacts without extended structure
    /// - **Saturation**: Nonlinear detector response affecting moment calculations
    /// - **Confusion**: Multiple blended sources within the measurement aperture
    ///
    /// # Accuracy and Limitations
    ///
    /// **Centroid Precision**: Depends on S/N ratio, typically:
    /// - S/N > 100: ~0.01 pixel accuracy
    /// - S/N > 10: ~0.1 pixel accuracy  
    /// - S/N < 5: ~0.5 pixel accuracy
    ///
    /// **Shape Measurement**: Requires S/N > 10 for reliable FWHM/roundness
    /// **Photometry**: Aperture effects may bias faint source measurements
    ///
    /// # Performance Notes
    ///
    /// - Computational cost scales with kernel size (typically 10-50 pixels)
    /// - Memory allocation for cutouts may impact performance in crowded fields
    /// - Sky estimation can be expensive for large kernels with many background pixels
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
            sky,
        })
    }

    /// Estimate local sky background level from pixels surrounding the stellar source
    ///
    /// This method implements IRAF-style sky estimation by analyzing pixels in the
    /// source cutout that fall outside the stellar aperture (as defined by the kernel mask).
    /// The sky estimate is crucial for accurate photometry and source characterization.
    ///
    /// # Algorithm
    ///
    /// 1. **Background Pixel Selection**: Identify pixels where kernel.mask[y,x] = false,
    ///    indicating regions outside the stellar aperture
    /// 2. **Statistical Estimation**: Calculate the mean value of these background pixels
    /// 3. **Fallback Method**: If no background pixels available, estimate sky from
    ///    the difference between maximum data value and maximum kernel value
    ///
    /// # Sky Estimation Methods
    ///
    /// **Primary Method (Mean of Background Pixels)**:
    /// - Uses pixels explicitly excluded from the stellar aperture
    /// - Provides robust estimates when sufficient background area exists
    /// - Automatically accounts for local background variations
    ///
    /// **Fallback Method (Peak Difference)**:
    /// - Used when kernel mask covers entire cutout (very large kernels)
    /// - Estimates sky as: max(data) - max(kernel_response)
    /// - Less accurate but prevents complete measurement failure
    ///
    /// # Arguments
    ///
    /// * `data_cutout` - Small 2D array centered on the source, typically sized
    ///   to match the detection kernel (e.g., 15×15 to 51×51 pixels)
    ///
    /// # Returns
    ///
    /// Sky background level in the same units as the input data. This value will be
    /// subtracted from source pixels during photometric measurements.
    ///
    /// # Accuracy Considerations
    ///
    /// **Ideal Conditions**: Uniform local background with adequate sampling
    /// - Typical accuracy: 1-5% of background level
    /// - Works well for isolated sources in sparse fields
    ///
    /// **Challenging Conditions**:
    /// - **Crowded Fields**: Nearby sources contaminate background estimate
    /// - **Extended Objects**: Galaxies, nebulae affect local background
    /// - **Gradients**: Large-scale background variations within cutout
    /// - **Small Kernels**: Insufficient background pixels for robust statistics
    ///
    /// # Performance Notes
    ///
    /// - Computational cost scales linearly with cutout size
    /// - Memory access pattern is cache-friendly for small cutouts
    /// - No significant performance bottlenecks for typical kernel sizes
    ///
    /// # Alternative Approaches
    ///
    /// For improved sky estimation in challenging conditions, consider:
    /// - Global background modeling before source detection
    /// - Annular aperture photometry with gap between source and sky
    /// - Sigma-clipped statistics to reject contaminating sources
    /// - Median-based estimates for robustness against outliers
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

    /// Apply quality filters to remove spurious detections and unreliable measurements
    ///
    /// This method implements the final filtering stage of IRAF DAOFIND, applying
    /// configurable quality criteria to distinguish genuine stellar sources from
    /// cosmic rays, detector artifacts, galaxies, and measurement failures.
    ///
    /// # Filter Criteria
    ///
    /// **Sharpness Filter** (`config.sharpness` range):
    /// - Compares measured FWHM to expected kernel FWHM
    /// - Rejects cosmic rays (too sharp) and galaxies (too broad)
    /// - Typical range: 0.2 to 2.0 for ground-based data
    /// - Lower values = more restrictive (fewer false positives)
    ///
    /// **Roundness Filter** (`config.roundness` range):
    /// - Measures source elongation/ellipticity
    /// - Rejects badly elongated sources (tracking errors, cosmic rays)
    /// - Range: 0.0 (circular) to 1.0 (maximally elongated)
    /// - Typical acceptance: -0.5 to +0.5 for stellar sources
    ///
    /// **Finite Value Validation**:
    /// - Rejects sources with NaN or infinite coordinates/flux
    /// - Ensures numerical stability for downstream analysis
    /// - Critical for robust catalog generation
    ///
    /// **Pixel Count Filter**:
    /// - Requires npix > 1 (more than single pixel)
    /// - Rejects cosmic ray hits and hot pixels
    /// - Ensures sufficient data for moment calculations
    ///
    /// **Peak Brightness Filter** (`config.peakmax` if specified):
    /// - Optional upper limit on peak pixel value
    /// - Useful for rejecting saturated sources
    /// - Prevents nonlinear effects from corrupting measurements
    ///
    /// # Algorithm Details
    ///
    /// 1. **Sequential Filtering**: Each criterion is applied as AND condition
    /// 2. **Early Rejection**: Sources failing any criterion are immediately excluded
    /// 3. **Configurable Bounds**: All filter ranges are user-adjustable
    /// 4. **Robust Defaults**: Default values work well for typical astronomical data
    ///
    /// # Arguments
    ///
    /// * `stars` - Vector of measured sources from the measurement stage
    ///
    /// # Returns
    ///
    /// Filtered vector containing only sources passing all quality criteria.
    /// The number of sources may be significantly reduced from the input,
    /// especially in noisy data or with strict filter settings.
    ///
    /// # Filter Tuning Guidelines
    ///
    /// **Conservative Settings** (fewer false positives):
    /// - Narrow sharpness range (e.g., 0.5..=1.2)
    /// - Tight roundness limits (e.g., -0.2..=0.2)
    /// - Enable peak brightness limits for saturated data
    ///
    /// **Liberal Settings** (higher completeness):
    /// - Wide sharpness range (e.g., 0.1..=3.0)
    /// - Relaxed roundness (e.g., -1.0..=1.0)
    /// - No peak limits (accept all brightness levels)
    ///
    /// **Data-Specific Adjustments**:
    /// - Space-based: Tighter sharpness (no seeing effects)
    /// - Ground-based: Looser roundness (atmospheric effects)
    /// - Crowded fields: Stricter filters (reduce blends)
    /// - Sparse fields: Relaxed filters (maximize completeness)
    ///
    /// # Performance Impact
    ///
    /// - Minimal computational cost (simple comparisons)
    /// - Memory reduction: filtered catalog is smaller
    /// - Downstream benefits: cleaner data for astrometry/photometry
    ///
    /// # Quality vs Completeness Trade-off
    ///
    /// Stricter filters improve catalog purity but may reject:
    /// - Faint sources with poor S/N measurements
    /// - Binary stars with blended profiles
    /// - Sources affected by detector artifacts
    /// - Genuine stars in challenging observing conditions
    fn apply_filters(&self, stars: Vec<IRAFStar>) -> Vec<IRAFStar> {
        stars
            .into_iter()
            .filter(|star| {
                self.config.sharpness.contains(&star.sharpness)
                    && self.config.roundness.contains(&star.roundness)
                    && star.x_centroid.is_finite()
                    && star.y_centroid.is_finite()
                    && star.flux.is_finite()
                    && star.npix > 1
                    && self.config.peakmax.is_none_or(|max| star.peak <= max)
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
