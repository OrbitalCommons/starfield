//! Star catalogs module
//!
//! This module provides functionality for loading and using star catalogs,
//! including efficient binary formats for optimized storage and loading.
//! It also provides an index of interesting astronomical features for targeting simulations.

use crate::coordinates::Equatorial;

pub mod features;
mod gaia;
pub mod hipparcos;
#[cfg(feature = "photometry")]
pub mod isophote_series;
pub mod minimal_catalog;
#[cfg(feature = "photometry")]
pub mod photometry;
#[cfg(feature = "photometry")]
pub mod radial_profile;
pub mod synthetic;

pub use features::{FeatureCatalog, FeatureType, SkyFeature};
pub use gaia::{GaiaCatalog, GaiaEntry};
pub use hipparcos::{HipparcosCatalog, HipparcosEntry};
#[cfg(feature = "photometry")]
pub use isophote_series::{IsophoteSample, IsophoteSeries};
pub use minimal_catalog::{MinimalCatalog, MinimalStar, ProgressUpdate};
#[cfg(feature = "photometry")]
pub use photometry::{Band, Photometry};
#[cfg(feature = "photometry")]
pub use radial_profile::RadialProfile;
pub use synthetic::{
    create_fov_catalog, create_synthetic_catalog, MagnitudeDistribution, SpatialDistribution,
    SyntheticCatalogConfig,
};

use rand::distr::{Distribution, Uniform};
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

/// Sérsic surface-brightness profile parameters for an extended source.
///
/// Represents the elliptical Sérsic model commonly used to describe
/// galaxy morphology in optical imaging. The brightness at angular
/// distance `r` along the major axis is proportional to
///
/// ```text
/// I(r) ∝ exp[ -b_n · ( (r / θ_half)^(1/n) - 1 ) ]
/// ```
///
/// where `b_n ≈ 2n − 1/3` is the constant that makes `θ_half` the
/// half-light radius. The `axis_ratio` flattens the profile along the
/// minor axis, and `position_angle_deg` rotates the major axis east of
/// north (J2000).
///
/// Common Sérsic indices: `n = 0.5` is Gaussian, `n = 1` is exponential
/// (typical late-type disks), `n = 4` is the de Vaucouleurs profile
/// (typical early-type bulges).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SersicProfile {
    /// Half-light radius along the major axis, in arcseconds.
    pub theta_half_arcsec: f64,
    /// Sérsic index *n* (dimensionless, typically ~0.5 to ~6 for galaxies).
    pub n: f64,
    /// Axis ratio b/a, where b ≤ a. `1.0` is circular, `0.0` is degenerate edge-on.
    pub axis_ratio: f64,
    /// Position angle of the major axis, degrees east of north (J2000).
    pub position_angle_deg: f64,
}

impl SersicProfile {
    /// The Sérsic constant `b_n` defined by the relation
    /// `γ(2n, b_n) = Γ(2n) / 2`, i.e. the value that makes
    /// `theta_half_arcsec` enclose half of the total light.
    ///
    /// Uses the Ciotti & Bertin (1999, A&A, 352, 447) Eq. 18 asymptotic
    /// series, which is accurate to better than ~1e-3 relative error for
    /// `n ≥ 0.36` and to better than ~1e-6 for `n ≥ 1`. Catalogs that fit
    /// pure-disk (`n ≈ 1`) and de Vaucouleurs (`n = 4`) profiles are
    /// trivially in range; the lower-`n` end is only relevant for highly
    /// concentrated dwarf-irregulars and is still well within the
    /// pixel-noise floor of any photometric simulation that consumes
    /// this value.
    pub fn b_n(&self) -> f64 {
        let n = self.n;
        2.0 * n - 1.0 / 3.0
            + 4.0 / (405.0 * n)
            + 46.0 / (25_515.0 * n.powi(2))
            + 131.0 / (1_148_175.0 * n.powi(3))
            - 2_194_697.0 / (30_690_717_750.0 * n.powi(4))
    }

    /// Surface brightness at offset `(dx_arcsec, dy_arcsec)` from the
    /// galaxy centre, in the same units as the (unspecified) central
    /// surface brightness `I_e` at the half-light radius — i.e. this
    /// returns `I(r) / I_e`, where `I(r) = I_e · exp[-b_n · ((r/θ_half)^(1/n) − 1)]`
    /// and `r` is the elliptical radius along the rotated axes.
    ///
    /// `(dx_arcsec, dy_arcsec)` are sky-tangent offsets from the centre:
    /// `+dx_arcsec` toward the east, `+dy_arcsec` toward the north
    /// (J2000). [`SersicProfile::position_angle_deg`] is interpreted as
    /// the standard astronomical convention — degrees east of north,
    /// measured from `+y` (north) toward `+x` (east). At
    /// `position_angle_deg = 0` the major axis is aligned with `+y`
    /// (north); at `position_angle_deg = 90` it is aligned with `+x`
    /// (east).
    ///
    /// This evaluator matches AstroPy's
    /// `astropy.modeling.functional_models.Sersic2D` model, with the
    /// convention translation
    /// `theta_AstroPy = 90° − position_angle_deg` (AstroPy measures
    /// `theta` counter-clockwise from `+x`; this trait stores PA
    /// east-of-north — clockwise from `+y` toward `+x` — so the two
    /// rotations are mirrored about the line `theta_AstroPy = 90°`)
    /// and `ellip_AstroPy = 1 − axis_ratio`. Multiply the returned
    /// value by the catalog's `I_e` to recover absolute surface
    /// brightness in physical units.
    ///
    /// # Converting total flux to `I_e`
    ///
    /// Most catalogs publish a *total integrated flux* `F_total` rather
    /// than `I_e`. Derive `I_e` from the closed-form Sérsic luminosity
    /// integral:
    ///
    /// ```text
    /// F_total = 2π · n · b_n^(-2n) · Γ(2n) · q · θ_eff² · exp(b_n) · I_e
    ///           ────────────────────────────────────────────────────
    ///                                   K
    ///
    /// I_e = F_total / K
    /// ```
    ///
    /// where `q = axis_ratio` and `θ_eff = theta_half_arcsec`.
    ///
    /// The `exp(b_n)` factor is easy to miss when re-deriving from
    /// `surface_brightness_at`, because the SB form here is written
    /// relative to `I_e` (`I(r) = I_e · exp[-b_n · ((r/θ_eff)^(1/n) − 1)]`)
    /// rather than relative to the central surface brightness
    /// `I_0 = I_e · exp(b_n)`. Forgetting it under-counts total flux by
    /// `exp(b_n)`, which at `n = 4` (de Vaucouleurs) is ≈ 2140. The bug
    /// is silent under any display stretch (asinh, percentile-clip,
    /// etc.) but corrupts aperture photometry. See [#128] for a
    /// companion convenience method.
    ///
    /// Reference: Graham & Driver (2005), "A concise reference to
    /// projected Sérsic R^(1/n) quantities", PASA 22, 118, Eq. (4)–(6).
    ///
    /// [#128]: https://github.com/OrbitalCommons/starfield/issues/128
    pub fn surface_brightness_at(&self, dx_arcsec: f64, dy_arcsec: f64) -> f64 {
        let bn = self.b_n();
        let a = self.theta_half_arcsec;
        let b = a * self.axis_ratio;
        let pa_rad = self.position_angle_deg.to_radians();
        let (sin_pa, cos_pa) = pa_rad.sin_cos();
        // Project (dx, dy) onto the major / minor axes. With PA measured
        // east of north (from +y toward +x), the major axis unit vector
        // is (sin_pa, cos_pa) and the minor axis unit vector is
        // (-cos_pa, sin_pa) — a +90° rotation of the major axis in the
        // standard right-handed sense.
        let x_maj = dx_arcsec * sin_pa + dy_arcsec * cos_pa;
        let x_min = -dx_arcsec * cos_pa + dy_arcsec * sin_pa;
        let z = ((x_maj / a).powi(2) + (x_min / b).powi(2)).sqrt();
        (-bn * (z.powf(1.0 / self.n) - 1.0)).exp()
    }

    /// Closed-form Sérsic luminosity coefficient `K` such that
    /// `F_total = K · I_e`.
    ///
    /// ```text
    /// K = 2π · n · b_n^(-2n) · Γ(2n) · q · θ_eff² · exp(b_n)
    /// ```
    ///
    /// (Graham & Driver 2005, Eq. 4–6.) Converts a catalog's *total
    /// integrated flux* to the `I_e` that
    /// [`SersicProfile::surface_brightness_at`] is implicitly normalised
    /// against:
    ///
    /// ```rust,ignore
    /// let i_e = total_flux / profile.total_flux_per_ie();
    /// let absolute_sb = i_e * profile.surface_brightness_at(dx, dy);
    /// ```
    ///
    /// The `exp(b_n)` factor is the most easily-dropped piece of this
    /// formula when re-deriving from the `I_e`-form Sérsic profile —
    /// at n=4 (de Vaucouleurs) `exp(b_n) ≈ 2140`, so omitting it
    /// under-counts the catalog flux by that factor. This helper
    /// exists chiefly to keep that mistake out of consumer code.
    ///
    /// `Γ(2n)` is evaluated via a Lanczos g=7 approximation (relative
    /// error < 1e-12 over `n ∈ [0.25, 50]`); the call is cheap to
    /// invoke once per profile but should still be cached outside any
    /// per-pixel inner loop.
    pub fn total_flux_per_ie(&self) -> f64 {
        let n = self.n;
        let bn = self.b_n();
        2.0 * std::f64::consts::PI
            * n
            * bn.powf(-2.0 * n)
            * gamma_lanczos(2.0 * n)
            * self.axis_ratio
            * self.theta_half_arcsec
            * self.theta_half_arcsec
            * bn.exp()
    }
}

/// Lanczos `g = 7` approximation to `Γ(x)`, valid for `x > 0`.
/// Relative error below 1e-12 over `x ∈ [0.5, 100]`. Uses the standard
/// reflection formula for `x < 0.5` to extend the domain — not strictly
/// needed for the `2n` argument used by [`SersicProfile::total_flux_per_ie`]
/// where `2n ≥ 1` for any physically reasonable Sérsic index, but cheap
/// to keep and useful for the `n = 0.25` dwarf-irregular regime.
fn gamma_lanczos(x: f64) -> f64 {
    const G: f64 = 7.0;
    // Coefficients reproduced from the canonical Lanczos g=7 derivation,
    // truncated to f64 precision.
    const COEF: [f64; 9] = [
        0.999_999_999_999_810,
        676.520_368_121_885_2,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_1,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma_lanczos(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = COEF[0];
        for (i, c) in COEF.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        let t = x + G + 0.5;
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// Trait for catalog entries that may be spatially extended on the sky.
///
/// Point sources (stars) get the default implementation, which returns
/// `None` — renderers should treat those as deltas convolved with the
/// optical PSF. Galaxies and other extended sources return
/// `Some(SersicProfile)` describing their morphology, and a
/// surface-brightness-aware renderer should integrate that profile over
/// the pixel grid instead of stamping a single point PSF.
///
/// Implementing the trait does not commit a catalog entry to *being*
/// extended at any given moment: a galaxy too small to resolve at the
/// instrument's plate scale, or one whose Sérsic fit failed in the
/// upstream catalog, can return `None` and be treated as a point source
/// (or skipped entirely) by downstream consumers.
pub trait ExtendedSource {
    /// Returns the Sérsic profile describing this source's spatial
    /// extent, or `None` if the source is a point or its morphology is
    /// unknown / unfit.
    fn sersic_profile(&self) -> Option<SersicProfile> {
        None
    }
}

impl ExtendedSource for StarData {}

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
    /// Minimal catalog with specified path (checks relative path and cache)
    Minimal(PathBuf),
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

        CatalogSource::Minimal(path) => {
            println!("Loading minimal catalog from: {}", path.display());
            let catalog = MinimalCatalog::load(&path)?;
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
    let ra_dist = Uniform::new(center_ra - half_fov, center_ra + half_fov).unwrap();
    let dec_dist = Uniform::new(center_dec - half_fov, center_dec + half_fov).unwrap();

    // For realistic magnitude distribution, use exponential distribution
    // For every step in magnitude, there are ~2.5x more stars
    let min_mag = 3.0; // Brightest stars (lower magnitude = brighter)
    let max_mag = 8.0; // Dimmest stars

    // We'll generate random values and transform them to follow stellar magnitude distribution
    let uniform = Uniform::new(0.0, 1.0).unwrap();

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
    use crate::catalogs::minimal_catalog::{MinimalCatalog, MinimalStar};
    use approx::assert_abs_diff_eq;

    /// Test the StarCatalog trait with a simple minimal catalog
    #[test]
    fn test_star_data() {
        // Create a minimal catalog
        let stars = vec![
            MinimalStar::new(1, 100.0, 10.0, -1.5), // Sirius-like
            MinimalStar::new(2, 50.0, -20.0, 0.5),  // Canopus-like
            MinimalStar::new(3, 150.0, 30.0, 1.2),  // Bright
            MinimalStar::new(4, 200.0, -45.0, 3.7), // Medium
            MinimalStar::new(5, 250.0, 60.0, 5.9),  // Dim
        ];

        let catalog = MinimalCatalog::from_stars(stars, "Test catalog");

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

    #[test]
    fn test_star_data_is_a_point_source_via_extended_source() {
        // The default ExtendedSource impl on StarData returns None,
        // marking it as a point source for renderers that dispatch on
        // morphology.
        let star = StarData::new(1, 12.34, 56.78, 9.0, Some(0.6));
        assert_eq!(star.sersic_profile(), None);
    }

    #[test]
    fn test_extended_source_default_impl_returns_none() {
        // Any type that derives only the default ExtendedSource impl
        // (no override) must report as a point source — the trait
        // contract for downstream dispatchers.
        struct PointSource;
        impl ExtendedSource for PointSource {}
        assert_eq!(PointSource.sersic_profile(), None);
    }

    #[test]
    fn test_sersic_profile_round_trips_through_struct_literal() {
        // Smoke-test the public field shape of SersicProfile so a
        // future field rename is caught here rather than far downstream.
        let p = SersicProfile {
            theta_half_arcsec: 3.5,
            n: 4.0,
            axis_ratio: 0.7,
            position_angle_deg: 32.0,
        };
        assert_eq!(p.theta_half_arcsec, 3.5);
        assert_eq!(p.n, 4.0);
        assert_eq!(p.axis_ratio, 0.7);
        assert_eq!(p.position_angle_deg, 32.0);
    }

    fn sersic_profile(n: f64, axis_ratio: f64, position_angle_deg: f64) -> SersicProfile {
        SersicProfile {
            theta_half_arcsec: 2.0,
            n,
            axis_ratio,
            position_angle_deg,
        }
    }

    #[test]
    fn test_b_n_matches_ciotti_bertin_reference_values() {
        // Reference values from scipy.special.gammaincinv(2*n, 0.5),
        // which solves the defining transcendental equation exactly.
        // The Ciotti-Bertin series degrades below n ≈ 0.36 (per the
        // docstring); accuracy improves rapidly with n. The test set
        // covers the catalog-relevant range and tightens the tolerance
        // as n grows, exposing any future change to the series itself.
        let p_half = sersic_profile(0.5, 1.0, 0.0);
        let p_one = sersic_profile(1.0, 1.0, 0.0);
        let p_two = sersic_profile(2.0, 1.0, 0.0);
        let p_four = sersic_profile(4.0, 1.0, 0.0);
        let p_six = sersic_profile(6.0, 1.0, 0.0);
        // Empirically measured series residuals (Ciotti-Bertin truncation):
        //   n=0.5 → 2.5e-4   n=1.0 → 4.2e-5   n=2.0 → 4.7e-6
        //   n=4.0 → 5.4e-7   n=6.0 → 1.6e-7
        // Tolerances are set ~2× above each residual so trivial rounding
        // shifts in the series constants don't trip the test, while a
        // genuine algorithmic regression still gets caught.
        assert_abs_diff_eq!(p_half.b_n(), 0.6931471806, epsilon = 5e-4);
        assert_abs_diff_eq!(p_one.b_n(), 1.6783469900, epsilon = 1e-4);
        assert_abs_diff_eq!(p_two.b_n(), 3.6720607489, epsilon = 1e-5);
        assert_abs_diff_eq!(p_four.b_n(), 7.6692494425, epsilon = 1e-6);
        assert_abs_diff_eq!(p_six.b_n(), 11.6683631530, epsilon = 1e-6);
    }

    #[test]
    fn test_surface_brightness_at_half_light_radius_equals_unity_along_major_axis() {
        // By construction, I(theta_half) / I_e = 1 along the major axis.
        // PA = 0 puts the major axis along +y (north), so a point at
        // (0, theta_half) is on the major axis at the half-light radius.
        let p = sersic_profile(4.0, 0.6, 0.0);
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, p.theta_half_arcsec),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_surface_brightness_at_circular_profile_is_axis_independent() {
        // axis_ratio = 1 ⇒ no preferred direction; SB depends only on
        // sqrt(dx^2 + dy^2), so points equidistant from the centre must
        // produce the same SB regardless of direction or PA.
        let p = sersic_profile(2.5, 1.0, 37.0);
        let r = 1.3;
        let east = p.surface_brightness_at(r, 0.0);
        let north = p.surface_brightness_at(0.0, r);
        let diag = p.surface_brightness_at(r / 2.0_f64.sqrt(), r / 2.0_f64.sqrt());
        assert_abs_diff_eq!(east, north, epsilon = 1e-12);
        assert_abs_diff_eq!(east, diag, epsilon = 1e-12);
    }

    #[test]
    fn test_surface_brightness_at_position_angle_rotates_major_axis_east_of_north() {
        // PA = 90° puts the major axis along +x (east). The half-light
        // surface brightness should now be reached at (theta_half, 0)
        // rather than (0, theta_half).
        let p = sersic_profile(4.0, 0.5, 90.0);
        assert_abs_diff_eq!(
            p.surface_brightness_at(p.theta_half_arcsec, 0.0),
            1.0,
            epsilon = 1e-12
        );
        // Same elliptical radius along the minor axis (now +y) requires
        // moving only theta_half * axis_ratio.
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, p.theta_half_arcsec * p.axis_ratio),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_surface_brightness_at_centre_is_e_to_the_b_n() {
        // I(0) / I_e = exp(b_n) by definition of the Sérsic profile.
        let p = sersic_profile(4.0, 0.6, 45.0);
        assert_abs_diff_eq!(
            p.surface_brightness_at(0.0, 0.0),
            p.b_n().exp(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_surface_brightness_at_matches_astropy_sersic2d_reference() {
        // Cross-checked against astropy.modeling.functional_models.Sersic2D
        // with: amplitude=1, r_eff=2.0, n=4, x_0=y_0=0,
        //       ellip = 1 - 0.6 = 0.4,
        //       theta = (90° − 45°) in radians  (the documented
        //                                        convention translation).
        // Reference values produced with scipy/astropy:
        //   model(1.0, 0.5) → 2.461462
        //   model(2.0, 2.0) → 0.499511
        let p = sersic_profile(4.0, 0.6, 45.0);
        assert_abs_diff_eq!(p.surface_brightness_at(1.0, 0.5), 2.461462, epsilon = 1e-5);
        assert_abs_diff_eq!(p.surface_brightness_at(2.0, 2.0), 0.499511, epsilon = 1e-5);
    }

    /// Pin the PA convention against axis-aligned cases that distinguish
    /// "major axis along +y at PA=0" (the documented behavior) from
    /// alternative conventions.
    ///
    /// This is the regression guard for #124 — the docstring used to
    /// claim `theta_AstroPy = position_angle_deg + 90°`, which gives the
    /// same answer as the corrected `90° − position_angle_deg` formula
    /// only at PA=0° (since 0+90 = 90−0). Any non-zero PA distinguishes
    /// them; PA=0° + axis_ratio<1 distinguishes "major along +y"
    /// (correct) from "major along +x" (would arise from an axis swap).
    #[test]
    fn test_position_angle_zero_aligns_major_axis_with_north() {
        // Axis ratio 0.5 ⇒ b = θ_half/2. At (θ_half, 0) the elliptical
        // radius is z = θ_half / b = 2, so SB drops well below the
        // half-light SB of 1.0; the strict inequality is the convention
        // anchor — an axis swap inverts it.
        let p_pa0 = sersic_profile(4.0, 0.5, 0.0);
        let on_major = p_pa0.surface_brightness_at(0.0, 2.0);
        let on_minor = p_pa0.surface_brightness_at(2.0, 0.0);
        assert_abs_diff_eq!(on_major, 1.0, epsilon = 1e-12);
        assert!(
            on_minor < on_major,
            "PA=0 should put major axis along +y; \
             got on_major={on_major}, on_minor={on_minor}"
        );

        // PA=90 swaps which axis is "major".
        let p_pa90 = sersic_profile(4.0, 0.5, 90.0);
        let on_major_90 = p_pa90.surface_brightness_at(2.0, 0.0);
        let on_minor_90 = p_pa90.surface_brightness_at(0.0, 2.0);
        assert_abs_diff_eq!(on_major_90, 1.0, epsilon = 1e-12);
        assert!(on_minor_90 < on_major_90);

        // By symmetry the off-axis values must agree across the two PAs.
        assert_abs_diff_eq!(on_minor, on_minor_90, epsilon = 1e-12);
    }

    #[test]
    fn test_surface_brightness_at_decays_monotonically_along_major_axis() {
        // Along the major axis (PA = 0 ⇒ +y), SB should strictly
        // decrease as |y| grows away from the centre.
        let p = sersic_profile(2.0, 0.7, 0.0);
        let samples: Vec<f64> = (0..20)
            .map(|i| p.surface_brightness_at(0.0, 0.25 * i as f64))
            .collect();
        for w in samples.windows(2) {
            assert!(w[0] > w[1], "expected monotonic decay, got {:?}", w);
        }
    }

    /// **Anchor for `gamma_lanczos` against integer factorials**:
    /// `Γ(n) = (n - 1)!` for positive integer `n`. The integer
    /// factorial sequence is the only ground-truth set we can write
    /// down without an external reference library, so it's the most
    /// honest anchor for the Lanczos approximation.
    #[test]
    fn test_gamma_lanczos_matches_factorial_for_integer_arguments() {
        let factorials = [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (5.0, 24.0),
            (6.0, 120.0),
            (10.0, 362_880.0),
        ];
        for (x, expected) in factorials {
            let g = super::gamma_lanczos(x);
            let rel = (g - expected).abs() / expected;
            assert!(
                rel < 1e-10,
                "Γ({x}) = {g}, expected {expected} (rel err {rel:.2e})"
            );
        }
    }

    /// Anchor for `gamma_lanczos(0.5) = √π`.
    #[test]
    fn test_gamma_lanczos_half_equals_sqrt_pi() {
        let g = super::gamma_lanczos(0.5);
        let expected = std::f64::consts::PI.sqrt();
        let rel = (g - expected).abs() / expected;
        assert!(rel < 1e-12, "Γ(0.5) = {g}, expected √π = {expected}");
    }

    /// **Physics anchor for `total_flux_per_ie`**: agrees with a
    /// hand-written term-by-term expansion of Graham & Driver (2005)
    /// Eq. 4–6 at the canonical de Vaucouleurs index n=4. Locks all
    /// factors (2π, n, b_n^(-2n), Γ(2n), q · θ_eff², exp(b_n)).
    #[test]
    fn test_total_flux_per_ie_matches_hand_expansion_at_n_equals_4() {
        let p = SersicProfile {
            theta_half_arcsec: 3.0,
            n: 4.0,
            axis_ratio: 0.7,
            position_angle_deg: 30.0,
        };
        let bn = p.b_n();
        let gamma_2n = super::gamma_lanczos(8.0);
        let expected =
            2.0 * std::f64::consts::PI * 4.0 * bn.powf(-8.0) * gamma_2n * 0.7 * 9.0 * bn.exp();
        let actual = p.total_flux_per_ie();
        let rel = (actual - expected).abs() / expected;
        assert!(rel < 1e-14, "K(actual)={actual} K(expected)={expected}");
    }

    /// **Round-trip flux normalisation**: pick `I_e`, compute
    /// `F_total = K · I_e`, recover `I_e' = F_total / K`, check
    /// equality. Locks the inverse-of-K relationship that
    /// renderer code relies on.
    #[test]
    fn test_total_flux_per_ie_round_trips_ie_to_total_to_ie() {
        for n in [1.0_f64, 2.0, 4.0, 6.0] {
            for q in [1.0_f64, 0.7, 0.4] {
                let p = SersicProfile {
                    theta_half_arcsec: 4.5,
                    n,
                    axis_ratio: q,
                    position_angle_deg: 17.0,
                };
                let i_e_in = 1234.5_f64;
                let k = p.total_flux_per_ie();
                let f_total = k * i_e_in;
                let i_e_out = f_total / k;
                assert!(
                    (i_e_in - i_e_out).abs() < 1e-9,
                    "round-trip I_e mismatch at n={n}, q={q}: in={i_e_in} out={i_e_out}"
                );
            }
        }
    }

    /// **Anchor against the `exp(b_n)`-omission failure mode**: this
    /// helper exists chiefly to keep the `exp(b_n)` factor out of
    /// consumer copy-paste mistakes. If a future refactor accidentally
    /// removes the `bn.exp()` term, this test fires before any
    /// downstream photometry silently shifts by `exp(b_n) ≈ 2140` at
    /// n=4.
    #[test]
    fn test_total_flux_per_ie_includes_exp_b_n_factor_at_n_equals_4() {
        let p = SersicProfile {
            theta_half_arcsec: 2.0,
            n: 4.0,
            axis_ratio: 1.0,
            position_angle_deg: 0.0,
        };
        let bn = p.b_n();
        let with_exp = p.total_flux_per_ie();
        let without_exp =
            2.0 * std::f64::consts::PI * 4.0 * bn.powf(-8.0) * super::gamma_lanczos(8.0) * 4.0; // θ_eff²
        let ratio = with_exp / without_exp;
        let rel = (ratio - bn.exp()).abs() / bn.exp();
        assert!(
            rel < 1e-12,
            "with_exp / without_exp = {ratio}, expected exp(b_n) = {} (n=4)",
            bn.exp()
        );
        assert!(
            (2000.0..2300.0).contains(&bn.exp()),
            "exp(b_n) at n=4 = {} expected to be ≈ 2140",
            bn.exp()
        );
    }

    /// Live cross-check of [`SersicProfile::surface_brightness_at`] against
    /// `astropy.modeling.functional_models.Sersic2D` via the in-process
    /// Python bridge.
    ///
    /// This is the "AstroPy-backed" sibling of the existing Skyfield
    /// comparison tests. It demonstrates that the same `PyRustBridge`
    /// works for both libraries — no parallel infrastructure needed,
    /// just `import astropy...` instead of `from skyfield.api ...`.
    ///
    /// Uses a **non-circular profile at PA=45°** to exercise the
    /// documented convention translation
    /// `theta_AstroPy = 90° − position_angle_deg`. A circular profile
    /// would tell us nothing about PA; PA=0° is degenerate because
    /// `90° − 0° = 0° + 90°`. PA=45° is the cleanest probe — the two
    /// candidate translations differ (45° vs 135°) and AstroPy's
    /// outputs differ accordingly, so a future drift in either the
    /// implementation or the docstring fires this test.
    #[cfg(feature = "python-tests")]
    #[test]
    fn test_surface_brightness_at_matches_astropy_sersic2d_via_bridge() {
        use crate::pybridge::{bridge::PyRustBridge, helpers::PythonResult};

        let bridge = PyRustBridge::new().expect("bridge");
        let raw = bridge
            .run_py_to_json(
                r#"
import numpy as np
from astropy.modeling.functional_models import Sersic2D

# Documented convention: theta_AstroPy = 90° - position_angle_deg
# (AstroPy measures theta CCW from +x; this trait stores PA east of
# north — CW from +y toward +x — so the rotations are mirrored.)
PA_DEG = 45.0
model = Sersic2D(amplitude=1.0, r_eff=2.0, n=4.0,
                 x_0=0.0, y_0=0.0,
                 ellip=1.0 - 0.6,
                 theta=np.deg2rad(90.0 - PA_DEG))

xs = np.array([1.0, 2.0, 3.0, 0.5, 4.0])
ys = np.array([0.5, 2.0, 1.5, 1.5, 0.0])
out = model(xs, ys).astype(np.float64)
rust.collect_array(out)
"#,
            )
            .expect("bridge run failed");

        let parsed = PythonResult::try_from(raw.as_str()).expect("bad json from bridge");
        let astropy_values: Vec<f64> = match parsed {
            PythonResult::Array { data, .. } => data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            other => panic!("expected Array, got {other:?}"),
        };

        let p = sersic_profile(4.0, 0.6, 45.0);
        let rust_values = [
            p.surface_brightness_at(1.0, 0.5),
            p.surface_brightness_at(2.0, 2.0),
            p.surface_brightness_at(3.0, 1.5),
            p.surface_brightness_at(0.5, 1.5),
            p.surface_brightness_at(4.0, 0.0),
        ];

        assert_eq!(astropy_values.len(), rust_values.len());
        for (rust, astropy) in rust_values.iter().zip(astropy_values.iter()) {
            assert_abs_diff_eq!(*rust, *astropy, epsilon = 1e-6);
        }
    }
}
