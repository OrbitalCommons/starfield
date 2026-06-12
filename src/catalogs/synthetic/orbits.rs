//! Synthetic small-body orbital population generators.
//!
//! The solar-system counterpart to the synthetic *star* generators in the
//! parent module: seeded, reproducible populations of heliocentric orbital
//! elements for injection/recovery testing, survey-bias studies, and
//! dynamical experiments. The default configuration reproduces the
//! scattered-disk/ETNO populations used in the Planet Nine literature
//! (Batygin & Brown 2016; Khain et al. 2018).
//!
//! # Distributions
//!
//! Each generated orbit is drawn independently:
//!
//! - **Semi-major axis `a` and perihelion `q`**: uniform on the `q <= a`
//!   wedge of the `[a_min, a_max] x [q_min, q_max]` rectangle. Eccentricity
//!   follows as `e = 1 - q/a`.
//! - **Inclination `i`**: one of [`InclinationModel::Planar`] (`i = 0`),
//!   [`InclinationModel::HalfNormal`] (`|N(0, sigma)|`), or
//!   [`InclinationModel::Rayleigh`] (the classic Brown 2001 debiased
//!   Kuiper-belt form).
//! - **Angles `Ω`, `ω`, `M`**: independent and uniform on `[0°, 360°)`.
//!
//! # Deterministic-N resampling
//!
//! Invalid `(a, q)` draws with `q > a` are *resampled in a loop*, never
//! silently skipped, so `generate` always returns exactly `count` records
//! for a given seed. (A skip-on-invalid loop returns fewer records than
//! requested with the same wedge-shaped bias, just undocumented.)
//!
//! # Example
//!
//! ```
//! use rand::rngs::StdRng;
//! use rand::SeedableRng;
//! use starfield::catalogs::synthetic::orbits::SyntheticOrbitPopulation;
//!
//! let mut rng = StdRng::seed_from_u64(42);
//! let population = SyntheticOrbitPopulation::scattered_disk()
//!     .with_count(100)
//!     .generate(&mut rng)
//!     .unwrap();
//!
//! assert_eq!(population.len(), 100);
//! for orbit in &population {
//!     assert!(orbit.perihelion() <= orbit.a);
//!     assert!(orbit.e >= 0.0 && orbit.e < 1.0);
//! }
//! ```

use rand::Rng;

use crate::keplerlib::{mpcorb_orbit, KeplerOrbit};
use crate::time::Time;
use crate::{Result, StarfieldError};

/// One synthetic heliocentric orbit as classical Keplerian elements.
///
/// Follows the JPL SBDB conventions used by [`crate::sbdb::SmallBodyOrbit`]:
/// distances in AU, angles in degrees.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticOrbit {
    /// Semi-major axis (AU)
    pub a: f64,
    /// Eccentricity
    pub e: f64,
    /// Inclination (degrees)
    pub i_deg: f64,
    /// Longitude of ascending node Ω (degrees)
    pub omega_big_deg: f64,
    /// Argument of perihelion ω (degrees)
    pub omega_deg: f64,
    /// Mean anomaly M (degrees)
    pub mean_anomaly_deg: f64,
}

impl SyntheticOrbit {
    /// Perihelion distance `q = a(1 - e)` in AU.
    pub fn perihelion(&self) -> f64 {
        self.a * (1.0 - self.e)
    }

    /// Aphelion distance `Q = a(1 + e)` in AU.
    pub fn aphelion(&self) -> f64 {
        self.a * (1.0 + self.e)
    }

    /// Longitude of perihelion `ϖ = ω + Ω` in degrees, wrapped to `[0°, 360°)`.
    pub fn longitude_of_perihelion_deg(&self) -> f64 {
        (self.omega_deg + self.omega_big_deg).rem_euclid(360.0)
    }

    /// Convert to a propagatable [`KeplerOrbit`].
    ///
    /// The elements are interpreted in the ecliptic frame (the MPC/SBDB
    /// convention); the returned orbit carries the ecliptic-to-equatorial
    /// rotation, exactly like [`mpcorb_orbit`]. `gm_km3_s2` is the central
    /// body's GM in km³/s² (use [`crate::constants::GM_SUN`] for
    /// heliocentric orbits).
    ///
    /// ```
    /// use rand::rngs::StdRng;
    /// use rand::SeedableRng;
    /// use starfield::catalogs::synthetic::orbits::SyntheticOrbitPopulation;
    /// use starfield::constants::GM_SUN;
    /// use starfield::time::Timescale;
    ///
    /// let ts = Timescale::default();
    /// let epoch = ts.tt_jd(2451545.0, None);
    /// let mut rng = StdRng::seed_from_u64(7);
    /// let population = SyntheticOrbitPopulation::scattered_disk()
    ///     .with_count(3)
    ///     .generate(&mut rng)
    ///     .unwrap();
    ///
    /// let orbit = population[0].to_kepler_orbit(&epoch, GM_SUN, Some("synthetic-0"));
    /// let position = orbit.at(&epoch);
    /// ```
    pub fn to_kepler_orbit(
        &self,
        epoch: &Time,
        gm_km3_s2: f64,
        target_name: Option<&str>,
    ) -> KeplerOrbit {
        mpcorb_orbit(
            self.a,
            self.e,
            self.i_deg,
            self.omega_big_deg,
            self.omega_deg,
            self.mean_anomaly_deg,
            epoch,
            gm_km3_s2,
            target_name,
        )
    }
}

/// Inclination distribution for a synthetic population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InclinationModel {
    /// All orbits in the reference plane (`i = 0`), for planar
    /// phase-space studies.
    Planar,
    /// Half-normal: `i = |N(0, sigma)|`. The form used by the Planet Nine
    /// scattered-disk simulations (e.g. Batygin & Brown 2016 with
    /// `sigma_deg = 15`). Mean inclination is `sigma * sqrt(2/pi)`.
    HalfNormal {
        /// Standard deviation of the underlying normal (degrees).
        sigma_deg: f64,
    },
    /// Rayleigh: `p(i) di ∝ i exp(-i²/2sigma²) di`, the standard debiased
    /// Kuiper-belt inclination form (Brown 2001). Mean inclination is
    /// `sigma * sqrt(pi/2)`.
    Rayleigh {
        /// Rayleigh scale parameter (degrees).
        sigma_deg: f64,
    },
}

/// Builder-style generator for synthetic small-body orbital populations.
///
/// See the [module docs](self) for the sampled distributions and the
/// deterministic-N resampling guarantee. Defaults (via
/// [`SyntheticOrbitPopulation::scattered_disk`] or `Default`) reproduce the
/// Batygin & Brown (2016) scattered-disk test-particle population:
/// `a` uniform in 150–550 AU, `q` uniform in 30–50 AU, half-normal
/// inclination with `sigma = 15°`.
///
/// ```
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
/// use starfield::catalogs::synthetic::orbits::{InclinationModel, SyntheticOrbitPopulation};
///
/// let mut rng = StdRng::seed_from_u64(1);
/// let belt = SyntheticOrbitPopulation::new()
///     .with_count(50)
///     .with_semi_major_axis_range(39.0, 48.0)
///     .with_perihelion_range(32.0, 45.0)
///     .with_inclination(InclinationModel::Rayleigh { sigma_deg: 12.0 })
///     .generate(&mut rng)
///     .unwrap();
/// assert_eq!(belt.len(), 50);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntheticOrbitPopulation {
    /// Number of orbits to generate (always met exactly).
    pub count: usize,
    /// Semi-major axis minimum (AU).
    pub a_min: f64,
    /// Semi-major axis maximum (AU).
    pub a_max: f64,
    /// Perihelion distance minimum (AU).
    pub q_min: f64,
    /// Perihelion distance maximum (AU).
    pub q_max: f64,
    /// Inclination distribution.
    pub inclination: InclinationModel,
}

impl Default for SyntheticOrbitPopulation {
    fn default() -> Self {
        Self {
            count: 100,
            a_min: 150.0,
            a_max: 550.0,
            q_min: 30.0,
            q_max: 50.0,
            inclination: InclinationModel::HalfNormal { sigma_deg: 15.0 },
        }
    }
}

impl SyntheticOrbitPopulation {
    /// New generator with the scattered-disk defaults (see type docs).
    pub fn new() -> Self {
        Self::default()
    }

    /// The Batygin & Brown (2016) scattered-disk/ETNO population: `a`
    /// uniform in 150–550 AU, `q` uniform in 30–50 AU, half-normal
    /// inclination with `sigma = 15°`. Identical to [`Self::new`]; named
    /// for discoverability.
    pub fn scattered_disk() -> Self {
        Self::default()
    }

    /// Set the number of orbits to generate.
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }

    /// Set the semi-major axis range (AU).
    pub fn with_semi_major_axis_range(mut self, a_min: f64, a_max: f64) -> Self {
        self.a_min = a_min;
        self.a_max = a_max;
        self
    }

    /// Set the perihelion distance range (AU).
    pub fn with_perihelion_range(mut self, q_min: f64, q_max: f64) -> Self {
        self.q_min = q_min;
        self.q_max = q_max;
        self
    }

    /// Set the inclination distribution.
    pub fn with_inclination(mut self, model: InclinationModel) -> Self {
        self.inclination = model;
        self
    }

    /// Validate the configuration without generating.
    ///
    /// Errors if a range is inverted or non-positive, if an inclination
    /// scale is negative, or if `q_min >= a_max` (the `q <= a` acceptance
    /// region would be empty and resampling could never terminate).
    pub fn validate(&self) -> Result<()> {
        let err = |msg: String| Err(StarfieldError::DataError(msg));
        // Written so NaN bounds also fail validation.
        let range_ok = |lo: f64, hi: f64| lo > 0.0 && lo <= hi;
        if !range_ok(self.a_min, self.a_max) {
            return err(format!(
                "invalid semi-major axis range [{}, {}] AU: need 0 < a_min <= a_max",
                self.a_min, self.a_max
            ));
        }
        if !range_ok(self.q_min, self.q_max) {
            return err(format!(
                "invalid perihelion range [{}, {}] AU: need 0 < q_min <= q_max",
                self.q_min, self.q_max
            ));
        }
        if self.q_min >= self.a_max {
            return err(format!(
                "empty (a, q) wedge: q_min = {} AU >= a_max = {} AU, no draw can satisfy q <= a",
                self.q_min, self.a_max
            ));
        }
        match self.inclination {
            InclinationModel::Planar => {}
            InclinationModel::HalfNormal { sigma_deg }
            | InclinationModel::Rayleigh { sigma_deg } => {
                // NaN also fails here.
                let sigma_ok = sigma_deg >= 0.0;
                if !sigma_ok {
                    return err(format!("inclination sigma must be >= 0, got {sigma_deg}"));
                }
            }
        }
        Ok(())
    }

    /// Generate the population.
    ///
    /// Returns exactly [`count`](Self::count) orbits: invalid `(a, q)`
    /// draws with `q > a` are resampled, not skipped, so the output is
    /// uniform on the `q <= a` wedge of the configured rectangle and the
    /// count is deterministic for a given seed.
    pub fn generate<R: Rng + ?Sized>(&self, rng: &mut R) -> Result<Vec<SyntheticOrbit>> {
        self.validate()?;

        let mut orbits = Vec::with_capacity(self.count);
        for _ in 0..self.count {
            // Resample (not skip) invalid q > a draws — see module docs.
            let (a, e) = loop {
                let a = uniform(rng, self.a_min, self.a_max);
                let q = uniform(rng, self.q_min, self.q_max);
                if q <= a {
                    // e = 1 - q/a in [0, 1); q >= q_min > 0 keeps e < 1.
                    break (a, 1.0 - q / a);
                }
            };

            let i_deg = match self.inclination {
                InclinationModel::Planar => 0.0,
                InclinationModel::HalfNormal { sigma_deg } => {
                    sigma_deg * standard_normal(rng).abs()
                }
                InclinationModel::Rayleigh { sigma_deg } => sigma_deg * rayleigh_unit(rng),
            };

            orbits.push(SyntheticOrbit {
                a,
                e,
                i_deg,
                omega_big_deg: uniform(rng, 0.0, 360.0),
                omega_deg: uniform(rng, 0.0, 360.0),
                mean_anomaly_deg: uniform(rng, 0.0, 360.0),
            });
        }
        Ok(orbits)
    }
}

/// Uniform draw on `[lo, hi)`.
fn uniform<R: Rng + ?Sized>(rng: &mut R, lo: f64, hi: f64) -> f64 {
    lo + rng.random::<f64>() * (hi - lo)
}

/// Standard normal draw via Box-Muller (avoids a rand_distr dependency),
/// matching the approach used by `statslib`.
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2: f64 = rng.random();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Unit-scale Rayleigh draw by inverse-CDF: `sqrt(-2 ln U)` with `U` in (0, 1].
fn rayleigh_unit<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u: f64 = (1.0 - rng.random::<f64>()).max(f64::MIN_POSITIVE);
    (-2.0 * u.ln()).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GM_SUN;
    use crate::time::Timescale;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_deterministic_count() {
        // Exactly N even though many (a, q) draws land in the rejected
        // q > a corner (a in [31, 60], q in [30, 59] overlaps heavily).
        let mut rng = StdRng::seed_from_u64(42);
        let pop = SyntheticOrbitPopulation::new()
            .with_count(500)
            .with_semi_major_axis_range(31.0, 60.0)
            .with_perihelion_range(30.0, 59.0)
            .generate(&mut rng)
            .unwrap();
        assert_eq!(pop.len(), 500);
    }

    #[test]
    fn test_same_seed_reproduces() {
        let config = SyntheticOrbitPopulation::scattered_disk().with_count(64);
        let a = config.generate(&mut StdRng::seed_from_u64(7)).unwrap();
        let b = config.generate(&mut StdRng::seed_from_u64(7)).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_elements_within_configured_ranges() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = SyntheticOrbitPopulation::scattered_disk().with_count(400);
        let pop = config.generate(&mut rng).unwrap();

        for orbit in &pop {
            assert!(
                orbit.a >= config.a_min && orbit.a < config.a_max,
                "a = {} out of [{}, {})",
                orbit.a,
                config.a_min,
                config.a_max
            );
            let q = orbit.perihelion();
            assert!(
                q >= config.q_min - 1e-9 && q < config.q_max,
                "q = {} out of [{}, {})",
                q,
                config.q_min,
                config.q_max
            );
            assert!(orbit.e >= 0.0 && orbit.e < 1.0, "e = {}", orbit.e);
            assert!(orbit.i_deg >= 0.0, "i = {}", orbit.i_deg);
            for angle in [orbit.omega_big_deg, orbit.omega_deg, orbit.mean_anomaly_deg] {
                assert!((0.0..360.0).contains(&angle), "angle = {angle}");
            }
        }
    }

    #[test]
    fn test_half_normal_inclination_mean() {
        // For half-normal with sigma = 15°, E[i] = sigma * sqrt(2/pi).
        let mut rng = StdRng::seed_from_u64(123);
        let pop = SyntheticOrbitPopulation::scattered_disk()
            .with_count(10_000)
            .generate(&mut rng)
            .unwrap();
        let expected = 15.0 * (2.0 / std::f64::consts::PI).sqrt();
        let actual: f64 = pop.iter().map(|o| o.i_deg).sum::<f64>() / pop.len() as f64;
        let relative_error = (actual - expected).abs() / expected;
        assert!(
            relative_error < 0.05,
            "mean i = {actual:.3} vs expected {expected:.3}"
        );
    }

    #[test]
    fn test_rayleigh_inclination_mean() {
        // For Rayleigh with sigma = 12°, E[i] = sigma * sqrt(pi/2).
        let mut rng = StdRng::seed_from_u64(456);
        let pop = SyntheticOrbitPopulation::new()
            .with_count(10_000)
            .with_inclination(InclinationModel::Rayleigh { sigma_deg: 12.0 })
            .generate(&mut rng)
            .unwrap();
        let expected = 12.0 * (std::f64::consts::PI / 2.0).sqrt();
        let actual: f64 = pop.iter().map(|o| o.i_deg).sum::<f64>() / pop.len() as f64;
        let relative_error = (actual - expected).abs() / expected;
        assert!(
            relative_error < 0.05,
            "mean i = {actual:.3} vs expected {expected:.3}"
        );
    }

    #[test]
    fn test_planar_inclination() {
        let mut rng = StdRng::seed_from_u64(9);
        let pop = SyntheticOrbitPopulation::new()
            .with_count(20)
            .with_inclination(InclinationModel::Planar)
            .generate(&mut rng)
            .unwrap();
        assert!(pop.iter().all(|o| o.i_deg == 0.0));
    }

    #[test]
    fn test_invalid_configs_rejected() {
        let mut rng = StdRng::seed_from_u64(1);
        // Inverted a range.
        assert!(SyntheticOrbitPopulation::new()
            .with_semi_major_axis_range(550.0, 150.0)
            .generate(&mut rng)
            .is_err());
        // Inverted q range.
        assert!(SyntheticOrbitPopulation::new()
            .with_perihelion_range(50.0, 30.0)
            .generate(&mut rng)
            .is_err());
        // Empty wedge: q_min >= a_max would loop forever if not rejected.
        assert!(SyntheticOrbitPopulation::new()
            .with_semi_major_axis_range(100.0, 150.0)
            .with_perihelion_range(150.0, 200.0)
            .generate(&mut rng)
            .is_err());
        // Negative inclination scale.
        assert!(SyntheticOrbitPopulation::new()
            .with_inclination(InclinationModel::HalfNormal { sigma_deg: -1.0 })
            .generate(&mut rng)
            .is_err());
    }

    #[test]
    fn test_derived_quantities() {
        let orbit = SyntheticOrbit {
            a: 500.0,
            e: 0.9,
            i_deg: 10.0,
            omega_big_deg: 300.0,
            omega_deg: 100.0,
            mean_anomaly_deg: 0.0,
        };
        assert!((orbit.perihelion() - 50.0).abs() < 1e-12);
        assert!((orbit.aphelion() - 950.0).abs() < 1e-12);
        // 300 + 100 = 400 wraps to 40.
        assert!((orbit.longitude_of_perihelion_deg() - 40.0).abs() < 1e-12);
    }

    #[test]
    fn test_to_kepler_orbit_distance_in_radial_bounds() {
        let ts = Timescale::default();
        let epoch = ts.tt_jd(2_451_545.0, None);
        let mut rng = StdRng::seed_from_u64(99);
        let pop = SyntheticOrbitPopulation::scattered_disk()
            .with_count(10)
            .generate(&mut rng)
            .unwrap();

        for (n, orbit) in pop.iter().enumerate() {
            let kepler = orbit.to_kepler_orbit(&epoch, GM_SUN, Some(&format!("synth-{n}")));
            let r = kepler.at(&epoch).position.norm();
            let (q, big_q) = (orbit.perihelion(), orbit.aphelion());
            assert!(
                r >= q * (1.0 - 1e-9) && r <= big_q * (1.0 + 1e-9),
                "r = {r} outside [{q}, {big_q}]"
            );
        }
    }
}
