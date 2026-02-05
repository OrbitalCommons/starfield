//! Distant stars with proper motion, parallax, and radial velocity
//!
//! Implements Skyfield's `starlib.py` — represent a fixed celestial object
//! and propagate its position accounting for space motion.
//!
//! # Example
//!
//! ```ignore
//! use starfield::starlib::Star;
//!
//! // Barnard's Star with full space motion
//! let barnard = Star::new(
//!     269.452_083_75,    // RA degrees
//!     4.693_390_889,     // Dec degrees
//!     -798.71,           // RA proper motion (mas/yr)
//!     10337.77,          // Dec proper motion (mas/yr)
//!     545.4,             // parallax (mas)
//!     -110.6,            // radial velocity (km/s)
//! );
//!
//! let earth = kernel.at("earth", &t)?;
//! let astrometric = barnard.observe_from(&earth, &t);
//! let apparent = astrometric.apparent(&mut kernel, &t)?;
//! let (ra, dec, dist) = apparent.radec(None);
//! ```

use nalgebra::Vector3;

use crate::constants::{ASEC2RAD, AU_KM, C, C_AUDAY, DAY_S, J2000};
use crate::positions::{Position, PositionKind};
use crate::relativity::light_time_difference;
use crate::time::Time;

/// A distant star or fixed object in the ICRS.
///
/// Stores the object's position and velocity in Cartesian AU and AU/day,
/// precomputed from its catalog coordinates and space motion parameters.
#[derive(Debug, Clone)]
pub struct Star {
    /// Right ascension in radians (ICRS)
    pub ra: f64,
    /// Declination in radians (ICRS)
    pub dec: f64,
    /// Proper motion in RA in milliarcseconds per year
    pub ra_mas_per_year: f64,
    /// Proper motion in Dec in milliarcseconds per year
    pub dec_mas_per_year: f64,
    /// Parallax in milliarcseconds
    pub parallax_mas: f64,
    /// Radial velocity in km/s
    pub radial_km_per_s: f64,
    /// Reference epoch as TT Julian date (default J2000.0)
    pub epoch: f64,
    /// Precomputed ICRS position in AU
    pub(crate) position_au: Vector3<f64>,
    /// Precomputed ICRS velocity in AU/day
    pub(crate) velocity_au_per_day: Vector3<f64>,
}

impl Star {
    /// Create a star from catalog coordinates.
    ///
    /// # Arguments
    /// * `ra_degrees` — Right ascension in degrees (ICRS)
    /// * `dec_degrees` — Declination in degrees (ICRS)
    /// * `ra_mas_per_year` — Proper motion in RA (mas/yr, includes cos(dec))
    /// * `dec_mas_per_year` — Proper motion in Dec (mas/yr)
    /// * `parallax_mas` — Parallax in milliarcseconds (0 → distance = 1 Gpc)
    /// * `radial_km_per_s` — Radial velocity in km/s
    pub fn new(
        ra_degrees: f64,
        dec_degrees: f64,
        ra_mas_per_year: f64,
        dec_mas_per_year: f64,
        parallax_mas: f64,
        radial_km_per_s: f64,
    ) -> Self {
        let ra = ra_degrees.to_radians();
        let dec = dec_degrees.to_radians();

        let mut star = Star {
            ra,
            dec,
            ra_mas_per_year,
            dec_mas_per_year,
            parallax_mas,
            radial_km_per_s,
            epoch: J2000,
            position_au: Vector3::zeros(),
            velocity_au_per_day: Vector3::zeros(),
        };
        star.compute_vectors();
        star
    }

    /// Create a star from RA in hours and Dec in degrees.
    pub fn from_ra_hours(
        ra_hours: f64,
        dec_degrees: f64,
        ra_mas_per_year: f64,
        dec_mas_per_year: f64,
        parallax_mas: f64,
        radial_km_per_s: f64,
    ) -> Self {
        Self::new(
            ra_hours * 15.0,
            dec_degrees,
            ra_mas_per_year,
            dec_mas_per_year,
            parallax_mas,
            radial_km_per_s,
        )
    }

    /// Set the reference epoch (TT Julian date).
    pub fn with_epoch(mut self, epoch: f64) -> Self {
        self.epoch = epoch;
        self.compute_vectors();
        self
    }

    /// Compute the astrometric position of this star as seen from an observer.
    ///
    /// The observer must be a Barycentric position. This method:
    /// 1. Propagates the star's position to the observation epoch
    /// 2. Applies light-time correction
    /// 3. Computes the relative position vector
    ///
    /// Returns an Astrometric position suitable for `.apparent()`.
    pub fn observe_from(&self, observer: &Position, time: &Time) -> Position {
        assert_eq!(
            observer.kind,
            PositionKind::Barycentric,
            "observe_from() requires a Barycentric position"
        );

        // Light-time correction
        let dt = light_time_difference(&self.position_au, &observer.position);

        // Propagate star position to observation epoch
        let t_elapsed = time.tdb() + dt - self.epoch;
        let position = self.position_au + self.velocity_au_per_day * t_elapsed;

        // Relative position and velocity
        let vector = position - observer.position;
        let velocity = -self.velocity_au_per_day + observer.velocity;

        let distance = vector.norm();
        let light_time = distance / C_AUDAY;

        Position::astrometric(vector, velocity, observer, -1, light_time)
    }

    /// Compute the ICRS position and velocity vectors from catalog parameters.
    fn compute_vectors(&mut self) {
        // Use 1 gigaparsec for stars with zero or negative parallax
        let parallax = if self.parallax_mas <= 0.0 {
            1.0e-6
        } else {
            self.parallax_mas
        };

        // Distance from parallax
        let dist = 1.0 / (parallax * 1.0e-3 * ASEC2RAD).sin();

        let cra = self.ra.cos();
        let sra = self.ra.sin();
        let cdc = self.dec.cos();
        let sdc = self.dec.sin();

        // Position in AU
        self.position_au = Vector3::new(dist * cdc * cra, dist * cdc * sra, dist * sdc);

        // Doppler factor
        let k = 1.0 / (1.0 - self.radial_km_per_s / (C * 1e-3));

        // Proper motion → velocity in AU/day
        let pmr = self.ra_mas_per_year / (parallax * 365.25) * k;
        let pmd = self.dec_mas_per_year / (parallax * 365.25) * k;
        let rvl = self.radial_km_per_s * DAY_S / AU_KM * k;

        self.velocity_au_per_day = Vector3::new(
            -pmr * sra - pmd * sdc * cra + rvl * cdc * cra,
            pmr * cra - pmd * sdc * sra + rvl * cdc * sra,
            pmd * cdc + rvl * sdc,
        );
    }

    /// Right ascension in degrees
    pub fn ra_degrees(&self) -> f64 {
        self.ra.to_degrees()
    }

    /// Right ascension in hours
    pub fn ra_hours(&self) -> f64 {
        self.ra.to_degrees() / 15.0
    }

    /// Declination in degrees
    pub fn dec_degrees(&self) -> f64 {
        self.dec.to_degrees()
    }
}

impl std::fmt::Display for Star {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Star(RA={:.6}h, Dec={:.6}°, parallax={:.1} mas)",
            self.ra_hours(),
            self.dec_degrees(),
            self.parallax_mas
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::kernel::SpiceKernel;
    use approx::assert_relative_eq;

    // Barnard's Star — high proper motion, well-characterized
    fn barnard() -> Star {
        Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        )
    }

    // Polaris — slow proper motion
    fn polaris() -> Star {
        Star::from_ra_hours(2.530_1, 89.264_1, 44.22, -11.74, 7.54, -17.4)
    }

    // A star with zero proper motion (distant quasar-like)
    fn distant_star() -> Star {
        Star::new(180.0, 45.0, 0.0, 0.0, 0.0, 0.0)
    }

    #[test]
    fn test_star_creation() {
        let s = barnard();
        assert_relative_eq!(s.ra_hours(), 17.963_471_675, epsilon = 1e-6);
        assert_relative_eq!(s.dec_degrees(), 4.693_390_889, epsilon = 1e-6);
    }

    #[test]
    fn test_star_from_degrees() {
        let s = Star::new(90.0, 45.0, 0.0, 0.0, 10.0, 0.0);
        assert_relative_eq!(s.ra_hours(), 6.0, epsilon = 1e-10);
        assert_relative_eq!(s.dec_degrees(), 45.0, epsilon = 1e-10);
    }

    #[test]
    fn test_position_vector_direction() {
        // Star at RA=0, Dec=0 should be along +x
        let s = Star::new(0.0, 0.0, 0.0, 0.0, 100.0, 0.0);
        assert!(s.position_au.x > 0.0);
        assert_relative_eq!(s.position_au.y, 0.0, epsilon = 1e-10);
        assert_relative_eq!(s.position_au.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_position_vector_ra90() {
        // Star at RA=90°, Dec=0 should be along +y
        let s = Star::new(90.0, 0.0, 0.0, 0.0, 100.0, 0.0);
        assert_relative_eq!(s.position_au.x, 0.0, epsilon = 1e-6);
        assert!(s.position_au.y > 0.0);
        assert_relative_eq!(s.position_au.z, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_position_vector_north_pole() {
        // Star at Dec=90° should be along +z
        let s = Star::new(0.0, 90.0, 0.0, 0.0, 100.0, 0.0);
        assert_relative_eq!(s.position_au.x, 0.0, epsilon = 1e-6);
        assert_relative_eq!(s.position_au.y, 0.0, epsilon = 1e-10);
        assert!(s.position_au.z > 0.0);
    }

    #[test]
    fn test_distance_from_parallax() {
        // 100 mas parallax → ~10 parsecs → ~2,062,648 AU
        let s = Star::new(0.0, 0.0, 0.0, 0.0, 100.0, 0.0);
        let dist = s.position_au.norm();
        let parsecs = dist / 206264.806_247; // AU per parsec
        assert_relative_eq!(parsecs, 10.0, epsilon = 0.1);
    }

    #[test]
    fn test_zero_parallax_gives_huge_distance() {
        let s = Star::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let dist = s.position_au.norm();
        // Should be ~1 gigaparsec
        assert!(dist > 1e12, "Distance should be enormous, got {}", dist);
    }

    #[test]
    fn test_velocity_zero_proper_motion() {
        let s = Star::new(0.0, 0.0, 0.0, 0.0, 100.0, 0.0);
        let v = s.velocity_au_per_day.norm();
        assert_relative_eq!(v, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_velocity_nonzero_for_barnard() {
        let s = barnard();
        let v = s.velocity_au_per_day.norm();
        assert!(v > 0.0, "Barnard's Star should have nonzero velocity");
    }

    #[test]
    fn test_display() {
        let s = barnard();
        let display = format!("{}", s);
        assert!(display.contains("Star"));
        assert!(display.contains("545.4"));
    }

    #[test]
    fn test_with_epoch() {
        let s = barnard().with_epoch(2451545.0 + 365.25 * 10.0);
        assert_relative_eq!(s.epoch, J2000 + 365.25 * 10.0, epsilon = 1e-10);
    }

    // --- Integration tests with DE421 ---

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    fn j2000_time() -> Time {
        crate::time::Timescale::default().tdb_jd(2451545.0)
    }

    #[test]
    fn test_observe_from_returns_astrometric() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();

        let s = barnard();
        let astro = s.observe_from(&earth, &t);
        assert_eq!(astro.kind, PositionKind::Astrometric);
        assert!(astro.light_time > 0.0);
    }

    #[test]
    fn test_distant_star_direction_preserved() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();

        // A very distant star: parallax shift should be negligible
        let s = distant_star();
        let astro = s.observe_from(&earth, &t);
        let (ra_h, dec_d, _) = astro.radec(None);

        // Should be very close to catalog RA/Dec
        assert_relative_eq!(ra_h, 12.0, epsilon = 0.01); // 180° = 12h
        assert_relative_eq!(dec_d, 45.0, epsilon = 0.01);
    }

    #[test]
    fn test_full_pipeline_barnard() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();

        let s = barnard();
        let astro = s.observe_from(&earth, &t);
        let apparent = astro.apparent(&mut kernel, &t).unwrap();
        let (ra_h, dec_d, _) = apparent.radec(None);

        // Barnard's Star is at roughly RA 17h58m, Dec +4°42'
        assert!(ra_h > 17.0 && ra_h < 18.5, "RA {} out of range", ra_h);
        assert!(dec_d > 3.0 && dec_d < 6.0, "Dec {} out of range", dec_d);
    }

    #[test]
    fn test_polaris_near_north_pole() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();

        let s = polaris();
        let astro = s.observe_from(&earth, &t);
        let (_, dec_d, _) = astro.radec(None);

        // Polaris should be very close to Dec +89°
        assert!(
            dec_d > 88.0 && dec_d < 90.0,
            "Polaris Dec {} out of range",
            dec_d
        );
    }

    #[test]
    fn test_proper_motion_changes_position_over_time() {
        let mut kernel = de421_kernel();
        let ts = crate::time::Timescale::default();

        let s = barnard();

        // Observe at J2000 and at J2000 + 50 years (within DE421 range)
        let t1 = ts.tdb_jd(2451545.0);
        let t2 = ts.tdb_jd(2451545.0 + 50.0 * 365.25);

        let earth1 = kernel.at("earth", &t1).unwrap();
        let earth2 = kernel.at("earth", &t2).unwrap();

        let astro1 = s.observe_from(&earth1, &t1);
        let astro2 = s.observe_from(&earth2, &t2);

        let (_ra1, dec1, _) = astro1.radec(None);
        let (_ra2, dec2, _) = astro2.radec(None);

        // Barnard's Star moves ~10.3"/yr in Dec → ~8.6' in 50 years ≈ 0.14°
        let dec_shift = (dec2 - dec1).abs();
        assert!(
            dec_shift > 0.1,
            "Barnard's Star Dec should shift significantly over 50 years, got {}°",
            dec_shift
        );
    }
}

#[cfg(feature = "python-tests")]
mod python_tests;
