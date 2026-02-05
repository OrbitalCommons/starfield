//! Position hierarchy for astronomical observations
//!
//! Implements Skyfield's ICRF → Barycentric → Astrometric → Apparent pipeline:
//!
//! 1. **Barycentric** — position relative to solar system barycenter (SSB)
//! 2. **Astrometric** — light-time corrected position (observer → target)
//! 3. **Apparent** — with gravitational deflection and aberration applied
//!
//! # Example
//!
//! ```ignore
//! let kernel = SpiceKernel::open("de421.bsp")?;
//! let ts = Timescale::default();
//! let t = ts.tdb_jd(2451545.0);
//!
//! let earth = kernel.at("earth", &t)?;
//! let mars_astrometric = earth.observe("mars", &mut kernel, &t)?;
//! let mars_apparent = mars_astrometric.apparent(&mut kernel, &t)?;
//! let (ra, dec, dist) = mars_apparent.radec(None);
//! ```

use nalgebra::Vector3;
use std::f64::consts::PI;

use crate::constants::{AU_KM, C_AUDAY, DAY_S};
use crate::framelib::inertial::{Ecliptic, Equatorial};
use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem::spk::jd_to_seconds;
use crate::relativity::{add_aberration, add_deflection, rmass, DEFLECTORS};
use crate::time::Time;

/// Maximum number of light-time iterations before giving up
const MAX_LIGHT_TIME_ITERATIONS: usize = 10;

/// Convergence threshold for light-time iteration (days)
const LIGHT_TIME_EPSILON: f64 = 1e-12;

/// Number of deflector bodies to include by default (sun, jupiter, saturn)
const DEFAULT_DEFLECTOR_COUNT: usize = 3;

/// The kind of position, determining what corrections have been applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionKind {
    /// Barycentric position relative to the solar system barycenter
    Barycentric,
    /// Astrometric position: light-time corrected, observer to target
    Astrometric,
    /// Apparent position: with deflection and aberration applied
    Apparent,
}

/// A position in the ICRF (International Celestial Reference Frame).
///
/// This struct represents a point in space at a moment in time, with
/// different levels of correction applied depending on its [`PositionKind`].
#[derive(Debug, Clone)]
pub struct Position {
    /// XYZ position in AU
    pub position: Vector3<f64>,
    /// XYZ velocity in AU/day
    pub velocity: Vector3<f64>,
    /// What corrections have been applied
    pub kind: PositionKind,
    /// NAIF ID of the center body (0 = SSB)
    pub center: i32,
    /// NAIF ID of the target body
    pub target: i32,
    /// Light travel time in days (set after observe)
    pub light_time: f64,
    /// Barycentric position of the observer (for apparent computation)
    observer_barycentric: Option<Box<Position>>,
}

impl Position {
    /// Create a new Barycentric position from raw SPK data.
    ///
    /// Position in AU, velocity in AU/day, both relative to SSB.
    pub fn barycentric(position: Vector3<f64>, velocity: Vector3<f64>, target: i32) -> Self {
        Position {
            position,
            velocity,
            kind: PositionKind::Barycentric,
            center: 0,
            target,
            light_time: 0.0,
            observer_barycentric: None,
        }
    }

    /// Create an Astrometric position from an observer and relative vectors.
    ///
    /// Used by `Star::observe_from()` and other non-ephemeris observations.
    pub fn astrometric(
        position: Vector3<f64>,
        velocity: Vector3<f64>,
        observer: &Position,
        target_id: i32,
        light_time: f64,
    ) -> Self {
        Position {
            position,
            velocity,
            kind: PositionKind::Astrometric,
            center: observer.target,
            target: target_id,
            light_time,
            observer_barycentric: Some(Box::new(observer.clone())),
        }
    }

    /// Compute the astrometric position of another body as seen from this position.
    ///
    /// Performs light-time iteration: finds where the target *was* when
    /// the light now arriving at the observer was emitted.
    ///
    /// Only valid on Barycentric positions.
    pub fn observe(
        &self,
        target_name: &str,
        kernel: &mut SpiceKernel,
        time: &Time,
    ) -> crate::jplephem::errors::Result<Position> {
        assert_eq!(
            self.kind,
            PositionKind::Barycentric,
            "observe() requires a Barycentric position"
        );

        let target_vf = kernel.get(target_name)?;
        let target_id = target_vf.target_id;
        let target_chain = target_vf.chain.clone();

        // Initial target position at observation time
        let tdb_seconds = jd_to_seconds(time.tdb());
        let (mut target_pos_km, mut target_vel_km_s) =
            kernel.compute_chain_pub(&target_chain, tdb_seconds)?;

        let observer_pos_km = self.position * AU_KM;

        // Compute initial distance
        let mut distance_km = (target_pos_km - observer_pos_km).norm();
        let mut light_time0 = 0.0;

        // Light-time iteration
        for _ in 0..MAX_LIGHT_TIME_ITERATIONS {
            let light_time = distance_km / (C_AUDAY * AU_KM);

            let delta = light_time - light_time0;
            if delta.abs() < LIGHT_TIME_EPSILON {
                // Converged — compute final relative position
                let rel_pos_km = target_pos_km - observer_pos_km;
                let rel_vel_km_s = target_vel_km_s - self.velocity * AU_KM / DAY_S;

                return Ok(Position {
                    position: Vector3::new(
                        rel_pos_km.x / AU_KM,
                        rel_pos_km.y / AU_KM,
                        rel_pos_km.z / AU_KM,
                    ),
                    velocity: Vector3::new(
                        rel_vel_km_s.x * DAY_S / AU_KM,
                        rel_vel_km_s.y * DAY_S / AU_KM,
                        rel_vel_km_s.z * DAY_S / AU_KM,
                    ),
                    kind: PositionKind::Astrometric,
                    center: self.target,
                    target: target_id,
                    light_time,
                    observer_barycentric: Some(Box::new(self.clone())),
                });
            }

            // Recompute target position at retarded time
            let retarded_tdb = time.tdb() - light_time;
            let retarded_seconds = jd_to_seconds(retarded_tdb);
            let (pos, vel) = kernel.compute_chain_pub(&target_chain, retarded_seconds)?;
            target_pos_km = pos;
            target_vel_km_s = vel;
            distance_km = (target_pos_km - observer_pos_km).norm();
            light_time0 = light_time;
        }

        Err(crate::jplephem::errors::JplephemError::Other(
            "Light-time iteration failed to converge".to_string(),
        ))
    }

    /// Compute the apparent position by applying gravitational deflection
    /// and stellar aberration.
    ///
    /// Only valid on Astrometric positions.
    pub fn apparent(
        &self,
        kernel: &mut SpiceKernel,
        time: &Time,
    ) -> crate::jplephem::errors::Result<Position> {
        assert_eq!(
            self.kind,
            PositionKind::Astrometric,
            "apparent() requires an Astrometric position"
        );

        let observer = self
            .observer_barycentric
            .as_ref()
            .expect("Astrometric position must have observer_barycentric set");

        let mut target_au = self.position;

        // Apply gravitational light deflection from major bodies
        let tdb_seconds = jd_to_seconds(time.tdb());

        for &deflector_name in DEFLECTORS.iter().take(DEFAULT_DEFLECTOR_COUNT) {
            if let Some(rm) = rmass(deflector_name) {
                // Try to look up the deflector; some kernels may not have all bodies
                let deflector_name_for_kernel = match deflector_name {
                    "jupiter" => "jupiter barycenter",
                    "saturn" => "saturn barycenter",
                    "uranus" => "uranus barycenter",
                    "neptune" => "neptune barycenter",
                    _ => deflector_name,
                };

                if let Ok(vf) = kernel.get(deflector_name_for_kernel) {
                    let chain = vf.chain.clone();
                    if let Ok((defl_pos_km, _)) = kernel.compute_chain_pub(&chain, tdb_seconds) {
                        let deflector_pos_au = Vector3::new(
                            defl_pos_km.x / AU_KM,
                            defl_pos_km.y / AU_KM,
                            defl_pos_km.z / AU_KM,
                        );

                        add_deflection(&mut target_au, &observer.position, &deflector_pos_au, rm);
                    }
                }
            }
        }

        // Apply stellar aberration
        add_aberration(&mut target_au, &observer.velocity, self.light_time);

        Ok(Position {
            position: target_au,
            velocity: self.velocity,
            kind: PositionKind::Apparent,
            center: self.center,
            target: self.target,
            light_time: self.light_time,
            observer_barycentric: self.observer_barycentric.clone(),
        })
    }

    /// Compute equatorial right ascension, declination, and distance.
    ///
    /// Returns `(ra_hours, dec_degrees, distance_au)`.
    ///
    /// When `epoch` is `None`, returns ICRF coordinates.
    /// When `epoch` is `Some(time)`, rotates by the precession-nutation matrix
    /// of that epoch (use the observation time for epoch-of-date coordinates).
    pub fn radec(&self, epoch: Option<&Time>) -> (f64, f64, f64) {
        let mut pos = self.position;

        // If an epoch is provided, rotate into the dynamical frame of that epoch
        if let Some(ep) = epoch {
            pos = ep.m_matrix() * pos;
        }

        let (r, dec, ra) = to_spherical(&pos);
        let ra_hours = ra.to_degrees() / 15.0;
        let dec_degrees = dec.to_degrees();
        (ra_hours, dec_degrees, r)
    }

    /// Distance from observer to target in AU
    pub fn distance(&self) -> f64 {
        self.position.norm()
    }

    /// Angular separation from another position, in radians
    pub fn separation_from(&self, other: &Position) -> f64 {
        let a = self.position.normalize();
        let b = other.position.normalize();
        let dot = a.dot(&b).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Compute ecliptic longitude and latitude of this position.
    ///
    /// Returns `(longitude_radians, latitude_radians, distance_au)`.
    /// Longitude is in [0, 2*PI), latitude in [-PI/2, PI/2].
    pub fn ecliptic_latlon(&self) -> (f64, f64, f64) {
        let eq = Equatorial::new(
            self.position.y.atan2(self.position.x),
            (self.position.z / self.position.norm()).asin(),
        );
        let ec: Ecliptic = eq.into();
        let mut lon = ec.lon;
        if lon < 0.0 {
            lon += 2.0 * PI;
        }
        (lon, ec.lat, self.position.norm())
    }
}

impl std::fmt::Display for Position {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kind = match self.kind {
            PositionKind::Barycentric => "Barycentric",
            PositionKind::Astrometric => "Astrometric",
            PositionKind::Apparent => "Apparent",
        };
        write!(
            f,
            "{} position ({} → {}): [{:.6}, {:.6}, {:.6}] AU",
            kind, self.center, self.target, self.position.x, self.position.y, self.position.z
        )
    }
}

/// Convert Cartesian XYZ to spherical coordinates.
///
/// Returns `(radius, declination, right_ascension)` where angles are in radians.
fn to_spherical(xyz: &Vector3<f64>) -> (f64, f64, f64) {
    let x = xyz.x;
    let y = xyz.y;
    let z = xyz.z;
    let r = xyz.norm();

    let dec = if r > 0.0 { (z / r).asin() } else { 0.0 };

    let mut ra = y.atan2(x);
    if ra < 0.0 {
        ra += 2.0 * PI;
    }

    (r, dec, ra)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_barycentric_creation() {
        let pos = Position::barycentric(
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.01, 0.0),
            399,
        );
        assert_eq!(pos.kind, PositionKind::Barycentric);
        assert_eq!(pos.center, 0);
        assert_eq!(pos.target, 399);
        assert_relative_eq!(pos.distance(), 1.0);
    }

    #[test]
    fn test_to_spherical_along_x() {
        let pos = Vector3::new(1.0, 0.0, 0.0);
        let (r, dec, ra) = to_spherical(&pos);
        assert_relative_eq!(r, 1.0);
        assert_relative_eq!(dec, 0.0, epsilon = 1e-15);
        assert_relative_eq!(ra, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_to_spherical_along_y() {
        let pos = Vector3::new(0.0, 1.0, 0.0);
        let (r, dec, ra) = to_spherical(&pos);
        assert_relative_eq!(r, 1.0);
        assert_relative_eq!(dec, 0.0, epsilon = 1e-15);
        assert_relative_eq!(ra, PI / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_to_spherical_along_z() {
        let pos = Vector3::new(0.0, 0.0, 1.0);
        let (r, dec, _ra) = to_spherical(&pos);
        assert_relative_eq!(r, 1.0);
        assert_relative_eq!(dec, PI / 2.0, epsilon = 1e-15);
    }

    #[test]
    fn test_to_spherical_negative_x() {
        // RA should wrap to PI
        let pos = Vector3::new(-1.0, 0.0, 0.0);
        let (_, _, ra) = to_spherical(&pos);
        assert_relative_eq!(ra, PI, epsilon = 1e-15);
    }

    #[test]
    fn test_radec_along_x() {
        let pos = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 0);
        let (ra_h, dec_d, dist) = pos.radec(None);
        assert_relative_eq!(ra_h, 0.0, epsilon = 1e-10);
        assert_relative_eq!(dec_d, 0.0, epsilon = 1e-10);
        assert_relative_eq!(dist, 1.0);
    }

    #[test]
    fn test_radec_along_y() {
        let pos = Position::barycentric(Vector3::new(0.0, 1.0, 0.0), Vector3::zeros(), 0);
        let (ra_h, dec_d, _) = pos.radec(None);
        assert_relative_eq!(ra_h, 6.0, epsilon = 1e-10); // 90 deg = 6 hours
        assert_relative_eq!(dec_d, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_radec_north_pole() {
        let pos = Position::barycentric(Vector3::new(0.0, 0.0, 1.0), Vector3::zeros(), 0);
        let (_, dec_d, _) = pos.radec(None);
        assert_relative_eq!(dec_d, 90.0, epsilon = 1e-10);
    }

    #[test]
    fn test_separation_from() {
        let a = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 0);
        let b = Position::barycentric(Vector3::new(0.0, 1.0, 0.0), Vector3::zeros(), 0);
        let sep = a.separation_from(&b);
        assert_relative_eq!(sep, PI / 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_separation_from_same_direction() {
        let a = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 0);
        let b = Position::barycentric(Vector3::new(2.0, 0.0, 0.0), Vector3::zeros(), 0);
        let sep = a.separation_from(&b);
        assert_relative_eq!(sep, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_display() {
        let pos = Position::barycentric(Vector3::new(1.0, 2.0, 3.0), Vector3::zeros(), 399);
        let s = format!("{}", pos);
        assert!(s.contains("Barycentric"));
        assert!(s.contains("399"));
    }

    // --- DE421 integration tests ---

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    fn j2000_time() -> crate::time::Time {
        crate::time::Timescale::default().tdb_jd(2451545.0)
    }

    #[test]
    fn test_kernel_at_returns_barycentric() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        assert_eq!(earth.kind, PositionKind::Barycentric);
        assert_eq!(earth.center, 0);

        // Earth should be roughly 1 AU from SSB
        let dist = earth.distance();
        assert!(
            dist > 0.9 && dist < 1.1,
            "Earth distance from SSB should be ~1 AU, got {}",
            dist
        );
    }

    #[test]
    fn test_observe_returns_astrometric() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        assert_eq!(mars.kind, PositionKind::Astrometric);
        assert_eq!(mars.center, 399); // center is Earth
        assert!(mars.light_time > 0.0, "Light time should be positive");

        // Mars distance from Earth at J2000 should be roughly 1-3 AU
        let dist = mars.distance();
        assert!(
            dist > 0.5 && dist < 3.0,
            "Earth-Mars distance should be 0.5-3 AU, got {}",
            dist
        );
    }

    #[test]
    fn test_observe_light_time_reasonable() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        // Light time to Mars should be roughly 3-22 minutes = 0.002-0.015 days
        let lt_minutes = mars.light_time * 24.0 * 60.0;
        assert!(
            lt_minutes > 2.0 && lt_minutes < 25.0,
            "Light time to Mars should be 3-22 min, got {} min",
            lt_minutes
        );
    }

    #[test]
    fn test_apparent_returns_apparent() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let mars_astro = earth.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();

        assert_eq!(mars_app.kind, PositionKind::Apparent);

        // Apparent position should be very close to astrometric
        // (deflection + aberration are small corrections)
        let diff = (mars_app.position - mars_astro.position).norm();
        assert!(
            diff < 0.001,
            "Apparent vs astrometric difference should be tiny, got {} AU",
            diff
        );
    }

    #[test]
    fn test_radec_mars_at_j2000() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let (ra_h, dec_d, dist) = mars.radec(None);

        // RA and Dec should be finite and in valid ranges
        assert!(ra_h >= 0.0 && ra_h < 24.0, "RA {} out of range", ra_h);
        assert!(
            dec_d >= -90.0 && dec_d <= 90.0,
            "Dec {} out of range",
            dec_d
        );
        assert!(dist > 0.0, "Distance should be positive");
    }

    #[test]
    fn test_full_pipeline_earth_to_mars() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        // Full pipeline: barycentric → astrometric → apparent → radec
        let earth = kernel.at("earth", &t).unwrap();
        let mars_astro = earth.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();
        let (ra_h, dec_d, _dist) = mars_app.radec(None);

        // Just verify the pipeline completes and produces reasonable values
        assert!(ra_h >= 0.0 && ra_h < 24.0);
        assert!(dec_d >= -90.0 && dec_d <= 90.0);
    }

    #[test]
    fn test_observe_jupiter() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let jupiter = earth
            .observe("jupiter barycenter", &mut kernel, &t)
            .unwrap();

        // Jupiter should be 3.9-6.5 AU from Earth
        let dist = jupiter.distance();
        assert!(
            dist > 3.5 && dist < 7.0,
            "Earth-Jupiter distance should be 3.5-7 AU, got {}",
            dist
        );
    }

    #[test]
    fn test_radec_epoch_of_date() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        let (ra_icrf, dec_icrf, _) = mars.radec(None);
        let (ra_date, dec_date, _) = mars.radec(Some(&t));

        // At J2000, epoch-of-date should be very close to ICRF
        // (precession is nearly zero at the reference epoch)
        assert!(
            (ra_icrf - ra_date).abs() < 0.01,
            "RA ICRF vs date should be close at J2000"
        );
        assert!(
            (dec_icrf - dec_date).abs() < 0.01,
            "Dec ICRF vs date should be close at J2000"
        );
    }
}
