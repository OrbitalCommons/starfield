//! Geographic observer positions on Earth
//!
//! Implements Skyfield's `toposlib.py` — represent an observer at a geographic
//! location on Earth and compute their GCRS position and local horizon coordinates.
//!
//! The key transformation chain is:
//! ```text
//! Geographic (lat/lon/elev) → ITRS xyz → rotate by C^T → GCRS xyz
//! ```
//!
//! # Example
//!
//! ```ignore
//! use starfield::toposlib::{Geoid, WGS84};
//!
//! let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
//! let t = ts.utc((2024, 6, 21, 12, 0, 0.0));
//! let observer = boston.at(&t);
//! let mars_app = observer.observe("mars", &mut kernel, &t)?.apparent(&mut kernel, &t)?;
//! let (alt, az, dist) = boston.altaz(&mars_app, &t);
//! ```

use nalgebra::Vector3;
use std::f64::consts::PI;

use crate::constants::{
    AU_M, DAY_S, EARTH_ANGVEL, EARTH_RADIUS, IERS_2010_INVERSE_EARTH_FLATTENING,
};
use crate::positions::{Position, PositionKind};
use crate::time::Time;

/// An Earth ellipsoid model used for geodetic-to-geocentric conversion.
#[derive(Debug, Clone)]
pub struct Geoid {
    /// Name of the geoid model
    pub name: &'static str,
    /// Equatorial radius in meters
    pub radius: f64,
    /// Inverse flattening (a / (a - b))
    pub inverse_flattening: f64,
    /// (1 - f)^2, precomputed
    one_minus_flattening_squared: f64,
}

impl Geoid {
    /// Create a new geoid model.
    pub const fn new(name: &'static str, radius: f64, inverse_flattening: f64) -> Self {
        let f = 1.0 / inverse_flattening;
        let omf = 1.0 - f;
        Geoid {
            name,
            radius,
            inverse_flattening,
            one_minus_flattening_squared: omf * omf,
        }
    }

    /// Create a geographic position on this ellipsoid.
    ///
    /// # Arguments
    /// * `latitude_degrees` — Geodetic latitude in degrees (positive north)
    /// * `longitude_degrees` — Geodetic longitude in degrees (positive east)
    /// * `elevation_m` — Height above ellipsoid in meters
    pub fn latlon(
        &self,
        latitude_degrees: f64,
        longitude_degrees: f64,
        elevation_m: f64,
    ) -> GeographicPosition {
        let lat = latitude_degrees * PI / 180.0;
        let lon = longitude_degrees * PI / 180.0;

        let sinphi = lat.sin();
        let cosphi = lat.cos();

        // Radius of curvature in the prime vertical
        let c =
            1.0 / (cosphi * cosphi + sinphi * sinphi * self.one_minus_flattening_squared).sqrt();
        let s = self.one_minus_flattening_squared * c;

        // Convert to AU
        let radius_au = self.radius / AU_M;
        let elevation_au = elevation_m / AU_M;

        // ITRS position
        let xy = (radius_au * c + elevation_au) * cosphi;
        let x = xy * lon.cos();
        let y = xy * lon.sin();
        let z = (radius_au * s + elevation_au) * sinphi;

        GeographicPosition {
            latitude: lat,
            longitude: lon,
            elevation_m,
            itrs_xyz: Vector3::new(x, y, z),
        }
    }
}

/// WGS84 ellipsoid (GPS standard)
pub const WGS84: Geoid = Geoid::new("WGS84", 6_378_137.0, 298.257_223_563);

/// IERS 2010 ellipsoid (used by Skyfield)
pub const IERS2010: Geoid =
    Geoid::new("IERS2010", EARTH_RADIUS, IERS_2010_INVERSE_EARTH_FLATTENING);

/// A geographic position on Earth's surface.
///
/// Holds the geodetic coordinates and precomputed ITRS position vector.
#[derive(Debug, Clone)]
pub struct GeographicPosition {
    /// Geodetic latitude in radians
    pub latitude: f64,
    /// Geodetic longitude in radians
    pub longitude: f64,
    /// Elevation above ellipsoid in meters
    pub elevation_m: f64,
    /// ITRS position in AU
    pub itrs_xyz: Vector3<f64>,
}

impl GeographicPosition {
    /// Compute the GCRS barycentric position of this observer at a given time.
    ///
    /// Rotates the ITRS position into GCRS using the C^T matrix, then adds
    /// Earth's barycentric position to get the observer's SSB position.
    pub fn at(
        &self,
        time: &Time,
        kernel: &mut crate::jplephem::kernel::SpiceKernel,
    ) -> crate::jplephem::errors::Result<Position> {
        // Get Earth's barycentric position
        let earth = kernel.at("earth", time)?;

        // Rotate ITRS → GCRS
        let ct = time.ct_matrix();
        let gcrs_pos = ct * self.itrs_xyz;

        // Velocity from Earth rotation: v = ω × r (in ITRS), then rotate to GCRS
        let angvel_au_day = EARTH_ANGVEL * DAY_S; // rad/day
        let itrs_vel = Vector3::new(
            -angvel_au_day * self.itrs_xyz.y,
            angvel_au_day * self.itrs_xyz.x,
            0.0,
        );
        let gcrs_vel = ct * itrs_vel;

        // Observer barycentric = Earth barycentric + geocentric offset
        Ok(Position::barycentric(
            earth.position + gcrs_pos,
            earth.velocity + gcrs_vel,
            399,
        ))
    }

    /// Compute altitude and azimuth of an apparent position as seen from this location.
    ///
    /// Returns `(altitude_degrees, azimuth_degrees, distance_au)`.
    ///
    /// Altitude: degrees above horizon (negative below).
    /// Azimuth: degrees clockwise from north (0=N, 90=E, 180=S, 270=W).
    pub fn altaz(&self, apparent: &Position, time: &Time) -> (f64, f64, f64) {
        assert_eq!(
            apparent.kind,
            PositionKind::Apparent,
            "altaz() requires an Apparent position"
        );

        // Rotate apparent GCRS position into ITRS
        let c = time.c_matrix();
        let itrs = c * apparent.position;

        // Build rotation from ITRS to local horizon (topocentric)
        let (alt, az) = self.itrs_to_horizon(&itrs);

        (alt * 180.0 / PI, az * 180.0 / PI, apparent.distance())
    }

    /// Rotate an ITRS direction vector into local horizon coordinates.
    ///
    /// Returns (altitude_radians, azimuth_radians).
    pub(crate) fn itrs_to_horizon(&self, itrs_direction: &Vector3<f64>) -> (f64, f64) {
        let slat = self.latitude.sin();
        let clat = self.latitude.cos();
        let slon = self.longitude.sin();
        let clon = self.longitude.cos();

        // Rotation from ITRS to local horizon (south, east, up)
        // R = R_y(90° - lat) × R_z(lon)
        let south = slat * clon * itrs_direction.x + slat * slon * itrs_direction.y
            - clat * itrs_direction.z;
        let east = -slon * itrs_direction.x + clon * itrs_direction.y;
        let up = clat * clon * itrs_direction.x
            + clat * slon * itrs_direction.y
            + slat * itrs_direction.z;

        let r_horiz = (south * south + east * east).sqrt();
        let alt = up.atan2(r_horiz);

        // Azimuth: measured clockwise from north
        // north = -south direction
        let mut az = east.atan2(-south);
        if az < 0.0 {
            az += 2.0 * PI;
        }

        (alt, az)
    }

    /// Local Apparent Sidereal Time in hours.
    pub fn lst_hours(&self, time: &Time) -> f64 {
        let gast = time.gast();
        let lst = gast + self.longitude * 12.0 / PI;
        lst.rem_euclid(24.0)
    }

    /// Apply atmospheric refraction correction to an observed altitude.
    ///
    /// Delegates to [`earthlib::refraction`](crate::earthlib::refraction).
    ///
    /// # Arguments
    /// * `altitude_degrees` — Observed altitude above horizon in degrees
    /// * `temperature_c` — Temperature in Celsius (default 10.0)
    /// * `pressure_mbar` — Atmospheric pressure in millibars (default 1010.0)
    ///
    /// Returns the refraction correction in degrees (always positive).
    pub fn refract(altitude_degrees: f64, temperature_c: f64, pressure_mbar: f64) -> f64 {
        crate::earthlib::refraction(altitude_degrees, temperature_c, pressure_mbar)
    }
}

impl std::fmt::Display for GeographicPosition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lat_d = self.latitude * 180.0 / PI;
        let lon_d = self.longitude * 180.0 / PI;
        let ns = if lat_d >= 0.0 { "N" } else { "S" };
        let ew = if lon_d >= 0.0 { "E" } else { "W" };
        write!(
            f,
            "{:.4}° {}, {:.4}° {}, {:.1} m",
            lat_d.abs(),
            ns,
            lon_d.abs(),
            ew,
            self.elevation_m
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::kernel::SpiceKernel;
    use approx::assert_relative_eq;

    #[test]
    fn test_wgs84_constants() {
        assert_relative_eq!(WGS84.radius, 6_378_137.0);
        assert_relative_eq!(WGS84.inverse_flattening, 298.257_223_563);
    }

    #[test]
    fn test_iers2010_constants() {
        assert_relative_eq!(IERS2010.radius, EARTH_RADIUS);
        assert_relative_eq!(
            IERS2010.inverse_flattening,
            IERS_2010_INVERSE_EARTH_FLATTENING
        );
    }

    #[test]
    fn test_latlon_equator_prime_meridian() {
        let pos = WGS84.latlon(0.0, 0.0, 0.0);
        // At equator, prime meridian: x = radius, y = 0, z = 0
        let expected_x = WGS84.radius / AU_M;
        assert_relative_eq!(pos.itrs_xyz.x, expected_x, epsilon = 1e-10);
        assert_relative_eq!(pos.itrs_xyz.y, 0.0, epsilon = 1e-15);
        assert_relative_eq!(pos.itrs_xyz.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_latlon_equator_90e() {
        let pos = WGS84.latlon(0.0, 90.0, 0.0);
        let expected_y = WGS84.radius / AU_M;
        assert_relative_eq!(pos.itrs_xyz.x, 0.0, epsilon = 1e-10);
        assert_relative_eq!(pos.itrs_xyz.y, expected_y, epsilon = 1e-10);
        assert_relative_eq!(pos.itrs_xyz.z, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_latlon_north_pole() {
        let pos = WGS84.latlon(90.0, 0.0, 0.0);
        // At pole, x ≈ 0, y = 0, z = polar radius
        assert_relative_eq!(pos.itrs_xyz.x, 0.0, epsilon = 1e-15);
        assert_relative_eq!(pos.itrs_xyz.y, 0.0, epsilon = 1e-15);
        // Polar radius should be less than equatorial radius
        let polar_radius_au = pos.itrs_xyz.z;
        let equatorial_radius_au = WGS84.radius / AU_M;
        assert!(
            polar_radius_au < equatorial_radius_au,
            "Polar radius {} should be less than equatorial radius {}",
            polar_radius_au,
            equatorial_radius_au
        );
        // Polar radius ≈ equatorial * (1 - 1/298.25) ≈ 6356752 m
        let expected_polar_m = WGS84.radius * (1.0 - 1.0 / WGS84.inverse_flattening);
        assert_relative_eq!(
            polar_radius_au * AU_M,
            expected_polar_m,
            epsilon = 100.0 // within 100m (we approximate)
        );
    }

    #[test]
    fn test_latlon_with_elevation() {
        let pos_ground = WGS84.latlon(0.0, 0.0, 0.0);
        let pos_high = WGS84.latlon(0.0, 0.0, 1000.0);
        let diff_m = (pos_high.itrs_xyz.x - pos_ground.itrs_xyz.x) * AU_M;
        assert_relative_eq!(diff_m, 1000.0, epsilon = 0.01);
    }

    #[test]
    fn test_latlon_symmetry() {
        let north = WGS84.latlon(45.0, 0.0, 0.0);
        let south = WGS84.latlon(-45.0, 0.0, 0.0);
        // Same x, same y, opposite z
        assert_relative_eq!(north.itrs_xyz.x, south.itrs_xyz.x, epsilon = 1e-15);
        assert_relative_eq!(north.itrs_xyz.z, -south.itrs_xyz.z, epsilon = 1e-15);
    }

    #[test]
    fn test_display() {
        let pos = WGS84.latlon(42.3583, -71.0603, 43.0);
        let s = format!("{}", pos);
        assert!(s.contains("N"));
        assert!(s.contains("W"));
    }

    #[test]
    fn test_refraction_at_horizon() {
        // At the horizon, refraction is about 34 arcminutes ≈ 0.57°
        let r = GeographicPosition::refract(0.0, 10.0, 1010.0);
        assert!(
            r > 0.4 && r < 0.7,
            "Horizon refraction should be ~0.57°, got {}",
            r
        );
    }

    #[test]
    fn test_refraction_at_zenith() {
        // At the zenith, refraction is nearly zero
        let r = GeographicPosition::refract(90.0, 10.0, 1010.0);
        assert!(r < 0.01, "Zenith refraction should be ~0, got {}", r);
    }

    #[test]
    fn test_refraction_below_horizon() {
        // Below -1°, no refraction
        let r = GeographicPosition::refract(-5.0, 10.0, 1010.0);
        assert_relative_eq!(r, 0.0);
    }

    #[test]
    fn test_horizon_rotation_up() {
        // A direction straight up at equator/prime-meridian is ITRS (1, 0, 0)
        let pos = WGS84.latlon(0.0, 0.0, 0.0);
        let up_itrs = Vector3::new(1.0, 0.0, 0.0);
        let (alt, _az) = pos.itrs_to_horizon(&up_itrs);
        assert_relative_eq!(alt, PI / 2.0, epsilon = 0.01);
    }

    #[test]
    fn test_horizon_rotation_north_at_equator() {
        // At equator, north is ITRS (0, 0, 1)
        let pos = WGS84.latlon(0.0, 0.0, 0.0);
        let north_itrs = Vector3::new(0.0, 0.0, 1.0);
        let (alt, az) = pos.itrs_to_horizon(&north_itrs);
        assert_relative_eq!(alt, 0.0, epsilon = 0.01);
        assert_relative_eq!(az, 0.0, epsilon = 0.01);
    }

    // --- Integration tests using DE421 ---

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    fn j2000_time() -> Time {
        crate::time::Timescale::default().tdb_jd(2451545.0)
    }

    #[test]
    fn test_observer_at_returns_barycentric() {
        let mut kernel = de421_kernel();
        let t = j2000_time();
        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();
        assert_eq!(observer.kind, PositionKind::Barycentric);
    }

    #[test]
    fn test_observer_near_earth() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let earth = kernel.at("earth", &t).unwrap();
        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();

        // Observer should be within ~1 Earth radius of Earth center
        let offset_au = (observer.position - earth.position).norm();
        let offset_m = offset_au * AU_M;
        assert!(
            offset_m > 6_000_000.0 && offset_m < 6_500_000.0,
            "Observer offset from Earth center should be ~6371 km, got {} m",
            offset_m
        );
    }

    #[test]
    fn test_observer_full_pipeline() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();

        // Observe Mars from Boston
        let mars_astro = observer.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();

        let (ra, dec, dist) = mars_app.radec(None);
        assert!(ra >= 0.0 && ra < 24.0, "RA out of range: {}", ra);
        assert!(dec >= -90.0 && dec <= 90.0, "Dec out of range: {}", dec);
        assert!(dist > 0.0, "Distance should be positive");
    }

    #[test]
    fn test_altaz_produces_valid_coordinates() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();
        let mars_astro = observer.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();

        let (alt, az, dist) = boston.altaz(&mars_app, &t);
        assert!(
            alt >= -90.0 && alt <= 90.0,
            "Altitude should be in [-90, 90], got {}",
            alt
        );
        assert!(
            az >= 0.0 && az < 360.0,
            "Azimuth should be in [0, 360), got {}",
            az
        );
        assert!(dist > 0.0, "Distance should be positive");
    }

    #[test]
    fn test_lst_in_valid_range() {
        let t = j2000_time();
        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let lst = boston.lst_hours(&t);
        assert!(
            lst >= 0.0 && lst < 24.0,
            "LST should be in [0, 24), got {}",
            lst
        );
    }

    #[test]
    fn test_lst_east_of_greenwich_is_ahead() {
        let t = j2000_time();
        let greenwich = WGS84.latlon(51.4769, 0.0, 0.0);
        let tokyo = WGS84.latlon(35.6762, 139.6503, 0.0);
        let lst_g = greenwich.lst_hours(&t);
        let lst_t = tokyo.lst_hours(&t);

        // Tokyo is ~9.3h east of Greenwich
        let diff = (lst_t - lst_g + 24.0) % 24.0;
        assert!(
            diff > 8.0 && diff < 11.0,
            "Tokyo LST should be ~9.3h ahead of Greenwich, got {}h",
            diff
        );
    }

    #[test]
    fn test_antipodal_observers_differ() {
        let mut kernel = de421_kernel();
        let t = j2000_time();

        let pos1 = WGS84.latlon(0.0, 0.0, 0.0);
        let pos2 = WGS84.latlon(0.0, 180.0, 0.0);
        let obs1 = pos1.at(&t, &mut kernel).unwrap();
        let obs2 = pos2.at(&t, &mut kernel).unwrap();

        // They should differ by about 2 Earth radii in GCRS
        let diff_au = (obs1.position - obs2.position).norm();
        let diff_m = diff_au * AU_M;
        assert!(
            diff_m > 12_000_000.0 && diff_m < 13_000_000.0,
            "Antipodal observers should differ by ~12742 km, got {} m",
            diff_m
        );
    }
}

#[cfg(feature = "python-tests")]
mod python_tests;
