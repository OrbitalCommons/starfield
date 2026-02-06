//! Visual magnitude calculations for planets
//!
//! Implements the Mallama & Hilton (2018) formulas for computing apparent
//! visual magnitudes of planets, matching Skyfield's `magnitudelib`.
//!
//! # Example
//!
//! ```ignore
//! use starfield::magnitudelib::planetary_magnitude;
//!
//! let earth = kernel.at("earth", &t)?;
//! let jupiter = earth.observe("jupiter barycenter", &mut kernel, &t)?;
//! let mag = planetary_magnitude(&jupiter)?;
//! ```

#[cfg(all(test, feature = "python-tests"))]
mod python_tests;

use nalgebra::Vector3;

use crate::constants::RAD2DEG;
use crate::positions::Position;
use crate::time::Time;

/// Saturn's north pole direction in ICRF (J2000 equatorial)
const SATURN_POLE: [f64; 3] = [0.08547883, 0.07323576, 0.99364475];
/// Uranus's north pole direction in ICRF (J2000 equatorial)
const URANUS_POLE: [f64; 3] = [-0.21199958, -0.94155916, -0.26176809];

/// Compute the apparent visual magnitude of a planet.
///
/// The `position` should be an astrometric (or apparent) position returned by
/// `earth.observe("planet", ...)`. The observer's barycentric position must be
/// available (set automatically by `observe()`).
///
/// The `time` parameter is needed for Neptune's year-dependent baseline
/// magnitude. For other planets it is ignored but should still be provided.
///
/// Returns `f64::NAN` for cases where the formula is undefined (e.g. Saturn
/// at large phase angles, Neptune before 2000 at large phase angles).
///
/// # Errors
///
/// Returns an error if the target body is not a supported planet (Mercury
/// through Neptune), or if the position lacks observer barycentric data.
pub fn planetary_magnitude(position: &Position, time: &Time) -> Result<f64, MagnitudeError> {
    let observer_bary = position
        .observer_barycentric
        .as_ref()
        .ok_or(MagnitudeError::MissingObserver)?;

    let observer_to_planet = position.position;

    // Treat the Sun as sitting at the Solar System Barycenter
    let sun_to_observer = observer_bary.position;
    let sun_to_planet = sun_to_observer + observer_to_planet;

    let r = sun_to_planet.norm();
    let delta = observer_to_planet.norm();
    let ph_ang = angle_between(&sun_to_planet, &observer_to_planet) * RAD2DEG;

    let target = position.target;

    match target {
        199 | 1 => Ok(mercury_magnitude(r, delta, ph_ang)),
        299 | 2 => Ok(venus_magnitude(r, delta, ph_ang)),
        399 => Ok(earth_magnitude(r, delta, ph_ang)),
        499 | 4 => Ok(mars_magnitude(r, delta, ph_ang)),
        599 | 5 => Ok(jupiter_magnitude(r, delta, ph_ang)),
        699 | 6 => {
            let pole = Vector3::from_column_slice(&SATURN_POLE);
            let sun_sub_lat = angle_between(&pole, &sun_to_planet) * RAD2DEG - 90.0;
            let obs_sub_lat = angle_between(&pole, &observer_to_planet) * RAD2DEG - 90.0;
            Ok(saturn_magnitude(r, delta, ph_ang, sun_sub_lat, obs_sub_lat))
        }
        799 | 7 => {
            let pole = Vector3::from_column_slice(&URANUS_POLE);
            let sun_sub_lat = angle_between(&pole, &sun_to_planet) * RAD2DEG - 90.0;
            let obs_sub_lat = angle_between(&pole, &observer_to_planet) * RAD2DEG - 90.0;
            Ok(uranus_magnitude(r, delta, ph_ang, sun_sub_lat, obs_sub_lat))
        }
        899 | 8 => {
            let year = time.j();
            Ok(neptune_magnitude(r, delta, ph_ang, year))
        }
        _ => Err(MagnitudeError::UnsupportedBody(target)),
    }
}

/// Error type for magnitude calculations
#[derive(Debug, thiserror::Error)]
pub enum MagnitudeError {
    #[error("Cannot compute magnitude: position has no observer barycentric data")]
    MissingObserver,
    #[error("Cannot compute magnitude for NAIF target {0}")]
    UnsupportedBody(i32),
}

/// Angle between two vectors in radians
fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let dot = a.normalize().dot(&b.normalize()).clamp(-1.0, 1.0);
    dot.acos()
}

/// Mercury magnitude (Mallama & Hilton 2018, Eq. 1)
fn mercury_magnitude(r: f64, delta: f64, ph_ang: f64) -> f64 {
    let distance_mag_factor = 5.0 * (r * delta).log10();
    let ph_ang_factor = 6.3280e-02 * ph_ang - 1.6336e-03 * ph_ang.powi(2)
        + 3.3644e-05 * ph_ang.powi(3)
        - 3.4265e-07 * ph_ang.powi(4)
        + 1.6893e-09 * ph_ang.powi(5)
        - 3.0334e-12 * ph_ang.powi(6);
    -0.613 + distance_mag_factor + ph_ang_factor
}

/// Venus magnitude (Mallama & Hilton 2018, Eqs. 2-3)
fn venus_magnitude(r: f64, delta: f64, ph_ang: f64) -> f64 {
    let distance_mag_factor = 5.0 * (r * delta).log10();
    let (a0, a1, a2, a3, a4) = if ph_ang < 163.7 {
        (0.0, -1.044e-03, 3.687e-04, -2.814e-06, 8.938e-09)
    } else {
        (236.05828 + 4.384, -2.81914e+00, 8.39034e-03, 0.0, 0.0)
    };
    // Horner's method
    let ph_ang_factor = a0 + ph_ang * (a1 + ph_ang * (a2 + ph_ang * (a3 + ph_ang * a4)));
    -4.384 + distance_mag_factor + ph_ang_factor
}

/// Earth magnitude (Mallama & Hilton 2018, Eq. 4)
fn earth_magnitude(r: f64, delta: f64, ph_ang: f64) -> f64 {
    let distance_mag_factor = 5.0 * (r * delta).log10();
    let ph_ang_factor = -1.060e-03 * ph_ang + 2.054e-04 * ph_ang.powi(2);
    -3.99 + distance_mag_factor + ph_ang_factor
}

/// Mars magnitude (Mallama & Hilton 2018, Eqs. 5-6)
fn mars_magnitude(r: f64, delta: f64, ph_ang: f64) -> f64 {
    let r_mag_factor = 2.5 * (r * r).log10();
    let delta_mag_factor = 2.5 * (delta * delta).log10();
    let distance_mag_factor = r_mag_factor + delta_mag_factor;

    let geocentric_phase_angle_limit = 50.0;

    let (a, b, v0) = if ph_ang <= geocentric_phase_angle_limit {
        (2.267e-02, -1.302e-04, -1.601)
    } else {
        (-0.02573, 0.0003445, -0.367)
    };
    let ph_ang_factor = a * ph_ang + b * ph_ang.powi(2);

    // Mars surface corrections not implemented (±0.06 mag error)
    v0 + distance_mag_factor + ph_ang_factor
}

/// Jupiter magnitude (Mallama & Hilton 2018, Eqs. 7-8)
fn jupiter_magnitude(r: f64, delta: f64, ph_ang: f64) -> f64 {
    let distance_mag_factor = 5.0 * (r * delta).log10();
    let geocentric_phase_angle_limit = 12.0;

    if ph_ang <= geocentric_phase_angle_limit {
        let ph_ang_factor = (6.16e-04 * ph_ang - 3.7e-04) * ph_ang;
        -9.395 + distance_mag_factor + ph_ang_factor
    } else {
        let ph_ang_pi = ph_ang / 180.0;
        let inner = ((((-1.876 * ph_ang_pi + 2.809) * ph_ang_pi - 0.062) * ph_ang_pi - 0.363)
            * ph_ang_pi
            - 1.507)
            * ph_ang_pi
            + 1.0;
        let ph_ang_factor = -2.5 * inner.log10();
        -9.428 + distance_mag_factor + ph_ang_factor
    }
}

/// Saturn magnitude with ring tilt (Mallama & Hilton 2018, Eqs. 9-12)
fn saturn_magnitude(r: f64, delta: f64, ph_ang: f64, sun_sub_lat: f64, earth_sub_lat: f64) -> f64 {
    let r_mag_factor = 2.5 * (r * r).log10();
    let delta_mag_factor = 2.5 * (delta * delta).log10();
    let distance_mag_factor = r_mag_factor + delta_mag_factor;

    let product = sun_sub_lat * earth_sub_lat;
    let sub_lat_geoc = if product >= 0.0 { product.sqrt() } else { 0.0 };

    let geocentric_phase_angle_limit = 6.5;
    let geocentric_inclination_limit = 27.0;

    if ph_ang <= geocentric_phase_angle_limit && sub_lat_geoc <= geocentric_inclination_limit {
        // Equation #10: globe+rings, geocentric circumstances
        -8.914 - 1.825 * (sub_lat_geoc / RAD2DEG).sin() + 0.026 * ph_ang
            - 0.378 * (sub_lat_geoc / RAD2DEG).sin() * (-2.25 * ph_ang).exp()
            + distance_mag_factor
    } else {
        f64::NAN
    }
}

/// Uranus magnitude (Mallama & Hilton 2018, Eqs. 13-14)
fn uranus_magnitude(r: f64, delta: f64, ph_ang: f64, sun_sub_lat: f64, earth_sub_lat: f64) -> f64 {
    let distance_mag_factor = 5.0 * (r * delta).log10();
    let sub_lat = (sun_sub_lat.abs() + earth_sub_lat.abs()) / 2.0;
    let sub_lat_factor = -0.00084 * sub_lat;
    let geocentric_phase_angle_limit = 3.1;

    let mut ap_mag = -7.110 + distance_mag_factor + sub_lat_factor;
    if ph_ang > geocentric_phase_angle_limit {
        ap_mag += (1.045e-4 * ph_ang + 6.587e-3) * ph_ang;
    }
    ap_mag
}

/// Neptune magnitude (Mallama & Hilton 2018, Eqs. 15-17)
fn neptune_magnitude(r: f64, delta: f64, ph_ang: f64, year: f64) -> f64 {
    let r_mag_factor = 2.5 * (r * r).log10();
    let delta_mag_factor = 2.5 * (delta * delta).log10();
    let distance_mag_factor = r_mag_factor + delta_mag_factor;

    let mut ap_mag = (-6.89 - 0.0054 * (year - 1980.0)).clamp(-7.00, -6.89);
    ap_mag += distance_mag_factor;

    let geocentric_phase_angle_limit = 1.9;
    if ph_ang > geocentric_phase_angle_limit {
        if year >= 2000.0 {
            ap_mag += 7.944e-3 * ph_ang + 9.617e-5 * ph_ang.powi(2);
        } else {
            return f64::NAN;
        }
    }

    ap_mag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::SpiceKernel;
    use crate::time::Timescale;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").unwrap()
    }

    #[test]
    fn test_mercury_magnitude_formula() {
        // At r=0.387, delta=1.0, ph_ang=90 degrees
        let mag = mercury_magnitude(0.387, 1.0, 90.0);
        assert!(mag.is_finite());
        assert!(mag > -3.0 && mag < 5.0, "Mercury mag={mag}");
    }

    #[test]
    fn test_venus_magnitude_formula() {
        let mag = venus_magnitude(0.723, 1.0, 60.0);
        assert!(mag.is_finite());
        assert!(mag > -5.0 && mag < 0.0, "Venus mag={mag}");
    }

    #[test]
    fn test_venus_magnitude_large_phase() {
        let mag = venus_magnitude(0.723, 1.0, 170.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_earth_magnitude_formula() {
        let mag = earth_magnitude(1.0, 1.0, 90.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_mars_magnitude_formula() {
        let mag = mars_magnitude(1.524, 1.0, 30.0);
        assert!(mag.is_finite());
        assert!(mag > -3.0 && mag < 3.0, "Mars mag={mag}");
    }

    #[test]
    fn test_jupiter_magnitude_small_phase() {
        let mag = jupiter_magnitude(5.2, 4.5, 5.0);
        assert!(mag.is_finite());
        assert!(mag > -3.0 && mag < 0.0, "Jupiter mag={mag}");
    }

    #[test]
    fn test_jupiter_magnitude_large_phase() {
        let mag = jupiter_magnitude(5.2, 4.5, 20.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_saturn_magnitude_with_rings() {
        let mag = saturn_magnitude(9.5, 9.0, 3.0, 15.0, 15.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_saturn_magnitude_nan_large_phase() {
        let mag = saturn_magnitude(9.5, 9.0, 10.0, 15.0, 15.0);
        assert!(mag.is_nan());
    }

    #[test]
    fn test_uranus_magnitude_formula() {
        let mag = uranus_magnitude(19.2, 18.5, 2.0, 30.0, 30.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_neptune_magnitude_modern() {
        let mag = neptune_magnitude(30.1, 29.5, 1.0, 2020.0);
        assert!(mag.is_finite());
    }

    #[test]
    fn test_neptune_magnitude_pre2000_nan() {
        let mag = neptune_magnitude(30.1, 29.5, 2.0, 1990.0);
        assert!(mag.is_nan());
    }

    #[test]
    fn test_planetary_magnitude_jupiter() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2459062.5); // 2020-Jul-31

        let earth = kernel.at("earth", &t).unwrap();
        let jupiter = earth
            .observe("jupiter barycenter", &mut kernel, &t)
            .unwrap();
        let mag = planetary_magnitude(&jupiter, &t).unwrap();

        // Jupiter should be around -2.7 magnitude at this date
        assert!(
            mag > -3.5 && mag < -2.0,
            "Jupiter magnitude should be ~-2.7, got {mag}"
        );
    }

    #[test]
    fn test_planetary_magnitude_mars() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0); // J2000

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let mag = planetary_magnitude(&mars, &t).unwrap();

        // Mars magnitude varies widely; just check it's reasonable
        assert!(
            mag > -3.0 && mag < 3.0,
            "Mars magnitude should be -3 to 3, got {mag}"
        );
    }

    #[test]
    fn test_planetary_magnitude_unsupported() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let earth = kernel.at("earth", &t).unwrap();
        let moon = earth.observe("moon", &mut kernel, &t).unwrap();
        let result = planetary_magnitude(&moon, &t);
        assert!(result.is_err());
    }

    #[test]
    fn test_angle_between_orthogonal() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        let angle = angle_between(&a, &b);
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_angle_between_parallel() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(2.0, 0.0, 0.0);
        let angle = angle_between(&a, &b);
        assert!(angle.abs() < 1e-10);
    }
}
