//! SGP4 satellite tracking
//!
//! This module provides TLE (Two-Line Element) parsing and SGP4 propagation
//! for Earth satellites, matching Python Skyfield's `sgp4lib.py`.
//!
//! # Example
//!
//! ```ignore
//! use starfield::sgp4lib::EarthSatellite;
//! use starfield::time::Timescale;
//!
//! let ts = Timescale::default();
//!
//! // ISS TLE from Celestrak
//! let line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9991";
//! let line2 = "2 25544  51.6400 200.0000 0006000  50.0000 310.0000 15.50000000000010";
//!
//! let iss = EarthSatellite::from_tle(line1, line2, Some("ISS"), &ts)?;
//! let t = ts.utc((2024, 1, 1, 12, 0, 0.0));
//! let pos = iss.at(&t)?;
//!
//! println!("ISS position: {:?} AU", pos.position);
//! ```

pub mod teme;

#[cfg(feature = "python-tests")]
mod python_tests;

use chrono::{Datelike, Timelike};
use nalgebra::Vector3;
use sgp4::{Constants, Elements, MinutesSinceEpoch, Prediction};

use crate::constants::{AU_KM, DAY_S};
use crate::positions::Position;
use crate::time::{Time, Timescale};
use crate::StarfieldError;

use teme::transform_teme_to_gcrs;

/// An Earth satellite loaded from a TLE and propagated with SGP4.
///
/// The satellite position is computed in the TEME (True Equator Mean Equinox)
/// reference frame by SGP4, then transformed to GCRS for integration with
/// other starfield positions.
#[derive(Debug, Clone)]
pub struct EarthSatellite {
    /// Satellite name (optional, from line 0 of 3LE or provided by user)
    pub name: Option<String>,

    /// The TLE epoch as a Time object
    pub epoch: Time,

    /// NORAD catalog ID
    pub norad_id: u64,

    /// Mean motion in revolutions per day (from TLE)
    pub revs_per_day: f64,

    /// The SGP4 propagator constants
    model: Constants,

    /// Original TLE elements (for reference)
    elements: Elements,
}

impl EarthSatellite {
    /// Create a satellite from TLE lines.
    ///
    /// # Arguments
    /// * `line1` - First line of TLE (69 characters)
    /// * `line2` - Second line of TLE (69 characters)
    /// * `name` - Optional satellite name
    /// * `ts` - Timescale for epoch conversion
    ///
    /// # Returns
    /// An `EarthSatellite` ready for propagation, or an error if the TLE is invalid.
    pub fn from_tle(
        line1: &str,
        line2: &str,
        name: Option<&str>,
        ts: &Timescale,
    ) -> Result<Self, StarfieldError> {
        // Combine lines for parsing (sgp4 crate expects newline-separated)
        let tle_string = format!("{}\n{}", line1.trim(), line2.trim());
        let elements_list = sgp4::parse_2les(&tle_string)
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse TLE: {:?}", e)))?;

        if elements_list.is_empty() {
            return Err(StarfieldError::DataError("No TLE elements found".into()));
        }

        let elements = elements_list.into_iter().next().unwrap();
        Self::from_elements(elements, name.map(String::from), ts)
    }

    /// Create a satellite from parsed SGP4 Elements.
    pub fn from_elements(
        elements: Elements,
        name: Option<String>,
        ts: &Timescale,
    ) -> Result<Self, StarfieldError> {
        // Build SGP4 propagator constants
        let model = Constants::from_elements(&elements).map_err(|e| {
            StarfieldError::CalculationError(format!("SGP4 initialization failed: {:?}", e))
        })?;

        // Convert epoch from chrono::NaiveDateTime to Time
        let epoch = Self::datetime_to_time(&elements.datetime, ts);

        // Mean motion in rev/day (TLE field is already in rev/day)
        let revs_per_day = elements.mean_motion;

        Ok(EarthSatellite {
            name,
            epoch,
            norad_id: elements.norad_id,
            revs_per_day,
            model,
            elements,
        })
    }

    /// Convert chrono NaiveDateTime to Time
    fn datetime_to_time(dt: &chrono::NaiveDateTime, ts: &Timescale) -> Time {
        let year = dt.year();
        let month = dt.month();
        let day = dt.day();
        let hour = dt.hour();
        let minute = dt.minute();
        let second = dt.second() as f64 + dt.nanosecond() as f64 / 1e9;

        ts.utc((year, month, day, hour, minute, second))
    }

    /// Compute the satellite's GCRS position at time `t`.
    ///
    /// Returns a `Position` with:
    /// - `kind = Barycentric` (Earth-centered, not SSB-centered)
    /// - `center = 399` (Earth)
    /// - `target = -(100000 + norad_id)` (negative to avoid NAIF ID conflicts)
    pub fn at(&self, t: &Time) -> Result<Position, StarfieldError> {
        // Compute minutes since TLE epoch
        let epoch_jd = self.epoch_jd();
        let t_jd = t.tt(); // Use TT for consistency with Skyfield
        let minutes_since_epoch = (t_jd - epoch_jd) * 1440.0; // days to minutes

        // Propagate with SGP4
        let prediction = self
            .model
            .propagate(MinutesSinceEpoch(minutes_since_epoch))
            .map_err(|e| {
                StarfieldError::CalculationError(format!("SGP4 propagation failed: {:?}", e))
            })?;

        // Convert from TEME (km, km/s) to GCRS (AU, AU/day)
        let (pos_gcrs, vel_gcrs) = self.prediction_to_gcrs(t, &prediction);

        // Create Position (Earth-centered)
        let target_id = self.target_id();
        Ok(Position::geocentric(pos_gcrs, vel_gcrs, target_id))
    }

    /// Get the TLE epoch as a Julian Date (TT scale)
    pub fn epoch_jd(&self) -> f64 {
        self.epoch.tt()
    }

    /// Get the target ID for this satellite
    ///
    /// Uses negative IDs to avoid conflicts with NAIF IDs:
    /// target = -(100000 + norad_id)
    pub fn target_id(&self) -> i32 {
        -((100000 + self.norad_id) as i32)
    }

    /// Get the raw TEME position and velocity in km and km/s
    pub fn position_and_velocity_teme_km(
        &self,
        t: &Time,
    ) -> Result<(Vector3<f64>, Vector3<f64>), StarfieldError> {
        let epoch_jd = self.epoch_jd();
        let t_jd = t.tt();
        let minutes_since_epoch = (t_jd - epoch_jd) * 1440.0;

        let prediction = self
            .model
            .propagate(MinutesSinceEpoch(minutes_since_epoch))
            .map_err(|e| {
                StarfieldError::CalculationError(format!("SGP4 propagation failed: {:?}", e))
            })?;

        let pos = Vector3::new(
            prediction.position[0],
            prediction.position[1],
            prediction.position[2],
        );
        let vel = Vector3::new(
            prediction.velocity[0],
            prediction.velocity[1],
            prediction.velocity[2],
        );

        Ok((pos, vel))
    }

    /// Convert SGP4 prediction (TEME km, km/s) to GCRS (AU, AU/day)
    fn prediction_to_gcrs(&self, t: &Time, pred: &Prediction) -> (Vector3<f64>, Vector3<f64>) {
        // Extract position and velocity from prediction
        let pos_teme = Vector3::new(pred.position[0], pred.position[1], pred.position[2]);
        let vel_teme = Vector3::new(pred.velocity[0], pred.velocity[1], pred.velocity[2]);

        // Transform TEME to GCRS (still in km, km/s)
        let (pos_gcrs_km, vel_gcrs_kms) = transform_teme_to_gcrs(t, &pos_teme, &vel_teme);

        // Convert to AU and AU/day
        let pos_gcrs_au = pos_gcrs_km / AU_KM;
        let vel_gcrs_au_day = vel_gcrs_kms * DAY_S / AU_KM;

        (pos_gcrs_au, vel_gcrs_au_day)
    }

    /// Get the original TLE elements
    pub fn elements(&self) -> &Elements {
        &self.elements
    }
}

/// Parse multiple satellites from a multi-line TLE string.
///
/// Supports both 2-line and 3-line element formats.
///
/// # Arguments
/// * `tle_data` - String containing one or more TLEs
/// * `ts` - Timescale for epoch conversion
///
/// # Returns
/// A vector of `EarthSatellite` objects.
pub fn parse_tle_file(
    tle_data: &str,
    ts: &Timescale,
) -> Result<Vec<EarthSatellite>, StarfieldError> {
    // Try 3LE format first (with names), fall back to 2LE
    let elements_result = sgp4::parse_3les(tle_data);
    let elements_list = match elements_result {
        Ok(list) if !list.is_empty() => list,
        _ => sgp4::parse_2les(tle_data)
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse TLE: {:?}", e)))?,
    };

    let mut satellites = Vec::with_capacity(elements_list.len());
    for elements in elements_list {
        let name = elements.object_name.clone();
        let sat = EarthSatellite::from_elements(elements, name, ts)?;
        satellites.push(sat);
    }

    Ok(satellites)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // ISS TLE for testing (from AIAA 2006-6753 Appendix C test case - slightly modified)
    // Using a simplified TLE with correct checksums
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    #[test]
    fn test_parse_tle() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");

        assert_eq!(sat.name, Some("ISS".to_string()));
        assert_eq!(sat.norad_id, 25544);
        // ISS orbits ~15.7 times per day (92-minute orbital period)
        assert_relative_eq!(sat.revs_per_day, 15.72, epsilon = 0.1);
    }

    #[test]
    fn test_target_id() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        // Target ID should be -(100000 + 25544) = -125544
        assert_eq!(sat.target_id(), -125544);
    }

    #[test]
    fn test_propagation_returns_position() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        // Propagate to some time near the epoch
        let t = ts.tt_jd(sat.epoch_jd() + 1.0, None); // 1 day after epoch
        let pos = sat.at(&t).expect("Propagation failed");

        // Position should be in reasonable range for LEO satellite
        // ISS is ~400 km altitude, so distance from Earth center ~6800 km
        // In AU: 6800 / 149597870.7 ≈ 4.5e-5 AU
        let dist_au = pos.position.norm();
        assert!(dist_au > 1e-5, "Distance too small: {} AU", dist_au);
        assert!(dist_au < 1e-4, "Distance too large: {} AU", dist_au);

        // Verify it's Earth-centered
        assert_eq!(pos.center, 399);
    }

    #[test]
    fn test_teme_position_reasonable() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        let t = ts.tt_jd(sat.epoch_jd(), None);
        let (pos_km, vel_kms) = sat
            .position_and_velocity_teme_km(&t)
            .expect("Failed to get TEME position");

        // Position magnitude should be ~6800 km (ISS altitude + Earth radius)
        let r = pos_km.norm();
        assert!(
            r > 6000.0 && r < 7500.0,
            "Position magnitude {} km out of range",
            r
        );

        // Velocity should be ~7.5 km/s for LEO
        let v = vel_kms.norm();
        assert!(
            v > 6.0 && v < 9.0,
            "Velocity magnitude {} km/s out of range",
            v
        );
    }
}
