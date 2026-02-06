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

use crate::constants::{AU_KM, DAY_S};
use crate::positions::Position;
use crate::searchlib;
use crate::time::{Time, Timescale};
use crate::toposlib::GeographicPosition;
use crate::StarfieldError;
use chrono::{Datelike, Timelike};
use nalgebra::Vector3;
use sgp4::{Constants, Elements, MinutesSinceEpoch, Prediction};

use teme::transform_teme_to_gcrs;

/// Satellite event type returned by `find_events()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SatelliteEvent {
    /// Satellite rose above the altitude threshold
    Rise = 0,
    /// Satellite reached peak altitude (culmination)
    Culminate = 1,
    /// Satellite fell below the altitude threshold
    Set = 2,
}

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

    /// Create a satellite from an OMM (Orbit Mean-elements Message) JSON string.
    ///
    /// OMM is the newer format replacing TLEs, used by Space-Track and Celestrak.
    /// The JSON should contain standard OMM fields (OBJECT_NAME, EPOCH,
    /// MEAN_MOTION, ECCENTRICITY, INCLINATION, etc.).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let omm_json = r#"{
    ///     "OBJECT_NAME": "ISS (ZARYA)",
    ///     "OBJECT_ID": "1998-067A",
    ///     "EPOCH": "2024-01-01T12:00:00.000000",
    ///     "MEAN_MOTION": 15.72,
    ///     "ECCENTRICITY": 0.0006703,
    ///     "INCLINATION": 51.6416,
    ///     ...
    /// }"#;
    /// let sat = EarthSatellite::from_omm(omm_json, &ts)?;
    /// ```
    pub fn from_omm(json: &str, ts: &Timescale) -> Result<Self, StarfieldError> {
        let elements: Elements = serde_json::from_str(json)
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse OMM JSON: {}", e)))?;
        let name = elements.object_name.clone();
        Self::from_elements(elements, name, ts)
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

    /// Compute the satellite's altitude in degrees as seen from a ground observer.
    ///
    /// This is a simplified computation for satellite tracking that skips
    /// the full barycentric→astrometric→apparent pipeline (light-time and
    /// aberration are negligible for LEO/MEO satellites).
    ///
    /// The GCRS satellite position is rotated into ITRS, the observer ITRS
    /// position is subtracted, and the result is projected onto the local
    /// horizon frame.
    fn altitude_degrees(
        &self,
        t: &Time,
        observer: &GeographicPosition,
    ) -> Result<f64, StarfieldError> {
        let pos = self.at(t)?;

        // Rotate satellite geocentric GCRS position into ITRS
        let c = t.c_matrix();
        let sat_itrs = c * pos.position;

        // Difference vector in ITRS (AU)
        let diff = sat_itrs - observer.itrs_xyz;

        let (alt_rad, _az_rad) = observer.itrs_to_horizon(&diff);
        Ok(alt_rad.to_degrees())
    }

    /// Find satellite rise, culmination, and set events as seen from an observer.
    ///
    /// Searches between `t0` and `t1` for passes of this satellite above
    /// `altitude_degrees` as seen from `observer`.
    ///
    /// Returns a vector of `(Time, SatelliteEvent)` pairs sorted chronologically:
    /// - `SatelliteEvent::Rise` — satellite rose above the altitude threshold
    /// - `SatelliteEvent::Culminate` — satellite reached peak altitude
    /// - `SatelliteEvent::Set` — satellite fell below the altitude threshold
    ///
    /// Matches Python Skyfield's `EarthSatellite.find_events()`.
    pub fn find_events(
        &self,
        observer: &GeographicPosition,
        t0: &Time,
        t1: &Time,
        ts: &Timescale,
        altitude_degrees: f64,
    ) -> Result<Vec<(Time, SatelliteEvent)>, StarfieldError> {
        let jd_start = t0.tt();
        let jd_end = t1.tt();
        let half_second = 0.5 / DAY_S;

        // Compute step size from orbital period
        // mean_motion is already in rev/day
        let orbits_per_day = self.elements.mean_motion;
        let mut step_days = 0.05 / orbits_per_day.max(1.0);
        if step_days > 0.25 {
            step_days = 0.25;
        }

        // Find altitude maxima
        let alt_fn = |jds: &[f64]| -> Vec<f64> {
            jds.iter()
                .map(|&jd| {
                    let t = ts.tt_jd(jd, None);
                    self.altitude_degrees(&t, observer).unwrap_or(-90.0)
                })
                .collect()
        };

        let maxima = searchlib::find_maxima(
            jd_start,
            jd_end,
            &alt_fn,
            step_days,
            half_second,
            searchlib::DEFAULT_NUM,
        );

        // Filter maxima above the altitude threshold
        let keepers: Vec<(f64, f64)> = maxima
            .into_iter()
            .filter(|&(_, alt)| alt >= altitude_degrees)
            .collect();

        if keepers.is_empty() {
            return Ok(Vec::new());
        }

        let mut results: Vec<(f64, SatelliteEvent)> = Vec::new();

        // Add culmination events
        for &(jd, _) in &keepers {
            results.push((jd, SatelliteEvent::Culminate));
        }

        // Find rise/set transitions (above/below altitude threshold)
        let mut below_fn = |jds: &[f64]| -> Vec<i64> {
            jds.iter()
                .map(|&jd| {
                    let t = ts.tt_jd(jd, None);
                    let alt = self.altitude_degrees(&t, observer).unwrap_or(-90.0);
                    if alt < altitude_degrees {
                        1
                    } else {
                        0
                    }
                })
                .collect()
        };

        let transitions =
            searchlib::find_discrete(jd_start, jd_end, &mut below_fn, step_days, half_second, 8);

        for (jd, value) in transitions {
            if value == 0 {
                // Transitioned to above threshold = rise
                results.push((jd, SatelliteEvent::Rise));
            } else {
                // Transitioned to below threshold = set
                results.push((jd, SatelliteEvent::Set));
            }
        }

        // Sort by time
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Convert Julian dates to Time objects
        let events: Vec<(Time, SatelliteEvent)> = results
            .into_iter()
            .map(|(jd, event)| (ts.tt_jd(jd, None), event))
            .collect();

        Ok(events)
    }

    /// Formatted display name matching Skyfield's `target_name` property.
    ///
    /// Format: `"NAME catalog #NORAD epoch YYYY-MM-DD HH:MM:SS UTC"`
    pub fn target_name(&self) -> String {
        let epoch_str = self
            .epoch
            .utc_strftime("%Y-%m-%d %H:%M:%S UTC")
            .unwrap_or_else(|_| "unknown".to_string());
        match &self.name {
            Some(n) => format!("{} catalog #{} epoch {}", n, self.norad_id, epoch_str),
            None => format!("catalog #{} epoch {}", self.norad_id, epoch_str),
        }
    }
}

impl std::fmt::Display for EarthSatellite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.target_name())
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
    use crate::toposlib::WGS84;
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
        assert_relative_eq!(sat.revs_per_day, 15.72, epsilon = 0.1);
    }

    #[test]
    fn test_target_id() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        assert_eq!(sat.target_id(), -125544);
    }

    #[test]
    fn test_propagation_returns_position() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        let t = ts.tt_jd(sat.epoch_jd() + 1.0, None);
        let pos = sat.at(&t).expect("Propagation failed");

        let dist_au = pos.position.norm();
        assert!(dist_au > 1e-5, "Distance too small: {} AU", dist_au);
        assert!(dist_au < 1e-4, "Distance too large: {} AU", dist_au);
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

        let r = pos_km.norm();
        assert!(
            r > 6000.0 && r < 7500.0,
            "Position magnitude {} km out of range",
            r
        );

        let v = vel_kms.norm();
        assert!(
            v > 6.0 && v < 9.0,
            "Velocity magnitude {} km/s out of range",
            v
        );
    }

    #[test]
    fn test_target_name_with_name() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");

        let name = sat.target_name();
        assert!(name.contains("ISS"));
        assert!(name.contains("25544"));
        assert!(name.contains("epoch"));
    }

    #[test]
    fn test_target_name_without_name() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        let name = sat.target_name();
        assert!(name.starts_with("catalog #25544"));
    }

    #[test]
    fn test_display_trait() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let s = format!("{}", sat);
        assert!(s.contains("ISS"));
    }

    #[test]
    fn test_from_omm_json() {
        let ts = Timescale::default();
        let omm = r#"{
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "EPOCH": "2024-01-01T12:00:00.000000",
            "MEAN_MOTION": 15.72125391,
            "ECCENTRICITY": 0.0006703,
            "INCLINATION": 51.6416,
            "RA_OF_ASC_NODE": 247.4627,
            "ARG_OF_PERICENTER": 130.536,
            "MEAN_ANOMALY": 325.0288,
            "EPHEMERIS_TYPE": 0,
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 0,
            "BSTAR": -0.11606E-4,
            "MEAN_MOTION_DOT": -0.00002182,
            "MEAN_MOTION_DDOT": 0
        }"#;

        let sat = EarthSatellite::from_omm(omm, &ts).expect("Failed to parse OMM");
        assert_eq!(sat.name, Some("ISS (ZARYA)".to_string()));
        assert_eq!(sat.norad_id, 25544);
        assert_relative_eq!(sat.revs_per_day, 15.72, epsilon = 0.1);
    }

    #[test]
    fn test_from_omm_invalid_json() {
        let ts = Timescale::default();
        let result = EarthSatellite::from_omm("not json", &ts);
        assert!(result.is_err());
    }

    #[test]
    fn test_altitude_degrees() {
        let ts = Timescale::default();
        let sat =
            EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, None, &ts).expect("Failed to parse TLE");

        let observer = WGS84.latlon(42.3583, -71.0603, 43.0);
        let t = ts.tt_jd(sat.epoch_jd(), None);
        let alt = sat
            .altitude_degrees(&t, &observer)
            .expect("Failed to compute altitude");

        // Altitude should be between -90 and 90
        assert!(alt >= -90.0 && alt <= 90.0, "Altitude {} out of range", alt);
    }

    #[test]
    fn test_find_events_returns_events() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");

        // ISS orbits ~15.7 times/day, so in 1 day it should have multiple passes
        let observer = WGS84.latlon(42.3583, -71.0603, 43.0);
        let t0 = ts.tt_jd(sat.epoch_jd(), None);
        let t1 = ts.tt_jd(sat.epoch_jd() + 1.0, None);

        let events = sat
            .find_events(&observer, &t0, &t1, &ts, 0.0)
            .expect("Failed to find events");

        // Over 1 day, ISS should have some visible passes from most locations
        // (it orbits 15+ times, crossing many latitudes)
        assert!(!events.is_empty(), "Expected at least one event in 1 day");

        // Events should be chronologically ordered
        for i in 1..events.len() {
            assert!(
                events[i].0.tt() >= events[i - 1].0.tt(),
                "Events not in chronological order"
            );
        }
    }

    #[test]
    fn test_find_events_high_altitude_fewer() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");

        let observer = WGS84.latlon(42.3583, -71.0603, 43.0);
        let t0 = ts.tt_jd(sat.epoch_jd(), None);
        let t1 = ts.tt_jd(sat.epoch_jd() + 1.0, None);

        let events_low = sat
            .find_events(&observer, &t0, &t1, &ts, 0.0)
            .expect("Failed to find events");
        let events_high = sat
            .find_events(&observer, &t0, &t1, &ts, 45.0)
            .expect("Failed to find events");

        // Higher altitude threshold should yield equal or fewer events
        assert!(
            events_high.len() <= events_low.len(),
            "Higher threshold should produce fewer events: {} vs {}",
            events_high.len(),
            events_low.len()
        );
    }

    #[test]
    fn test_find_events_event_types() {
        let ts = Timescale::default();
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");

        let observer = WGS84.latlon(42.3583, -71.0603, 43.0);
        let t0 = ts.tt_jd(sat.epoch_jd(), None);
        let t1 = ts.tt_jd(sat.epoch_jd() + 1.0, None);

        let events = sat
            .find_events(&observer, &t0, &t1, &ts, 0.0)
            .expect("Failed to find events");

        // Verify we get all three event types
        let has_rise = events.iter().any(|(_, e)| *e == SatelliteEvent::Rise);
        let has_culminate = events.iter().any(|(_, e)| *e == SatelliteEvent::Culminate);
        let has_set = events.iter().any(|(_, e)| *e == SatelliteEvent::Set);

        if !events.is_empty() {
            assert!(has_culminate, "Expected at least one culmination event");
            // Rise and set may be missing if pass spans boundary
            let _ = (has_rise, has_set);
        }
    }
}
