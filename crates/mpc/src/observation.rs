//! Parser for MPC 80-column optical observation records
//!
//! Format: <https://www.minorplanetcenter.net/iau/info/OpticalObs.html>
//!
//! Cols 1-12:  Designation (packed)
//! Col 13:     Discovery asterisk
//! Col 14:     Publishable note
//! Col 15:     Observation type (C=CCD, P=photographic, etc.)
//! Cols 16-32: Date (YYYY MM DD.ddddd)
//! Cols 33-44: RA J2000 (HH MM SS.ddd)
//! Cols 45-56: Dec J2000 (sDD MM SS.dd)
//! Cols 66-71: Magnitude + band
//! Cols 78-80: Observatory code

use starfield_core::{Result, StarfieldError};

/// A single astrometric observation from the MPC
#[derive(Debug, Clone)]
pub struct Observation {
    /// Packed designation (cols 1-12)
    pub designation: String,
    /// Whether this is the discovery observation
    pub is_discovery: bool,
    /// Observation type: 'C' = CCD, 'P' = photographic, etc.
    pub obs_type: Option<char>,
    /// Observation date as (year, month, fractional day)
    pub date: (i32, u32, f64),
    /// Right ascension in degrees (J2000)
    pub ra_deg: f64,
    /// Declination in degrees (J2000)
    pub dec_deg: f64,
    /// Observed magnitude
    pub magnitude: Option<f64>,
    /// Photometric band (V, R, B, etc.)
    pub band: Option<char>,
    /// 3-character observatory code
    pub observatory_code: String,
}

/// Parse RA from "HH MM SS.ddd" format to degrees
fn parse_ra(s: &str) -> Option<f64> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let h: f64 = parts[0].parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts[2].parse().ok()?;
    Some((h + m / 60.0 + sec / 3600.0) * 15.0)
}

/// Parse Dec from "sDD MM SS.dd" format to degrees
fn parse_dec(s: &str) -> Option<f64> {
    let s = s.trim();
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let deg_str = parts[0];
    let sign: f64 = if deg_str.starts_with('-') { -1.0 } else { 1.0 };
    let d: f64 = deg_str.trim_start_matches(['+', '-']).parse().ok()?;
    let m: f64 = parts[1].parse().ok()?;
    let sec: f64 = parts[2].parse().ok()?;
    Some(sign * (d + m / 60.0 + sec / 3600.0))
}

/// Parse a date string "YYYY MM DD.ddddd" to (year, month, fractional_day)
fn parse_date(s: &str) -> Option<(i32, u32, f64)> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }
    let year: i32 = parts[0].parse().ok()?;
    let month: u32 = parts[1].parse().ok()?;
    let day: f64 = parts[2].parse().ok()?;
    Some((year, month, day))
}

/// Parse a single 80-column observation record.
///
/// Returns `None` for lines that don't match the expected format.
pub fn parse_observation_line(line: &str) -> Option<Observation> {
    if line.len() < 80 {
        return None;
    }

    let designation = line[..12].trim().to_string();
    if designation.is_empty() {
        return None;
    }

    let is_discovery = line.as_bytes().get(12) == Some(&b'*');
    let obs_type = line.as_bytes().get(14).and_then(|&b| {
        let c = b as char;
        if c.is_ascii_alphabetic() {
            Some(c)
        } else {
            None
        }
    });

    let date = parse_date(line[15..32].trim())?;
    let ra_deg = parse_ra(line[32..44].trim())?;
    let dec_deg = parse_dec(line[44..56].trim())?;

    let mag_str = if line.len() >= 71 {
        line[65..70].trim()
    } else {
        ""
    };
    let magnitude = mag_str.parse::<f64>().ok();

    let band = if line.len() >= 72 {
        let b = line.as_bytes()[70];
        if b.is_ascii_alphabetic() {
            Some(b as char)
        } else {
            None
        }
    } else {
        None
    };

    let observatory_code = if line.len() >= 80 {
        line[77..80].trim().to_string()
    } else {
        String::new()
    };

    Some(Observation {
        designation,
        is_discovery,
        obs_type,
        date,
        ra_deg,
        dec_deg,
        magnitude,
        band,
        observatory_code,
    })
}

/// Parse multiple 80-column observation records.
pub fn parse_observations(text: &str) -> Result<Vec<Observation>> {
    let obs: Vec<Observation> = text.lines().filter_map(parse_observation_line).collect();

    if obs.is_empty() {
        return Err(StarfieldError::DataError(
            "No observation records found".to_string(),
        ));
    }

    Ok(obs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic 80-column observation records following MPC format.
    // Each line is exactly 80 characters with fields at fixed positions.
    //                         1111111111222222222233333333334444444444555555555566666666667777777778
    //                1234567890123456789012345678901234567890123456789012345678901234567890123456789 0
    const CERES_OBS_1: &str =
        "00001         C2024 01 15.12345 06 23 45.123-08 12 34.56          18.5V      568";
    const CERES_OBS_2: &str =
        "00001       * C2024 01 15.12345 06 23 45.123-08 12 34.56          18.5R      568";

    #[test]
    fn test_parse_ra() {
        let ra = parse_ra("06 23 45.123").unwrap();
        // 6h 23m 45.123s = (6 + 23/60 + 45.123/3600) * 15
        let expected = (6.0 + 23.0 / 60.0 + 45.123 / 3600.0) * 15.0;
        assert!((ra - expected).abs() < 1e-6);
    }

    #[test]
    fn test_parse_dec_negative() {
        let dec = parse_dec("-08 12 34.56").unwrap();
        let expected = -(8.0 + 12.0 / 60.0 + 34.56 / 3600.0);
        assert!((dec - expected).abs() < 1e-6);
    }

    #[test]
    fn test_parse_dec_positive() {
        let dec = parse_dec("+16 17 25.5").unwrap();
        let expected = 16.0 + 17.0 / 60.0 + 25.5 / 3600.0;
        assert!((dec - expected).abs() < 1e-6);
    }

    #[test]
    fn test_parse_observation() {
        let obs = parse_observation_line(CERES_OBS_1).unwrap();
        assert_eq!(obs.designation, "00001");
        assert!(!obs.is_discovery);
        assert_eq!(obs.obs_type, Some('C'));
        assert_eq!(obs.date.0, 2024);
        assert_eq!(obs.date.1, 1);
        assert!((obs.date.2 - 15.12345).abs() < 1e-5);
        assert!(obs.ra_deg > 95.0 && obs.ra_deg < 96.0);
        assert!(obs.dec_deg < -8.0 && obs.dec_deg > -9.0);
        assert!((obs.magnitude.unwrap() - 18.5).abs() < 1e-2);
        assert_eq!(obs.band, Some('V'));
        assert_eq!(obs.observatory_code, "568");
    }

    #[test]
    fn test_discovery_flag() {
        let obs = parse_observation_line(CERES_OBS_2).unwrap();
        assert!(obs.is_discovery);
        assert_eq!(obs.band, Some('R'));
    }

    #[test]
    fn test_skip_short_lines() {
        assert!(parse_observation_line("too short").is_none());
        assert!(parse_observation_line("").is_none());
    }

    #[test]
    fn test_parse_date() {
        let (y, m, d) = parse_date("2024 01 15.12345").unwrap();
        assert_eq!(y, 2024);
        assert_eq!(m, 1);
        assert!((d - 15.12345).abs() < 1e-5);
    }
}
