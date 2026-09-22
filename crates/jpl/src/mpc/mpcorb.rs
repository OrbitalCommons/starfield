//! Parser for MPCORB.DAT fixed-width records
//!
//! Format: <https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html>
//!
//! Each record is a single line with fields at fixed column positions.

use starfield_core::{Result, StarfieldError};

/// Orbital elements for a minor planet from MPCORB.DAT
#[derive(Debug, Clone)]
pub struct MpcOrbRecord {
    /// Packed designation (cols 1-7)
    pub designation: String,
    /// Absolute magnitude H (cols 9-13)
    pub h_magnitude: Option<f64>,
    /// Slope parameter G (cols 15-19)
    pub g_slope: Option<f64>,
    /// Epoch in packed form (cols 21-25)
    pub epoch_packed: String,
    /// Mean anomaly at epoch, degrees (cols 27-35)
    pub mean_anomaly: f64,
    /// Argument of perihelion, J2000, degrees (cols 38-46)
    pub arg_perihelion: f64,
    /// Longitude of ascending node, J2000, degrees (cols 49-57)
    pub long_asc_node: f64,
    /// Inclination to ecliptic, J2000, degrees (cols 60-68)
    pub inclination: f64,
    /// Orbital eccentricity (cols 71-79)
    pub eccentricity: f64,
    /// Mean daily motion, degrees/day (cols 81-91)
    pub mean_motion: f64,
    /// Semimajor axis, AU (cols 93-103)
    pub semimajor_axis: f64,
    /// Reference designation (cols 108-116)
    pub reference: String,
    /// Number of observations (cols 118-122)
    pub num_observations: Option<u32>,
    /// Number of oppositions (cols 124-126)
    pub num_oppositions: Option<u32>,
    /// Arc span string, e.g. "1801-2026" (cols 128-136)
    pub arc: String,
    /// RMS residual, arcsec (cols 138-141)
    pub residual_rms: Option<f64>,
    /// Computer name (cols 151-160)
    pub computer: String,
    /// Readable designation (cols 167-194)
    pub readable_designation: String,
    /// Date of last observation, YYYYMMDD (cols 195-202)
    pub last_obs_date: String,
}

fn slice_field(line: &str, start: usize, end: usize) -> &str {
    if line.len() >= end {
        line[start..end].trim()
    } else if line.len() > start {
        line[start..].trim()
    } else {
        ""
    }
}

/// Parse a single MPCORB.DAT record line.
///
/// Returns `None` for header lines, blank lines, or separator lines.
pub fn parse_mpcorb_line(line: &str) -> Option<MpcOrbRecord> {
    // Skip short lines, headers, blank lines, and separator lines
    if line.len() < 160 || line.starts_with('-') || line.starts_with(' ') {
        return None;
    }

    let designation = slice_field(line, 0, 7).to_string();
    if designation.is_empty() {
        return None;
    }

    let h_magnitude = slice_field(line, 8, 13).parse::<f64>().ok();
    let g_slope = slice_field(line, 14, 19).parse::<f64>().ok();
    let epoch_packed = slice_field(line, 20, 25).to_string();

    let mean_anomaly = slice_field(line, 26, 35)
        .parse::<f64>()
        .map_err(|e| {
            StarfieldError::DataError(format!("Bad mean anomaly for {}: {}", designation, e))
        })
        .ok()?;

    let arg_perihelion = slice_field(line, 37, 46).parse::<f64>().ok()?;
    let long_asc_node = slice_field(line, 48, 57).parse::<f64>().ok()?;
    let inclination = slice_field(line, 59, 68).parse::<f64>().ok()?;
    let eccentricity = slice_field(line, 70, 79).parse::<f64>().ok()?;
    let mean_motion = slice_field(line, 80, 91).parse::<f64>().ok()?;
    let semimajor_axis = slice_field(line, 92, 103).parse::<f64>().ok()?;

    let reference = slice_field(line, 107, 116).to_string();
    let num_observations = slice_field(line, 117, 122).parse::<u32>().ok();
    let num_oppositions = slice_field(line, 123, 126).parse::<u32>().ok();
    let arc = slice_field(line, 127, 136).to_string();
    let residual_rms = slice_field(line, 137, 141).parse::<f64>().ok();
    let computer = slice_field(line, 150, 160).to_string();
    let readable_designation = slice_field(line, 166, 194).to_string();
    let last_obs_date = slice_field(line, 194, 202).to_string();

    Some(MpcOrbRecord {
        designation,
        h_magnitude,
        g_slope,
        epoch_packed,
        mean_anomaly,
        arg_perihelion,
        long_asc_node,
        inclination,
        eccentricity,
        mean_motion,
        semimajor_axis,
        reference,
        num_observations,
        num_oppositions,
        arc,
        residual_rms,
        computer,
        readable_designation,
        last_obs_date,
    })
}

/// Parse the full MPCORB.DAT content, skipping the header block.
pub fn parse_mpcorb(text: &str) -> Result<Vec<MpcOrbRecord>> {
    let records: Vec<MpcOrbRecord> = text.lines().filter_map(parse_mpcorb_line).collect();

    if records.is_empty() {
        return Err(StarfieldError::DataError(
            "No MPCORB records found".to_string(),
        ));
    }

    Ok(records)
}

/// Unpack an MPC packed epoch into (year, month, day).
///
/// Packed epoch format: first two chars = century+year (e.g. "K25" = 2025),
/// third char = month (1-9, A=10, B=11, C=12),
/// fourth-fifth chars = day (packed similarly).
pub fn unpack_epoch(packed: &str) -> Option<(i32, u32, f64)> {
    if packed.len() < 5 {
        return None;
    }

    let bytes = packed.as_bytes();
    let century = match bytes[0] {
        b'I' => 1800,
        b'J' => 1900,
        b'K' => 2000,
        _ => return None,
    };

    let year_tens = (bytes[1] as char).to_digit(10)?;
    let year_ones = (bytes[2] as char).to_digit(10)?;
    let year = century + (year_tens * 10 + year_ones) as i32;

    let month = unpack_digit(bytes[3])?;
    let day = unpack_digit(bytes[4])? as f64;

    Some((year, month, day))
}

fn unpack_digit(b: u8) -> Option<u32> {
    match b {
        b'1'..=b'9' => Some((b - b'0') as u32),
        b'A'..=b'V' => Some((b - b'A') as u32 + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real records from MPCORB.DAT (fetched 2026-03-09)
    const CERES_LINE: &str = "00001    3.35  0.15 K25BL 231.53975   73.29974   80.24963   10.58789  0.0795763  0.21429712   2.7656157  0 MPO964264  7384 126 1801-2026 0.69 M-v 30k MPCORBFIT  4000      (1) Ceres              20260103";
    const PALLAS_LINE: &str = "00002    4.11  0.15 K25BL 211.52977  310.93340  172.88859   34.92833  0.2306430  0.21379713   2.7699258  0 MPO964264  9023 124 1804-2025 0.64 M-c 28k MPCORBFIT  4000      (2) Pallas             20251214";
    const VESTA_LINE: &str = "00004    3.25  0.15 K25BL  26.80969  151.53711  103.70232    7.14406  0.0901676  0.27158812   2.3615413  0 MPO964264  7603 112 1821-2025 0.69 M-p 18k MPCORBFIT  4000      (4) Vesta              20250624";

    #[test]
    fn test_parse_ceres() {
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        assert_eq!(rec.designation, "00001");
        assert!((rec.h_magnitude.unwrap() - 3.35).abs() < 1e-2);
        assert!((rec.g_slope.unwrap() - 0.15).abs() < 1e-2);
        assert_eq!(rec.epoch_packed, "K25BL");
        assert!((rec.mean_anomaly - 231.53975).abs() < 1e-4);
        assert!((rec.arg_perihelion - 73.29974).abs() < 1e-4);
        assert!((rec.long_asc_node - 80.24963).abs() < 1e-4);
        assert!((rec.inclination - 10.58789).abs() < 1e-4);
        assert!((rec.eccentricity - 0.0795763).abs() < 1e-7);
        assert!((rec.mean_motion - 0.21429712).abs() < 1e-7);
        assert!((rec.semimajor_axis - 2.7656157).abs() < 1e-6);
        assert_eq!(rec.num_observations, Some(7384));
        assert_eq!(rec.num_oppositions, Some(126));
        assert_eq!(rec.arc, "1801-2026");
        assert!((rec.residual_rms.unwrap() - 0.69).abs() < 1e-2);
        assert!(rec.readable_designation.contains("Ceres"));
        assert_eq!(rec.last_obs_date, "20260103");
    }

    #[test]
    fn test_parse_pallas() {
        let rec = parse_mpcorb_line(PALLAS_LINE).unwrap();
        assert_eq!(rec.designation, "00002");
        assert!((rec.inclination - 34.92833).abs() < 1e-4);
        assert!((rec.eccentricity - 0.2306430).abs() < 1e-7);
        assert!(rec.readable_designation.contains("Pallas"));
    }

    #[test]
    fn test_parse_vesta() {
        let rec = parse_mpcorb_line(VESTA_LINE).unwrap();
        assert_eq!(rec.designation, "00004");
        assert!((rec.semimajor_axis - 2.3615413).abs() < 1e-6);
        assert!(rec.readable_designation.contains("Vesta"));
    }

    #[test]
    fn test_parse_batch_with_headers() {
        let input = format!(
            "MINOR PLANET CENTER ORBIT DATABASE\n\n{}\n{}\n{}",
            CERES_LINE, PALLAS_LINE, VESTA_LINE
        );
        let records = parse_mpcorb(&input).unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_skip_header_lines() {
        assert!(parse_mpcorb_line("MINOR PLANET CENTER ORBIT DATABASE").is_none());
        assert!(parse_mpcorb_line("").is_none());
        assert!(parse_mpcorb_line("---").is_none());
        assert!(parse_mpcorb_line("   Software programs may include").is_none());
    }

    #[test]
    fn test_unpack_epoch() {
        let (y, m, d) = unpack_epoch("K25BL").unwrap();
        assert_eq!(y, 2025);
        assert_eq!(m, 11); // B = 11
        assert_eq!(d, 21.0); // L = 21
    }

    #[test]
    fn test_unpack_epoch_j_century() {
        let (y, m, _) = unpack_epoch("J9611").unwrap();
        assert_eq!(y, 1996);
        assert_eq!(m, 1);
    }

    #[test]
    fn test_ceres_physical_sanity() {
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        // Ceres orbits in the main belt at ~2.77 AU
        assert!(rec.semimajor_axis > 2.5 && rec.semimajor_axis < 3.0);
        // Near-circular orbit
        assert!(rec.eccentricity < 0.15);
        // Low inclination
        assert!(rec.inclination < 15.0);
    }
}
