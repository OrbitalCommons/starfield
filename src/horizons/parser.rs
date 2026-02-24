//! Parser for HORIZONS text output.
//!
//! HORIZONS returns ephemeris data as formatted text with the actual data
//! delimited by `$$SOE` (Start of Ephemeris) and `$$EOE` (End of Ephemeris)
//! markers. This module extracts and parses the data between those markers.

use crate::{Result, StarfieldError};

/// A single row of Cartesian state vector data from HORIZONS
#[derive(Debug, Clone)]
pub struct VectorRow {
    /// Julian Date (TDB)
    pub jd_tdb: f64,
    /// Calendar date string (e.g., "A.D. 2024-Jan-01 00:00:00.0000")
    pub calendar_date: String,
    /// X position
    pub x: f64,
    /// Y position
    pub y: f64,
    /// Z position
    pub z: f64,
    /// X velocity
    pub vx: f64,
    /// Y velocity
    pub vy: f64,
    /// Z velocity
    pub vz: f64,
    /// One-way light time (seconds)
    pub light_time: Option<f64>,
    /// Range from coordinate center
    pub range: Option<f64>,
    /// Range rate (radial velocity)
    pub range_rate: Option<f64>,
}

/// A single row of observer-table data from HORIZONS
#[derive(Debug, Clone)]
pub struct ObserverRow {
    /// Julian Date (TDB or UT depending on request)
    pub jd: f64,
    /// Calendar date string
    pub calendar_date: String,
    /// All parsed fields as key-value pairs.
    /// Keys depend on the QUANTITIES requested.
    pub fields: Vec<(String, String)>,
}

/// A single row of osculating orbital elements from HORIZONS
#[derive(Debug, Clone)]
pub struct ElementsRow {
    /// Julian Date (TDB)
    pub jd_tdb: f64,
    /// Calendar date string
    pub calendar_date: String,
    /// Eccentricity
    pub eccentricity: f64,
    /// Periapsis distance (AU or km depending on OUT_UNITS)
    pub periapsis_dist: f64,
    /// Inclination (degrees)
    pub inclination: f64,
    /// Longitude of ascending node (degrees)
    pub long_asc_node: f64,
    /// Argument of perihelion (degrees)
    pub arg_perihelion: f64,
    /// Time of periapsis passage (Julian Date TDB)
    pub time_periapsis: f64,
    /// Mean motion (degrees per time unit)
    pub mean_motion: f64,
    /// Mean anomaly (degrees)
    pub mean_anomaly: f64,
    /// True anomaly (degrees)
    pub true_anomaly: f64,
    /// Semi-major axis (AU or km)
    pub semi_major_axis: Option<f64>,
    /// Apoapsis distance (AU or km)
    pub apoapsis_dist: Option<f64>,
    /// Orbital period (time units)
    pub period: Option<f64>,
}

/// Extract the ephemeris data block between $$SOE and $$EOE markers
pub fn extract_ephemeris_block(result: &str) -> Result<&str> {
    let soe = result.find("$$SOE").ok_or_else(|| {
        StarfieldError::DataError("HORIZONS response missing $$SOE marker".to_string())
    })?;
    let eoe = result.find("$$EOE").ok_or_else(|| {
        StarfieldError::DataError("HORIZONS response missing $$EOE marker".to_string())
    })?;

    if eoe <= soe {
        return Err(StarfieldError::DataError(
            "$$EOE appears before $$SOE in HORIZONS response".to_string(),
        ));
    }

    // Skip past the $$SOE line
    let start = soe + "$$SOE".len();
    Ok(result[start..eoe].trim())
}

/// Parse CSV-format vector rows from a HORIZONS ephemeris block.
///
/// Expects CSV output from a VECTORS request with VEC_TABLE='3' (state + extras).
/// Each record spans multiple CSV lines in the HORIZONS output.
/// With CSV_FORMAT=YES, each row is: JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ, LT, RG, RR,
pub fn parse_vector_rows(block: &str) -> Result<Vec<VectorRow>> {
    let mut rows = Vec::new();

    // In CSV mode with VEC_TABLE=3, each record is a single CSV line:
    // JDTDB, CalendarDate(TDB), X, Y, Z, VX, VY, VZ, LT, RG, RR,
    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

        // We need at least JDTDB + calendar date + X/Y/Z + VX/VY/VZ = 8 fields
        if fields.len() < 8 {
            continue;
        }

        let jd_tdb = parse_f64(fields[0], "JDTDB")?;
        let calendar_date = fields[1].trim().to_string();
        let x = parse_f64(fields[2], "X")?;
        let y = parse_f64(fields[3], "Y")?;
        let z = parse_f64(fields[4], "Z")?;
        let vx = parse_f64(fields[5], "VX")?;
        let vy = parse_f64(fields[6], "VY")?;
        let vz = parse_f64(fields[7], "VZ")?;

        let light_time = fields.get(8).and_then(|f| f.trim().parse().ok());
        let range = fields.get(9).and_then(|f| f.trim().parse().ok());
        let range_rate = fields.get(10).and_then(|f| f.trim().parse().ok());

        rows.push(VectorRow {
            jd_tdb,
            calendar_date,
            x,
            y,
            z,
            vx,
            vy,
            vz,
            light_time,
            range,
            range_rate,
        });
    }

    if rows.is_empty() {
        return Err(StarfieldError::DataError(
            "No vector rows parsed from HORIZONS output".to_string(),
        ));
    }

    Ok(rows)
}

/// Parse CSV-format elements rows from a HORIZONS ephemeris block.
///
/// Expects CSV output from an ELEMENTS request.
/// Fields: JDTDB, Calendar Date, EC, QR, IN, OM, W, Tp, N, MA, TA, A, AD, PR,
pub fn parse_elements_rows(block: &str) -> Result<Vec<ElementsRow>> {
    let mut rows = Vec::new();

    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

        // JDTDB, Cal, EC, QR, IN, OM, W, Tp, N, MA, TA = 11 required fields
        if fields.len() < 11 {
            continue;
        }

        let jd_tdb = parse_f64(fields[0], "JDTDB")?;
        let calendar_date = fields[1].trim().to_string();
        let eccentricity = parse_f64(fields[2], "EC")?;
        let periapsis_dist = parse_f64(fields[3], "QR")?;
        let inclination = parse_f64(fields[4], "IN")?;
        let long_asc_node = parse_f64(fields[5], "OM")?;
        let arg_perihelion = parse_f64(fields[6], "W")?;
        let time_periapsis = parse_f64(fields[7], "Tp")?;
        let mean_motion = parse_f64(fields[8], "N")?;
        let mean_anomaly = parse_f64(fields[9], "MA")?;
        let true_anomaly = parse_f64(fields[10], "TA")?;

        let semi_major_axis = fields.get(11).and_then(|f| f.trim().parse().ok());
        let apoapsis_dist = fields.get(12).and_then(|f| f.trim().parse().ok());
        let period = fields.get(13).and_then(|f| f.trim().parse().ok());

        rows.push(ElementsRow {
            jd_tdb,
            calendar_date,
            eccentricity,
            periapsis_dist,
            inclination,
            long_asc_node,
            arg_perihelion,
            time_periapsis,
            mean_motion,
            mean_anomaly,
            true_anomaly,
            semi_major_axis,
            apoapsis_dist,
            period,
        });
    }

    if rows.is_empty() {
        return Err(StarfieldError::DataError(
            "No elements rows parsed from HORIZONS output".to_string(),
        ));
    }

    Ok(rows)
}

/// Parse CSV-format observer rows from a HORIZONS ephemeris block.
///
/// Observer output is highly variable depending on the QUANTITIES requested.
/// This parser preserves all fields as string key-value pairs for maximum
/// flexibility. The header line above $$SOE in the full HORIZONS result
/// describes the columns.
pub fn parse_observer_rows(block: &str, column_names: &[String]) -> Result<Vec<ObserverRow>> {
    let mut rows = Vec::new();

    for line in block.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split(',').map(|f| f.trim()).collect();

        // Need at least JD + calendar date
        if fields.len() < 2 {
            continue;
        }

        let jd = match fields[0].trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let calendar_date = fields[1].trim().to_string();

        let mut named_fields = Vec::new();
        for (i, value) in fields.iter().enumerate().skip(2) {
            let name = column_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("col_{}", i));
            named_fields.push((name, value.trim().to_string()));
        }

        rows.push(ObserverRow {
            jd,
            calendar_date,
            fields: named_fields,
        });
    }

    if rows.is_empty() {
        return Err(StarfieldError::DataError(
            "No observer rows parsed from HORIZONS output".to_string(),
        ));
    }

    Ok(rows)
}

/// Extract column header names from the HORIZONS result text.
///
/// The header line appears just before $$SOE and contains comma-separated
/// column names when CSV_FORMAT=YES.
pub fn extract_column_names(result: &str) -> Vec<String> {
    // Find the line just before $$SOE that contains column headers
    // In CSV mode, this is typically 2 lines above $$SOE
    if let Some(soe_pos) = result.find("$$SOE") {
        let before_soe = &result[..soe_pos];
        // Walk backwards through non-empty lines
        for line in before_soe.lines().rev() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('*') {
                continue;
            }
            // If this line contains commas, treat it as the header
            if trimmed.contains(',') {
                return trimmed.split(',').map(|s| s.trim().to_string()).collect();
            }
            break;
        }
    }
    Vec::new()
}

/// Parse a float field, returning a descriptive error on failure
fn parse_f64(s: &str, field_name: &str) -> Result<f64> {
    s.trim().parse::<f64>().map_err(|_| {
        StarfieldError::DataError(format!(
            "Failed to parse HORIZONS field '{}': '{}'",
            field_name,
            s.trim()
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_VECTOR_RESULT: &str = r#"
Some header text
*******************************************************************************
$$SOE
 2460310.500000000, A.D. 2024-Jan-01 00:00:00.0000,  1.326568771901361E+00,  5.455289002498498E-01, -3.687818081498823E-02,
 -4.614386613412735E-03,  1.215675362101498E-02,  3.666153662394858E-04,  8.779022975988899E+02,  1.437395883016715E+00, -7.233449301483814E-03,
 2460311.500000000, A.D. 2024-Jan-02 00:00:00.0000,  1.321949252701283E+00,  5.576645505098201E-01, -3.650923764498134E-02,
 -4.717340113418924E-03,  1.210584432101125E-02,  3.659843772394121E-04,  8.800843285988132E+02,  1.439962814016513E+00, -7.128930451483291E-03,
$$EOE
*******************************************************************************
"#;

    #[test]
    fn test_extract_ephemeris_block() {
        let block = extract_ephemeris_block(SAMPLE_VECTOR_RESULT).unwrap();
        assert!(block.contains("2460310.500000000"));
        assert!(block.contains("2460311.500000000"));
        assert!(!block.contains("$$SOE"));
        assert!(!block.contains("$$EOE"));
    }

    #[test]
    fn test_extract_block_missing_soe() {
        let result = "no markers here";
        assert!(extract_ephemeris_block(result).is_err());
    }

    #[test]
    fn test_parse_vector_rows() {
        // HORIZONS CSV vector output: each record is on one long line
        let csv_block = concat!(
            " 2460310.500000000, A.D. 2024-Jan-01 00:00:00.0000,",
            "  1.326568771901361E+00,  5.455289002498498E-01, -3.687818081498823E-02,",
            " -4.614386613412735E-03,  1.215675362101498E-02,  3.666153662394858E-04,",
            "  8.779022975988899E+02,  1.437395883016715E+00, -7.233449301483814E-03,\n",
            " 2460311.500000000, A.D. 2024-Jan-02 00:00:00.0000,",
            "  1.321949252701283E+00,  5.576645505098201E-01, -3.650923764498134E-02,",
            " -4.717340113418924E-03,  1.210584432101125E-02,  3.659843772394121E-04,",
            "  8.800843285988132E+02,  1.439962814016513E+00, -7.128930451483291E-03,",
        );

        let rows = parse_vector_rows(csv_block).unwrap();
        assert_eq!(rows.len(), 2);

        let r = &rows[0];
        assert!((r.jd_tdb - 2460310.5).abs() < 1e-6);
        assert!(r.calendar_date.contains("2024-Jan-01"));
        assert!((r.x - 1.326568771901361).abs() < 1e-10);
        assert!((r.vx - (-4.614386613412735e-3)).abs() < 1e-15);
        assert!(r.light_time.is_some());
        assert!(r.range.is_some());
        assert!(r.range_rate.is_some());
    }

    #[test]
    fn test_parse_elements_rows() {
        let csv_block = " 2460310.500000000, A.D. 2024-Jan-01 00:00:00.0000,  9.339510776498570E-02,  1.381216248476880E+00,  1.848158649553816E+00,  4.951408556629253E+01,  2.867137790700476E+02,  2459928.15625,  5.240760835577432E-01,  2.001564523577853E+02,  1.965743625481955E+02,  1.523662486197090E+00,  1.665908723917300E+00,  6.870988429561428E+02,\n";

        let rows = parse_elements_rows(csv_block).unwrap();
        assert_eq!(rows.len(), 1);

        let r = &rows[0];
        assert!((r.jd_tdb - 2460310.5).abs() < 1e-6);
        assert!((r.eccentricity - 9.339510776498570e-2).abs() < 1e-10);
        assert!((r.inclination - 1.848158649553816).abs() < 1e-10);
        assert!(r.semi_major_axis.is_some());
        assert!((r.semi_major_axis.unwrap() - 1.523662486197090).abs() < 1e-10);
        assert!(r.period.is_some());
    }

    #[test]
    fn test_extract_column_names() {
        let result =
            "Header\n  Date__(UT)__HR:MN, , R.A._____(ICRF)___DEC, Unk,\n$$SOE\ndata\n$$EOE\n";
        let names = extract_column_names(result);
        assert!(names.len() >= 2);
        assert!(names[0].contains("Date"));
    }

    #[test]
    fn test_parse_f64() {
        assert!((parse_f64("  1.5E+02  ", "test").unwrap() - 150.0).abs() < 1e-10);
        assert!(parse_f64("not_a_number", "test").is_err());
    }
}
