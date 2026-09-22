//! Parser for MPC observatory code records
//!
//! Source: <https://minorplanetcenter.net/iau/lists/ObsCodes.html>
//!
//! Each line is fixed-width:
//!   Cols 1-3:   Observatory code
//!   Cols 4-13:  Longitude (degrees, east positive)
//!   Cols 14-21: cos(latitude) * rho
//!   Cols 22-30: sin(latitude) * rho
//!   Cols 31+:   Name

use starfield_core::{Result, StarfieldError};

/// An MPC observatory with its geographic position
#[derive(Debug, Clone)]
pub struct Observatory {
    /// 3-character MPC observatory code
    pub code: String,
    /// East longitude in degrees (0-360)
    pub longitude: Option<f64>,
    /// cos(latitude) * distance-from-center (Earth radii)
    pub cos_lat_rho: Option<f64>,
    /// sin(latitude) * distance-from-center (Earth radii)
    pub sin_lat_rho: Option<f64>,
    /// Observatory name
    pub name: String,
}

/// Parse a single observatory code line.
///
/// Returns `None` for header/blank lines that should be skipped.
pub fn parse_observatory_line(line: &str) -> Option<Observatory> {
    if line.len() < 30 {
        return None;
    }

    let code = line[..3].trim().to_string();
    if code.is_empty() || code == "Cod" {
        return None;
    }

    let lon_str = line[3..13].trim();
    let cos_str = line[13..21].trim();
    let sin_str = line[21..30].trim();
    let name = line[30..].trim().to_string();

    let longitude = lon_str.parse::<f64>().ok();
    let cos_lat_rho = cos_str.parse::<f64>().ok();
    let sin_lat_rho = sin_str.parse::<f64>().ok();

    Some(Observatory {
        code,
        longitude,
        cos_lat_rho,
        sin_lat_rho,
        name,
    })
}

/// Parse a full observatory codes file (text content, not HTML).
pub fn parse_observatory_codes(text: &str) -> Result<Vec<Observatory>> {
    let observatories: Vec<Observatory> = text.lines().filter_map(parse_observatory_line).collect();

    if observatories.is_empty() {
        return Err(StarfieldError::DataError(
            "No observatory records found".to_string(),
        ));
    }

    Ok(observatories)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real data excerpt from https://minorplanetcenter.net/iau/lists/ObsCodes.html
    const SAMPLE_OBSERVATORY_LINES: &str = "\
000   0.0000 0.62411 +0.77873 Greenwich
001   0.1542 0.62992 +0.77411 Crowborough
002   0.62   0.622   +0.781   Rayleigh
003   3.90   0.725   +0.687   Montpellier
004   1.4625 0.72520 +0.68627 Toulouse
005   2.231000.659891+0.748875Meudon
006   2.124170.751042+0.658129Fabra Observatory, Barcelona
007   2.336750.659470+0.749223Paris
008   3.0355 0.80172 +0.59578 Algiers-Bouzareah
009   7.4417 0.6838  +0.7272  Berne-Uecht
010   6.921240.723655+0.688135Caussols
011   8.7975 0.67920 +0.73161 Wetzikon
012   4.358210.633333+0.771306Uccle
013   4.483970.614813+0.786029Leiden
014   5.395090.728859+0.682384Marseilles
015   5.129290.615770+0.785285Utrecht
016   5.9893 0.68006 +0.73076 Besancon
017   6.849240.641946+0.764282Hoher List
018   6.7612 0.62779 +0.77578 Dusseldorf-Bilk
019   6.9575 0.68331 +0.72779 Neuchatel";

    #[test]
    fn test_parse_greenwich() {
        let obs = parse_observatory_line("000   0.0000 0.62411 +0.77873 Greenwich").unwrap();
        assert_eq!(obs.code, "000");
        assert!((obs.longitude.unwrap() - 0.0).abs() < 1e-4);
        assert!((obs.cos_lat_rho.unwrap() - 0.62411).abs() < 1e-5);
        assert!((obs.sin_lat_rho.unwrap() - 0.77873).abs() < 1e-5);
        assert_eq!(obs.name, "Greenwich");
    }

    #[test]
    fn test_parse_compact_spacing() {
        // Meudon has no space between some fields
        let obs = parse_observatory_line("005   2.231000.659891+0.748875Meudon").unwrap();
        assert_eq!(obs.code, "005");
        assert_eq!(obs.name, "Meudon");
    }

    #[test]
    fn test_parse_batch() {
        let result = parse_observatory_codes(SAMPLE_OBSERVATORY_LINES).unwrap();
        assert_eq!(result.len(), 20);
        assert_eq!(result[0].code, "000");
        assert_eq!(result[19].code, "019");
    }

    #[test]
    fn test_skip_short_lines() {
        assert!(parse_observatory_line("").is_none());
        assert!(parse_observatory_line("short").is_none());
    }
}
