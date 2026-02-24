//! HTTP client for the NASA JPL HORIZONS ephemeris computation service.
//!
//! HORIZONS computes positions, velocities, and observational quantities
//! for over 1.5 million solar system objects. This module provides a
//! type-safe Rust interface to both the ephemeris API and the lookup API.
//!
//! No API key or authentication is required.

use crate::{Result, StarfieldError};
use serde::Deserialize;
use std::collections::HashMap;

/// Base URL for the HORIZONS ephemeris API
const HORIZONS_API_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";

/// Base URL for the HORIZONS lookup API
const HORIZONS_LOOKUP_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api";

/// Target body specification for the COMMAND parameter.
///
/// Different syntax rules apply to major bodies vs small bodies.
/// The semicolon suffix for small bodies is handled automatically.
#[derive(Debug, Clone)]
pub enum Command {
    /// Major body by NAIF ID (e.g., 499 for Mars, 10 for Sun, 301 for Moon)
    MajorBody(i32),
    /// Asteroid by IAU number (semicolon appended automatically)
    Asteroid(u32),
    /// Comet by designation string (e.g., "73P")
    Comet(String),
    /// Object by provisional designation (e.g., "1999 AN10")
    Designation(String),
    /// Object by name (case-insensitive search, semicolon appended)
    Name(String),
}

impl Command {
    /// Convert to the query string value expected by the HORIZONS API
    pub fn to_query_value(&self) -> String {
        match self {
            Command::MajorBody(id) => format!("{}", id),
            Command::Asteroid(num) => format!("{};", num),
            Command::Comet(des) => format!("{};", des),
            Command::Designation(des) => format!("DES={};", des),
            Command::Name(name) => format!("{};", name),
        }
    }
}

/// Ephemeris output type
#[derive(Debug, Clone, Copy)]
pub enum EphemType {
    /// Observer-table: sky-plane observables (RA/Dec, magnitude, etc.)
    Observer,
    /// Vectors: Cartesian state vectors (X, Y, Z, VX, VY, VZ)
    Vectors,
    /// Elements: osculating Keplerian orbital elements
    Elements,
}

impl EphemType {
    fn as_str(&self) -> &'static str {
        match self {
            EphemType::Observer => "OBSERVER",
            EphemType::Vectors => "VECTORS",
            EphemType::Elements => "ELEMENTS",
        }
    }
}

/// Observer or coordinate center specification
#[derive(Debug, Clone)]
pub enum Center {
    /// Body center by NAIF ID (e.g., 399 for geocentric, 0 for SSB)
    BodyCenter(i32),
    /// Observatory site code at a body (e.g., "675@399" for Palomar)
    Site(String),
    /// Geocentric (equivalent to BodyCenter(500@399))
    Geocentric,
    /// Solar System Barycenter
    SolarSystemBarycenter,
}

impl Center {
    fn to_query_value(&self) -> String {
        match self {
            Center::BodyCenter(id) => format!("500@{}", id),
            Center::Site(code) => code.clone(),
            Center::Geocentric => "500@399".to_string(),
            Center::SolarSystemBarycenter => "500@0".to_string(),
        }
    }
}

/// Time specification for ephemeris requests
#[derive(Debug, Clone)]
pub enum TimeSpec {
    /// Time range with start, stop, and step size
    Range {
        /// Start time (e.g., "2024-01-01", "2024-01-01 12:00", "JD2451545.0")
        start: String,
        /// Stop time
        stop: String,
        /// Step size (e.g., "1 d", "1 h", "30 m", "10", "1 MONTH")
        step: String,
    },
    /// Discrete list of Julian Day numbers (TDB)
    JulianDayList(Vec<f64>),
}

/// Output distance/time units for vector and elements ephemerides
#[derive(Debug, Clone, Copy)]
pub enum OutputUnits {
    /// Kilometers and seconds
    KmS,
    /// Astronomical units and days
    AuD,
    /// Kilometers and days
    KmD,
}

impl OutputUnits {
    fn as_str(&self) -> &'static str {
        match self {
            OutputUnits::KmS => "KM-S",
            OutputUnits::AuD => "AU-D",
            OutputUnits::KmD => "KM-D",
        }
    }
}

/// Reference plane for vector or elements output
#[derive(Debug, Clone, Copy)]
pub enum ReferencePlane {
    /// Ecliptic and mean equinox of reference epoch
    Ecliptic,
    /// Body-centered reference frame (ICRF)
    Frame,
    /// Body equator and node of date
    BodyEquator,
}

impl ReferencePlane {
    fn as_str(&self) -> &'static str {
        match self {
            ReferencePlane::Ecliptic => "ECLIPTIC",
            ReferencePlane::Frame => "FRAME",
            ReferencePlane::BodyEquator => "BODY EQUATOR",
        }
    }
}

/// Vector table content type
#[derive(Debug, Clone, Copy)]
pub enum VecTable {
    /// Position only: X, Y, Z
    Position,
    /// State vector: X, Y, Z, VX, VY, VZ
    State,
    /// State + extras: X, Y, Z, VX, VY, VZ, LT, RG, RR (default)
    StateExtras,
    /// Position + extras: X, Y, Z, LT, RG, RR
    PositionExtras,
    /// Velocity only: VX, VY, VZ
    Velocity,
    /// Extras only: LT, RG, RR
    Extras,
}

impl VecTable {
    fn as_str(&self) -> &'static str {
        match self {
            VecTable::Position => "1",
            VecTable::State => "2",
            VecTable::StateExtras => "3",
            VecTable::PositionExtras => "4",
            VecTable::Velocity => "5",
            VecTable::Extras => "6",
        }
    }
}

/// Aberration correction for vector output
#[derive(Debug, Clone, Copy)]
pub enum VecCorrection {
    /// Geometric (no correction)
    None,
    /// Light-time corrected (astrometric)
    LightTime,
    /// Light-time + stellar aberration (apparent)
    LightTimeAberration,
}

impl VecCorrection {
    fn as_str(&self) -> &'static str {
        match self {
            VecCorrection::None => "NONE",
            VecCorrection::LightTime => "LT",
            VecCorrection::LightTimeAberration => "LT+S",
        }
    }
}

/// Request builder for HORIZONS ephemeris queries
#[derive(Debug, Clone)]
pub struct EphemerisRequest {
    /// Target body
    pub command: Command,
    /// Ephemeris type
    pub ephem_type: EphemType,
    /// Coordinate center
    pub center: Center,
    /// Time specification
    pub time_spec: TimeSpec,
    /// Include object data header (default: false)
    pub obj_data: bool,
    /// Vector table type
    pub vec_table: Option<VecTable>,
    /// Output units
    pub out_units: Option<OutputUnits>,
    /// Vector aberration correction
    pub vec_corr: Option<VecCorrection>,
    /// Reference plane
    pub ref_plane: Option<ReferencePlane>,
    /// Observer quantity codes (comma-separated, e.g., "1,9,20,23")
    pub quantities: Option<String>,
    /// RA/Dec angle format: "HMS" or "DEG"
    pub ang_format: Option<String>,
    /// Enable CSV-format output
    pub csv_format: bool,
    /// Extra precision in RA/Dec
    pub extra_prec: bool,
}

impl EphemerisRequest {
    /// Create a request for Cartesian state vectors
    pub fn vectors(command: Command, center: Center, time_spec: TimeSpec) -> Self {
        Self {
            command,
            ephem_type: EphemType::Vectors,
            center,
            time_spec,
            obj_data: false,
            vec_table: Some(VecTable::StateExtras),
            out_units: Some(OutputUnits::AuD),
            vec_corr: Some(VecCorrection::None),
            ref_plane: Some(ReferencePlane::Ecliptic),
            quantities: None,
            ang_format: None,
            csv_format: true,
            extra_prec: false,
        }
    }

    /// Create a request for observer-table data (RA/Dec, magnitude, etc.)
    pub fn observer(command: Command, center: Center, time_spec: TimeSpec) -> Self {
        Self {
            command,
            ephem_type: EphemType::Observer,
            center,
            time_spec,
            obj_data: false,
            vec_table: None,
            out_units: None,
            vec_corr: None,
            ref_plane: None,
            quantities: Some("1,9,20,23".to_string()),
            ang_format: Some("DEG".to_string()),
            csv_format: true,
            extra_prec: true,
        }
    }

    /// Create a request for osculating Keplerian orbital elements
    pub fn elements(command: Command, center: Center, time_spec: TimeSpec) -> Self {
        Self {
            command,
            ephem_type: EphemType::Elements,
            center,
            time_spec,
            obj_data: false,
            vec_table: None,
            out_units: Some(OutputUnits::AuD),
            vec_corr: None,
            ref_plane: Some(ReferencePlane::Ecliptic),
            quantities: None,
            ang_format: None,
            csv_format: true,
            extra_prec: false,
        }
    }

    /// Build query parameters for the HTTP request
    fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params: Vec<(String, String)> = Vec::new();

        params.push(("format".into(), "json".into()));
        params.push((
            "COMMAND".into(),
            format!("'{}'", self.command.to_query_value()),
        ));
        params.push(("MAKE_EPHEM".into(), "YES".into()));
        params.push(("EPHEM_TYPE".into(), self.ephem_type.as_str().into()));
        params.push((
            "CENTER".into(),
            format!("'{}'", self.center.to_query_value()),
        ));

        match &self.time_spec {
            TimeSpec::Range { start, stop, step } => {
                params.push(("START_TIME".into(), format!("'{}'", start)));
                params.push(("STOP_TIME".into(), format!("'{}'", stop)));
                params.push(("STEP_SIZE".into(), format!("'{}'", step)));
            }
            TimeSpec::JulianDayList(jds) => {
                let tlist: Vec<String> = jds.iter().map(|jd| format!("{}", jd)).collect();
                params.push(("TLIST".into(), tlist.join(",")));
            }
        }

        if self.obj_data {
            params.push(("OBJ_DATA".into(), "YES".into()));
        } else {
            params.push(("OBJ_DATA".into(), "NO".into()));
        }

        if let Some(vt) = &self.vec_table {
            params.push(("VEC_TABLE".into(), format!("'{}'", vt.as_str())));
        }

        if let Some(units) = &self.out_units {
            params.push(("OUT_UNITS".into(), format!("'{}'", units.as_str())));
        }

        if let Some(corr) = &self.vec_corr {
            params.push(("VEC_CORR".into(), format!("'{}'", corr.as_str())));
        }

        if let Some(plane) = &self.ref_plane {
            params.push(("REF_PLANE".into(), format!("'{}'", plane.as_str())));
        }

        if let Some(quant) = &self.quantities {
            params.push(("QUANTITIES".into(), format!("'{}'", quant)));
        }

        if let Some(fmt) = &self.ang_format {
            params.push(("ANG_FORMAT".into(), format!("'{}'", fmt)));
        }

        if self.csv_format {
            params.push(("CSV_FORMAT".into(), "YES".into()));
        }

        if self.extra_prec {
            params.push(("EXTRA_PREC".into(), "YES".into()));
        }

        params
    }
}

/// API response signature common to all HORIZONS endpoints
#[derive(Debug, Clone, Deserialize)]
pub struct Signature {
    pub source: String,
    pub version: String,
}

/// Raw JSON response from the HORIZONS ephemeris API
#[derive(Debug, Clone, Deserialize)]
pub struct HorizonsResponse {
    pub signature: Option<Signature>,
    /// Full text output (for OBSERVER, VECTORS, ELEMENTS, APPROACH types)
    pub result: Option<String>,
    /// Base64-encoded SPK binary (for SPK ephemeris type)
    pub spk: Option<String>,
    /// Suggested filename for SPK output
    pub spk_file_id: Option<String>,
}

/// Object group filter for the lookup API
#[derive(Debug, Clone, Copy)]
pub enum ObjectGroup {
    /// Asteroids only
    Asteroid,
    /// Comets only
    Comet,
    /// Planets only
    Planet,
    /// Natural satellites only
    Satellite,
    /// Spacecraft only
    Spacecraft,
    /// All major bodies (planets + satellites + spacecraft)
    MajorBody,
    /// All small bodies (asteroids + comets)
    SmallBody,
}

impl ObjectGroup {
    fn as_str(&self) -> &'static str {
        match self {
            ObjectGroup::Asteroid => "ast",
            ObjectGroup::Comet => "com",
            ObjectGroup::Planet => "pln",
            ObjectGroup::Satellite => "sat",
            ObjectGroup::Spacecraft => "sct",
            ObjectGroup::MajorBody => "mb",
            ObjectGroup::SmallBody => "sb",
        }
    }
}

/// A single match from the lookup API
#[derive(Debug, Clone, Deserialize)]
pub struct LookupMatch {
    /// Primary designation
    pub pdes: Option<String>,
    /// Object name
    pub name: Option<String>,
    /// SPK-ID
    pub spkid: Option<String>,
    /// Alternate designations
    pub alias: Option<Vec<String>>,
}

/// Response from the HORIZONS lookup API
#[derive(Debug, Clone, Deserialize)]
pub struct LookupResponse {
    pub signature: Option<Signature>,
    /// Number of matches (as string from the API)
    pub count: Option<String>,
    /// Match results (present when count >= 1)
    pub result: Option<Vec<LookupMatch>>,
}

impl LookupResponse {
    /// Get the match count as a number
    pub fn count(&self) -> usize {
        self.count
            .as_ref()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0)
    }
}

/// HTTP client for the HORIZONS API
pub struct HorizonsClient {
    client: reqwest::blocking::Client,
}

impl HorizonsClient {
    /// Create a new HORIZONS API client
    pub fn new() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|e| {
                StarfieldError::DataError(format!("Failed to create HTTP client: {}", e))
            })?;
        Ok(Self { client })
    }

    /// Execute an ephemeris query and return the raw response
    pub fn query(&self, request: &EphemerisRequest) -> Result<HorizonsResponse> {
        let params = request.to_query_params();
        let response = self
            .client
            .get(HORIZONS_API_URL)
            .query(&params)
            .send()
            .map_err(|e| StarfieldError::DataError(format!("HORIZONS request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(StarfieldError::DataError(format!(
                "HORIZONS API returned HTTP {}",
                response.status()
            )));
        }

        let body: HorizonsResponse = response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse HORIZONS response: {}", e))
        })?;

        // Check for HORIZONS-level errors in the result text
        if let Some(ref result) = body.result {
            if result.contains("Cannot interpret target body")
                || result.contains("No ephemeris for target")
                || result.contains("Ambiguous target name")
                || result.contains("No matches found")
            {
                return Err(StarfieldError::DataError(format!(
                    "HORIZONS error: {}",
                    extract_error_message(result)
                )));
            }
        }

        Ok(body)
    }

    /// Look up an object by name, designation, or SPK-ID
    pub fn lookup(&self, sstr: &str, group: Option<ObjectGroup>) -> Result<LookupResponse> {
        let mut params: HashMap<&str, String> = HashMap::new();
        params.insert("sstr", sstr.to_string());
        params.insert("format", "json".to_string());
        if let Some(g) = group {
            params.insert("group", g.as_str().to_string());
        }

        let response = self
            .client
            .get(HORIZONS_LOOKUP_URL)
            .query(&params)
            .send()
            .map_err(|e| StarfieldError::DataError(format!("HORIZONS lookup failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(StarfieldError::DataError(format!(
                "HORIZONS lookup API returned HTTP {}",
                response.status()
            )));
        }

        let body: LookupResponse = response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse lookup response: {}", e))
        })?;

        Ok(body)
    }
}

/// Extract a concise error message from HORIZONS result text
fn extract_error_message(result: &str) -> String {
    // HORIZONS embeds errors in the full text output.
    // Try to find the most relevant line.
    for line in result.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Cannot interpret")
            || trimmed.starts_with("No ephemeris")
            || trimmed.starts_with("Ambiguous target")
            || trimmed.starts_with("No matches")
            || trimmed.starts_with("No site matches")
        {
            return trimmed.to_string();
        }
    }
    // Fallback: first 200 chars
    result.chars().take(200).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_major_body() {
        assert_eq!(Command::MajorBody(499).to_query_value(), "499");
        assert_eq!(Command::MajorBody(10).to_query_value(), "10");
        assert_eq!(Command::MajorBody(0).to_query_value(), "0");
        assert_eq!(Command::MajorBody(-170).to_query_value(), "-170");
    }

    #[test]
    fn test_command_asteroid() {
        assert_eq!(Command::Asteroid(433).to_query_value(), "433;");
        assert_eq!(Command::Asteroid(1).to_query_value(), "1;");
    }

    #[test]
    fn test_command_comet() {
        assert_eq!(Command::Comet("73P".to_string()).to_query_value(), "73P;");
    }

    #[test]
    fn test_command_designation() {
        assert_eq!(
            Command::Designation("1999 AN10".to_string()).to_query_value(),
            "DES=1999 AN10;"
        );
    }

    #[test]
    fn test_command_name() {
        assert_eq!(
            Command::Name("Apophis".to_string()).to_query_value(),
            "Apophis;"
        );
    }

    #[test]
    fn test_center_values() {
        assert_eq!(Center::Geocentric.to_query_value(), "500@399");
        assert_eq!(Center::SolarSystemBarycenter.to_query_value(), "500@0");
        assert_eq!(Center::BodyCenter(10).to_query_value(), "500@10");
        assert_eq!(
            Center::Site("675@399".to_string()).to_query_value(),
            "675@399"
        );
    }

    #[test]
    fn test_vectors_request_params() {
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("COMMAND").unwrap(), "'499'");
        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "VECTORS");
        assert_eq!(map.get("CENTER").unwrap(), "'500@0'");
        assert_eq!(map.get("START_TIME").unwrap(), "'2024-01-01'");
        assert_eq!(map.get("STOP_TIME").unwrap(), "'2024-01-02'");
        assert_eq!(map.get("STEP_SIZE").unwrap(), "'1 d'");
        assert_eq!(map.get("CSV_FORMAT").unwrap(), "YES");
        assert_eq!(map.get("OUT_UNITS").unwrap(), "'AU-D'");
    }

    #[test]
    fn test_observer_request_params() {
        let req = EphemerisRequest::observer(
            Command::MajorBody(499),
            Center::Geocentric,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "OBSERVER");
        assert_eq!(map.get("CENTER").unwrap(), "'500@399'");
        assert_eq!(map.get("QUANTITIES").unwrap(), "'1,9,20,23'");
        assert_eq!(map.get("ANG_FORMAT").unwrap(), "'DEG'");
        assert_eq!(map.get("EXTRA_PREC").unwrap(), "YES");
    }

    #[test]
    fn test_elements_request_params() {
        let req = EphemerisRequest::elements(
            Command::Asteroid(433),
            Center::BodyCenter(10),
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-02-01".into(),
                step: "1 d".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("COMMAND").unwrap(), "'433;'");
        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "ELEMENTS");
        assert_eq!(map.get("REF_PLANE").unwrap(), "'ECLIPTIC'");
    }

    #[test]
    fn test_tlist_params() {
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::JulianDayList(vec![2451545.0, 2451546.0]),
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("TLIST").unwrap(), "2451545,2451546");
        assert!(map.get("START_TIME").is_none());
    }

    #[test]
    fn test_error_extraction() {
        let result = "Some header\n  Cannot interpret target body\nMore text";
        assert_eq!(
            extract_error_message(result),
            "Cannot interpret target body"
        );
    }

    #[test]
    fn test_lookup_response_count() {
        let resp = LookupResponse {
            signature: None,
            count: Some("5".to_string()),
            result: None,
        };
        assert_eq!(resp.count(), 5);

        let resp_none = LookupResponse {
            signature: None,
            count: None,
            result: None,
        };
        assert_eq!(resp_none.count(), 0);
    }

    #[test]
    #[ignore]
    fn test_horizons_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(HORIZONS_API_URL)
            .send()
            .expect("HORIZONS API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    #[ignore]
    fn test_lookup_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(HORIZONS_LOOKUP_URL)
            .send()
            .expect("HORIZONS lookup API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }
}
