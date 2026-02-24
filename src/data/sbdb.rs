//! HTTP client for the JPL Small-Body Database (SBDB) API ecosystem.
//!
//! Provides access to asteroid and comet data from NASA JPL's Small-Body
//! Database, including orbital elements, physical parameters, close approaches,
//! fireball events, and impact risk monitoring (Sentry).
//!
//! All endpoints are HTTP GET, return JSON, and require no authentication.

use crate::sbdb::types::*;
use crate::{Result, StarfieldError};
use serde_json::Value;
use std::collections::HashMap;

const SBDB_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sbdb.api";
const CAD_API_URL: &str = "https://ssd-api.jpl.nasa.gov/cad.api";
const FIREBALL_API_URL: &str = "https://ssd-api.jpl.nasa.gov/fireball.api";
const SENTRY_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sentry.api";
const SBDB_QUERY_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sbdb_query.api";

/// Client for the JPL Small-Body Database API ecosystem
pub struct SbdbClient {
    client: reqwest::blocking::Client,
}

impl SbdbClient {
    /// Create a new SBDB API client
    pub fn new() -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| {
                StarfieldError::DataError(format!("Failed to create HTTP client: {}", e))
            })?;
        Ok(Self { client })
    }

    /// Look up a single small body by search string (name, designation, SPK-ID).
    ///
    /// Returns identification, orbital elements, and optionally physical parameters
    /// and close-approach data.
    pub fn lookup(&self, sstr: &str) -> Result<SbdbLookupResponse> {
        let params = [
            ("sstr", sstr.to_string()),
            ("phys-par", "true".to_string()),
            ("ca-data", "true".to_string()),
            ("discovery", "true".to_string()),
        ];

        let json = self.get_json(SBDB_API_URL, &params)?;
        parse_sbdb_response(&json)
    }

    /// Look up a single small body with minimal data (just identification and orbit).
    pub fn lookup_basic(&self, sstr: &str) -> Result<SbdbLookupResponse> {
        let params = [("sstr", sstr.to_string())];
        let json = self.get_json(SBDB_API_URL, &params)?;
        parse_sbdb_response(&json)
    }

    /// Query close approach data with configurable filters.
    pub fn close_approaches(&self, params: &CadParams) -> Result<CadResponse> {
        let query = params.to_query_params();
        let json = self.get_json(CAD_API_URL, &query)?;
        parse_cad_response(&json)
    }

    /// Query fireball/bolide impact event data.
    pub fn fireballs(&self, params: &FireballParams) -> Result<FireballResponse> {
        let query = params.to_query_params();
        let json = self.get_json(FIREBALL_API_URL, &query)?;
        parse_fireball_response(&json)
    }

    /// Get all objects currently on the Sentry impact monitoring list.
    pub fn sentry_summary(&self) -> Result<SentryResponse> {
        let params: [(String, String); 0] = [];
        let json = self.get_json(SENTRY_API_URL, &params)?;
        parse_sentry_response(&json)
    }

    /// Get Sentry impact risk data for a specific object.
    pub fn sentry_object(&self, des: &str) -> Result<SentryResponse> {
        let params = [("des", des.to_string())];
        let json = self.get_json(SENTRY_API_URL, &params)?;
        parse_sentry_response(&json)
    }

    /// Execute a bulk query against the small-body database.
    pub fn query(
        &self,
        params: &super::super::sbdb::query::SbdbQueryParams,
    ) -> Result<SbdbQueryResponse> {
        let query = params.to_query_params();
        let json = self.get_json(SBDB_QUERY_API_URL, &query)?;
        parse_sbdb_query_response(&json)
    }

    /// Perform a GET request and parse the JSON response
    fn get_json<K: AsRef<str>, V: AsRef<str>>(
        &self,
        url: &str,
        params: &[(K, V)],
    ) -> Result<Value> {
        let query: Vec<(&str, &str)> = params
            .iter()
            .map(|(k, v)| (k.as_ref(), v.as_ref()))
            .collect();

        let response = self
            .client
            .get(url)
            .query(&query)
            .send()
            .map_err(|e| StarfieldError::DataError(format!("SBDB request failed: {}", e)))?;

        let status = response.status();
        if status.as_u16() == 300 {
            return Err(StarfieldError::DataError(
                "Ambiguous search: multiple objects matched. Try a more specific query."
                    .to_string(),
            ));
        }
        if !status.is_success() {
            return Err(StarfieldError::DataError(format!(
                "SBDB API returned HTTP {}",
                status
            )));
        }

        response.json::<Value>().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse SBDB JSON response: {}", e))
        })
    }
}

// ── SBDB Single Lookup ──────────────────────────────────────────────────────

/// Response from the SBDB single-object lookup API
#[derive(Debug, Clone)]
pub struct SbdbLookupResponse {
    /// Object identification
    pub object: SmallBodyObject,
    /// Orbital elements
    pub orbit: Option<SmallBodyOrbit>,
    /// Physical parameters
    pub phys_par: Option<PhysicalParams>,
    /// Close approach records
    pub close_approaches: Option<Vec<CloseApproachRecord>>,
}

fn parse_sbdb_response(json: &Value) -> Result<SbdbLookupResponse> {
    let obj = json
        .get("object")
        .ok_or_else(|| StarfieldError::DataError("Missing 'object' in SBDB response".into()))?;

    let object = SmallBodyObject {
        designation: json_str(obj, "des").unwrap_or_default(),
        spkid: json_str(obj, "spkid"),
        fullname: json_str(obj, "fullname"),
        shortname: json_str(obj, "shortname"),
        kind: json_str(obj, "kind"),
        neo: obj.get("neo").and_then(|v| v.as_bool()).unwrap_or(false),
        pha: obj.get("pha").and_then(|v| v.as_bool()).unwrap_or(false),
        orbit_class: obj
            .get("orbit_class")
            .and_then(|oc| oc.get("code"))
            .and_then(|c| c.as_str())
            .map(OrbitClass::from_code),
    };

    let orbit = json.get("orbit").map(parse_orbit);
    let phys_par = json.get("phys_par").map(parse_phys_par);

    let close_approaches = json.get("ca_data").and_then(|ca| {
        let fields = ca.get("fields")?.as_array()?;
        let data = ca.get("data")?.as_array()?;

        let field_names: Vec<String> = fields
            .iter()
            .filter_map(|f| f.as_str().map(String::from))
            .collect();
        let index = build_field_index(&field_names);

        let records: Vec<CloseApproachRecord> = data
            .iter()
            .filter_map(|row| {
                let row_arr = row.as_array()?;
                Some(parse_ca_row(&index, row_arr, &object.designation))
            })
            .collect();

        Some(records)
    });

    Ok(SbdbLookupResponse {
        object,
        orbit,
        phys_par,
        close_approaches,
    })
}

fn parse_orbit(o: &Value) -> SmallBodyOrbit {
    let elements = o.get("elements").and_then(|e| e.as_array());

    let mut orbit = SmallBodyOrbit {
        orbit_id: json_str(o, "orbit_id"),
        epoch_jd: json_str(o, "epoch").and_then(|s| s.parse().ok()),
        eccentricity: None,
        semi_major_axis: None,
        perihelion_dist: None,
        inclination: None,
        long_asc_node: None,
        arg_perihelion: None,
        mean_anomaly: None,
        time_perihelion: None,
        mean_motion: None,
        period: None,
        aphelion_dist: None,
        moid_au: None,
        first_obs: json_str(o, "first_obs"),
        last_obs: json_str(o, "last_obs"),
        n_obs_used: json_str(o, "n_obs_used").and_then(|s| s.parse().ok()),
        data_arc_days: json_str(o, "data_arc").and_then(|s| s.parse().ok()),
        condition_code: json_str(o, "condition_code"),
        rms: json_str(o, "rms").and_then(|s| s.parse().ok()),
    };

    if let Some(elems) = elements {
        for elem in elems {
            let name = elem.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let value: Option<f64> = elem
                .get("value")
                .and_then(|v| v.as_str())
                .and_then(|s| s.parse().ok());

            match name {
                "e" => orbit.eccentricity = value,
                "a" => orbit.semi_major_axis = value,
                "q" => orbit.perihelion_dist = value,
                "i" => orbit.inclination = value,
                "om" => orbit.long_asc_node = value,
                "w" => orbit.arg_perihelion = value,
                "ma" => orbit.mean_anomaly = value,
                "tp" => orbit.time_perihelion = value,
                "n" => orbit.mean_motion = value,
                "per" => orbit.period = value,
                "ad" => orbit.aphelion_dist = value,
                "moid" => orbit.moid_au = value,
                _ => {}
            }
        }
    }

    orbit
}

fn parse_phys_par(pp: &Value) -> PhysicalParams {
    let items = pp.as_array();
    let mut params = PhysicalParams {
        abs_magnitude_h: None,
        magnitude_slope_g: None,
        diameter_km: None,
        albedo: None,
        rotation_period_h: None,
        spectral_type: None,
    };

    if let Some(items) = items {
        for item in items {
            let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let value_str = item.get("value").and_then(|v| v.as_str());

            match name {
                "H" => params.abs_magnitude_h = value_str.and_then(|s| s.parse().ok()),
                "G" => params.magnitude_slope_g = value_str.and_then(|s| s.parse().ok()),
                "diameter" => params.diameter_km = value_str.and_then(|s| s.parse().ok()),
                "albedo" => params.albedo = value_str.and_then(|s| s.parse().ok()),
                "rot_per" => params.rotation_period_h = value_str.and_then(|s| s.parse().ok()),
                "spec_T" | "spec_B" => {
                    params.spectral_type = value_str.map(String::from);
                }
                _ => {}
            }
        }
    }

    params
}

// ── Close Approach Data (CAD) ───────────────────────────────────────────────

/// Parameters for close-approach queries
#[derive(Debug, Clone, Default)]
pub struct CadParams {
    /// Minimum date filter (YYYY-MM-DD or YYYY-MMM-DD or "now")
    pub date_min: Option<String>,
    /// Maximum date filter
    pub date_max: Option<String>,
    /// Maximum close approach distance (AU or LD with suffix)
    pub dist_max: Option<String>,
    /// Minimum close approach distance
    pub dist_min: Option<String>,
    /// Minimum absolute magnitude H
    pub h_min: Option<f64>,
    /// Maximum absolute magnitude H
    pub h_max: Option<f64>,
    /// Close approach body (default: Earth)
    pub body: Option<String>,
    /// Sort field (e.g., "dist", "date", "h")
    pub sort: Option<String>,
    /// Maximum number of results
    pub limit: Option<u32>,
    /// Include full object name
    pub fullname: bool,
    /// Include diameter data
    pub diameter: bool,
}

impl CadParams {
    fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();
        if let Some(ref v) = self.date_min {
            params.push(("date-min".into(), v.clone()));
        }
        if let Some(ref v) = self.date_max {
            params.push(("date-max".into(), v.clone()));
        }
        if let Some(ref v) = self.dist_max {
            params.push(("dist-max".into(), v.clone()));
        }
        if let Some(ref v) = self.dist_min {
            params.push(("dist-min".into(), v.clone()));
        }
        if let Some(v) = self.h_min {
            params.push(("h-min".into(), v.to_string()));
        }
        if let Some(v) = self.h_max {
            params.push(("h-max".into(), v.to_string()));
        }
        if let Some(ref v) = self.body {
            params.push(("body".into(), v.clone()));
        }
        if let Some(ref v) = self.sort {
            params.push(("sort".into(), v.clone()));
        }
        if let Some(v) = self.limit {
            params.push(("limit".into(), v.to_string()));
        }
        if self.fullname {
            params.push(("fullname".into(), "true".into()));
        }
        if self.diameter {
            params.push(("diameter".into(), "true".into()));
        }
        params
    }
}

/// Response from the close approach data API
#[derive(Debug, Clone)]
pub struct CadResponse {
    /// Total number of records
    pub count: u32,
    /// Close approach records
    pub records: Vec<CloseApproachRecord>,
}

fn parse_cad_response(json: &Value) -> Result<CadResponse> {
    let count: u32 = json
        .get("count")
        .map(|c| {
            c.as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| c.as_u64().map(|n| n as u32))
                .unwrap_or(0)
        })
        .unwrap_or(0);

    let fields = json
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let data = json.get("data").and_then(|d| d.as_array());

    let index = build_field_index(&fields);
    let mut records = Vec::new();

    if let Some(rows) = data {
        for row in rows {
            if let Some(arr) = row.as_array() {
                records.push(parse_cad_row(&index, arr));
            }
        }
    }

    Ok(CadResponse { count, records })
}

fn parse_cad_row(index: &HashMap<&str, usize>, row: &[Value]) -> CloseApproachRecord {
    CloseApproachRecord {
        designation: get_str(index, row, "des").unwrap_or_default(),
        orbit_id: get_str(index, row, "orbit_id"),
        jd_tdb: get_f64(index, row, "jd"),
        date: get_str(index, row, "cd").unwrap_or_default(),
        dist_au: get_f64(index, row, "dist").unwrap_or(0.0),
        dist_min_au: get_f64(index, row, "dist_min"),
        dist_max_au: get_f64(index, row, "dist_max"),
        v_rel_km_s: get_f64(index, row, "v_rel"),
        v_inf_km_s: get_f64(index, row, "v_inf"),
        h_mag: get_f64(index, row, "h"),
        diameter_km: get_f64(index, row, "diameter"),
        fullname: get_str(index, row, "fullname"),
        body: get_str(index, row, "body").unwrap_or_else(|| "Earth".to_string()),
    }
}

fn parse_ca_row(
    index: &HashMap<&str, usize>,
    row: &[Value],
    designation: &str,
) -> CloseApproachRecord {
    CloseApproachRecord {
        designation: designation.to_string(),
        orbit_id: None,
        jd_tdb: get_f64(index, row, "jd"),
        date: get_str(index, row, "cd").unwrap_or_default(),
        dist_au: get_f64(index, row, "dist").unwrap_or(0.0),
        dist_min_au: get_f64(index, row, "dist_min"),
        dist_max_au: get_f64(index, row, "dist_max"),
        v_rel_km_s: get_f64(index, row, "v_rel"),
        v_inf_km_s: get_f64(index, row, "v_inf"),
        h_mag: None,
        diameter_km: None,
        fullname: None,
        body: get_str(index, row, "body").unwrap_or_else(|| "Earth".to_string()),
    }
}

// ── Fireball ────────────────────────────────────────────────────────────────

/// Parameters for fireball/bolide queries
#[derive(Debug, Clone, Default)]
pub struct FireballParams {
    /// Minimum date (YYYY-MM-DD)
    pub date_min: Option<String>,
    /// Maximum date
    pub date_max: Option<String>,
    /// Minimum radiated energy (joules * 10^10)
    pub energy_min: Option<f64>,
    /// Maximum radiated energy
    pub energy_max: Option<f64>,
    /// Include velocity components
    pub vel_comp: bool,
    /// Require location data
    pub req_loc: bool,
    /// Sort field
    pub sort: Option<String>,
    /// Maximum number of results
    pub limit: Option<u32>,
}

impl FireballParams {
    fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();
        if let Some(ref v) = self.date_min {
            params.push(("date-min".into(), v.clone()));
        }
        if let Some(ref v) = self.date_max {
            params.push(("date-max".into(), v.clone()));
        }
        if let Some(v) = self.energy_min {
            params.push(("energy-min".into(), v.to_string()));
        }
        if let Some(v) = self.energy_max {
            params.push(("energy-max".into(), v.to_string()));
        }
        if self.vel_comp {
            params.push(("vel-comp".into(), "true".into()));
        }
        if self.req_loc {
            params.push(("req-loc".into(), "true".into()));
        }
        if let Some(ref v) = self.sort {
            params.push(("sort".into(), v.clone()));
        }
        if let Some(v) = self.limit {
            params.push(("limit".into(), v.to_string()));
        }
        params
    }
}

/// Response from the fireball data API
#[derive(Debug, Clone)]
pub struct FireballResponse {
    /// Total number of records
    pub count: u32,
    /// Fireball event records
    pub records: Vec<FireballRecord>,
}

fn parse_fireball_response(json: &Value) -> Result<FireballResponse> {
    let count = parse_count(json);

    let fields = json
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let data = json.get("data").and_then(|d| d.as_array());
    let index = build_field_index(&fields);
    let mut records = Vec::new();

    if let Some(rows) = data {
        for row in rows {
            if let Some(arr) = row.as_array() {
                records.push(FireballRecord {
                    date: get_str(&index, arr, "date").unwrap_or_default(),
                    energy_joules_e10: get_f64(&index, arr, "energy"),
                    impact_energy_kt: get_f64(&index, arr, "impact-e"),
                    latitude: get_f64(&index, arr, "lat"),
                    lat_dir: get_str(&index, arr, "lat-dir"),
                    longitude: get_f64(&index, arr, "lon"),
                    lon_dir: get_str(&index, arr, "lon-dir"),
                    altitude_km: get_f64(&index, arr, "alt"),
                    velocity_km_s: get_f64(&index, arr, "vel"),
                });
            }
        }
    }

    Ok(FireballResponse { count, records })
}

// ── Sentry ──────────────────────────────────────────────────────────────────

/// Response from the Sentry impact risk API
#[derive(Debug, Clone)]
pub struct SentryResponse {
    /// Total number of entries
    pub count: u32,
    /// Sentry risk entries
    pub entries: Vec<SentryEntry>,
}

fn parse_sentry_response(json: &Value) -> Result<SentryResponse> {
    let count = parse_count(json);
    let mut entries = Vec::new();

    // Sentry summary returns "data" as array of objects
    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
        for item in data {
            entries.push(SentryEntry {
                designation: json_str(item, "des").unwrap_or_default(),
                fullname: json_str(item, "fullname"),
                h_mag: json_str(item, "h").and_then(|s| s.parse().ok()),
                diameter_km: json_str(item, "diameter")
                    .or_else(|| json_str(item, "size"))
                    .and_then(|s| s.parse().ok()),
                n_imp: json_str(item, "n_imp").and_then(|s| s.parse().ok()),
                ip: json_str(item, "ip").and_then(|s| s.parse().ok()),
                ps_cum: json_str(item, "ps_cum").and_then(|s| s.parse().ok()),
                ps_max: json_str(item, "ps_max").and_then(|s| s.parse().ok()),
                ts_max: json_str(item, "ts_max").and_then(|s| s.parse().ok()),
                last_obs: json_str(item, "last_obs"),
                ip_range: json_str(item, "range"),
            });
        }
    }

    Ok(SentryResponse { count, entries })
}

// ── SBDB Query ──────────────────────────────────────────────────────────────

/// Response from the SBDB bulk query API
#[derive(Debug, Clone)]
pub struct SbdbQueryResponse {
    /// Total number of matching objects
    pub count: u32,
    /// Field names for the returned columns
    pub fields: Vec<String>,
    /// Raw data rows (each row is a Vec of JSON values)
    pub data: Vec<Vec<Value>>,
}

fn parse_sbdb_query_response(json: &Value) -> Result<SbdbQueryResponse> {
    let count = parse_count(json);

    let fields = json
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let data = json
        .get("data")
        .and_then(|d| d.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|r| r.as_array().cloned())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(SbdbQueryResponse {
        count,
        fields,
        data,
    })
}

// ── Shared Helpers ──────────────────────────────────────────────────────────

/// Build a field name -> index mapping from a fields array
fn build_field_index(fields: &[String]) -> HashMap<&str, usize> {
    fields
        .iter()
        .enumerate()
        .map(|(i, f)| (f.as_str(), i))
        .collect()
}

/// Extract a string value from a tabular row by field name
fn get_str(index: &HashMap<&str, usize>, row: &[Value], field: &str) -> Option<String> {
    index
        .get(field)
        .and_then(|&i| row.get(i))
        .and_then(|v| v.as_str().map(String::from))
}

/// Extract a float value from a tabular row by field name
fn get_f64(index: &HashMap<&str, usize>, row: &[Value], field: &str) -> Option<f64> {
    index.get(field).and_then(|&i| row.get(i)).and_then(|v| {
        v.as_str()
            .and_then(|s| s.parse().ok())
            .or_else(|| v.as_f64())
    })
}

/// Extract a string field from a JSON object
fn json_str(obj: &Value, field: &str) -> Option<String> {
    obj.get(field).and_then(|v| v.as_str()).map(String::from)
}

/// Parse the "count" field from a JSON response (handles both string and integer)
fn parse_count(json: &Value) -> u32 {
    json.get("count")
        .map(|c| {
            c.as_str()
                .and_then(|s| s.parse().ok())
                .or_else(|| c.as_u64().map(|n| n as u32))
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cad_params_default() {
        let params = CadParams::default();
        assert!(params.to_query_params().is_empty());
    }

    #[test]
    fn test_cad_params_with_filters() {
        let params = CadParams {
            date_min: Some("2024-01-01".into()),
            date_max: Some("2024-12-31".into()),
            dist_max: Some("0.05".into()),
            limit: Some(10),
            fullname: true,
            ..Default::default()
        };
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("date-min").unwrap(), "2024-01-01");
        assert_eq!(map.get("date-max").unwrap(), "2024-12-31");
        assert_eq!(map.get("dist-max").unwrap(), "0.05");
        assert_eq!(map.get("limit").unwrap(), "10");
        assert_eq!(map.get("fullname").unwrap(), "true");
    }

    #[test]
    fn test_fireball_params() {
        let params = FireballParams {
            date_min: Some("2020-01-01".into()),
            req_loc: true,
            limit: Some(5),
            ..Default::default()
        };
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("date-min").unwrap(), "2020-01-01");
        assert_eq!(map.get("req-loc").unwrap(), "true");
        assert_eq!(map.get("limit").unwrap(), "5");
    }

    #[test]
    fn test_parse_count_string() {
        let json: Value = serde_json::from_str(r#"{"count": "42"}"#).unwrap();
        assert_eq!(parse_count(&json), 42);
    }

    #[test]
    fn test_parse_count_integer() {
        let json: Value = serde_json::from_str(r#"{"count": 42}"#).unwrap();
        assert_eq!(parse_count(&json), 42);
    }

    #[test]
    fn test_parse_count_missing() {
        let json: Value = serde_json::from_str(r#"{}"#).unwrap();
        assert_eq!(parse_count(&json), 0);
    }

    #[test]
    fn test_build_field_index() {
        let fields: Vec<String> = vec!["des".into(), "cd".into(), "dist".into()];
        let index = build_field_index(&fields);
        assert_eq!(index.get("des"), Some(&0));
        assert_eq!(index.get("cd"), Some(&1));
        assert_eq!(index.get("dist"), Some(&2));
        assert_eq!(index.get("other"), None);
    }

    #[test]
    fn test_get_str_and_f64() {
        let fields: Vec<String> = vec!["name".into(), "value".into()];
        let index = build_field_index(&fields);
        let row: Vec<Value> = vec![Value::String("test".into()), Value::String("3.14".into())];

        assert_eq!(get_str(&index, &row, "name"), Some("test".to_string()));
        assert!((get_f64(&index, &row, "value").unwrap() - 3.14).abs() < 1e-10);
        assert_eq!(get_str(&index, &row, "missing"), None);
    }

    #[test]
    fn test_parse_cad_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "fields": ["des", "cd", "dist", "v_rel", "h", "body"],
            "data": [
                ["2024 AA", "2024-Jan-01 12:00", "0.001", "5.2", "28.5", "Earth"],
                ["2024 BB", "2024-Feb-15 06:00", "0.002", "8.1", "25.0", "Earth"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_cad_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.records.len(), 2);
        assert_eq!(resp.records[0].designation, "2024 AA");
        assert!((resp.records[0].dist_au - 0.001).abs() < 1e-10);
        assert_eq!(resp.records[1].body, "Earth");
    }

    #[test]
    fn test_parse_fireball_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "1",
            "fields": ["date", "energy", "impact-e", "lat", "lat-dir", "lon", "lon-dir", "alt", "vel"],
            "data": [
                ["2024-01-01 12:00:00", "0.5", "0.01", "45.0", "N", "90.0", "E", "30.0", "15.0"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_fireball_response(&json).unwrap();
        assert_eq!(resp.count, 1);
        assert_eq!(resp.records.len(), 1);
        assert_eq!(resp.records[0].date, "2024-01-01 12:00:00");
        assert!((resp.records[0].energy_joules_e10.unwrap() - 0.5).abs() < 1e-10);
        assert_eq!(resp.records[0].lat_dir.as_deref(), Some("N"));
    }

    #[test]
    fn test_parse_sentry_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "1",
            "data": [
                {"des": "99942", "fullname": "99942 Apophis", "h": "19.7", "n_imp": "2", "ip": "5.2e-06", "ps_cum": "-3.12", "ps_max": "-3.12", "ts_max": "0", "last_obs": "2024-01-01"}
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_sentry_response(&json).unwrap();
        assert_eq!(resp.count, 1);
        assert_eq!(resp.entries.len(), 1);
        assert_eq!(resp.entries[0].designation, "99942");
        assert!((resp.entries[0].h_mag.unwrap() - 19.7).abs() < 0.1);
    }

    #[test]
    fn test_parse_sbdb_query_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "fields": ["spkid", "full_name", "e", "a"],
            "data": [
                ["2000433", "433 Eros", "0.2229", "1.4583"],
                ["2000001", "1 Ceres", "0.0758", "2.7691"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_sbdb_query_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.fields.len(), 4);
        assert_eq!(resp.data.len(), 2);
    }

    #[test]
    fn test_parse_sbdb_object() {
        let json: Value = serde_json::from_str(
            r#"{
            "object": {
                "des": "433",
                "spkid": "2000433",
                "fullname": "433 Eros (A898 PA)",
                "shortname": "433 Eros",
                "kind": "an",
                "neo": true,
                "pha": false,
                "orbit_class": {"name": "Amor", "code": "AMO"}
            },
            "orbit": {
                "orbit_id": "780",
                "epoch": "2460400.5",
                "elements": [
                    {"name": "e", "value": "0.2229", "label": "e", "title": "eccentricity", "units": null},
                    {"name": "a", "value": "1.4583", "label": "a", "title": "semi-major axis", "units": "au"},
                    {"name": "i", "value": "10.83", "label": "i", "title": "inclination", "units": "deg"}
                ]
            }
        }"#,
        )
        .unwrap();

        let resp = parse_sbdb_response(&json).unwrap();
        assert_eq!(resp.object.designation, "433");
        assert!(resp.object.neo);
        assert!(!resp.object.pha);
        assert_eq!(resp.object.orbit_class, Some(OrbitClass::Amor));

        let orbit = resp.orbit.unwrap();
        assert!((orbit.eccentricity.unwrap() - 0.2229).abs() < 1e-4);
        assert!((orbit.semi_major_axis.unwrap() - 1.4583).abs() < 1e-4);
        assert!((orbit.inclination.unwrap() - 10.83).abs() < 0.01);
    }

    #[test]
    #[ignore]
    fn test_sbdb_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(SBDB_API_URL)
            .send()
            .expect("SBDB API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    #[ignore]
    fn test_cad_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(CAD_API_URL)
            .send()
            .expect("CAD API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }
}
