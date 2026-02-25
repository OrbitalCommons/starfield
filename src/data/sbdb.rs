//! HTTP client for the JPL Small-Body Database (SBDB) API ecosystem.
//!
//! Provides access to asteroid and comet data from NASA JPL's Small-Body
//! Database, including orbital elements, physical parameters, close approaches,
//! fireball events, impact risk monitoring (Sentry), and radar astrometry.
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
const SCOUT_API_URL: &str = "https://ssd-api.jpl.nasa.gov/scout.api";
const MDESIGN_API_URL: &str = "https://ssd-api.jpl.nasa.gov/mdesign.api";
const RADAR_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sb_radar.api";
const SB_IDENT_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sb_ident.api";
const SBWOBS_API_URL: &str = "https://ssd-api.jpl.nasa.gov/sbwobs.api";
const NHATS_API_URL: &str = "https://ssd-api.jpl.nasa.gov/nhats.api";

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

    /// Find the most accessible small bodies for missions (Mode A).
    ///
    /// Returns targets ranked by the specified optimality criterion for the
    /// given launch year(s).
    pub fn mission_accessible(
        &self,
        params: &MissionAccessibleParams,
    ) -> Result<MissionAccessibleResponse> {
        let mut query: Vec<(String, String)> =
            vec![("crit".into(), params.crit.as_api_value().to_string())];
        if !params.year.is_empty() {
            let years: Vec<String> = params.year.iter().map(|y| y.to_string()).collect();
            query.push(("year".into(), years.join(",")));
        }
        if let Some(lim) = params.lim {
            query.push(("lim".into(), lim.to_string()));
        }
        let json = self.get_json(MDESIGN_API_URL, &query)?;
        parse_mission_accessible_response(&json)
    }

    /// Look up pre-computed mission parameters for a specific object (Mode Q).
    ///
    /// Returns selected mission trajectories with departure/arrival velocities,
    /// phase angles, and other parameters.
    pub fn mission_query(&self, des: &str) -> Result<MissionQueryResponse> {
        let params = [("des", des.to_string())];
        let json = self.get_json(MDESIGN_API_URL, &params)?;
        parse_mission_query_response(&json)
    }

    /// Find small bodies approaching a given heliocentric orbit (Mode T).
    ///
    /// Searches for flyby/extension targets within a specified time span
    /// and distance threshold.
    pub fn mission_flyby(&self, params: &MissionFlybyParams) -> Result<MissionFlybyResponse> {
        let mut query: Vec<(String, String)> = vec![
            ("ec".into(), params.ec.to_string()),
            ("qr".into(), params.qr.to_string()),
            ("tp".into(), params.tp.to_string()),
            ("in".into(), params.inc.to_string()),
            ("om".into(), params.om.to_string()),
            ("w".into(), params.w.to_string()),
            ("jd0".into(), params.jd0.to_string()),
            ("jdf".into(), params.jdf.to_string()),
        ];
        if let Some(maxout) = params.maxout {
            query.push(("maxout".into(), maxout.to_string()));
        }
        if let Some(maxdist) = params.maxdist {
            query.push(("maxdist".into(), maxdist.to_string()));
        }
        let json = self.get_json(MDESIGN_API_URL, &query)?;
        parse_mission_flyby_response(&json)
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

    /// Get a summary of all objects on the NEOCP (Scout Mode S).
    ///
    /// Returns analysis data for unconfirmed objects on the Minor Planet Center's
    /// Near-Earth Object Confirmation Page.
    pub fn scout_summary(&self) -> Result<ScoutSummaryResponse> {
        let params: [(String, String); 0] = [];
        let json = self.get_json(SCOUT_API_URL, &params)?;
        parse_scout_summary_response(&json)
    }

    /// Get detailed Scout analysis for a specific NEOCP object (Scout Mode O).
    ///
    /// The `tdes` parameter is the NEOCP temporary designation (e.g., "P10uUSw").
    pub fn scout_object(&self, tdes: &str) -> Result<ScoutObjectResponse> {
        let params = [("tdes", tdes.to_string()), ("orbits", "true".to_string())];
        let json = self.get_json(SCOUT_API_URL, &params)?;
        parse_scout_object_response(&json)
    }

    /// Query radar astrometry measurement data for small bodies.
    pub fn radar(&self, params: &RadarParams) -> Result<RadarResponse> {
        let query = params.to_query_params();
        let json = self.get_json(RADAR_API_URL, &query)?;
        parse_radar_response(&json)
    }

    /// Identify known small bodies within a specified field of view.
    ///
    /// Given an observer location, observation time, and field of view,
    /// returns small bodies that fall within that region of sky.
    pub fn identify(&self, params: &SbIdentParams) -> Result<SbIdentResponse> {
        let query = params.to_query_params();
        let json = self.get_json(SB_IDENT_API_URL, &query)?;
        parse_sb_ident_response(&json)
    }

    /// Query which small bodies are observable from a given location on a given night.
    ///
    /// Returns night information (sunset/sunrise, twilight times, moon data) and
    /// a list of observable objects with their ephemeris data.
    pub fn observability(&self, params: &ObservabilityParams) -> Result<ObservabilityResponse> {
        let query = params.to_query_params();
        let json = self.get_json(SBWOBS_API_URL, &query)?;
        parse_observability_response(&json)
    }

    /// Query NHATS accessible targets summary (Mode S).
    ///
    /// Returns a list of near-Earth asteroids accessible for human exploration,
    /// filtered by mission constraints such as delta-v, duration, and stay time.
    pub fn nhats_summary(&self, params: &NhatsParams) -> Result<NhatsSummaryResponse> {
        let query = params.to_query_params();
        let json = self.get_json(NHATS_API_URL, &query)?;
        parse_nhats_summary_response(&json)
    }

    /// Query NHATS trajectory detail for a specific object (Mode O).
    ///
    /// Returns detailed trajectory information for a specific near-Earth asteroid,
    /// including minimum delta-v and minimum duration trajectories.
    pub fn nhats_object(&self, des: &str) -> Result<NhatsObjectResponse> {
        let params = [("des", des.to_string())];
        let json = self.get_json(NHATS_API_URL, &params)?;
        parse_nhats_object_response(&json)
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

// ── SB Radar ────────────────────────────────────────────────────────────────

fn parse_radar_response(json: &Value) -> Result<RadarResponse> {
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
                records.push(RadarRecord {
                    designation: get_str(&index, arr, "des").unwrap_or_default(),
                    epoch: get_str(&index, arr, "epoch").unwrap_or_default(),
                    value: get_f64(&index, arr, "value"),
                    sigma: get_f64(&index, arr, "sigma"),
                    units: get_str(&index, arr, "units"),
                    freq: get_f64(&index, arr, "freq"),
                    rcvr: get_str(&index, arr, "rcvr"),
                    xmit: get_str(&index, arr, "xmit"),
                    bp: get_str(&index, arr, "bp"),
                    observer: get_str(&index, arr, "observer"),
                    notes: get_str(&index, arr, "notes"),
                    reference: get_str(&index, arr, "ref"),
                    fullname: get_str(&index, arr, "fullname"),
                    modified: get_str(&index, arr, "modified"),
                    longitude: get_f64(&index, arr, "longitude"),
                    latitude: get_f64(&index, arr, "latitude"),
                    altitude: get_f64(&index, arr, "altitude")
                        .or_else(|| get_f64(&index, arr, "d_xy")),
                });
            }
        }
    }

    Ok(RadarResponse { count, records })
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

// ── Scout (NEOCP Analysis) ──────────────────────────────────────────────────

fn parse_scout_summary_entry(item: &Value) -> ScoutSummaryEntry {
    ScoutSummaryEntry {
        object_name: json_str(item, "objectName").unwrap_or_default(),
        n_obs: json_str(item, "nObs")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("nObs").and_then(|v| v.as_u64()).map(|n| n as u32)),
        arc: json_str(item, "arc")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("arc").and_then(|v| v.as_f64())),
        rms_n: json_str(item, "rmsN")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("rmsN").and_then(|v| v.as_f64())),
        h_mag: json_str(item, "H")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("H").and_then(|v| v.as_f64())),
        rating: json_str(item, "rating")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("rating")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32)
            }),
        moid: json_str(item, "moid")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("moid").and_then(|v| v.as_f64())),
        ca_dist: json_str(item, "caDist")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("caDist").and_then(|v| v.as_f64())),
        v_inf: json_str(item, "vInf")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("vInf").and_then(|v| v.as_f64())),
        pha_score: json_str(item, "phaScore")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("phaScore")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
            }),
        neo_score: json_str(item, "neoScore")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("neoScore")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
            }),
        geocentric_score: json_str(item, "geocentricScore")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("geocentricScore")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
            }),
        ieo_score: json_str(item, "ieoScore")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("ieoScore")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
            }),
        tisserand_score: json_str(item, "tisserandScore")
            .and_then(|s| s.parse().ok())
            .or_else(|| {
                item.get("tisserandScore")
                    .and_then(|v| v.as_i64())
                    .map(|n| n as i32)
            }),
        last_run: json_str(item, "lastRun"),
        ra: json_str(item, "ra"),
        dec: json_str(item, "dec"),
        elong: json_str(item, "elong"),
        rate: json_str(item, "rate")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("rate").and_then(|v| v.as_f64())),
        v_mag: json_str(item, "Vmag")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("Vmag").and_then(|v| v.as_f64())),
        unc: json_str(item, "unc")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("unc").and_then(|v| v.as_f64())),
        unc_p1: json_str(item, "uncP1")
            .and_then(|s| s.parse().ok())
            .or_else(|| item.get("uncP1").and_then(|v| v.as_f64())),
    }
}

fn parse_scout_summary_response(json: &Value) -> Result<ScoutSummaryResponse> {
    let count = parse_count(json);
    let mut entries = Vec::new();

    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
        for item in data {
            entries.push(parse_scout_summary_entry(item));
        }
    }

    Ok(ScoutSummaryResponse {
        count,
        data: entries,
    })
}

// ── Mission Design ──────────────────────────────────────────────────────────

fn parse_mission_accessible_response(json: &Value) -> Result<MissionAccessibleResponse> {
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
    let mut entries = Vec::new();

    if let Some(rows) = data {
        for row in rows {
            if let Some(arr) = row.as_array() {
                entries.push(parse_mission_accessible_row(&index, arr));
            }
        }
    }

    Ok(crate::sbdb::types::MissionAccessibleResponse {
        count,
        data: entries,
    })
}

fn parse_scout_object_response(json: &Value) -> Result<ScoutObjectResponse> {
    let summary = parse_scout_summary_entry(json);

    let orbits = json.get("orbits").map(|o| {
        let count = o
            .get("count")
            .and_then(|c| {
                c.as_str()
                    .and_then(|s| s.parse().ok())
                    .or_else(|| c.as_u64().map(|n| n as u32))
            })
            .unwrap_or(0);

        let fields = o
            .get("fields")
            .and_then(|f| f.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let data = o
            .get("data")
            .and_then(|d| d.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| r.as_array().cloned())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        ScoutOrbitData {
            count,
            fields,
            data,
        }
    });

    Ok(ScoutObjectResponse {
        detail: ScoutObjectDetail {
            summary,
            neo1km_score: json_str(json, "neo1kmScore"),
            t_ephem: json_str(json, "tEphem"),
            orbits,
        },
    })
}

fn parse_mission_accessible_row(
    index: &HashMap<&str, usize>,
    row: &[Value],
) -> MissionAccessibleEntry {
    let neo_str = get_str(index, row, "neo").unwrap_or_default();
    let pha_str = get_str(index, row, "pha").unwrap_or_default();

    MissionAccessibleEntry {
        name: get_str(index, row, "name").unwrap_or_default(),
        pdes: get_str(index, row, "pdes"),
        date0: get_str(index, row, "date0").unwrap_or_default(),
        mjd0: get_f64(index, row, "MJD0").unwrap_or(0.0),
        datef: get_str(index, row, "datef").unwrap_or_default(),
        mjdf: get_f64(index, row, "MJDF").unwrap_or(0.0),
        c3_dep: get_f64(index, row, "c3_dep").unwrap_or(0.0),
        vinf_dep: get_f64(index, row, "vinf_dep").unwrap_or(0.0),
        vinf_arr: get_f64(index, row, "vinf_arr").unwrap_or(0.0),
        dv_tot: get_f64(index, row, "dv_tot").unwrap_or(0.0),
        tof: get_f64(index, row, "tof").unwrap_or(0.0),
        class: get_str(index, row, "class"),
        h_mag: get_f64(index, row, "H"),
        condition_code: get_str(index, row, "condition_code"),
        neo: neo_str == "Y",
        pha: pha_str == "Y",
    }
}

fn parse_mission_query_response(json: &Value) -> Result<MissionQueryResponse> {
    let obj = json
        .get("object")
        .ok_or_else(|| StarfieldError::DataError("Missing 'object' in mdesign response".into()))?;

    let object = MissionQueryObject {
        des: json_str(obj, "des").unwrap_or_default(),
        fullname: json_str(obj, "fullname"),
        spkid: json_str(obj, "spkid"),
        orbit_class: json_str(obj, "orbit_class"),
        condition_code: json_str(obj, "condition_code"),
        data_arc: json_str(obj, "data_arc"),
        orbit_id: json_str(obj, "orbit_id"),
        md_orbit_id: json_str(obj, "md_orbit_id"),
    };

    let fields = json
        .get("fields")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let selected_missions = json
        .get("selectedMissions")
        .and_then(|sm| sm.as_array())
        .map(|rows| {
            rows.iter()
                .filter_map(|row| {
                    row.as_array().map(|arr| {
                        arr.iter()
                            .map(|v| {
                                v.as_f64().unwrap_or_else(|| {
                                    v.as_str().and_then(|s| s.parse().ok()).unwrap_or(f64::NAN)
                                })
                            })
                            .collect()
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(crate::sbdb::types::MissionQueryResponse {
        object,
        fields,
        selected_missions,
    })
}

fn parse_mission_flyby_response(json: &Value) -> Result<MissionFlybyResponse> {
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
    let mut entries = Vec::new();

    if let Some(rows) = data {
        for row in rows {
            if let Some(arr) = row.as_array() {
                entries.push(parse_mission_flyby_row(&index, arr));
            }
        }
    }

    Ok(crate::sbdb::types::MissionFlybyResponse {
        count,
        data: entries,
    })
}

fn parse_mission_flyby_row(index: &HashMap<&str, usize>, row: &[Value]) -> MissionFlybyEntry {
    let neo_str = get_str(index, row, "neo").unwrap_or_default();
    let pha_str = get_str(index, row, "pha").unwrap_or_default();

    MissionFlybyEntry {
        full_name: get_str(index, row, "full_name").unwrap_or_default(),
        pdes: get_str(index, row, "pdes"),
        spkid: get_str(index, row, "spkid"),
        date: get_str(index, row, "date").unwrap_or_default(),
        jd: get_f64(index, row, "jd").unwrap_or(0.0),
        min_dist_au: get_f64(index, row, "min_dist_au").unwrap_or(0.0),
        min_dist_km: get_f64(index, row, "min_dist_km"),
        rel_vel: get_f64(index, row, "rel_vel").unwrap_or(0.0),
        class: get_str(index, row, "class"),
        h_mag: get_f64(index, row, "H"),
        condition_code: get_str(index, row, "condition_code"),
        neo: neo_str == "Y",
        pha: pha_str == "Y",
    }
}

// ── SB Identification ───────────────────────────────────────────────────────

fn parse_sb_ident_response(json: &Value) -> Result<SbIdentResponse> {
    let observer_obj = json.get("observer");
    let observer = SbIdentObserverInfo {
        obs_date: observer_obj.and_then(|o| json_str(o, "obs_date")),
        location: observer_obj.and_then(|o| json_str(o, "location")),
        fov_center: observer_obj.and_then(|o| json_str(o, "fov_center")),
        fov_offset: observer_obj.and_then(|o| json_str(o, "fov_offset")),
        frame: observer_obj.and_then(|o| json_str(o, "frame")),
    };

    let n_first_pass = json
        .get("n_first_pass")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(0) as u32;

    let n_second_pass = json
        .get("n_second_pass")
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
        .unwrap_or(0) as u32;

    let data_first_pass = parse_sb_ident_data(json, "fields_first", "data_first_pass");
    let data_second_pass = parse_sb_ident_data(json, "fields_second", "data_second_pass");

    let elem_first_pass = parse_sb_ident_elements(json, "elem_fields_first", "elem_first_pass");
    let elem_second_pass = parse_sb_ident_elements(json, "elem_fields_second", "elem_second_pass");

    Ok(SbIdentResponse {
        observer,
        n_first_pass,
        n_second_pass,
        data_first_pass,
        data_second_pass,
        elem_first_pass,
        elem_second_pass,
    })
}

fn parse_sb_ident_data(json: &Value, fields_key: &str, data_key: &str) -> Vec<SbIdentEntry> {
    let fields = match json.get(fields_key).and_then(|f| f.as_array()) {
        Some(f) => f
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>(),
        None => return Vec::new(),
    };

    let data = match json.get(data_key).and_then(|d| d.as_array()) {
        Some(d) => d,
        None => return Vec::new(),
    };

    let index = build_field_index(&fields);
    let mut entries = Vec::new();

    for row in data {
        if let Some(arr) = row.as_array() {
            entries.push(SbIdentEntry {
                name: get_str(&index, arr, "Object name").unwrap_or_default(),
                ra: get_str(&index, arr, "Astrometric RA"),
                dec: get_str(&index, arr, "Astrometric Dec"),
                ra_offset: get_f64(&index, arr, "RA offset (arcsec)"),
                dec_offset: get_f64(&index, arr, "Dec offset (arcsec)"),
                total_offset: get_f64(&index, arr, "total offset (arcsec)"),
                vmag: get_f64(&index, arr, "visual magnitude V"),
                ra_rate: get_f64(&index, arr, "RA rate (deg/sec)"),
                dec_rate: get_f64(&index, arr, "Dec rate (deg/sec)"),
                ra_err: get_f64(&index, arr, "RA error estimate (arcsec)"),
                dec_err: get_f64(&index, arr, "Dec error estimate (arcsec)"),
            });
        }
    }

    entries
}

fn parse_sb_ident_elements(
    json: &Value,
    fields_key: &str,
    data_key: &str,
) -> Vec<SbIdentOrbitalElements> {
    let fields = match json.get(fields_key).and_then(|f| f.as_array()) {
        Some(f) => f
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect::<Vec<_>>(),
        None => return Vec::new(),
    };

    let data = match json.get(data_key).and_then(|d| d.as_array()) {
        Some(d) => d,
        None => return Vec::new(),
    };

    let index = build_field_index(&fields);
    let mut entries = Vec::new();

    for row in data {
        if let Some(arr) = row.as_array() {
            entries.push(SbIdentOrbitalElements {
                name: get_str(&index, arr, "Object name").unwrap_or_default(),
                h: get_f64(&index, arr, "H"),
                g: get_f64(&index, arr, "G"),
                e: get_f64(&index, arr, "e"),
                q: get_f64(&index, arr, "q (AU)"),
                tp: get_f64(&index, arr, "tp (JD)"),
                om: get_f64(&index, arr, "om (deg)"),
                w: get_f64(&index, arr, "w (deg)"),
                i: get_f64(&index, arr, "i (deg)"),
                epoch: get_f64(&index, arr, "epoch (JD)"),
            });
        }
    }

    entries
}

// ── SB Observability ────────────────────────────────────────────────────────

fn parse_observability_response(json: &Value) -> Result<ObservabilityResponse> {
    let night_info = json
        .get("obs_night")
        .map(parse_night_info)
        .unwrap_or_else(default_night_info);

    let total = json
        .get("total_objects")
        .and_then(|v| {
            v.as_u64()
                .map(|n| n as u32)
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
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
    let mut objects = Vec::new();

    if let Some(rows) = data {
        for row in rows {
            if let Some(arr) = row.as_array() {
                objects.push(parse_observable_object(&index, arr));
            }
        }
    }

    Ok(ObservabilityResponse {
        night_info,
        count: total,
        objects,
    })
}

fn parse_night_info(night: &Value) -> ObservabilityNightInfo {
    ObservabilityNightInfo {
        sun_set: json_str(night, "sun_set"),
        sun_rise: json_str(night, "sun_rise"),
        sun_set_az: json_str(night, "sun_set_az"),
        sun_rise_az: json_str(night, "sun_rise_az"),
        begin_astronomical: json_str(night, "begin_astronomical"),
        end_astronomical: json_str(night, "end_astronomical"),
        begin_civil: json_str(night, "begin_civil"),
        end_civil: json_str(night, "end_civil"),
        begin_nautical: json_str(night, "begin_nautical"),
        end_nautical: json_str(night, "end_nautical"),
        moon_rise: json_str(night, "moon_rise"),
        moon_rise_phase: json_str(night, "moon_rise_phase"),
        moon_set: json_str(night, "moon_set"),
        moon_set_phase: json_str(night, "moon_set_phase"),
        transit: json_str(night, "transit"),
        transit_phase: json_str(night, "transit_phase"),
        begin_dark: json_str(night, "begin_dark"),
        mid_dark: json_str(night, "mid_dark"),
        end_dark: json_str(night, "end_dark"),
        dark_time: json_str(night, "dark_time"),
    }
}

fn default_night_info() -> ObservabilityNightInfo {
    ObservabilityNightInfo {
        sun_set: None,
        sun_rise: None,
        sun_set_az: None,
        sun_rise_az: None,
        begin_astronomical: None,
        end_astronomical: None,
        begin_civil: None,
        end_civil: None,
        begin_nautical: None,
        end_nautical: None,
        moon_rise: None,
        moon_rise_phase: None,
        moon_set: None,
        moon_set_phase: None,
        transit: None,
        transit_phase: None,
        begin_dark: None,
        mid_dark: None,
        end_dark: None,
        dark_time: None,
    }
}

fn parse_observable_object(index: &HashMap<&str, usize>, row: &[Value]) -> ObservableObject {
    ObservableObject {
        des: get_str(index, row, "des").unwrap_or_default(),
        fullname: get_str(index, row, "fullname"),
        rise: get_str(index, row, "rise"),
        transit: get_str(index, row, "trans"),
        set: get_str(index, row, "set"),
        max_time: get_str(index, row, "maxt"),
        ra: get_str(index, row, "ra"),
        dec: get_str(index, row, "dec"),
        vmag: get_f64(index, row, "vmag"),
        helio_range_au: get_f64(index, row, "helio"),
        topo_range_au: get_f64(index, row, "topo"),
        sun_angle: get_f64(index, row, "oes"),
        moon_angle: get_f64(index, row, "oem"),
        galactic_lat: get_f64(index, row, "glat"),
    }
}

// ── NHATS ────────────────────────────────────────────────────────────────────

use crate::sbdb::types::{
    NhatsDvDur, NhatsObjectResponse, NhatsParams, NhatsSummaryEntry, NhatsSummaryResponse,
    NhatsTrajectory,
};

fn parse_nhats_dv_dur(obj: &Value) -> NhatsDvDur {
    NhatsDvDur {
        dv: json_str(obj, "dv").and_then(|s| s.parse().ok()),
        dur: json_str(obj, "dur").and_then(|s| s.parse().ok()),
    }
}

fn parse_nhats_trajectory(obj: &Value) -> NhatsTrajectory {
    NhatsTrajectory {
        tid: json_str(obj, "tid"),
        dv_total: json_str(obj, "dv_total").and_then(|s| s.parse().ok()),
        dur_total: json_str(obj, "dur_total").and_then(|s| s.parse().ok()),
        dur_out: json_str(obj, "dur_out").and_then(|s| s.parse().ok()),
        dur_at: json_str(obj, "dur_at").and_then(|s| s.parse().ok()),
        dur_ret: json_str(obj, "dur_ret").and_then(|s| s.parse().ok()),
        launch: json_str(obj, "launch"),
        c3: json_str(obj, "c3").and_then(|s| s.parse().ok()),
        v_dep_earth: json_str(obj, "v_dep_earth").and_then(|s| s.parse().ok()),
        dv_dep_park: json_str(obj, "dv_dep_park").and_then(|s| s.parse().ok()),
        vrel_arr_neo: json_str(obj, "vrel_arr_neo").and_then(|s| s.parse().ok()),
        vrel_dep_neo: json_str(obj, "vrel_dep_neo").and_then(|s| s.parse().ok()),
        vrel_arr_earth: json_str(obj, "vrel_arr_earth").and_then(|s| s.parse().ok()),
        v_arr_earth: json_str(obj, "v_arr_earth").and_then(|s| s.parse().ok()),
        dec_dep: json_str(obj, "dec_dep").and_then(|s| s.parse().ok()),
        dec_arr: json_str(obj, "dec_arr").and_then(|s| s.parse().ok()),
    }
}

fn parse_nhats_summary_response(json: &Value) -> Result<NhatsSummaryResponse> {
    let count = parse_count(json);
    let mut data = Vec::new();

    if let Some(arr) = json.get("data").and_then(|d| d.as_array()) {
        for item in arr {
            let n_via_traj = json_str(item, "n_via_traj")
                .and_then(|s| s.parse().ok())
                .or_else(|| {
                    item.get("n_via_traj")
                        .and_then(|v| v.as_u64())
                        .map(|n| n as u32)
                });

            data.push(NhatsSummaryEntry {
                des: json_str(item, "des").unwrap_or_default(),
                fullname: json_str(item, "fullname"),
                orbit_id: json_str(item, "orbit_id"),
                h: json_str(item, "h").and_then(|s| s.parse().ok()),
                min_size: json_str(item, "min_size").and_then(|s| s.parse().ok()),
                max_size: json_str(item, "max_size").and_then(|s| s.parse().ok()),
                size: json_str(item, "size").and_then(|s| s.parse().ok()),
                occ: json_str(item, "occ").and_then(|s| s.parse().ok()),
                min_dv: item.get("min_dv").map(parse_nhats_dv_dur),
                min_dur: item.get("min_dur").map(parse_nhats_dv_dur),
                n_via_traj,
                obs_start: json_str(item, "obs_start"),
                obs_end: json_str(item, "obs_end"),
            });
        }
    }

    Ok(NhatsSummaryResponse { count, data })
}

fn parse_nhats_object_response(json: &Value) -> Result<NhatsObjectResponse> {
    let n_via_traj = json_str(json, "n_via_traj")
        .and_then(|s| s.parse().ok())
        .or_else(|| {
            json.get("n_via_traj")
                .and_then(|v| v.as_u64())
                .map(|n| n as u32)
        });

    Ok(NhatsObjectResponse {
        des: json_str(json, "des").unwrap_or_default(),
        fullname: json_str(json, "fullname"),
        orbit_id: json_str(json, "orbit_id"),
        h: json_str(json, "h").and_then(|s| s.parse().ok()),
        min_size: json_str(json, "min_size").and_then(|s| s.parse().ok()),
        max_size: json_str(json, "max_size").and_then(|s| s.parse().ok()),
        size: json_str(json, "size").and_then(|s| s.parse().ok()),
        occ: json_str(json, "occ").and_then(|s| s.parse().ok()),
        n_via_traj,
        min_dv_traj: json.get("min_dv_traj").map(parse_nhats_trajectory),
        min_dur_traj: json.get("min_dur_traj").map(parse_nhats_trajectory),
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
    fn test_parse_scout_summary_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "data": [
                {
                    "objectName": "P10uUSw",
                    "nObs": 12,
                    "arc": 1.5,
                    "rmsN": 0.42,
                    "H": 25.3,
                    "rating": 67,
                    "moid": 0.0012,
                    "caDist": 0.003,
                    "vInf": 12.5,
                    "phaScore": 80,
                    "neoScore": 95,
                    "geocentricScore": 5,
                    "ieoScore": 0,
                    "tisserandScore": 10,
                    "lastRun": "2024-06-15 12:30:00",
                    "ra": "180.5",
                    "dec": "-22.3",
                    "elong": "145.2",
                    "rate": 2.1,
                    "Vmag": 21.5,
                    "unc": 15.3,
                    "uncP1": 45.7
                },
                {
                    "objectName": "Q20xYZa",
                    "nObs": 5,
                    "arc": 0.3,
                    "rmsN": 1.1,
                    "H": 28.0,
                    "rating": 12,
                    "moid": null,
                    "caDist": null,
                    "vInf": null,
                    "phaScore": 0,
                    "neoScore": 50,
                    "geocentricScore": 40,
                    "ieoScore": 0,
                    "tisserandScore": 0,
                    "lastRun": "2024-06-14 08:00:00",
                    "ra": "90.0",
                    "dec": "45.0",
                    "elong": "60.0",
                    "rate": 0.5,
                    "Vmag": 22.8,
                    "unc": 120.0,
                    "uncP1": 500.0
                }
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_scout_summary_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.data.len(), 2);

        let first = &resp.data[0];
        assert_eq!(first.object_name, "P10uUSw");
        assert_eq!(first.n_obs, Some(12));
        assert!((first.arc.unwrap() - 1.5).abs() < 1e-10);
        assert!((first.rms_n.unwrap() - 0.42).abs() < 1e-10);
        assert!((first.h_mag.unwrap() - 25.3).abs() < 0.1);
        assert_eq!(first.rating, Some(67));
        assert!((first.moid.unwrap() - 0.0012).abs() < 1e-10);
        assert!((first.ca_dist.unwrap() - 0.003).abs() < 1e-10);
        assert!((first.v_inf.unwrap() - 12.5).abs() < 0.1);
        assert_eq!(first.pha_score, Some(80));
        assert_eq!(first.neo_score, Some(95));
        assert_eq!(first.geocentric_score, Some(5));
        assert_eq!(first.ieo_score, Some(0));
        assert_eq!(first.tisserand_score, Some(10));
        assert_eq!(first.last_run.as_deref(), Some("2024-06-15 12:30:00"));
        assert_eq!(first.ra.as_deref(), Some("180.5"));
        assert_eq!(first.dec.as_deref(), Some("-22.3"));
        assert_eq!(first.elong.as_deref(), Some("145.2"));
        assert!((first.rate.unwrap() - 2.1).abs() < 0.1);
        assert!((first.v_mag.unwrap() - 21.5).abs() < 0.1);
        assert!((first.unc.unwrap() - 15.3).abs() < 0.1);
        assert!((first.unc_p1.unwrap() - 45.7).abs() < 0.1);

        let second = &resp.data[1];
        assert_eq!(second.object_name, "Q20xYZa");
        assert_eq!(second.n_obs, Some(5));
        assert_eq!(second.moid, None);
        assert_eq!(second.ca_dist, None);
        assert_eq!(second.v_inf, None);
    }

    #[test]
    fn test_parse_scout_summary_string_values() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "1",
            "data": [
                {
                    "objectName": "A10bCdE",
                    "nObs": "8",
                    "arc": "2.5",
                    "rmsN": "0.55",
                    "H": "26.1",
                    "rating": "45",
                    "moid": "0.005",
                    "caDist": "0.01",
                    "vInf": "8.2",
                    "phaScore": "60",
                    "neoScore": "70",
                    "geocentricScore": "15",
                    "ieoScore": "3",
                    "tisserandScore": "5",
                    "lastRun": "2024-07-01 00:00:00",
                    "ra": "270.0",
                    "dec": "10.0",
                    "elong": "90.0",
                    "rate": "1.0",
                    "Vmag": "20.0",
                    "unc": "30.0",
                    "uncP1": "100.0"
                }
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_scout_summary_response(&json).unwrap();
        assert_eq!(resp.count, 1);
        let entry = &resp.data[0];
        assert_eq!(entry.object_name, "A10bCdE");
        assert_eq!(entry.n_obs, Some(8));
        assert!((entry.arc.unwrap() - 2.5).abs() < 1e-10);
        assert_eq!(entry.pha_score, Some(60));
        assert_eq!(entry.ieo_score, Some(3));
    }

    #[test]
    fn test_parse_scout_object_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "objectName": "P10uUSw",
            "nObs": 12,
            "arc": 1.5,
            "rmsN": 0.42,
            "H": 25.3,
            "rating": 67,
            "moid": 0.0012,
            "caDist": 0.003,
            "vInf": 12.5,
            "phaScore": 80,
            "neoScore": 95,
            "geocentricScore": 5,
            "ieoScore": 0,
            "tisserandScore": 10,
            "lastRun": "2024-06-15 12:30:00",
            "ra": "180.5",
            "dec": "-22.3",
            "elong": "145.2",
            "rate": 2.1,
            "Vmag": 21.5,
            "unc": 15.3,
            "uncP1": 45.7,
            "neo1kmScore": "0.015",
            "tEphem": "2024-06-15 12:00:00",
            "orbits": {
                "count": "3",
                "fields": ["idx", "epoch", "ec", "qr", "tp", "om", "w", "inc", "H"],
                "data": [
                    [0, "2460476.5", "0.35", "0.85", "2460450.0", "120.5", "45.2", "12.3", "25.3"],
                    [1, "2460476.5", "0.42", "0.90", "2460451.0", "121.0", "46.0", "13.0", "25.5"],
                    [2, "2460476.5", "0.30", "0.80", "2460449.0", "119.8", "44.5", "11.8", "25.1"]
                ]
            }
        }"#,
        )
        .unwrap();

        let resp = parse_scout_object_response(&json).unwrap();
        assert_eq!(resp.detail.summary.object_name, "P10uUSw");
        assert_eq!(resp.detail.summary.n_obs, Some(12));
        assert!((resp.detail.summary.h_mag.unwrap() - 25.3).abs() < 0.1);
        assert_eq!(resp.detail.neo1km_score.as_deref(), Some("0.015"));
        assert_eq!(resp.detail.t_ephem.as_deref(), Some("2024-06-15 12:00:00"));

        let orbits = resp.detail.orbits.unwrap();
        assert_eq!(orbits.count, 3);
        assert_eq!(orbits.fields.len(), 9);
        assert_eq!(orbits.fields[0], "idx");
        assert_eq!(orbits.data.len(), 3);
    }

    #[test]
    fn test_parse_scout_object_no_orbits() {
        let json: Value = serde_json::from_str(
            r#"{
            "objectName": "Z99test",
            "nObs": 3,
            "arc": 0.1,
            "H": 30.0,
            "rating": 5,
            "neoScore": 10,
            "lastRun": "2024-01-01 00:00:00"
        }"#,
        )
        .unwrap();

        let resp = parse_scout_object_response(&json).unwrap();
        assert_eq!(resp.detail.summary.object_name, "Z99test");
        assert_eq!(resp.detail.summary.n_obs, Some(3));
        assert!(resp.detail.orbits.is_none());
        assert!(resp.detail.neo1km_score.is_none());
        assert!(resp.detail.t_ephem.is_none());
    }

    #[test]
    fn test_parse_scout_summary_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "0",
            "data": []
        }"#,
        )
        .unwrap();

        let resp = parse_scout_summary_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.data.is_empty());
    }

    #[test]
    #[ignore]
    fn test_scout_api_live_summary() {
        let client = SbdbClient::new().unwrap();
        let resp = client.scout_summary().unwrap();
        assert!(resp.count > 0 || resp.data.is_empty());
        for entry in &resp.data {
            assert!(!entry.object_name.is_empty());
        }
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

    #[test]
    fn test_parse_mission_accessible_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "fields": ["name", "pdes", "date0", "MJD0", "datef", "MJDF", "c3_dep", "vinf_dep", "vinf_arr", "dv_tot", "tof", "class", "H", "condition_code", "neo", "pha"],
            "data": [
                ["2000 SG344", "2000 SG344", "2029-Apr-17", "62573.0", "2030-Jan-08", "62839.0", "0.112", "0.335", "0.674", "1.009", "266.0", "APO", "24.7", "1", "Y", "N"],
                ["2015 JD3", "2015 JD3", "2029-May-01", "62587.0", "2030-Mar-22", "62912.0", "0.542", "0.737", "1.231", "1.968", "325.0", "APO", "28.4", "6", "Y", "N"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_mission_accessible_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].name, "2000 SG344");
        assert_eq!(resp.data[0].pdes.as_deref(), Some("2000 SG344"));
        assert_eq!(resp.data[0].date0, "2029-Apr-17");
        assert!((resp.data[0].mjd0 - 62573.0).abs() < 1e-10);
        assert!((resp.data[0].c3_dep - 0.112).abs() < 1e-10);
        assert!((resp.data[0].vinf_dep - 0.335).abs() < 1e-10);
        assert!((resp.data[0].vinf_arr - 0.674).abs() < 1e-10);
        assert!((resp.data[0].dv_tot - 1.009).abs() < 1e-10);
        assert!((resp.data[0].tof - 266.0).abs() < 1e-10);
        assert_eq!(resp.data[0].class.as_deref(), Some("APO"));
        assert!((resp.data[0].h_mag.unwrap() - 24.7).abs() < 0.1);
        assert_eq!(resp.data[0].condition_code.as_deref(), Some("1"));
        assert!(resp.data[0].neo);
        assert!(!resp.data[0].pha);
        assert!(!resp.data[1].pha);
    }

    #[test]
    fn test_parse_mission_query_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "object": {
                "des": "433",
                "fullname": "433 Eros (A898 PA)",
                "spkid": "2000433",
                "orbit_class": "AMO",
                "condition_code": "0",
                "data_arc": "46857",
                "orbit_id": "780",
                "md_orbit_id": "780"
            },
            "fields": ["MJD0", "MJDf", "vinf_dep", "vinf_arr", "phase_ang", "earth_dist", "elong_arr", "decl_dep", "approach_ang"],
            "selectedMissions": [
                [60300.0, 60500.0, 5.12, 8.45, 32.1, 1.23, 145.6, -12.3, 78.9],
                [60400.0, 60650.0, 4.98, 7.32, 28.5, 0.98, 160.2, 5.6, 82.1]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_mission_query_response(&json).unwrap();
        assert_eq!(resp.object.des, "433");
        assert_eq!(resp.object.fullname.as_deref(), Some("433 Eros (A898 PA)"));
        assert_eq!(resp.object.spkid.as_deref(), Some("2000433"));
        assert_eq!(resp.object.orbit_class.as_deref(), Some("AMO"));
        assert_eq!(resp.object.condition_code.as_deref(), Some("0"));
        assert_eq!(resp.object.orbit_id.as_deref(), Some("780"));
        assert_eq!(resp.fields.len(), 9);
        assert_eq!(resp.fields[0], "MJD0");
        assert_eq!(resp.selected_missions.len(), 2);
        assert!((resp.selected_missions[0][0] - 60300.0).abs() < 1e-10);
        assert!((resp.selected_missions[0][2] - 5.12).abs() < 1e-10);
        assert!((resp.selected_missions[1][3] - 7.32).abs() < 1e-10);
    }

    #[test]
    fn test_parse_mission_flyby_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "fields": ["full_name", "pdes", "spkid", "date", "jd", "min_dist_au", "min_dist_km", "rel_vel", "class", "H", "condition_code", "neo", "pha"],
            "data": [
                ["(2015 JD3)", "2015 JD3", "3713011", "2030-Mar-22", "2462912.5", "0.00321", "480230", "2.15", "APO", "28.4", "6", "Y", "N"],
                ["433 Eros (A898 PA)", "433", "2000433", "2030-Jun-10", "2462992.5", "0.15234", "22785000", "8.91", "AMO", "11.2", "0", "Y", "N"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_mission_flyby_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].full_name, "(2015 JD3)");
        assert_eq!(resp.data[0].pdes.as_deref(), Some("2015 JD3"));
        assert_eq!(resp.data[0].spkid.as_deref(), Some("3713011"));
        assert_eq!(resp.data[0].date, "2030-Mar-22");
        assert!((resp.data[0].jd - 2462912.5).abs() < 1e-10);
        assert!((resp.data[0].min_dist_au - 0.00321).abs() < 1e-10);
        assert!((resp.data[0].min_dist_km.unwrap() - 480230.0).abs() < 1.0);
        assert!((resp.data[0].rel_vel - 2.15).abs() < 1e-10);
        assert_eq!(resp.data[0].class.as_deref(), Some("APO"));
        assert!(resp.data[0].neo);
        assert!(!resp.data[0].pha);
        assert_eq!(resp.data[1].full_name, "433 Eros (A898 PA)");
        assert!((resp.data[1].h_mag.unwrap() - 11.2).abs() < 0.1);
    }

    #[test]
    fn test_parse_mission_accessible_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "0",
            "fields": ["name", "date0", "MJD0"],
            "data": []
        }"#,
        )
        .unwrap();

        let resp = parse_mission_accessible_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.data.is_empty());
    }

    #[test]
    fn test_parse_mission_flyby_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "0",
            "fields": ["full_name", "date", "jd"],
            "data": []
        }"#,
        )
        .unwrap();

        let resp = parse_mission_flyby_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.data.is_empty());
    }

    #[test]
    #[ignore]
    fn test_mdesign_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(MDESIGN_API_URL)
            .send()
            .expect("Mission Design API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    fn test_radar_params_default() {
        let params = RadarParams::default();
        assert!(params.to_query_params().is_empty());
    }

    #[test]
    fn test_radar_params_with_filters() {
        let params = RadarParams {
            des: Some("433".into()),
            measurement_type: Some("R".into()),
            fullname: true,
            observer: true,
            coords: true,
            ..Default::default()
        };
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("des").unwrap(), "433");
        assert_eq!(map.get("type").unwrap(), "R");
        assert_eq!(map.get("fullname").unwrap(), "true");
        assert_eq!(map.get("observer").unwrap(), "true");
        assert_eq!(map.get("coords").unwrap(), "true");
    }

    #[test]
    fn test_parse_radar_response_standard() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "2",
            "fields": ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp"],
            "data": [
                ["433", "2005-Jan-26 07:29", "22.34560", "0.50000", "us", "8560", "-14", "-14", "C"],
                ["433", "2005-Jan-26 07:29", "-19.52300", "0.10000", "Hz", "8560", "-14", "-14", "C"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_radar_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.records.len(), 2);
        assert_eq!(resp.records[0].designation, "433");
        assert_eq!(resp.records[0].epoch, "2005-Jan-26 07:29");
        assert!((resp.records[0].value.unwrap() - 22.34560).abs() < 1e-5);
        assert!((resp.records[0].sigma.unwrap() - 0.5).abs() < 1e-5);
        assert_eq!(resp.records[0].units.as_deref(), Some("us"));
        assert!((resp.records[0].freq.unwrap() - 8560.0).abs() < 1e-1);
        assert_eq!(resp.records[0].rcvr.as_deref(), Some("-14"));
        assert_eq!(resp.records[0].xmit.as_deref(), Some("-14"));
        assert_eq!(resp.records[0].bp.as_deref(), Some("C"));

        assert_eq!(resp.records[1].units.as_deref(), Some("Hz"));
        assert!((resp.records[1].value.unwrap() - (-19.523)).abs() < 1e-3);
    }

    #[test]
    fn test_parse_radar_response_with_optional_fields() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "1",
            "fields": ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp", "observer", "notes", "ref", "fullname", "modified", "longitude", "latitude", "altitude"],
            "data": [
                ["1566", "1968-Jun-14 00:00", "0.14270", "0.00060", "us", "2380", "-1", "-1", "C", "R. Goldstein", "first asteroid detection", "Goldstein (1968)", "1566 Icarus", "2023-01-15", "243.205", "35.426", "1000.0"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_radar_response(&json).unwrap();
        assert_eq!(resp.count, 1);
        assert_eq!(resp.records.len(), 1);

        let rec = &resp.records[0];
        assert_eq!(rec.designation, "1566");
        assert_eq!(rec.observer.as_deref(), Some("R. Goldstein"));
        assert_eq!(rec.notes.as_deref(), Some("first asteroid detection"));
        assert_eq!(rec.reference.as_deref(), Some("Goldstein (1968)"));
        assert_eq!(rec.fullname.as_deref(), Some("1566 Icarus"));
        assert_eq!(rec.modified.as_deref(), Some("2023-01-15"));
        assert!((rec.longitude.unwrap() - 243.205).abs() < 1e-3);
        assert!((rec.latitude.unwrap() - 35.426).abs() < 1e-3);
        assert!((rec.altitude.unwrap() - 1000.0).abs() < 1e-1);
    }

    #[test]
    fn test_parse_radar_response_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": "0",
            "fields": ["des", "epoch", "value", "sigma", "units", "freq", "rcvr", "xmit", "bp"],
            "data": []
        }"#,
        )
        .unwrap();

        let resp = parse_radar_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.records.is_empty());
    }

    #[test]
    #[ignore]
    fn test_radar_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(RADAR_API_URL)
            .send()
            .expect("Radar API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    #[ignore]
    fn test_radar_api_live_query() {
        let client = SbdbClient::new().unwrap();
        let params = RadarParams {
            des: Some("433".into()),
            ..Default::default()
        };
        let resp = client.radar(&params).unwrap();
        assert!(resp.count > 0, "Expected radar data for asteroid 433 Eros");
        assert_eq!(resp.records.len(), resp.count as usize);
        assert_eq!(resp.records[0].designation, "433");
    }

    #[test]
    fn test_parse_sb_ident_response_first_pass() {
        let json: Value = serde_json::from_str(
            r#"{
            "signature": {"version": "1.1", "source": "NASA/JPL Small-Body Identification API"},
            "observer": {
                "obs_date": "2024-01-01 00:00:00",
                "location": "F51 (Pan-STARRS 1)",
                "fov_center": "05 00 00.0 +20 00 00.0",
                "fov_offset": "1.0 x 1.0 deg",
                "frame": "J2000"
            },
            "n_first_pass": 3,
            "n_second_pass": 0,
            "fields_first": ["Object name", "Astrometric RA", "Astrometric Dec", "RA offset (arcsec)", "Dec offset (arcsec)", "total offset (arcsec)", "visual magnitude V", "RA rate (deg/sec)", "Dec rate (deg/sec)", "RA error estimate (arcsec)", "Dec error estimate (arcsec)"],
            "data_first_pass": [
                ["(1036) Ganymed", "04 58 12.34", "+19 45 30.1", "-108.5", "-870.0", "876.7", "15.2", "0.000023", "-0.000012", "2.5", "1.8"],
                ["(4179) Toutatis", "05 01 45.67", "+20 12 15.3", "26.3", "735.0", "735.5", "18.7", "0.000015", "0.000008", "5.1", "3.2"],
                ["2024 AA1", "05 00 30.00", "+20 05 00.0", "7.5", "300.0", "300.1", "21.5", "0.000045", "-0.000020", "12.0", "10.0"]
            ],
            "data_second_pass": null
        }"#,
        )
        .unwrap();

        let resp = parse_sb_ident_response(&json).unwrap();
        assert_eq!(resp.n_first_pass, 3);
        assert_eq!(resp.n_second_pass, 0);
        assert_eq!(resp.data_first_pass.len(), 3);
        assert!(resp.data_second_pass.is_empty());

        assert_eq!(
            resp.observer.obs_date.as_deref(),
            Some("2024-01-01 00:00:00")
        );
        assert_eq!(resp.observer.frame.as_deref(), Some("J2000"));

        let entry = &resp.data_first_pass[0];
        assert_eq!(entry.name, "(1036) Ganymed");
        assert_eq!(entry.ra.as_deref(), Some("04 58 12.34"));
        assert_eq!(entry.dec.as_deref(), Some("+19 45 30.1"));
        assert!((entry.ra_offset.unwrap() - (-108.5)).abs() < 1e-10);
        assert!((entry.total_offset.unwrap() - 876.7).abs() < 1e-10);
        assert!((entry.vmag.unwrap() - 15.2).abs() < 1e-10);
        assert!(entry.ra_rate.is_some());
        assert!(entry.dec_rate.is_some());
        assert!(entry.ra_err.is_some());
        assert!(entry.dec_err.is_some());
    }

    #[test]
    fn test_parse_sb_ident_response_two_pass() {
        let json: Value = serde_json::from_str(
            r#"{
            "observer": {
                "obs_date": "2024-06-15 12:00:00",
                "location": "Geocentric"
            },
            "n_first_pass": 2,
            "n_second_pass": 1,
            "fields_first": ["Object name", "Astrometric RA", "Astrometric Dec", "visual magnitude V"],
            "data_first_pass": [
                ["(433) Eros", "12 30 00.00", "+05 15 00.0", "12.5"],
                ["(1862) Apollo", "12 31 00.00", "+05 10 00.0", "14.8"]
            ],
            "fields_second": ["Object name", "Astrometric RA", "Astrometric Dec", "visual magnitude V"],
            "data_second_pass": [
                ["(433) Eros", "12 30 00.12", "+05 14 59.8", "12.5"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_sb_ident_response(&json).unwrap();
        assert_eq!(resp.n_first_pass, 2);
        assert_eq!(resp.n_second_pass, 1);
        assert_eq!(resp.data_first_pass.len(), 2);
        assert_eq!(resp.data_second_pass.len(), 1);
        assert_eq!(resp.data_second_pass[0].name, "(433) Eros");
    }

    #[test]
    fn test_parse_sb_ident_response_with_elements() {
        let json: Value = serde_json::from_str(
            r#"{
            "observer": {"obs_date": "2024-01-01"},
            "n_first_pass": 1,
            "n_second_pass": 0,
            "fields_first": ["Object name", "visual magnitude V"],
            "data_first_pass": [
                ["(433) Eros", "12.5"]
            ],
            "elem_fields_first": ["Object name", "H", "G", "e", "q (AU)", "tp (JD)", "om (deg)", "w (deg)", "i (deg)", "epoch (JD)"],
            "elem_first_pass": [
                ["(433) Eros", "10.31", "0.46", "0.2229", "1.1334", "2460200.5", "304.32", "178.82", "10.83", "2460400.5"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_sb_ident_response(&json).unwrap();
        assert_eq!(resp.elem_first_pass.len(), 1);
        let elem = &resp.elem_first_pass[0];
        assert_eq!(elem.name, "(433) Eros");
        assert!((elem.h.unwrap() - 10.31).abs() < 1e-10);
        assert!((elem.g.unwrap() - 0.46).abs() < 1e-10);
        assert!((elem.e.unwrap() - 0.2229).abs() < 1e-4);
        assert!((elem.q.unwrap() - 1.1334).abs() < 1e-4);
        assert!((elem.i.unwrap() - 10.83).abs() < 0.01);
    }

    #[test]
    fn test_parse_sb_ident_response_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "observer": {},
            "n_first_pass": 0,
            "n_second_pass": 0
        }"#,
        )
        .unwrap();

        let resp = parse_sb_ident_response(&json).unwrap();
        assert_eq!(resp.n_first_pass, 0);
        assert_eq!(resp.n_second_pass, 0);
        assert!(resp.data_first_pass.is_empty());
        assert!(resp.data_second_pass.is_empty());
        assert!(resp.elem_first_pass.is_empty());
        assert!(resp.elem_second_pass.is_empty());
    }

    #[test]
    #[ignore]
    fn test_sb_ident_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(SB_IDENT_API_URL)
            .send()
            .expect("SB Ident API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    #[ignore]
    fn test_sb_ident_api_live_query() {
        let client = SbdbClient::new().unwrap();
        let params = SbIdentParams {
            observer: SbIdentObserver::MpcCode("F51".into()),
            fov: SbIdentFov::Center {
                ra_center: "05-00-00".into(),
                dec_center: "20-00-00".into(),
                ra_hwidth: Some(0.5),
                dec_hwidth: Some(0.5),
            },
            obs_time: "2024-01-01".into(),
            vmag_lim: Some(22.0),
            two_pass: false,
            mag_required: Some(true),
            sb_kind: Some("a".into()),
            sb_group: None,
            req_elem: false,
        };
        let resp = client.identify(&params).unwrap();
        assert!(resp.n_first_pass > 0 || resp.data_first_pass.is_empty());
        assert!(resp.observer.obs_date.is_some());
    }

    #[test]
    fn test_parse_observability_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "signature": {"version": "1.0", "source": "NASA/JPL ... API"},
            "location": "Mt. Lemmon Survey (G96)",
            "total_objects": 2,
            "shown_objects": 2,
            "obs_night": {
                "sun_set": "2026-Mar-01 01:15",
                "sun_rise": "2026-Mar-01 13:30",
                "sun_set_az": "258.3",
                "sun_rise_az": "101.7",
                "begin_astronomical": "2026-Mar-01 02:45",
                "end_astronomical": "2026-Mar-01 12:00",
                "begin_civil": "2026-Mar-01 01:40",
                "end_civil": "2026-Mar-01 13:05",
                "begin_nautical": "2026-Mar-01 02:12",
                "end_nautical": "2026-Mar-01 12:33",
                "moon_rise": "2026-Mar-01 15:20",
                "moon_rise_phase": "0.85",
                "moon_set": "2026-Mar-01 04:10",
                "moon_set_phase": "0.84",
                "transit": "2026-Mar-01 09:45",
                "transit_phase": "0.85",
                "begin_dark": "2026-Mar-01 04:10",
                "mid_dark": "2026-Mar-01 07:20",
                "end_dark": "2026-Mar-01 12:00",
                "dark_time": "7.83"
            },
            "fields": ["des", "fullname", "rise", "trans", "set", "maxt", "ra", "dec", "vmag", "helio", "topo", "oes", "oem", "glat"],
            "data": [
                ["1", "1 Ceres", "2026-Mar-01 02:00", "2026-Mar-01 07:30", "2026-Mar-01 12:00", "10:00", "06 45 12.3", "+23 15 45", "8.5", "2.769", "2.105", "145.2", "67.8", "12.3"],
                ["4", "4 Vesta", "2026-Mar-01 03:00", "2026-Mar-01 08:00", "2026-Mar-01 11:30", "08:30", "10 12 34.5", "-05 30 22", "7.2", "1.876", "1.234", "160.5", "89.1", "-25.7"]
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_observability_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.objects.len(), 2);

        // Night info
        assert_eq!(
            resp.night_info.sun_set.as_deref(),
            Some("2026-Mar-01 01:15")
        );
        assert_eq!(
            resp.night_info.sun_rise.as_deref(),
            Some("2026-Mar-01 13:30")
        );
        assert_eq!(
            resp.night_info.begin_astronomical.as_deref(),
            Some("2026-Mar-01 02:45")
        );
        assert_eq!(resp.night_info.moon_rise_phase.as_deref(), Some("0.85"));
        assert_eq!(resp.night_info.dark_time.as_deref(), Some("7.83"));

        // First object: Ceres
        let ceres = &resp.objects[0];
        assert_eq!(ceres.des, "1");
        assert_eq!(ceres.fullname.as_deref(), Some("1 Ceres"));
        assert_eq!(ceres.rise.as_deref(), Some("2026-Mar-01 02:00"));
        assert_eq!(ceres.transit.as_deref(), Some("2026-Mar-01 07:30"));
        assert_eq!(ceres.ra.as_deref(), Some("06 45 12.3"));
        assert!((ceres.vmag.unwrap() - 8.5).abs() < 0.01);
        assert!((ceres.helio_range_au.unwrap() - 2.769).abs() < 0.001);
        assert!((ceres.topo_range_au.unwrap() - 2.105).abs() < 0.001);
        assert!((ceres.sun_angle.unwrap() - 145.2).abs() < 0.1);
        assert!((ceres.moon_angle.unwrap() - 67.8).abs() < 0.1);
        assert!((ceres.galactic_lat.unwrap() - 12.3).abs() < 0.1);

        // Second object: Vesta
        let vesta = &resp.objects[1];
        assert_eq!(vesta.des, "4");
        assert!((vesta.vmag.unwrap() - 7.2).abs() < 0.01);
        assert!((vesta.galactic_lat.unwrap() - (-25.7)).abs() < 0.1);
    }

    #[test]
    fn test_parse_observability_response_empty() {
        let json: Value = serde_json::from_str(
            r#"{
            "signature": {"version": "1.0", "source": "NASA/JPL ... API"},
            "total_objects": 0,
            "shown_objects": 0,
            "fields": ["des", "fullname", "rise", "trans", "set", "maxt", "ra", "dec", "vmag", "helio", "topo", "oes", "oem", "glat"],
            "data": []
        }"#,
        )
        .unwrap();

        let resp = parse_observability_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.objects.is_empty());
        assert!(resp.night_info.sun_set.is_none());
    }

    #[test]
    #[ignore]
    fn test_sbwobs_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(SBWOBS_API_URL)
            .send()
            .expect("SBWOBS API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    fn test_nhats_params_default() {
        let params = NhatsParams::default();
        assert!(params.to_query_params().is_empty());
    }

    #[test]
    fn test_nhats_params_with_filters() {
        let params = NhatsParams {
            dv: Some(6),
            dur: Some(360),
            stay: Some(16),
            launch: Some("2025-2040".into()),
            h: Some(26),
            occ: Some(6),
        };
        let query = params.to_query_params();
        let map: HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("dv").unwrap(), "6");
        assert_eq!(map.get("dur").unwrap(), "360");
        assert_eq!(map.get("stay").unwrap(), "16");
        assert_eq!(map.get("launch").unwrap(), "2025-2040");
        assert_eq!(map.get("h").unwrap(), "26");
        assert_eq!(map.get("occ").unwrap(), "6");
    }

    #[test]
    fn test_parse_nhats_summary_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "count": 2,
            "data": [
                {
                    "des": "2000 SG344",
                    "fullname": "2000 SG344",
                    "orbit_id": "178",
                    "h": "24.7",
                    "min_size": "0.024",
                    "max_size": "0.054",
                    "occ": "0",
                    "min_dv": {"dv": "3.961", "dur": "338.0"},
                    "min_dur": {"dv": "6.474", "dur": "75.0"},
                    "n_via_traj": 198,
                    "obs_start": "2028-01-15",
                    "obs_end": "2029-06-10"
                },
                {
                    "des": "2006 RH120",
                    "h": "29.5",
                    "n_via_traj": "42"
                }
            ]
        }"#,
        )
        .unwrap();

        let resp = parse_nhats_summary_response(&json).unwrap();
        assert_eq!(resp.count, 2);
        assert_eq!(resp.data.len(), 2);
        assert_eq!(resp.data[0].des, "2000 SG344");
        assert!((resp.data[0].h.unwrap() - 24.7).abs() < 0.1);
        assert!((resp.data[0].min_dv.as_ref().unwrap().dv.unwrap() - 3.961).abs() < 0.001);
        assert_eq!(resp.data[0].n_via_traj, Some(198));
        assert_eq!(resp.data[0].obs_start.as_deref(), Some("2028-01-15"));
        assert_eq!(resp.data[1].des, "2006 RH120");
        assert_eq!(resp.data[1].n_via_traj, Some(42));
    }

    #[test]
    fn test_parse_nhats_object_response() {
        let json: Value = serde_json::from_str(
            r#"{
            "des": "2000 SG344",
            "fullname": "2000 SG344",
            "orbit_id": "178",
            "h": "24.7",
            "min_size": "0.024",
            "max_size": "0.054",
            "occ": "0",
            "n_via_traj": 198,
            "min_dv_traj": {
                "tid": "1234",
                "dv_total": "3.961",
                "dur_total": "338.0",
                "dur_out": "120.0",
                "dur_at": "16.0",
                "dur_ret": "202.0",
                "launch": "2028-06-15",
                "c3": "1.5",
                "v_dep_earth": "1.22",
                "dv_dep_park": "3.20",
                "vrel_arr_neo": "0.38",
                "vrel_dep_neo": "0.38",
                "vrel_arr_earth": "2.15",
                "v_arr_earth": "11.25",
                "dec_dep": "28.5",
                "dec_arr": "-15.2"
            },
            "min_dur_traj": {
                "tid": "5678",
                "dv_total": "6.474",
                "dur_total": "75.0"
            }
        }"#,
        )
        .unwrap();

        let resp = parse_nhats_object_response(&json).unwrap();
        assert_eq!(resp.des, "2000 SG344");
        assert!((resp.h.unwrap() - 24.7).abs() < 0.1);
        assert_eq!(resp.n_via_traj, Some(198));

        let min_dv = resp.min_dv_traj.as_ref().unwrap();
        assert_eq!(min_dv.tid.as_deref(), Some("1234"));
        assert!((min_dv.dv_total.unwrap() - 3.961).abs() < 0.001);
        assert!((min_dv.dur_out.unwrap() - 120.0).abs() < 0.1);
        assert!((min_dv.c3.unwrap() - 1.5).abs() < 0.1);
        assert!((min_dv.dec_dep.unwrap() - 28.5).abs() < 0.1);

        let min_dur = resp.min_dur_traj.as_ref().unwrap();
        assert_eq!(min_dur.tid.as_deref(), Some("5678"));
        assert!((min_dur.dv_total.unwrap() - 6.474).abs() < 0.001);
    }

    #[test]
    fn test_parse_nhats_summary_empty() {
        let json: Value = serde_json::from_str(r#"{"count": 0, "data": []}"#).unwrap();
        let resp = parse_nhats_summary_response(&json).unwrap();
        assert_eq!(resp.count, 0);
        assert!(resp.data.is_empty());
    }

    #[test]
    #[ignore]
    fn test_nhats_api_reachable() {
        let client = reqwest::blocking::Client::new();
        let resp = client
            .head(NHATS_API_URL)
            .send()
            .expect("NHATS API unreachable");
        assert!(resp.status().is_success() || resp.status().as_u16() == 405);
    }

    #[test]
    #[ignore]
    fn test_nhats_summary_live() {
        let client = SbdbClient::new().unwrap();
        let params = NhatsParams {
            dv: Some(6),
            dur: Some(360),
            ..Default::default()
        };
        let resp = client.nhats_summary(&params).unwrap();
        assert!(resp.count > 0);
        assert!(!resp.data.is_empty());
        assert!(!resp.data[0].des.is_empty());
    }

    #[test]
    #[ignore]
    fn test_nhats_object_live() {
        let client = SbdbClient::new().unwrap();
        let resp = client.nhats_object("2000 SG344").unwrap();
        assert_eq!(resp.des, "2000 SG344");
        assert!(resp.n_via_traj.is_some());
    }
}
