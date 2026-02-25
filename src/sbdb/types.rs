//! Shared domain types for the JPL Small-Body Database API ecosystem.
//!
//! These types represent orbital elements, physical parameters, and object
//! identification data common across multiple SBDB API endpoints.

use serde::Deserialize;

/// API response signature present in all SBDB API responses
#[derive(Debug, Clone, Deserialize)]
pub struct Signature {
    pub source: String,
    pub version: String,
}

/// Orbit class of a small body
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrbitClass {
    /// IEO - Atira (Interior Earth Object)
    Atira,
    /// ATE - Aten
    Aten,
    /// APO - Apollo
    Apollo,
    /// AMO - Amor
    Amor,
    /// MCA - Mars-crossing Asteroid
    MarsCrosser,
    /// MBA - Main Belt Asteroid
    MainBelt,
    /// JFC - Jupiter-family Comet
    JupiterFamilyComet,
    /// HTC - Halley-type Comet
    HalleyTypeComet,
    /// ETc - Encke-type Comet
    EnckeTypeComet,
    /// COM - Comet (general)
    Comet,
    /// TJN - Jupiter Trojan
    JupiterTrojan,
    /// CEN - Centaur
    Centaur,
    /// TNO - Trans-Neptunian Object
    TransNeptunian,
    /// AST - Asteroid (generic)
    Asteroid,
    /// PAA - Parabolic Asteroid
    ParabolicAsteroid,
    /// HYA - Hyperbolic Asteroid
    HyperbolicAsteroid,
    /// Unrecognized orbit class code
    Other(String),
}

impl OrbitClass {
    /// Parse an orbit class from the SBDB API code string
    pub fn from_code(code: &str) -> Self {
        match code {
            "IEO" => OrbitClass::Atira,
            "ATE" => OrbitClass::Aten,
            "APO" => OrbitClass::Apollo,
            "AMO" => OrbitClass::Amor,
            "MCA" => OrbitClass::MarsCrosser,
            "MBA" => OrbitClass::MainBelt,
            "JFC" | "JFc" => OrbitClass::JupiterFamilyComet,
            "HTC" => OrbitClass::HalleyTypeComet,
            "ETc" => OrbitClass::EnckeTypeComet,
            "COM" => OrbitClass::Comet,
            "TJN" => OrbitClass::JupiterTrojan,
            "CEN" => OrbitClass::Centaur,
            "TNO" => OrbitClass::TransNeptunian,
            "AST" => OrbitClass::Asteroid,
            "PAA" => OrbitClass::ParabolicAsteroid,
            "HYA" => OrbitClass::HyperbolicAsteroid,
            other => OrbitClass::Other(other.to_string()),
        }
    }

    /// Convert to the SBDB API code string
    pub fn as_code(&self) -> &str {
        match self {
            OrbitClass::Atira => "IEO",
            OrbitClass::Aten => "ATE",
            OrbitClass::Apollo => "APO",
            OrbitClass::Amor => "AMO",
            OrbitClass::MarsCrosser => "MCA",
            OrbitClass::MainBelt => "MBA",
            OrbitClass::JupiterFamilyComet => "JFC",
            OrbitClass::HalleyTypeComet => "HTC",
            OrbitClass::EnckeTypeComet => "ETc",
            OrbitClass::Comet => "COM",
            OrbitClass::JupiterTrojan => "TJN",
            OrbitClass::Centaur => "CEN",
            OrbitClass::TransNeptunian => "TNO",
            OrbitClass::Asteroid => "AST",
            OrbitClass::ParabolicAsteroid => "PAA",
            OrbitClass::HyperbolicAsteroid => "HYA",
            OrbitClass::Other(code) => code.as_str(),
        }
    }
}

/// Small body identification data from the SBDB `object` field
#[derive(Debug, Clone)]
pub struct SmallBodyObject {
    /// Primary designation (e.g., "433", "2015 TB145")
    pub designation: String,
    /// SPK-ID
    pub spkid: Option<String>,
    /// Full name (e.g., "433 Eros (A898 PA)")
    pub fullname: Option<String>,
    /// Short name (e.g., "433 Eros")
    pub shortname: Option<String>,
    /// Object kind code (an/au/cn/cu)
    pub kind: Option<String>,
    /// Is a Near-Earth Object
    pub neo: bool,
    /// Is a Potentially Hazardous Asteroid
    pub pha: bool,
    /// Orbit class
    pub orbit_class: Option<OrbitClass>,
}

/// Orbital elements for a small body
#[derive(Debug, Clone)]
pub struct SmallBodyOrbit {
    /// Orbit solution ID
    pub orbit_id: Option<String>,
    /// Epoch (Julian Date TDB)
    pub epoch_jd: Option<f64>,
    /// Eccentricity
    pub eccentricity: Option<f64>,
    /// Semi-major axis (AU)
    pub semi_major_axis: Option<f64>,
    /// Perihelion distance (AU)
    pub perihelion_dist: Option<f64>,
    /// Inclination (degrees)
    pub inclination: Option<f64>,
    /// Longitude of ascending node (degrees)
    pub long_asc_node: Option<f64>,
    /// Argument of perihelion (degrees)
    pub arg_perihelion: Option<f64>,
    /// Mean anomaly (degrees)
    pub mean_anomaly: Option<f64>,
    /// Time of perihelion passage (Julian Date TDB)
    pub time_perihelion: Option<f64>,
    /// Mean motion (degrees/day)
    pub mean_motion: Option<f64>,
    /// Orbital period (days)
    pub period: Option<f64>,
    /// Aphelion distance (AU)
    pub aphelion_dist: Option<f64>,
    /// Minimum orbit intersection distance with Earth (AU)
    pub moid_au: Option<f64>,
    /// First observation date
    pub first_obs: Option<String>,
    /// Last observation date
    pub last_obs: Option<String>,
    /// Number of observations used
    pub n_obs_used: Option<u32>,
    /// Data arc span (days)
    pub data_arc_days: Option<u32>,
    /// Orbit condition code (0-9, 0 is best)
    pub condition_code: Option<String>,
    /// RMS of weighted residuals
    pub rms: Option<f64>,
}

/// Physical parameters for a small body
#[derive(Debug, Clone)]
pub struct PhysicalParams {
    /// Absolute magnitude H
    pub abs_magnitude_h: Option<f64>,
    /// Magnitude slope parameter G
    pub magnitude_slope_g: Option<f64>,
    /// Diameter (km)
    pub diameter_km: Option<f64>,
    /// Geometric albedo
    pub albedo: Option<f64>,
    /// Rotation period (hours)
    pub rotation_period_h: Option<f64>,
    /// Spectral type
    pub spectral_type: Option<String>,
}

/// A close approach record
#[derive(Debug, Clone)]
pub struct CloseApproachRecord {
    /// Object designation
    pub designation: String,
    /// Orbit solution ID
    pub orbit_id: Option<String>,
    /// Julian Date (TDB) of closest approach
    pub jd_tdb: Option<f64>,
    /// Calendar date/time of closest approach
    pub date: String,
    /// Nominal close approach distance (AU)
    pub dist_au: f64,
    /// Minimum possible distance (AU)
    pub dist_min_au: Option<f64>,
    /// Maximum possible distance (AU)
    pub dist_max_au: Option<f64>,
    /// Relative velocity at close approach (km/s)
    pub v_rel_km_s: Option<f64>,
    /// Velocity at infinity (km/s)
    pub v_inf_km_s: Option<f64>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
    /// Estimated diameter (km)
    pub diameter_km: Option<f64>,
    /// Full name of the object
    pub fullname: Option<String>,
    /// Close approach body (e.g., "Earth", "Mars")
    pub body: String,
}

/// A fireball/bolide event record
#[derive(Debug, Clone)]
pub struct FireballRecord {
    /// Date/time of peak brightness
    pub date: String,
    /// Radiated energy (joules * 10^10)
    pub energy_joules_e10: Option<f64>,
    /// Estimated total impact energy (kilotons of TNT)
    pub impact_energy_kt: Option<f64>,
    /// Latitude (degrees, positive = N)
    pub latitude: Option<f64>,
    /// Latitude direction (N or S)
    pub lat_dir: Option<String>,
    /// Longitude (degrees, positive = E)
    pub longitude: Option<f64>,
    /// Longitude direction (E or W)
    pub lon_dir: Option<String>,
    /// Altitude (km)
    pub altitude_km: Option<f64>,
    /// Velocity (km/s)
    pub velocity_km_s: Option<f64>,
}

// ── Mission Design API Types ────────────────────────────────────────────────

/// Optimality criterion for mission accessible target search (Mode A)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionDesignCriterion {
    /// Minimize departure V-infinity
    MinDepartureVinf = 1,
    /// Minimize arrival V-infinity
    MinArrivalVinf = 2,
    /// Minimize total delta-v
    MinTotalDv = 3,
    /// Minimize TOF + minimize departure V-infinity
    MinTofMinDepVinf = 4,
    /// Minimize TOF + minimize arrival V-infinity
    MinTofMinArrVinf = 5,
    /// Minimize TOF + minimize total delta-v
    MinTofMinTotalDv = 6,
}

impl MissionDesignCriterion {
    /// Convert to the integer value used by the API
    pub fn as_api_value(&self) -> u32 {
        *self as u32
    }
}

/// Parameters for Mission Design accessible target search (Mode A)
#[derive(Debug, Clone)]
pub struct MissionAccessibleParams {
    /// Optimality criterion for ranking results
    pub crit: MissionDesignCriterion,
    /// Launch year(s) to search
    pub year: Vec<u32>,
    /// Maximum number of records to return
    pub lim: Option<u32>,
}

/// A single accessible target entry from the Mission Design API (Mode A)
#[derive(Debug, Clone)]
pub struct MissionAccessibleEntry {
    /// Object name
    pub name: String,
    /// Primary designation
    pub pdes: Option<String>,
    /// Departure date (calendar)
    pub date0: String,
    /// Departure date (Modified Julian Date)
    pub mjd0: f64,
    /// Arrival date (calendar)
    pub datef: String,
    /// Arrival date (Modified Julian Date)
    pub mjdf: f64,
    /// Departure C3 (km^2/s^2)
    pub c3_dep: f64,
    /// Departure V-infinity (km/s)
    pub vinf_dep: f64,
    /// Arrival V-infinity (km/s)
    pub vinf_arr: f64,
    /// Total delta-v (km/s)
    pub dv_tot: f64,
    /// Time of flight (days)
    pub tof: f64,
    /// Orbit class code
    pub class: Option<String>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
    /// Orbit condition code
    pub condition_code: Option<String>,
    /// Near-Earth Object flag
    pub neo: bool,
    /// Potentially Hazardous Asteroid flag
    pub pha: bool,
}

/// Response from the Mission Design accessible target search (Mode A)
#[derive(Debug, Clone)]
pub struct MissionAccessibleResponse {
    /// Number of records returned
    pub count: u32,
    /// Accessible target entries
    pub data: Vec<MissionAccessibleEntry>,
}

/// Object info returned in a Mission Design query response (Mode Q)
#[derive(Debug, Clone)]
pub struct MissionQueryObject {
    /// Primary designation
    pub des: String,
    /// Full name
    pub fullname: Option<String>,
    /// SPK-ID
    pub spkid: Option<String>,
    /// Orbit class
    pub orbit_class: Option<String>,
    /// Orbit condition code
    pub condition_code: Option<String>,
    /// Data arc (days)
    pub data_arc: Option<String>,
    /// Orbit solution ID
    pub orbit_id: Option<String>,
    /// Mission design orbit ID
    pub md_orbit_id: Option<String>,
}

/// Response from the Mission Design query for a specific object (Mode Q)
#[derive(Debug, Clone)]
pub struct MissionQueryResponse {
    /// Object identification
    pub object: MissionQueryObject,
    /// Field names for the selected missions table
    pub fields: Vec<String>,
    /// Selected mission data rows (tabular, matches fields order)
    pub selected_missions: Vec<Vec<f64>>,
}

/// Parameters for Mission Design flyby/extension target search (Mode T)
#[derive(Debug, Clone)]
pub struct MissionFlybyParams {
    /// Eccentricity of reference orbit
    pub ec: f64,
    /// Perihelion distance (AU)
    pub qr: f64,
    /// Time of perihelion passage (Julian Date)
    pub tp: f64,
    /// Inclination (degrees)
    pub inc: f64,
    /// Longitude of ascending node (degrees)
    pub om: f64,
    /// Argument of periapsis (degrees)
    pub w: f64,
    /// Start of time span (Julian Date)
    pub jd0: f64,
    /// End of time span (Julian Date)
    pub jdf: f64,
    /// Maximum number of output records
    pub maxout: Option<u32>,
    /// Maximum close-approach distance (AU)
    pub maxdist: Option<f64>,
}

/// A flyby/extension target entry from the Mission Design API (Mode T)
#[derive(Debug, Clone)]
pub struct MissionFlybyEntry {
    /// Full object name
    pub full_name: String,
    /// Primary designation
    pub pdes: Option<String>,
    /// SPK-ID
    pub spkid: Option<String>,
    /// Close approach date (calendar)
    pub date: String,
    /// Close approach date (Julian Date)
    pub jd: f64,
    /// Minimum distance (AU)
    pub min_dist_au: f64,
    /// Minimum distance (km)
    pub min_dist_km: Option<f64>,
    /// Relative velocity (km/s)
    pub rel_vel: f64,
    /// Orbit class code
    pub class: Option<String>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
    /// Orbit condition code
    pub condition_code: Option<String>,
    /// Near-Earth Object flag
    pub neo: bool,
    /// Potentially Hazardous Asteroid flag
    pub pha: bool,
}

/// Response from the Mission Design flyby/extension target search (Mode T)
#[derive(Debug, Clone)]
pub struct MissionFlybyResponse {
    /// Number of records returned
    pub count: u32,
    /// Flyby target entries
    pub data: Vec<MissionFlybyEntry>,
}

/// A Sentry impact risk entry
#[derive(Debug, Clone)]
pub struct SentryEntry {
    /// Object designation
    pub designation: String,
    /// Full name
    pub fullname: Option<String>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
    /// Estimated diameter (km)
    pub diameter_km: Option<f64>,
    /// Number of potential impacts
    pub n_imp: Option<u32>,
    /// Cumulative impact probability
    pub ip: Option<f64>,
    /// Cumulative Palermo Scale
    pub ps_cum: Option<f64>,
    /// Maximum Palermo Scale
    pub ps_max: Option<f64>,
    /// Maximum Torino Scale
    pub ts_max: Option<u32>,
    /// Last observation date
    pub last_obs: Option<String>,
    /// Range of potential impact years
    pub ip_range: Option<String>,
}

/// A summary entry from the Scout NEOCP analysis API (Mode S)
#[derive(Debug, Clone)]
pub struct ScoutSummaryEntry {
    /// NEOCP temporary designation
    pub object_name: String,
    /// Number of observations
    pub n_obs: Option<u32>,
    /// Observation arc (days)
    pub arc: Option<f64>,
    /// Normalized RMS residual
    pub rms_n: Option<f64>,
    /// Estimated absolute magnitude
    pub h_mag: Option<f64>,
    /// Interest rating (0-100, higher = more interesting)
    pub rating: Option<u32>,
    /// Minimum orbit intersection distance (AU)
    pub moid: Option<f64>,
    /// Close approach distance (AU)
    pub ca_dist: Option<f64>,
    /// Velocity at infinity (km/s)
    pub v_inf: Option<f64>,
    /// PHA likelihood score
    pub pha_score: Option<i32>,
    /// NEO likelihood score
    pub neo_score: Option<i32>,
    /// Geocentric orbit likelihood
    pub geocentric_score: Option<i32>,
    /// Interior Earth orbit likelihood
    pub ieo_score: Option<i32>,
    /// Tisserand parameter score (comet vs asteroid)
    pub tisserand_score: Option<i32>,
    /// Last analysis run time
    pub last_run: Option<String>,
    /// Right ascension
    pub ra: Option<String>,
    /// Declination
    pub dec: Option<String>,
    /// Solar elongation
    pub elong: Option<String>,
    /// Rate of motion
    pub rate: Option<f64>,
    /// Estimated visual magnitude
    pub v_mag: Option<f64>,
    /// Positional uncertainty (arcsec)
    pub unc: Option<f64>,
    /// Positional uncertainty at +1 day (arcsec)
    pub unc_p1: Option<f64>,
}

/// Detailed data for a single Scout NEOCP object (Mode O)
#[derive(Debug, Clone)]
pub struct ScoutObjectDetail {
    /// All summary-level fields
    pub summary: ScoutSummaryEntry,
    /// NEO 1km impact score
    pub neo1km_score: Option<String>,
    /// Ephemeris time
    pub t_ephem: Option<String>,
    /// Sampled orbit data (fields + rows)
    pub orbits: Option<ScoutOrbitData>,
}

/// Sampled orbit data from Scout object detail
#[derive(Debug, Clone)]
pub struct ScoutOrbitData {
    /// Number of sampled orbits
    pub count: u32,
    /// Field names for orbit columns
    pub fields: Vec<String>,
    /// Raw orbit data rows
    pub data: Vec<Vec<serde_json::Value>>,
}

/// Response from the Scout summary endpoint (Mode S)
#[derive(Debug, Clone)]
pub struct ScoutSummaryResponse {
    /// Total number of NEOCP objects
    pub count: u32,
    /// Summary entries for each object
    pub data: Vec<ScoutSummaryEntry>,
}

/// Response from the Scout object detail endpoint (Mode O)
#[derive(Debug, Clone)]
pub struct ScoutObjectResponse {
    /// Detailed object data
    pub detail: ScoutObjectDetail,
}

/// Parameters for SB Radar API queries
#[derive(Debug, Clone, Default)]
pub struct RadarParams {
    /// Select by SPK-ID
    pub spk: Option<i64>,
    /// Select by designation
    pub des: Option<String>,
    /// Object type: a, an, au, c, cn, cu, n, u
    pub kind: Option<String>,
    /// Reference point: P (peak/center) or C (center of mass)
    pub bp: Option<String>,
    /// Measurement type: R (range/delay) or P (Doppler)
    pub measurement_type: Option<String>,
    /// Include observer field
    pub observer: bool,
    /// Include notes field
    pub notes: bool,
    /// Include reference field
    pub ref_field: bool,
    /// Include full object name
    pub fullname: bool,
    /// Include modification date
    pub modified: bool,
    /// Include station geodetic coordinates
    pub coords: bool,
    /// Use cylindrical station coordinates instead of geodetic
    pub c_coords: bool,
}

impl RadarParams {
    /// Convert parameters to query string pairs
    pub fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();
        if let Some(v) = self.spk {
            params.push(("spk".into(), v.to_string()));
        }
        if let Some(ref v) = self.des {
            params.push(("des".into(), v.clone()));
        }
        if let Some(ref v) = self.kind {
            params.push(("kind".into(), v.clone()));
        }
        if let Some(ref v) = self.bp {
            params.push(("bp".into(), v.clone()));
        }
        if let Some(ref v) = self.measurement_type {
            params.push(("type".into(), v.clone()));
        }
        if self.observer {
            params.push(("observer".into(), "true".into()));
        }
        if self.notes {
            params.push(("notes".into(), "true".into()));
        }
        if self.ref_field {
            params.push(("ref".into(), "true".into()));
        }
        if self.fullname {
            params.push(("fullname".into(), "true".into()));
        }
        if self.modified {
            params.push(("modified".into(), "true".into()));
        }
        if self.coords {
            params.push(("coords".into(), "true".into()));
        }
        if self.c_coords {
            params.push(("c-coords".into(), "true".into()));
        }
        params
    }
}

/// Observer location for the SB Identification API
#[derive(Debug, Clone)]
pub enum SbIdentObserver {
    /// MPC observatory code (e.g., "F51" for Pan-STARRS)
    MpcCode(String),
    /// Geodetic coordinates (lat degrees north-positive, lon degrees east-positive, alt km)
    Geodetic { lat: f64, lon: f64, alt: f64 },
    /// Geocentric state vector: position (km) and optional velocity (km/s), J2000 equatorial
    Geocentric(String),
    /// Heliocentric state vector: position (AU) and optional velocity (AU/d), J2000 equatorial
    Heliocentric(String),
}

/// Field of view specification for the SB Identification API
#[derive(Debug, Clone)]
pub enum SbIdentFov {
    /// FOV defined by RA/Dec edge limits (sexagesimal strings)
    Edges {
        /// RA edges: comma-separated `hh-mm-ss[.ss]` values
        ra_lim: String,
        /// Dec edges: comma-separated `dd-mm-ss[.ss]` values
        dec_lim: String,
    },
    /// FOV defined by center point and half-widths
    Center {
        /// Center RA: `hh-mm-ss[.ss]`
        ra_center: String,
        /// Center Dec: `dd-mm-ss[.ss]`
        dec_center: String,
        /// Half-width in RA (degrees, default 0.5)
        ra_hwidth: Option<f64>,
        /// Half-width in Dec (degrees, default 0.5)
        dec_hwidth: Option<f64>,
    },
}

/// Parameters for the SB Identification API query
#[derive(Debug, Clone)]
pub struct SbIdentParams {
    /// Observer location
    pub observer: SbIdentObserver,
    /// Field of view specification
    pub fov: SbIdentFov,
    /// Observation time: `YYYY-MM-DD[_hh:mm:ss]` or Julian Date
    pub obs_time: String,
    /// Visual magnitude limit threshold
    pub vmag_lim: Option<f64>,
    /// Enable second pass with high-fidelity numerical integration
    pub two_pass: bool,
    /// Require magnitude parameters for returned objects
    pub mag_required: Option<bool>,
    /// Object type filter: `a` (asteroids only)
    pub sb_kind: Option<String>,
    /// Group filter: e.g., `neo`
    pub sb_group: Option<String>,
    /// Include osculating orbital elements in the response
    pub req_elem: bool,
}

impl SbIdentParams {
    /// Convert parameters to query string key-value pairs for the HTTP request
    pub fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();

        match &self.observer {
            SbIdentObserver::MpcCode(code) => {
                params.push(("mpc-code".into(), code.clone()));
            }
            SbIdentObserver::Geodetic { lat, lon, alt } => {
                params.push(("lat".into(), lat.to_string()));
                params.push(("lon".into(), lon.to_string()));
                params.push(("alt".into(), alt.to_string()));
            }
            SbIdentObserver::Geocentric(state) => {
                params.push(("xobs".into(), state.clone()));
            }
            SbIdentObserver::Heliocentric(state) => {
                params.push(("xobs-hel".into(), state.clone()));
            }
        }

        match &self.fov {
            SbIdentFov::Edges { ra_lim, dec_lim } => {
                params.push(("fov-ra-lim".into(), ra_lim.clone()));
                params.push(("fov-dec-lim".into(), dec_lim.clone()));
            }
            SbIdentFov::Center {
                ra_center,
                dec_center,
                ra_hwidth,
                dec_hwidth,
            } => {
                params.push(("fov-ra-center".into(), ra_center.clone()));
                params.push(("fov-dec-center".into(), dec_center.clone()));
                if let Some(w) = ra_hwidth {
                    params.push(("fov-ra-hwidth".into(), w.to_string()));
                }
                if let Some(w) = dec_hwidth {
                    params.push(("fov-dec-hwidth".into(), w.to_string()));
                }
            }
        }

        params.push(("obs-time".into(), self.obs_time.clone()));

        if let Some(v) = self.vmag_lim {
            params.push(("vmag-lim".into(), v.to_string()));
        }
        if self.two_pass {
            params.push(("two-pass".into(), "true".into()));
            params.push(("suppress-first-pass".into(), "false".into()));
        }
        if let Some(v) = self.mag_required {
            params.push(("mag-required".into(), v.to_string()));
        }
        if let Some(ref v) = self.sb_kind {
            params.push(("sb-kind".into(), v.clone()));
        }
        if let Some(ref v) = self.sb_group {
            params.push(("sb-group".into(), v.clone()));
        }
        if self.req_elem {
            params.push(("req-elem".into(), "true".into()));
        }

        params
    }
}

/// A radar astrometry measurement record for a small body
#[derive(Debug, Clone)]
pub struct RadarRecord {
    /// Object designation
    pub designation: String,
    /// Observation epoch
    pub epoch: String,
    /// Measured value (delay in microseconds or Doppler shift in Hz)
    pub value: Option<f64>,
    /// Measurement uncertainty (1-sigma)
    pub sigma: Option<f64>,
    /// Units of measurement: "us" (microseconds) or "Hz"
    pub units: Option<String>,
    /// Transmission frequency (MHz)
    pub freq: Option<f64>,
    /// Receiving station DSN code
    pub rcvr: Option<String>,
    /// Transmitting station DSN code
    pub xmit: Option<String>,
    /// Reference point: P (peak/center) or C (center of mass)
    pub bp: Option<String>,
    /// Observer information
    pub observer: Option<String>,
    /// Notes
    pub notes: Option<String>,
    /// Literature reference
    pub reference: Option<String>,
    /// Full object name
    pub fullname: Option<String>,
    /// Last modification date
    pub modified: Option<String>,
    /// Station longitude (degrees)
    pub longitude: Option<f64>,
    /// Station latitude (degrees)
    pub latitude: Option<f64>,
    /// Station altitude or cylindrical d_xy
    pub altitude: Option<f64>,
}

/// Response from the SB Radar API
#[derive(Debug, Clone)]
pub struct RadarResponse {
    /// Total number of records
    pub count: u32,
    /// Radar measurement records
    pub records: Vec<RadarRecord>,
}

/// A single identified small body entry from the SB Identification API
#[derive(Debug, Clone)]
pub struct SbIdentEntry {
    /// Object name/designation
    pub name: String,
    /// Astrometric right ascension (sexagesimal or degrees)
    pub ra: Option<String>,
    /// Astrometric declination (sexagesimal or degrees)
    pub dec: Option<String>,
    /// RA offset from FOV center (arcsec)
    pub ra_offset: Option<f64>,
    /// Dec offset from FOV center (arcsec)
    pub dec_offset: Option<f64>,
    /// Total offset from FOV center (arcsec)
    pub total_offset: Option<f64>,
    /// Visual magnitude
    pub vmag: Option<f64>,
    /// RA rate of motion (deg/s)
    pub ra_rate: Option<f64>,
    /// Dec rate of motion (deg/s)
    pub dec_rate: Option<f64>,
    /// RA error estimate (arcsec, first-pass only)
    pub ra_err: Option<f64>,
    /// Dec error estimate (arcsec, first-pass only)
    pub dec_err: Option<f64>,
}

/// Orbital elements for an identified small body (when `req_elem` is true)
#[derive(Debug, Clone)]
pub struct SbIdentOrbitalElements {
    /// Object name/designation
    pub name: String,
    /// Absolute magnitude H
    pub h: Option<f64>,
    /// Magnitude slope parameter G
    pub g: Option<f64>,
    /// Eccentricity
    pub e: Option<f64>,
    /// Perihelion distance (AU)
    pub q: Option<f64>,
    /// Time of perihelion passage (JD)
    pub tp: Option<f64>,
    /// Longitude of ascending node (degrees)
    pub om: Option<f64>,
    /// Argument of perihelion (degrees)
    pub w: Option<f64>,
    /// Inclination (degrees)
    pub i: Option<f64>,
    /// Epoch (JD)
    pub epoch: Option<f64>,
}

/// Observer information returned in the SB Identification response
#[derive(Debug, Clone)]
pub struct SbIdentObserverInfo {
    /// Observation date/time
    pub obs_date: Option<String>,
    /// Observer location description
    pub location: Option<String>,
    /// FOV center coordinate
    pub fov_center: Option<String>,
    /// FOV offset description
    pub fov_offset: Option<String>,
    /// Reference frame (typically "J2000")
    pub frame: Option<String>,
}

/// Response from the SB Identification API
#[derive(Debug, Clone)]
pub struct SbIdentResponse {
    /// Observer information
    pub observer: SbIdentObserverInfo,
    /// Number of objects found in first pass
    pub n_first_pass: u32,
    /// Number of objects found in second pass (when two-pass enabled)
    pub n_second_pass: u32,
    /// Identified objects from first-pass search
    pub data_first_pass: Vec<SbIdentEntry>,
    /// Identified objects from second-pass search (when two-pass enabled)
    pub data_second_pass: Vec<SbIdentEntry>,
    /// Orbital elements from first-pass (when req_elem is true)
    pub elem_first_pass: Vec<SbIdentOrbitalElements>,
    /// Orbital elements from second-pass (when req_elem is true)
    pub elem_second_pass: Vec<SbIdentOrbitalElements>,
}

/// Observer location for the SB Observability API
#[derive(Debug, Clone)]
pub enum ObservabilityObserver {
    /// MPC observatory code (e.g., "F51" for Pan-STARRS 1)
    MpcCode(String),
    /// Geodetic coordinates
    Geodetic {
        /// Latitude in degrees, north-positive [-90, 90]
        lat: f64,
        /// Longitude in degrees, east-positive [-180, 180]
        lon: f64,
        /// Altitude above WGS-84 ellipsoid in km
        alt: f64,
    },
}

/// Parameters for the SB Observability API query
#[derive(Debug, Clone)]
pub struct ObservabilityParams {
    /// Observer location (required)
    pub observer: ObservabilityObserver,
    /// Observation date: YYYY-MM-DD or YYYY-MM-DD_hh:mm:ss (required)
    pub obs_time: String,
    /// End observation time
    pub obs_end: Option<String>,
    /// Require sun below horizon (default: true)
    pub optical: Option<bool>,
    /// Minimum solar elongation (deg), required if optical=false
    pub elong_min: Option<f64>,
    /// Maximum solar elongation (deg)
    pub elong_max: Option<f64>,
    /// Minimum galactic latitude (deg)
    pub glat_min: Option<f64>,
    /// Maximum galactic latitude (deg)
    pub glat_max: Option<f64>,
    /// Minimum elevation above horizon (deg, default: 30)
    pub elev_min: Option<f64>,
    /// Minimum observable time (minutes, default: 0)
    pub time_min: Option<u32>,
    /// Minimum visual magnitude
    pub vmag_min: Option<f64>,
    /// Maximum visual magnitude
    pub vmag_max: Option<f64>,
    /// Require magnitude data
    pub mag_required: Option<bool>,
    /// Minimum heliocentric distance (AU)
    pub helio_min: Option<f64>,
    /// Maximum heliocentric distance (AU)
    pub helio_max: Option<f64>,
    /// Minimum topocentric distance (AU)
    pub dist_min: Option<f64>,
    /// Maximum topocentric distance (AU)
    pub dist_max: Option<f64>,
    /// Maximum number of output records
    pub maxoutput: Option<u32>,
    /// Sort field for output
    pub output_sort: Option<String>,
    /// Sort in descending order
    pub output_sort_r: Option<bool>,
    /// Small body kind filter (a=asteroid, c=comet)
    pub sb_kind: Option<String>,
    /// Small body group filter (e.g., "neo", "pha")
    pub sb_group: Option<String>,
}

impl ObservabilityParams {
    /// Convert parameters to query string key-value pairs
    pub fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();

        match &self.observer {
            ObservabilityObserver::MpcCode(code) => {
                params.push(("mpc-code".into(), code.clone()));
            }
            ObservabilityObserver::Geodetic { lat, lon, alt } => {
                params.push(("lat".into(), lat.to_string()));
                params.push(("lon".into(), lon.to_string()));
                params.push(("alt".into(), alt.to_string()));
            }
        }

        params.push(("obs-time".into(), self.obs_time.clone()));

        if let Some(ref v) = self.obs_end {
            params.push(("obs-end".into(), v.clone()));
        }
        if let Some(v) = self.optical {
            params.push(("optical".into(), v.to_string()));
        }
        if let Some(v) = self.elong_min {
            params.push(("elong-min".into(), v.to_string()));
        }
        if let Some(v) = self.elong_max {
            params.push(("elong-max".into(), v.to_string()));
        }
        if let Some(v) = self.glat_min {
            params.push(("glat-min".into(), v.to_string()));
        }
        if let Some(v) = self.glat_max {
            params.push(("glat-max".into(), v.to_string()));
        }
        if let Some(v) = self.elev_min {
            params.push(("elev-min".into(), v.to_string()));
        }
        if let Some(v) = self.time_min {
            params.push(("time-min".into(), v.to_string()));
        }
        if let Some(v) = self.vmag_min {
            params.push(("vmag-min".into(), v.to_string()));
        }
        if let Some(v) = self.vmag_max {
            params.push(("vmag-max".into(), v.to_string()));
        }
        if let Some(v) = self.mag_required {
            params.push(("mag-required".into(), v.to_string()));
        }
        if let Some(v) = self.helio_min {
            params.push(("helio-min".into(), v.to_string()));
        }
        if let Some(v) = self.helio_max {
            params.push(("helio-max".into(), v.to_string()));
        }
        if let Some(v) = self.dist_min {
            params.push(("dist-min".into(), v.to_string()));
        }
        if let Some(v) = self.dist_max {
            params.push(("dist-max".into(), v.to_string()));
        }
        if let Some(v) = self.maxoutput {
            params.push(("maxoutput".into(), v.to_string()));
        }
        if let Some(ref v) = self.output_sort {
            params.push(("output-sort".into(), v.clone()));
        }
        if let Some(v) = self.output_sort_r {
            params.push(("output-sort-r".into(), v.to_string()));
        }
        if let Some(ref v) = self.sb_kind {
            params.push(("sb-kind".into(), v.clone()));
        }
        if let Some(ref v) = self.sb_group {
            params.push(("sb-group".into(), v.clone()));
        }

        params
    }
}

/// Night information from the SB Observability API response
#[derive(Debug, Clone)]
pub struct ObservabilityNightInfo {
    /// Sunset time (UT)
    pub sun_set: Option<String>,
    /// Sunrise time (UT)
    pub sun_rise: Option<String>,
    /// Sunset azimuth (deg)
    pub sun_set_az: Option<String>,
    /// Sunrise azimuth (deg)
    pub sun_rise_az: Option<String>,
    /// Begin astronomical twilight (UT)
    pub begin_astronomical: Option<String>,
    /// End astronomical twilight (UT)
    pub end_astronomical: Option<String>,
    /// Begin civil twilight (UT)
    pub begin_civil: Option<String>,
    /// End civil twilight (UT)
    pub end_civil: Option<String>,
    /// Begin nautical twilight (UT)
    pub begin_nautical: Option<String>,
    /// End nautical twilight (UT)
    pub end_nautical: Option<String>,
    /// Moon rise time (UT)
    pub moon_rise: Option<String>,
    /// Moon rise phase
    pub moon_rise_phase: Option<String>,
    /// Moon set time (UT)
    pub moon_set: Option<String>,
    /// Moon set phase
    pub moon_set_phase: Option<String>,
    /// Moon transit time (UT)
    pub transit: Option<String>,
    /// Moon transit phase
    pub transit_phase: Option<String>,
    /// Begin dark time (UT)
    pub begin_dark: Option<String>,
    /// Mid dark time (UT)
    pub mid_dark: Option<String>,
    /// End dark time (UT)
    pub end_dark: Option<String>,
    /// Total dark time (hours)
    pub dark_time: Option<String>,
}

/// A single observable object from the SB Observability API
#[derive(Debug, Clone)]
pub struct ObservableObject {
    /// Object designation
    pub des: String,
    /// Full name
    pub fullname: Option<String>,
    /// Rise time (UT)
    pub rise: Option<String>,
    /// Transit time (UT)
    pub transit: Option<String>,
    /// Set time (UT)
    pub set: Option<String>,
    /// Maximum observable time
    pub max_time: Option<String>,
    /// Right ascension
    pub ra: Option<String>,
    /// Declination
    pub dec: Option<String>,
    /// Visual magnitude
    pub vmag: Option<f64>,
    /// Heliocentric range (AU)
    pub helio_range_au: Option<f64>,
    /// Topocentric range (AU)
    pub topo_range_au: Option<f64>,
    /// Object-Observer-Sun angle (deg)
    pub sun_angle: Option<f64>,
    /// Object-Observer-Moon angle (deg)
    pub moon_angle: Option<f64>,
    /// Galactic latitude (deg)
    pub galactic_lat: Option<f64>,
}

/// Response from the SB Observability API
#[derive(Debug, Clone)]
pub struct ObservabilityResponse {
    /// Night information (sunset/sunrise, twilight, moon data)
    pub night_info: ObservabilityNightInfo,
    /// Total number of observable objects
    pub count: u32,
    /// Observable objects
    pub objects: Vec<ObservableObject>,
}

/// Parameters for NHATS (Near-Earth Asteroid Human Space Flight Accessible Targets Survey) queries
#[derive(Debug, Clone, Default)]
pub struct NhatsParams {
    /// Maximum total mission delta-v in km/s (4-12, default 12)
    pub dv: Option<u32>,
    /// Maximum total mission duration in days (60-450 in steps of 30, default 450)
    pub dur: Option<u32>,
    /// Minimum stay time at asteroid in days (8, 16, 24, or 32; default 8)
    pub stay: Option<u32>,
    /// Launch window (e.g., "2020-2045", "2025-2030")
    pub launch: Option<String>,
    /// Maximum absolute magnitude H (16-30, Mode S only)
    pub h: Option<u32>,
    /// Maximum orbit condition code (0-8, Mode S only)
    pub occ: Option<u32>,
}

impl NhatsParams {
    /// Convert parameters to query string pairs for the HTTP request
    pub fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();
        if let Some(v) = self.dv {
            params.push(("dv".into(), v.to_string()));
        }
        if let Some(v) = self.dur {
            params.push(("dur".into(), v.to_string()));
        }
        if let Some(v) = self.stay {
            params.push(("stay".into(), v.to_string()));
        }
        if let Some(ref v) = self.launch {
            params.push(("launch".into(), v.clone()));
        }
        if let Some(v) = self.h {
            params.push(("h".into(), v.to_string()));
        }
        if let Some(v) = self.occ {
            params.push(("occ".into(), v.to_string()));
        }
        params
    }
}

/// A delta-v / duration pair used in NHATS summary entries
#[derive(Debug, Clone)]
pub struct NhatsDvDur {
    /// Delta-v (km/s)
    pub dv: Option<f64>,
    /// Duration (days)
    pub dur: Option<f64>,
}

/// A trajectory record from the NHATS API (Mode O detail)
#[derive(Debug, Clone)]
pub struct NhatsTrajectory {
    /// Trajectory ID
    pub tid: Option<String>,
    /// Total mission delta-v (km/s)
    pub dv_total: Option<f64>,
    /// Total mission duration (days)
    pub dur_total: Option<f64>,
    /// Outbound duration (days)
    pub dur_out: Option<f64>,
    /// Stay duration at asteroid (days)
    pub dur_at: Option<f64>,
    /// Return duration (days)
    pub dur_ret: Option<f64>,
    /// Launch date (YYYY-MM-DD)
    pub launch: Option<String>,
    /// Launch C3 energy (km^2/s^2)
    pub c3: Option<f64>,
    /// Departure velocity from Earth (km/s)
    pub v_dep_earth: Option<f64>,
    /// Delta-v to depart parking orbit (km/s)
    pub dv_dep_park: Option<f64>,
    /// Relative arrival velocity at NEO (km/s)
    pub vrel_arr_neo: Option<f64>,
    /// Relative departure velocity from NEO (km/s)
    pub vrel_dep_neo: Option<f64>,
    /// Relative arrival velocity at Earth (km/s)
    pub vrel_arr_earth: Option<f64>,
    /// Arrival velocity at Earth (km/s)
    pub v_arr_earth: Option<f64>,
    /// Departure declination (degrees)
    pub dec_dep: Option<f64>,
    /// Arrival declination (degrees)
    pub dec_arr: Option<f64>,
}

/// A summary entry from the NHATS API (Mode S)
#[derive(Debug, Clone)]
pub struct NhatsSummaryEntry {
    /// Object designation
    pub des: String,
    /// Full name
    pub fullname: Option<String>,
    /// Orbit solution ID
    pub orbit_id: Option<String>,
    /// Absolute magnitude H
    pub h: Option<f64>,
    /// Minimum estimated size (meters)
    pub min_size: Option<f64>,
    /// Maximum estimated size (meters)
    pub max_size: Option<f64>,
    /// Measured size (meters), if available
    pub size: Option<f64>,
    /// Orbit condition code
    pub occ: Option<u32>,
    /// Minimum delta-v trajectory summary (dv + dur)
    pub min_dv: Option<NhatsDvDur>,
    /// Minimum duration trajectory summary (dv + dur)
    pub min_dur: Option<NhatsDvDur>,
    /// Number of viable trajectories
    pub n_via_traj: Option<u32>,
    /// Observation window start date
    pub obs_start: Option<String>,
    /// Observation window end date
    pub obs_end: Option<String>,
}

/// Response from the NHATS API in summary mode (Mode S)
#[derive(Debug, Clone)]
pub struct NhatsSummaryResponse {
    /// Total number of accessible objects
    pub count: u32,
    /// List of accessible NEA entries
    pub data: Vec<NhatsSummaryEntry>,
}

/// Response from the NHATS API in object detail mode (Mode O)
#[derive(Debug, Clone)]
pub struct NhatsObjectResponse {
    /// Object designation
    pub des: String,
    /// Full name
    pub fullname: Option<String>,
    /// Orbit solution ID
    pub orbit_id: Option<String>,
    /// Absolute magnitude H
    pub h: Option<f64>,
    /// Minimum estimated size (meters)
    pub min_size: Option<f64>,
    /// Maximum estimated size (meters)
    pub max_size: Option<f64>,
    /// Measured size (meters), if available
    pub size: Option<f64>,
    /// Orbit condition code
    pub occ: Option<u32>,
    /// Number of viable trajectories
    pub n_via_traj: Option<u32>,
    /// Minimum delta-v trajectory detail
    pub min_dv_traj: Option<NhatsTrajectory>,
    /// Minimum duration trajectory detail
    pub min_dur_traj: Option<NhatsTrajectory>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbit_class_from_code() {
        assert_eq!(OrbitClass::from_code("APO"), OrbitClass::Apollo);
        assert_eq!(OrbitClass::from_code("AMO"), OrbitClass::Amor);
        assert_eq!(OrbitClass::from_code("MBA"), OrbitClass::MainBelt);
        assert_eq!(OrbitClass::from_code("TNO"), OrbitClass::TransNeptunian);
        assert_eq!(
            OrbitClass::from_code("XYZ"),
            OrbitClass::Other("XYZ".to_string())
        );
    }

    #[test]
    fn test_mission_design_criterion_values() {
        assert_eq!(MissionDesignCriterion::MinDepartureVinf.as_api_value(), 1);
        assert_eq!(MissionDesignCriterion::MinArrivalVinf.as_api_value(), 2);
        assert_eq!(MissionDesignCriterion::MinTotalDv.as_api_value(), 3);
        assert_eq!(MissionDesignCriterion::MinTofMinDepVinf.as_api_value(), 4);
        assert_eq!(MissionDesignCriterion::MinTofMinArrVinf.as_api_value(), 5);
        assert_eq!(MissionDesignCriterion::MinTofMinTotalDv.as_api_value(), 6);
    }

    #[test]
    fn test_orbit_class_roundtrip() {
        let classes = [
            OrbitClass::Atira,
            OrbitClass::Aten,
            OrbitClass::Apollo,
            OrbitClass::Amor,
            OrbitClass::MainBelt,
            OrbitClass::Centaur,
        ];
        for class in &classes {
            assert_eq!(&OrbitClass::from_code(class.as_code()), class);
        }
    }

    #[test]
    fn test_sb_ident_params_mpc_code() {
        let params = SbIdentParams {
            observer: SbIdentObserver::MpcCode("F51".into()),
            fov: SbIdentFov::Center {
                ra_center: "05-00-00".into(),
                dec_center: "20-00-00".into(),
                ra_hwidth: Some(1.0),
                dec_hwidth: Some(1.0),
            },
            obs_time: "2024-01-01".into(),
            vmag_lim: Some(20.0),
            two_pass: false,
            mag_required: None,
            sb_kind: None,
            sb_group: None,
            req_elem: false,
        };
        let query = params.to_query_params();
        let map: std::collections::HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("mpc-code").unwrap(), "F51");
        assert_eq!(map.get("fov-ra-center").unwrap(), "05-00-00");
        assert_eq!(map.get("fov-dec-center").unwrap(), "20-00-00");
        assert_eq!(map.get("fov-ra-hwidth").unwrap(), "1");
        assert_eq!(map.get("obs-time").unwrap(), "2024-01-01");
        assert_eq!(map.get("vmag-lim").unwrap(), "20");
        assert!(map.get("two-pass").is_none());
    }

    #[test]
    fn test_sb_ident_params_geodetic_edges() {
        let params = SbIdentParams {
            observer: SbIdentObserver::Geodetic {
                lat: 34.05,
                lon: -118.25,
                alt: 0.0,
            },
            fov: SbIdentFov::Edges {
                ra_lim: "05-00-00,06-00-00".into(),
                dec_lim: "19-00-00,21-00-00".into(),
            },
            obs_time: "2024-06-15_12:00:00".into(),
            vmag_lim: None,
            two_pass: true,
            mag_required: Some(true),
            sb_kind: Some("a".into()),
            sb_group: Some("neo".into()),
            req_elem: true,
        };
        let query = params.to_query_params();
        let map: std::collections::HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("lat").unwrap(), "34.05");
        assert_eq!(map.get("lon").unwrap(), "-118.25");
        assert_eq!(map.get("alt").unwrap(), "0");
        assert_eq!(map.get("fov-ra-lim").unwrap(), "05-00-00,06-00-00");
        assert_eq!(map.get("fov-dec-lim").unwrap(), "19-00-00,21-00-00");
        assert_eq!(map.get("two-pass").unwrap(), "true");
        assert_eq!(map.get("suppress-first-pass").unwrap(), "false");
        assert_eq!(map.get("mag-required").unwrap(), "true");
        assert_eq!(map.get("sb-kind").unwrap(), "a");
        assert_eq!(map.get("sb-group").unwrap(), "neo");
        assert_eq!(map.get("req-elem").unwrap(), "true");
    }

    #[test]
    fn test_observability_params_mpc_code() {
        let params = ObservabilityParams {
            observer: ObservabilityObserver::MpcCode("F51".into()),
            obs_time: "2026-03-01".into(),
            obs_end: None,
            optical: None,
            elong_min: None,
            elong_max: None,
            glat_min: None,
            glat_max: None,
            elev_min: None,
            time_min: None,
            vmag_min: None,
            vmag_max: Some(18.0),
            mag_required: None,
            helio_min: None,
            helio_max: None,
            dist_min: None,
            dist_max: None,
            maxoutput: Some(10),
            output_sort: Some("vmag".into()),
            output_sort_r: None,
            sb_kind: None,
            sb_group: None,
        };
        let query = params.to_query_params();
        let map: std::collections::HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("mpc-code").unwrap(), "F51");
        assert_eq!(map.get("obs-time").unwrap(), "2026-03-01");
        assert_eq!(map.get("vmag-max").unwrap(), "18");
        assert_eq!(map.get("maxoutput").unwrap(), "10");
        assert_eq!(map.get("output-sort").unwrap(), "vmag");
        assert!(!map.contains_key("lat"));
    }

    #[test]
    fn test_observability_params_geodetic() {
        let params = ObservabilityParams {
            observer: ObservabilityObserver::Geodetic {
                lat: 34.05,
                lon: -118.25,
                alt: 0.1,
            },
            obs_time: "2026-06-15".into(),
            obs_end: None,
            optical: Some(false),
            elong_min: Some(30.0),
            elong_max: None,
            glat_min: None,
            glat_max: None,
            elev_min: Some(20.0),
            time_min: None,
            vmag_min: None,
            vmag_max: None,
            mag_required: None,
            helio_min: None,
            helio_max: None,
            dist_min: None,
            dist_max: None,
            maxoutput: None,
            output_sort: None,
            output_sort_r: None,
            sb_kind: Some("a".into()),
            sb_group: Some("neo".into()),
        };
        let query = params.to_query_params();
        let map: std::collections::HashMap<String, String> = query.into_iter().collect();

        assert_eq!(map.get("lat").unwrap(), "34.05");
        assert_eq!(map.get("lon").unwrap(), "-118.25");
        assert_eq!(map.get("alt").unwrap(), "0.1");
        assert_eq!(map.get("optical").unwrap(), "false");
        assert_eq!(map.get("elong-min").unwrap(), "30");
        assert_eq!(map.get("elev-min").unwrap(), "20");
        assert_eq!(map.get("sb-kind").unwrap(), "a");
        assert_eq!(map.get("sb-group").unwrap(), "neo");
        assert!(!map.contains_key("mpc-code"));
    }
}
