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
}
