//! HTTP client for the NASA JPL HORIZONS ephemeris computation service.
//!
//! HORIZONS computes positions, velocities, and observational quantities
//! for over 1.5 million solar system objects. This module provides a
//! type-safe Rust interface to both the ephemeris API and the lookup API.
//!
//! No API key or authentication is required.

use crate::jplephem::kernel::SpiceKernel;
use crate::{Result, StarfieldError};
use base64::Engine;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Base URL for the HORIZONS ephemeris API
const HORIZONS_API_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";

/// Base URL for the HORIZONS File API (POST endpoint for large requests)
const HORIZONS_FILE_API_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons_file.api";

/// Base URL for the HORIZONS lookup API
const HORIZONS_LOOKUP_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons_lookup.api";

/// URL length threshold above which `query_auto` switches from GET to POST.
/// Standard HTTP servers and proxies typically support URLs up to ~2000 chars,
/// so we use a conservative threshold to leave room for encoding overhead.
const AUTO_POST_URL_THRESHOLD: usize = 1500;

/// Ecliptic reference frame for user-defined orbital elements
#[derive(Debug, Clone, Copy)]
pub enum EclipticFrame {
    /// J2000 ecliptic and equinox
    J2000,
    /// B1950 ecliptic and equinox
    B1950,
}

impl EclipticFrame {
    fn as_str(&self) -> &'static str {
        match self {
            EclipticFrame::J2000 => "J2000",
            EclipticFrame::B1950 => "B1950",
        }
    }
}

/// One of three element-set parameterizations accepted by HORIZONS
/// for user-defined heliocentric ecliptic orbital elements.
#[derive(Debug, Clone)]
pub enum ElementSet {
    /// Periapsis distance (AU) and time of perihelion passage (JD TDB)
    Periapsis {
        /// Perihelion distance in AU
        perihelion_dist: f64,
        /// Time of perihelion passage (Julian Date, TDB)
        time_perihelion: f64,
    },
    /// Semi-major axis (AU) and mean anomaly (degrees)
    SemiMajor {
        /// Semi-major axis in AU
        semi_major_axis: f64,
        /// Mean anomaly in degrees
        mean_anomaly: f64,
    },
    /// Mean motion (degrees/day) and mean anomaly (degrees)
    MeanMotion {
        /// Mean motion in degrees/day
        mean_motion: f64,
        /// Mean anomaly in degrees
        mean_anomaly: f64,
    },
}

/// Optional magnitude parameters for user-defined asteroid objects
#[derive(Debug, Clone)]
pub struct AsteroidMagnitude {
    /// Absolute magnitude parameter H
    pub h: f64,
    /// Magnitude slope parameter G (default 0.15)
    pub g: f64,
}

/// Optional magnitude parameters for user-defined comet objects
#[derive(Debug, Clone)]
pub struct CometMagnitude {
    /// Total absolute magnitude
    pub m1: f64,
    /// Nuclear absolute magnitude
    pub m2: f64,
    /// Total magnitude scaling factor
    pub k1: f64,
    /// Nuclear magnitude scaling factor
    pub k2: f64,
}

/// User-defined orbital elements for generating ephemerides of
/// hypothetical or custom objects via the HORIZONS API.
///
/// HORIZONS accepts heliocentric ecliptic orbital elements with three
/// possible parameterizations for the orbit shape (see [`ElementSet`]).
///
/// # Example
///
/// ```
/// use starfield::horizons::{UserDefinedElements, ElementSet, EclipticFrame};
///
/// // Eros-like elements using semi-major axis + mean anomaly
/// let elements = UserDefinedElements::from_semi_major(
///     2451544.5,  // epoch (J2000.0)
///     0.2229,     // eccentricity
///     1.4583,     // semi-major axis (AU)
///     178.8,      // mean anomaly (degrees)
///     304.3,      // longitude of ascending node (degrees)
///     178.9,      // argument of perihelion (degrees)
///     10.83,      // inclination (degrees)
/// );
/// ```
#[derive(Debug, Clone)]
pub struct UserDefinedElements {
    /// User-chosen object name
    pub object_name: String,
    /// Epoch of osculating elements (Julian Date, TDB)
    pub epoch: f64,
    /// Eccentricity (dimensionless)
    pub eccentricity: f64,
    /// Orbit shape parameterization
    pub element_set: ElementSet,
    /// Inclination in degrees
    pub inclination: f64,
    /// Longitude of ascending node in degrees
    pub long_asc_node: f64,
    /// Argument of perihelion in degrees
    pub arg_perihelion: f64,
    /// Ecliptic reference frame (default J2000)
    pub ecliptic_frame: EclipticFrame,
    /// Optional asteroid magnitude parameters
    pub asteroid_magnitude: Option<AsteroidMagnitude>,
    /// Optional comet magnitude parameters
    pub comet_magnitude: Option<CometMagnitude>,
}

impl UserDefinedElements {
    /// Create user-defined elements using periapsis distance and time of perihelion.
    ///
    /// This corresponds to HORIZONS element set A (QR + TP).
    pub fn from_periapsis(
        epoch: f64,
        eccentricity: f64,
        perihelion_dist: f64,
        time_perihelion: f64,
        long_asc_node: f64,
        arg_perihelion: f64,
        inclination: f64,
    ) -> Self {
        Self {
            object_name: "UserObject".to_string(),
            epoch,
            eccentricity,
            element_set: ElementSet::Periapsis {
                perihelion_dist,
                time_perihelion,
            },
            inclination,
            long_asc_node,
            arg_perihelion,
            ecliptic_frame: EclipticFrame::J2000,
            asteroid_magnitude: None,
            comet_magnitude: None,
        }
    }

    /// Create user-defined elements using semi-major axis and mean anomaly.
    ///
    /// This corresponds to HORIZONS element set B (A + MA).
    pub fn from_semi_major(
        epoch: f64,
        eccentricity: f64,
        semi_major_axis: f64,
        mean_anomaly: f64,
        long_asc_node: f64,
        arg_perihelion: f64,
        inclination: f64,
    ) -> Self {
        Self {
            object_name: "UserObject".to_string(),
            epoch,
            eccentricity,
            element_set: ElementSet::SemiMajor {
                semi_major_axis,
                mean_anomaly,
            },
            inclination,
            long_asc_node,
            arg_perihelion,
            ecliptic_frame: EclipticFrame::J2000,
            asteroid_magnitude: None,
            comet_magnitude: None,
        }
    }

    /// Set the object name (returned in HORIZONS output headers)
    pub fn with_name(mut self, name: &str) -> Self {
        self.object_name = name.to_string();
        self
    }

    /// Set asteroid magnitude parameters (H, G)
    pub fn with_asteroid_magnitude(mut self, h: f64, g: f64) -> Self {
        self.asteroid_magnitude = Some(AsteroidMagnitude { h, g });
        self.comet_magnitude = None;
        self
    }

    /// Set comet magnitude parameters (M1, M2, K1, K2)
    pub fn with_comet_magnitude(mut self, m1: f64, m2: f64, k1: f64, k2: f64) -> Self {
        self.comet_magnitude = Some(CometMagnitude { m1, m2, k1, k2 });
        self.asteroid_magnitude = None;
        self
    }

    /// Set the ecliptic reference frame
    pub fn with_ecliptic_frame(mut self, frame: EclipticFrame) -> Self {
        self.ecliptic_frame = frame;
        self
    }

    /// Serialize the orbital elements as HORIZONS API query parameters
    fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = Vec::new();

        params.push(("OBJECT".into(), format!("'{}'", self.object_name)));
        params.push(("EPOCH".into(), format!("{}", self.epoch)));
        params.push((
            "ECLIP".into(),
            format!("'{}'", self.ecliptic_frame.as_str()),
        ));
        params.push(("EC".into(), format!("{}", self.eccentricity)));

        match &self.element_set {
            ElementSet::Periapsis {
                perihelion_dist,
                time_perihelion,
            } => {
                params.push(("QR".into(), format!("{}", perihelion_dist)));
                params.push(("TP".into(), format!("{}", time_perihelion)));
            }
            ElementSet::SemiMajor {
                semi_major_axis,
                mean_anomaly,
            } => {
                params.push(("A".into(), format!("{}", semi_major_axis)));
                params.push(("MA".into(), format!("{}", mean_anomaly)));
            }
            ElementSet::MeanMotion {
                mean_motion,
                mean_anomaly,
            } => {
                params.push(("N".into(), format!("{}", mean_motion)));
                params.push(("MA".into(), format!("{}", mean_anomaly)));
            }
        }

        params.push(("OM".into(), format!("{}", self.long_asc_node)));
        params.push(("W".into(), format!("{}", self.arg_perihelion)));
        params.push(("IN".into(), format!("{}", self.inclination)));

        if let Some(ref mag) = self.asteroid_magnitude {
            params.push(("H".into(), format!("{}", mag.h)));
            params.push(("G".into(), format!("{}", mag.g)));
        }

        if let Some(ref mag) = self.comet_magnitude {
            params.push(("M1".into(), format!("{}", mag.m1)));
            params.push(("M2".into(), format!("{}", mag.m2)));
            params.push(("K1".into(), format!("{}", mag.k1)));
            params.push(("K2".into(), format!("{}", mag.k2)));
        }

        params
    }
}

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
    /// User-supplied TLE (Two-Line Element) data for SGP4 propagation.
    ///
    /// The string contains the full TLE: optional name line, line 1, and line 2,
    /// separated by newlines. HORIZONS accepts up to 600 TLE pairs.
    Tle(String),
    /// User-defined object from custom orbital elements
    UserDefined(UserDefinedElements),
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
            Command::Tle(_) => "TLE".to_string(),
            Command::UserDefined(_) => ";".to_string(),
        }
    }

    /// Return additional query parameters for user-defined objects.
    /// Returns an empty vec for all other command types.
    fn extra_params(&self) -> Vec<(String, String)> {
        match self {
            Command::UserDefined(elements) => elements.to_query_params(),
            _ => Vec::new(),
        }
    }

    /// Create a TLE command from individual lines.
    ///
    /// # Arguments
    /// * `name` - Optional satellite name (line 0 of a 3-line element set)
    /// * `line1` - TLE line 1 (must start with '1')
    /// * `line2` - TLE line 2 (must start with '2')
    ///
    /// # Errors
    /// Returns `StarfieldError::DataError` if line1 does not start with '1'
    /// or line2 does not start with '2'.
    pub fn from_tle(name: Option<&str>, line1: &str, line2: &str) -> Result<Self> {
        let l1 = line1.trim();
        let l2 = line2.trim();

        if !l1.starts_with('1') {
            return Err(StarfieldError::DataError(
                "TLE line 1 must start with '1'".into(),
            ));
        }
        if !l2.starts_with('2') {
            return Err(StarfieldError::DataError(
                "TLE line 2 must start with '2'".into(),
            ));
        }

        let tle_string = match name {
            Some(n) => format!("{}\n{}\n{}", n.trim(), l1, l2),
            None => format!("{}\n{}", l1, l2),
        };

        Ok(Command::Tle(tle_string))
    }
}

/// Ephemeris output type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EphemType {
    /// Observer-table: sky-plane observables (RA/Dec, magnitude, etc.)
    Observer,
    /// Vectors: Cartesian state vectors (X, Y, Z, VX, VY, VZ)
    Vectors,
    /// Elements: osculating Keplerian orbital elements
    Elements,
    /// SPK: binary SPICE kernel file (small bodies only)
    Spk,
    /// Approach: close-approach tables for small bodies
    Approach,
}

impl EphemType {
    fn as_str(&self) -> &'static str {
        match self {
            EphemType::Observer => "OBSERVER",
            EphemType::Vectors => "VECTORS",
            EphemType::Elements => "ELEMENTS",
            EphemType::Spk => "SPK",
            EphemType::Approach => "APPROACH",
        }
    }
}

/// Close-approach table detail level
#[derive(Debug, Clone, Copy)]
pub enum CaTableType {
    /// Standard columns: Date, Body, CA Dist, MinDist, MaxDist, Vrel, TCA3Sg, Nsigs, P_i/p
    Standard,
    /// Extended columns: adds JDTDB, B-plane parameters (SMaA, SMiA, B.T, B.R, Theta)
    Extended,
}

impl CaTableType {
    fn as_str(&self) -> &'static str {
        match self {
            CaTableType::Standard => "STANDARD",
            CaTableType::Extended => "EXTENDED",
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
    /// Close-approach table type (APPROACH only)
    pub ca_table_type: Option<CaTableType>,
    /// Max 3-sigma time-of-CA uncertainty in minutes (APPROACH only)
    pub tca3sg_limit: Option<u32>,
    /// Small-body close-approach distance limit in AU (APPROACH only)
    pub calim_sb: Option<f64>,
    /// Per-planet close-approach distance limits in AU (APPROACH only).
    /// Order: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto, Moon.
    pub calim_pl: Option<[f64; 10]>,
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
            ca_table_type: None,
            tca3sg_limit: None,
            calim_sb: None,
            calim_pl: None,
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
            ca_table_type: None,
            tca3sg_limit: None,
            calim_sb: None,
            calim_pl: None,
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
            ca_table_type: None,
            tca3sg_limit: None,
            calim_sb: None,
            calim_pl: None,
        }
    }

    /// Create a request for close-approach tables
    ///
    /// Generates tables of close approaches between a small body and
    /// major solar system bodies (planets and the 16 largest asteroids).
    pub fn approach(command: Command, center: Center, time_spec: TimeSpec) -> Self {
        Self {
            command,
            ephem_type: EphemType::Approach,
            center,
            time_spec,
            obj_data: false,
            vec_table: None,
            out_units: None,
            vec_corr: None,
            ref_plane: None,
            quantities: None,
            ang_format: None,
            csv_format: false,
            extra_prec: false,
            ca_table_type: Some(CaTableType::Standard),
            tca3sg_limit: None,
            calim_sb: None,
            calim_pl: None,
        }
    }

    /// Create a request for a binary SPK ephemeris file (small bodies only)
    ///
    /// SPK requests only use `COMMAND`, `START_TIME`, and `STOP_TIME`.
    /// All other ephemeris parameters are ignored by the HORIZONS API.
    pub fn spk(command: Command, start: &str, stop: &str) -> Self {
        Self {
            command,
            ephem_type: EphemType::Spk,
            center: Center::SolarSystemBarycenter,
            time_spec: TimeSpec::Range {
                start: start.to_string(),
                stop: stop.to_string(),
                step: "1 d".to_string(),
            },
            obj_data: false,
            vec_table: None,
            out_units: None,
            vec_corr: None,
            ref_plane: None,
            quantities: None,
            ang_format: None,
            csv_format: false,
            extra_prec: false,
            ca_table_type: None,
            tca3sg_limit: None,
            calim_sb: None,
            calim_pl: None,
        }
    }

    /// Build a HORIZONS input file string for the File API (POST endpoint).
    ///
    /// The input file uses key=value pairs between `!$$SOF` and `!$$EOF` markers.
    /// Unlike the GET query parameters, the `format` key is not included here
    /// because it is sent as a separate form field in the POST request.
    pub fn to_input_file(&self) -> String {
        let params = self.to_horizons_params();
        let mut lines = Vec::with_capacity(params.len() + 2);
        lines.push("!$$SOF".to_string());
        for (key, value) in &params {
            lines.push(format!("{}={}", key, value));
        }
        lines.push("!$$EOF".to_string());
        lines.join("\n")
    }

    /// Estimate the URL length that a GET request would produce.
    ///
    /// This is used by `query_auto` to decide whether to use GET or POST.
    pub fn estimated_url_length(&self) -> usize {
        let params = self.to_query_params();
        // Base URL + '?'
        let mut len = HORIZONS_API_URL.len() + 1;
        for (i, (key, value)) in params.iter().enumerate() {
            if i > 0 {
                len += 1; // '&'
            }
            len += key.len() + 1 + value.len(); // key=value
        }
        len
    }

    /// Build the HORIZONS parameter key-value pairs (without the `format` key).
    ///
    /// Shared between `to_query_params` (GET) and `to_input_file` (POST).
    fn to_horizons_params(&self) -> Vec<(String, String)> {
        let mut params: Vec<(String, String)> = Vec::new();

        params.push((
            "COMMAND".into(),
            format!("'{}'", self.command.to_query_value()),
        ));

        // For TLE commands, include the TLE data as a separate parameter
        if let Command::Tle(ref tle_data) = self.command {
            params.push(("TLE".into(), format!("'{}'", tle_data)));
        }

        params.push(("MAKE_EPHEM".into(), "YES".into()));
        params.push(("EPHEM_TYPE".into(), self.ephem_type.as_str().into()));
        params.push((
            "CENTER".into(),
            format!("'{}'", self.center.to_query_value()),
        ));

        // Append user-defined element parameters if present
        params.extend(self.command.extra_params());

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

        if let Some(ca_type) = &self.ca_table_type {
            params.push(("CA_TABLE_TYPE".into(), format!("'{}'", ca_type.as_str())));
        }

        if let Some(limit) = self.tca3sg_limit {
            params.push(("TCA3SG_LIMIT".into(), format!("'{}'", limit)));
        }

        if let Some(sb) = self.calim_sb {
            params.push(("CALIM_SB".into(), format!("'{}'", sb)));
        }

        if let Some(pl) = &self.calim_pl {
            let values: Vec<String> = pl.iter().map(|v| format!("{}", v)).collect();
            params.push(("CALIM_PL".into(), format!("'{}'", values.join(","))));
        }

        params
    }

    /// Build query parameters for the HTTP GET request.
    ///
    /// Prepends `format=json` to the shared HORIZONS parameter list.
    fn to_query_params(&self) -> Vec<(String, String)> {
        let mut params = vec![("format".into(), "json".into())];
        params.extend(self.to_horizons_params());
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

/// Decoded response from a HORIZONS SPK ephemeris request
#[derive(Debug, Clone)]
pub struct SpkResponse {
    /// Raw binary SPK file content
    pub raw_spk: Vec<u8>,
    /// Suggested filename from the API (e.g., "2000433.bsp")
    pub spk_file_id: Option<String>,
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

    /// Execute an ephemeris query via the File API (POST endpoint).
    ///
    /// This sends the request parameters as a HORIZONS input file in the body
    /// of a POST request. Use this for large requests (e.g., long TLIST values)
    /// that would exceed URL length limits on the standard GET endpoint.
    pub fn query_file(&self, request: &EphemerisRequest) -> Result<HorizonsResponse> {
        let input_file = request.to_input_file();
        let form = reqwest::blocking::multipart::Form::new()
            .text("input", input_file)
            .text("format", "json");

        let response = self
            .client
            .post(HORIZONS_FILE_API_URL)
            .multipart(form)
            .send()
            .map_err(|e| {
                StarfieldError::DataError(format!("HORIZONS file request failed: {}", e))
            })?;

        if !response.status().is_success() {
            return Err(StarfieldError::DataError(format!(
                "HORIZONS File API returned HTTP {}",
                response.status()
            )));
        }

        let body: HorizonsResponse = response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse HORIZONS file response: {}", e))
        })?;

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

    /// Automatically choose GET or POST based on estimated URL length.
    ///
    /// For small requests that fit comfortably in a URL, this uses the standard
    /// GET endpoint. For large requests (e.g., with many Julian Day entries in
    /// TLIST) that would produce URLs longer than ~1500 characters, this falls
    /// back to the File API POST endpoint.
    pub fn query_auto(&self, request: &EphemerisRequest) -> Result<HorizonsResponse> {
        if request.estimated_url_length() > AUTO_POST_URL_THRESHOLD {
            self.query_file(request)
        } else {
            self.query(request)
        }
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

    /// Generate an SPK binary ephemeris and return the decoded bytes
    ///
    /// Sends an SPK-type query to HORIZONS, extracts the base64-encoded `spk`
    /// field from the JSON response, and decodes it to raw bytes.
    /// Only works for small bodies (asteroids and comets).
    pub fn generate_spk(&self, request: &EphemerisRequest) -> Result<SpkResponse> {
        if request.ephem_type != EphemType::Spk {
            return Err(StarfieldError::DataError(
                "generate_spk requires EphemType::Spk".to_string(),
            ));
        }

        let response = self.query(request)?;

        let spk_b64 = response.spk.ok_or_else(|| {
            StarfieldError::DataError(
                "HORIZONS response missing 'spk' field for SPK request".to_string(),
            )
        })?;

        let raw_spk = base64::engine::general_purpose::STANDARD
            .decode(&spk_b64)
            .map_err(|e| {
                StarfieldError::DataError(format!("Failed to decode SPK base64 data: {}", e))
            })?;

        Ok(SpkResponse {
            raw_spk,
            spk_file_id: response.spk_file_id,
        })
    }

    /// Generate an SPK binary ephemeris and save it to a file
    ///
    /// Combines `generate_spk` with writing the decoded bytes to disk.
    pub fn generate_spk_to_file(
        &self,
        request: &EphemerisRequest,
        path: &Path,
    ) -> Result<SpkResponse> {
        let spk_response = self.generate_spk(request)?;
        std::fs::write(path, &spk_response.raw_spk)?;
        Ok(spk_response)
    }

    /// Generate an SPK binary ephemeris and load it as a SpiceKernel
    ///
    /// Combines `generate_spk` with parsing the binary data into a
    /// ready-to-use `SpiceKernel` for position computations.
    pub fn generate_spk_kernel(&self, request: &EphemerisRequest) -> Result<SpiceKernel> {
        let spk_response = self.generate_spk(request)?;
        SpiceKernel::from_bytes(&spk_response.raw_spk)
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse SPK kernel: {}", e)))
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
    fn test_command_user_defined() {
        let elements = UserDefinedElements::from_semi_major(
            2451544.5, 0.2229, 1.4583, 178.8, 304.3, 178.9, 10.83,
        );
        assert_eq!(Command::UserDefined(elements).to_query_value(), ";");
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

    // ISS TLE for HORIZONS TLE input testing
    const ISS_TLE_LINE1: &str =
        "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993";
    const ISS_TLE_LINE2: &str =
        "2 25544  51.6420  30.2134 0002345 210.5678 149.4246 15.49456789123456";

    #[test]
    fn test_command_tle_query_value() {
        let cmd = Command::Tle("test data".to_string());
        assert_eq!(cmd.to_query_value(), "TLE");
    }

    #[test]
    fn test_command_from_tle_with_name() {
        let cmd = Command::from_tle(Some("ISS (ZARYA)"), ISS_TLE_LINE1, ISS_TLE_LINE2)
            .expect("from_tle should succeed");
        assert_eq!(cmd.to_query_value(), "TLE");
        if let Command::Tle(data) = &cmd {
            assert!(data.starts_with("ISS (ZARYA)\n1 25544"));
            assert!(data.contains("\n2 25544"));
            // Three lines: name, line1, line2
            assert_eq!(data.lines().count(), 3);
        } else {
            panic!("Expected Command::Tle variant");
        }
    }

    #[test]
    fn test_command_from_tle_without_name() {
        let cmd =
            Command::from_tle(None, ISS_TLE_LINE1, ISS_TLE_LINE2).expect("from_tle should succeed");
        if let Command::Tle(data) = &cmd {
            assert!(data.starts_with("1 25544"));
            // Two lines: line1, line2
            assert_eq!(data.lines().count(), 2);
        } else {
            panic!("Expected Command::Tle variant");
        }
    }

    #[test]
    fn test_command_from_tle_validates_line1() {
        let result = Command::from_tle(None, "2 bad line", ISS_TLE_LINE2);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("line 1 must start with '1'"));
    }

    #[test]
    fn test_command_from_tle_validates_line2() {
        let result = Command::from_tle(None, ISS_TLE_LINE1, "1 bad line");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("line 2 must start with '2'"));
    }

    #[test]
    fn test_tle_request_params() {
        let cmd = Command::from_tle(Some("ISS (ZARYA)"), ISS_TLE_LINE1, ISS_TLE_LINE2)
            .expect("from_tle should succeed");
        let req = EphemerisRequest::vectors(
            cmd,
            Center::Geocentric,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 h".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        // COMMAND should be 'TLE'
        assert_eq!(map.get("COMMAND").unwrap(), "'TLE'");

        // TLE parameter should contain the full TLE data
        let tle_param = map.get("TLE").expect("TLE parameter missing");
        assert!(tle_param.contains("ISS (ZARYA)"));
        assert!(tle_param.contains("1 25544"));
        assert!(tle_param.contains("2 25544"));
    }

    #[test]
    fn test_tle_request_no_tle_param_for_non_tle() {
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

        // Non-TLE commands should not have a TLE parameter
        assert!(map.get("TLE").is_none());
    }

    #[test]
    #[ignore]
    fn test_horizons_tle_iss_query() {
        let cmd = Command::from_tle(Some("ISS (ZARYA)"), ISS_TLE_LINE1, ISS_TLE_LINE2)
            .expect("from_tle should succeed");
        let req = EphemerisRequest::vectors(
            cmd,
            Center::Geocentric,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 h".into(),
            },
        );
        let client = HorizonsClient::new().expect("Failed to create client");
        let response = client.query(&req).expect("HORIZONS TLE query failed");
        let result = response.result.expect("No result in response");
        assert!(
            result.contains("$$SOE"),
            "Response should contain ephemeris data"
        );
        assert!(
            result.contains("$$EOE"),
            "Response should contain ephemeris end marker"
        );
    }

    #[test]
    fn test_approach_request_params() {
        let req = EphemerisRequest::approach(
            Command::Asteroid(99942),
            Center::BodyCenter(10),
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2035-01-01".into(),
                step: "1 d".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("COMMAND").unwrap(), "'99942;'");
        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "APPROACH");
        assert_eq!(map.get("CA_TABLE_TYPE").unwrap(), "'STANDARD'");
        assert!(!map.contains_key("CSV_FORMAT"));
    }

    #[test]
    fn test_approach_request_extended_with_limits() {
        let mut req = EphemerisRequest::approach(
            Command::Asteroid(99942),
            Center::BodyCenter(10),
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2050-01-01".into(),
                step: "1 d".into(),
            },
        );
        req.ca_table_type = Some(CaTableType::Extended);
        req.tca3sg_limit = Some(14400);
        req.calim_sb = Some(0.2);
        req.calim_pl = Some([0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 0.1, 0.003]);

        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("CA_TABLE_TYPE").unwrap(), "'EXTENDED'");
        assert_eq!(map.get("TCA3SG_LIMIT").unwrap(), "'14400'");
        assert_eq!(map.get("CALIM_SB").unwrap(), "'0.2'");
        assert_eq!(
            map.get("CALIM_PL").unwrap(),
            "'0.1,0.1,0.1,0.1,1,1,1,1,0.1,0.003'"
        );
    }

    #[test]
    fn test_ca_table_type_as_str() {
        assert_eq!(CaTableType::Standard.as_str(), "STANDARD");
        assert_eq!(CaTableType::Extended.as_str(), "EXTENDED");
    }

    #[test]
    fn test_user_defined_periapsis_params() {
        let elements = UserDefinedElements::from_periapsis(
            2451544.5, // epoch J2000.0
            0.9671,    // eccentricity
            0.5871,    // perihelion distance AU
            2451000.5, // time of perihelion JD
            58.15,     // longitude of ascending node
            111.87,    // argument of perihelion
            162.26,    // inclination
        );
        let params = elements.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("OBJECT").unwrap(), "'UserObject'");
        assert_eq!(map.get("EPOCH").unwrap(), "2451544.5");
        assert_eq!(map.get("ECLIP").unwrap(), "'J2000'");
        assert_eq!(map.get("EC").unwrap(), "0.9671");
        assert_eq!(map.get("QR").unwrap(), "0.5871");
        assert_eq!(map.get("TP").unwrap(), "2451000.5");
        assert_eq!(map.get("OM").unwrap(), "58.15");
        assert_eq!(map.get("W").unwrap(), "111.87");
        assert_eq!(map.get("IN").unwrap(), "162.26");
        assert!(map.get("A").is_none());
        assert!(map.get("MA").is_none());
        assert!(map.get("N").is_none());
    }

    #[test]
    fn test_user_defined_semi_major_params() {
        let elements = UserDefinedElements::from_semi_major(
            2451544.5, 0.2229, 1.4583, 178.8, 304.3, 178.9, 10.83,
        )
        .with_name("Eros")
        .with_asteroid_magnitude(11.16, 0.46);

        let params = elements.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("OBJECT").unwrap(), "'Eros'");
        assert_eq!(map.get("EPOCH").unwrap(), "2451544.5");
        assert_eq!(map.get("ECLIP").unwrap(), "'J2000'");
        assert_eq!(map.get("EC").unwrap(), "0.2229");
        assert_eq!(map.get("A").unwrap(), "1.4583");
        assert_eq!(map.get("MA").unwrap(), "178.8");
        assert_eq!(map.get("OM").unwrap(), "304.3");
        assert_eq!(map.get("W").unwrap(), "178.9");
        assert_eq!(map.get("IN").unwrap(), "10.83");
        assert_eq!(map.get("H").unwrap(), "11.16");
        assert_eq!(map.get("G").unwrap(), "0.46");
        assert!(map.get("QR").is_none());
        assert!(map.get("TP").is_none());
        assert!(map.get("N").is_none());
    }

    #[test]
    fn test_user_defined_mean_motion_params() {
        let elements = UserDefinedElements {
            object_name: "TestObj".to_string(),
            epoch: 2460000.5,
            eccentricity: 0.1,
            element_set: ElementSet::MeanMotion {
                mean_motion: 0.524,
                mean_anomaly: 45.0,
            },
            inclination: 5.0,
            long_asc_node: 100.0,
            arg_perihelion: 200.0,
            ecliptic_frame: EclipticFrame::J2000,
            asteroid_magnitude: None,
            comet_magnitude: None,
        };
        let params = elements.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("OBJECT").unwrap(), "'TestObj'");
        assert_eq!(map.get("N").unwrap(), "0.524");
        assert_eq!(map.get("MA").unwrap(), "45");
        assert!(map.get("A").is_none());
        assert!(map.get("QR").is_none());
        assert!(map.get("TP").is_none());
    }

    #[test]
    fn test_user_defined_comet_magnitude() {
        let elements =
            UserDefinedElements::from_periapsis(2451544.5, 0.995, 0.23, 2451000.5, 0.0, 0.0, 0.0)
                .with_comet_magnitude(5.0, 12.0, 10.0, 5.0);

        let params = elements.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("M1").unwrap(), "5");
        assert_eq!(map.get("M2").unwrap(), "12");
        assert_eq!(map.get("K1").unwrap(), "10");
        assert_eq!(map.get("K2").unwrap(), "5");
        assert!(map.get("H").is_none());
        assert!(map.get("G").is_none());
    }

    #[test]
    fn test_user_defined_ecliptic_frame() {
        let elements =
            UserDefinedElements::from_semi_major(2451544.5, 0.2, 1.5, 90.0, 0.0, 0.0, 0.0)
                .with_ecliptic_frame(EclipticFrame::B1950);

        let params = elements.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("ECLIP").unwrap(), "'B1950'");
    }

    #[test]
    fn test_user_defined_in_ephemeris_request() {
        let elements = UserDefinedElements::from_semi_major(
            2451544.5, 0.2229, 1.4583, 178.8, 304.3, 178.9, 10.83,
        )
        .with_name("Eros");

        let req = EphemerisRequest::vectors(
            Command::UserDefined(elements),
            Center::SolarSystemBarycenter,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        // COMMAND should be the semicolon sentinel for user-defined
        assert_eq!(map.get("COMMAND").unwrap(), "';'");
        // Element parameters should be present
        assert_eq!(map.get("OBJECT").unwrap(), "'Eros'");
        assert_eq!(map.get("EPOCH").unwrap(), "2451544.5");
        assert_eq!(map.get("EC").unwrap(), "0.2229");
        assert_eq!(map.get("A").unwrap(), "1.4583");
        assert_eq!(map.get("MA").unwrap(), "178.8");
        // Standard ephemeris params should also be present
        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "VECTORS");
        assert_eq!(map.get("CENTER").unwrap(), "'500@0'");
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

    #[test]
    fn test_spk_ephem_type_serializes() {
        let req = EphemerisRequest::spk(Command::Asteroid(433), "2024-01-01", "2025-01-01");
        let params = req.to_query_params();
        let map: HashMap<String, String> = params.into_iter().collect();

        assert_eq!(map.get("EPHEM_TYPE").unwrap(), "SPK");
        assert_eq!(map.get("COMMAND").unwrap(), "'433;'");
        assert_eq!(map.get("START_TIME").unwrap(), "'2024-01-01'");
        assert_eq!(map.get("STOP_TIME").unwrap(), "'2025-01-01'");
    }

    #[test]
    fn test_spk_base64_decode() {
        let original_bytes: Vec<u8> = vec![
            0x44, 0x41, 0x46, 0x2F, 0x53, 0x50, 0x4B, 0x20, // "DAF/SPK "
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        ];
        let encoded = base64::engine::general_purpose::STANDARD.encode(&original_bytes);
        let decoded = base64::engine::general_purpose::STANDARD
            .decode(&encoded)
            .unwrap();
        assert_eq!(decoded, original_bytes);
    }

    #[test]
    fn test_spk_response_struct() {
        let resp = SpkResponse {
            raw_spk: vec![0xDE, 0xAD, 0xBE, 0xEF],
            spk_file_id: Some("2000433.bsp".to_string()),
        };
        assert_eq!(resp.raw_spk.len(), 4);
        assert_eq!(resp.spk_file_id.as_deref(), Some("2000433.bsp"));
    }

    #[test]
    fn test_input_file_vectors_range() {
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let input = req.to_input_file();

        assert!(input.starts_with("!$$SOF"));
        assert!(input.ends_with("!$$EOF"));
        assert!(input.contains("COMMAND='499'"));
        assert!(input.contains("MAKE_EPHEM=YES"));
        assert!(input.contains("EPHEM_TYPE=VECTORS"));
        assert!(input.contains("CENTER='500@0'"));
        assert!(input.contains("START_TIME='2024-01-01'"));
        assert!(input.contains("STOP_TIME='2024-01-02'"));
        assert!(input.contains("STEP_SIZE='1 d'"));
        assert!(input.contains("CSV_FORMAT=YES"));
        assert!(input.contains("OBJ_DATA=NO"));
        assert!(input.contains("OUT_UNITS='AU-D'"));
        assert!(input.contains("VEC_TABLE='3'"));
        assert!(input.contains("VEC_CORR='NONE'"));
        assert!(input.contains("REF_PLANE='ECLIPTIC'"));
        // The input file should NOT contain format=json (that goes as a form field)
        assert!(!input.contains("format=json"));
    }

    #[test]
    fn test_input_file_tlist() {
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::JulianDayList(vec![2451545.0, 2451546.0, 2451547.0]),
        );
        let input = req.to_input_file();

        assert!(input.starts_with("!$$SOF"));
        assert!(input.ends_with("!$$EOF"));
        assert!(input.contains("TLIST=2451545,2451546,2451547"));
        assert!(!input.contains("START_TIME"));
    }

    #[test]
    fn test_input_file_observer() {
        let req = EphemerisRequest::observer(
            Command::MajorBody(499),
            Center::Geocentric,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let input = req.to_input_file();

        assert!(input.contains("EPHEM_TYPE=OBSERVER"));
        assert!(input.contains("QUANTITIES='1,9,20,23'"));
        assert!(input.contains("ANG_FORMAT='DEG'"));
        assert!(input.contains("EXTRA_PREC=YES"));
    }

    #[test]
    fn test_query_auto_picks_get_for_small_request() {
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let url_len = req.estimated_url_length();
        assert!(
            url_len < AUTO_POST_URL_THRESHOLD,
            "Small request URL length {} should be below threshold {}",
            url_len,
            AUTO_POST_URL_THRESHOLD
        );
    }

    #[test]
    fn test_query_auto_picks_post_for_large_request() {
        // Create a request with many Julian Days to push URL length past the threshold
        let jds: Vec<f64> = (0..200).map(|i| 2451545.0 + i as f64).collect();
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::JulianDayList(jds),
        );
        let url_len = req.estimated_url_length();
        assert!(
            url_len > AUTO_POST_URL_THRESHOLD,
            "Large request URL length {} should exceed threshold {}",
            url_len,
            AUTO_POST_URL_THRESHOLD
        );
    }

    #[test]
    fn test_query_params_still_include_format_json() {
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
        assert_eq!(
            map.get("format").unwrap(),
            "json",
            "GET params must include format=json"
        );
    }

    #[test]
    #[ignore]
    fn test_generate_spk_asteroid_433_eros() {
        let client = HorizonsClient::new().expect("Failed to create HORIZONS client");
        let request = EphemerisRequest::spk(Command::Asteroid(433), "2024-01-01", "2025-01-01");
        let spk_response = client
            .generate_spk(&request)
            .expect("Failed to generate SPK for 433 Eros");

        // SPK binary files start with the DAF file record
        assert!(
            spk_response.raw_spk.len() > 1024,
            "SPK file too small: {} bytes",
            spk_response.raw_spk.len()
        );
        let header = std::str::from_utf8(&spk_response.raw_spk[..7]).unwrap_or("");
        assert_eq!(
            header, "DAF/SPK",
            "SPK file should start with DAF/SPK magic"
        );

        // Verify we can parse it as a SpiceKernel
        let _kernel = SpiceKernel::from_bytes(&spk_response.raw_spk)
            .expect("Failed to parse generated SPK as SpiceKernel");
    }

    #[test]
    #[ignore]
    fn test_file_api_post_mars_vectors() {
        let client = HorizonsClient::new().expect("Failed to create client");
        let req = EphemerisRequest::vectors(
            Command::MajorBody(499),
            Center::SolarSystemBarycenter,
            TimeSpec::Range {
                start: "2024-01-01".into(),
                stop: "2024-01-02".into(),
                step: "1 d".into(),
            },
        );
        let response = client.query_file(&req).expect("File API query failed");
        assert!(
            response.result.is_some(),
            "Expected result text in File API response"
        );
        let result = response.result.unwrap();
        assert!(
            result.contains("$$SOE"),
            "Expected ephemeris data block in result"
        );
    }
}
