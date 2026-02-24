//! HORIZONS API client for NASA JPL's solar system ephemeris service.
//!
//! This module provides access to the [HORIZONS](https://ssd.jpl.nasa.gov/horizons/)
//! on-line solar system data and ephemeris computation service. It can compute
//! positions, velocities, and observational quantities for over 1.5 million
//! solar system objects including planets, moons, asteroids, comets, and spacecraft.
//!
//! No API key or authentication is required.
//!
//! # Quick Start
//!
//! ```no_run
//! use starfield::horizons::{HorizonsClient, EphemerisRequest, Command, Center, TimeSpec};
//! use starfield::horizons::parser;
//!
//! let client = HorizonsClient::new().unwrap();
//!
//! // Get Mars state vectors relative to the Solar System Barycenter
//! let request = EphemerisRequest::vectors(
//!     Command::MajorBody(499),
//!     Center::SolarSystemBarycenter,
//!     TimeSpec::Range {
//!         start: "2024-01-01".into(),
//!         stop: "2024-01-02".into(),
//!         step: "1 d".into(),
//!     },
//! );
//!
//! let response = client.query(&request).unwrap();
//! let block = parser::extract_ephemeris_block(response.result.as_ref().unwrap()).unwrap();
//! let rows = parser::parse_vector_rows(block).unwrap();
//!
//! for row in &rows {
//!     println!("JD {}: ({}, {}, {}) AU", row.jd_tdb, row.x, row.y, row.z);
//! }
//! ```

pub mod parser;

pub use crate::data::horizons::{
    Center, Command, EphemType, EphemerisRequest, HorizonsClient, HorizonsResponse, LookupMatch,
    LookupResponse, ObjectGroup, OutputUnits, ReferencePlane, Signature, TimeSpec, VecCorrection,
    VecTable,
};

pub use parser::{ElementsRow, ObserverRow, VectorRow};
