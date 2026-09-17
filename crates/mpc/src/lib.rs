//! Minor Planet Center (MPC) data client
//!
//! Provides access to the MPC's astronomical data products:
//!
//! - **MPCORB catalog** — orbital elements for all numbered and unnumbered
//!   minor planets (asteroids, TNOs, comets)
//! - **Observatory codes** — geographic positions of all registered observatories
//! - **Observation records** — 80-column astrometric observation format parser
//!
//! No API key or authentication is required. Bulk data files are downloaded
//! directly from <https://minorplanetcenter.net>.
//!
//! # Quick Start
//!
//! ```no_run
//! use starfield_mpc::MpcClient;
//!
//! let mut client = MpcClient::new().unwrap();
//! let observatories = client.fetch_observatory_codes().unwrap();
//! println!("Loaded {} observatories", observatories.len());
//! ```

pub mod client;
pub mod ephemeris;
pub mod mpcorb;
pub mod observation;
pub mod observatory;

pub use client::MpcClient;
pub use ephemeris::hg_apparent_magnitude;
pub use mpcorb::{parse_mpcorb, parse_mpcorb_line, unpack_epoch, MpcOrbRecord};
pub use observation::{parse_observation_line, parse_observations, Observation};
pub use observatory::{parse_observatory_codes, parse_observatory_line, Observatory};
