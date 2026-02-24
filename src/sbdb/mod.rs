//! JPL Small-Body Database (SBDB) API client.
//!
//! This module provides access to NASA JPL's Small-Body Database, covering
//! approximately 1.5 million asteroids and comets. It supports five API endpoints:
//!
//! - **SBDB lookup** — detailed data for a single object
//! - **SBDB Query** — bulk filtered queries across all objects
//! - **Close Approach Data (CAD)** — asteroid/comet close approaches to planets
//! - **Fireball** — atmospheric impact events
//! - **Sentry** — Earth impact risk monitoring
//!
//! No API key or authentication is required.
//!
//! # Quick Start
//!
//! ```no_run
//! use starfield::sbdb::SbdbClient;
//!
//! let client = SbdbClient::new().unwrap();
//!
//! // Look up asteroid Eros
//! let eros = client.lookup("Eros").unwrap();
//! println!("{} ({})", eros.object.fullname.unwrap_or_default(), eros.object.designation);
//!
//! // Query upcoming close approaches
//! use starfield::sbdb::CadParams;
//! let params = CadParams {
//!     date_min: Some("now".into()),
//!     dist_max: Some("0.05".into()),
//!     limit: Some(10),
//!     ..Default::default()
//! };
//! let approaches = client.close_approaches(&params).unwrap();
//! println!("{} close approaches found", approaches.count);
//! ```

pub mod query;
pub mod types;

pub use crate::data::sbdb::{
    CadParams, CadResponse, FireballParams, FireballResponse, SbdbClient, SbdbLookupResponse,
    SbdbQueryResponse, SentryResponse,
};

pub use types::{
    CloseApproachRecord, FireballRecord, OrbitClass, PhysicalParams, SentryEntry, Signature,
    SmallBodyObject, SmallBodyOrbit,
};
