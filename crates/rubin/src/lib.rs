//! Clients for Vera C. Rubin Observatory LSST alert brokers.
//!
//! The Rubin Observatory distributes transient alerts through seven community
//! brokers. Each broker receives the full Kafka alert stream and exposes
//! filtered/enriched data through its own API.
//!
//! This crate provides typed Rust clients for the REST APIs of each broker.
//!
//! # Brokers
//!
//! | Broker | Auth | Status |
//! |--------|------|--------|
//! | [ALeRCE](https://alerce.online) | None | Public REST API |
//! | [ANTARES](https://antares.noirlab.edu) | None (search) | Public REST API |
//! | [Fink](https://ztf.fink-portal.org) | None | Public REST API |
//! | [Lasair](https://lasair-ztf.lsst.ac.uk) | API token | Requires registration |
//! | [Pitt-Google](https://pitt-broker.readthedocs.io) | GCP credentials | Google Cloud native |
//! | [AMPEL](https://ampel-ztf.zeuthen.desy.de) | Bearer token | GitHub org membership |
//! | [Babamul](https://babamul.caltech.edu) | API token | Invitation only |
//!
//! # Example
//!
//! ```no_run
//! use starfield_rubin::AlerceClient;
//!
//! let client = AlerceClient::new().unwrap();
//! let obj = client.get_object("ZTF21aakilyd").unwrap();
//! println!("RA={}, Dec={}", obj.meanra, obj.meandec);
//! ```

pub mod alerce;
pub mod ampel;
pub mod antares;
pub mod babamul;
pub mod fink;
pub mod lasair;
pub mod pitt_google;

pub use alerce::AlerceClient;
pub use ampel::AmpelClient;
pub use antares::AntaresClient;
pub use babamul::BabamulClient;
pub use fink::FinkClient;
pub use lasair::LasairClient;
pub use pitt_google::PittGoogleClient;

/// All broker API base URLs, for reachability testing.
pub fn all_broker_api_urls() -> Vec<(&'static str, &'static str)> {
    vec![
        ("ALeRCE", alerce::API_BASE_URL),
        ("ANTARES", antares::API_BASE_URL),
        ("Fink", fink::API_BASE_URL),
        ("Lasair", lasair::API_BASE_URL),
        ("AMPEL catalog", ampel::CATALOG_API_URL),
        ("AMPEL archive", ampel::ARCHIVE_API_URL),
        ("Babamul", babamul::API_BASE_URL),
    ]
}

/// All broker documentation/registration URLs, for link verification.
pub fn all_broker_doc_urls() -> Vec<(&'static str, &'static str)> {
    vec![
        ("ALeRCE docs", alerce::DOCS_URL),
        ("ANTARES portal", antares::PORTAL_URL),
        ("ANTARES registration", antares::REGISTRATION_URL),
        ("Fink portal", fink::PORTAL_URL),
        ("Lasair portal", lasair::PORTAL_URL),
        ("Pitt-Google docs", pitt_google::DOCS_URL),
        ("AMPEL catalog docs", ampel::CATALOG_DOCS_URL),
        ("AMPEL archive docs", ampel::ARCHIVE_DOCS_URL),
        ("Babamul portal", babamul::PORTAL_URL),
        ("Babamul docs", babamul::DOCS_URL),
    ]
}
