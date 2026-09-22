//! ANTARES alert broker client.
//!
//! [ANTARES](https://antares.noirlab.edu) is operated by NOIRLab and the
//! University of Arizona. It cross-matches alerts with multi-wavelength
//! catalogs and provides ElasticSearch-powered queries.
//!
//! **Authentication:** None required for search queries.
//! Streaming access requires credentials from the ANTARES team.
//!
//! **Registration:** <https://antares.noirlab.edu/register>
//!
//! **Documentation:** <https://nsf-noirlab.gitlab.io/csdc/antares/client/>

use serde::Deserialize;
use serde_json::Value;
use starfield_core::data::source_utils::{build_http_client, check_response_status};
use starfield_core::{Result, StarfieldError};

/// REST API base URL.
pub const API_BASE_URL: &str = "https://api.antares.noirlab.edu/v1";

/// Web portal URL.
pub const PORTAL_URL: &str = "https://antares.noirlab.edu";

/// Registration URL.
pub const REGISTRATION_URL: &str = "https://antares.noirlab.edu/register";

/// Client for the ANTARES broker REST API.
pub struct AntaresClient {
    client: reqwest::blocking::Client,
}

/// An ANTARES locus (aggregated alert object).
#[derive(Debug, Deserialize)]
pub struct AntaresLocus {
    /// ANTARES locus identifier.
    #[serde(default)]
    pub locus_id: Option<String>,
    /// Right ascension (degrees).
    #[serde(default)]
    pub ra: Option<f64>,
    /// Declination (degrees).
    #[serde(default)]
    pub dec: Option<f64>,
    /// Associated properties as raw JSON.
    #[serde(default)]
    pub properties: Value,
}

impl AntaresClient {
    /// Create a new ANTARES client.
    pub fn new() -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self { client })
    }

    /// Look up a locus by its ANTARES ID.
    pub fn get_by_id(&self, locus_id: &str) -> Result<Value> {
        let url = format!("{}/loci/{}", API_BASE_URL, locus_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ANTARES request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ANTARES response: {}", e))
        })
    }

    /// Look up a locus by ZTF object ID.
    pub fn get_by_ztf_object_id(&self, ztf_id: &str) -> Result<Value> {
        let url = format!("{}/loci", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .query(&[("ztf_object_id", ztf_id)])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ANTARES request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ANTARES response: {}", e))
        })
    }

    /// Perform a cone search.
    ///
    /// `ra` and `dec` are in degrees, `radius` is in arcseconds.
    pub fn cone_search(&self, ra: f64, dec: f64, radius_arcsec: f64) -> Result<Value> {
        let url = format!("{}/loci", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .query(&[
                    ("ra", ra.to_string()),
                    ("dec", dec.to_string()),
                    ("radius", radius_arcsec.to_string()),
                ])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ANTARES request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ANTARES response: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(AntaresClient::new().is_ok());
    }

    #[test]
    fn test_api_url_matches_docs() {
        assert!(
            API_BASE_URL.contains("antares.noirlab.edu"),
            "API_BASE_URL should point to antares.noirlab.edu"
        );
        assert!(
            PORTAL_URL.contains("antares.noirlab.edu"),
            "PORTAL_URL should point to antares.noirlab.edu"
        );
    }
}
