//! Babamul alert broker client.
//!
//! [Babamul](https://babamul.caltech.edu) is a Caltech-hosted broker built on
//! [SkyPortal](https://skyportal.io). It provides a rich REST API for managing
//! sources, photometry, spectra, and group-based access control.
//!
//! **Authentication:** API token required (invitation only).
//! Contact `babamul@lists.astro.caltech.edu` to request access.
//! Once registered, find your token in the SkyPortal profile page.
//!
//! **Portal:** <https://babamul.caltech.edu>
//!
//! **Documentation:** <https://docs.babamul.dev>

use serde::Deserialize;
use serde_json::Value;
use starfield_core::data::source_utils::{build_http_client, check_response_status};
use starfield_core::{Result, StarfieldError};

/// REST API base URL.
pub const API_BASE_URL: &str = "https://babamul.caltech.edu/api";

/// Web portal URL.
pub const PORTAL_URL: &str = "https://babamul.caltech.edu";

/// Documentation URL.
pub const DOCS_URL: &str = "https://docs.babamul.dev";

/// A SkyPortal source summary.
#[derive(Debug, Deserialize)]
pub struct BabamulSource {
    /// Source identifier.
    pub id: String,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Redshift (if available).
    #[serde(default)]
    pub redshift: Option<f64>,
    /// Classification (if available).
    #[serde(default)]
    pub classification: Option<Vec<Value>>,
}

/// Client for the Babamul (SkyPortal) broker REST API.
///
/// Requires an API token. This broker is invitation-only — contact
/// `babamul@lists.astro.caltech.edu` to request access.
pub struct BabamulClient {
    client: reqwest::blocking::Client,
    token: String,
}

impl BabamulClient {
    /// Create a new Babamul client with the given API token.
    pub fn new(token: &str) -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            token: token.to_string(),
        })
    }

    fn auth_header(&self) -> String {
        format!("token {}", self.token)
    }

    /// Get a source by its identifier.
    pub fn get_source(&self, source_id: &str) -> Result<Value> {
        let url = format!("{}/sources/{}", API_BASE_URL, source_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .header("Authorization", self.auth_header())
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Babamul request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Babamul response: {}", e))
        })
    }

    /// Get photometry for a source.
    pub fn get_photometry(&self, source_id: &str) -> Result<Value> {
        let url = format!("{}/sources/{}/photometry", API_BASE_URL, source_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .header("Authorization", self.auth_header())
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Babamul request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Babamul response: {}", e))
        })
    }

    /// Get spectra for a source.
    pub fn get_spectra(&self, source_id: &str) -> Result<Value> {
        let url = format!("{}/sources/{}/spectra", API_BASE_URL, source_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .header("Authorization", self.auth_header())
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Babamul request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Babamul response: {}", e))
        })
    }

    /// Get candidates (unvetted sources) with optional filters.
    ///
    /// Pass filter parameters as query key-value pairs, e.g.
    /// `[("numPerPage", "10"), ("savedStatus", "all")]`.
    pub fn get_candidates(&self, params: &[(&str, &str)]) -> Result<Value> {
        let url = format!("{}/candidates", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .header("Authorization", self.auth_header())
                .query(params)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Babamul request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Babamul response: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(BabamulClient::new("test-token").is_ok());
    }

    #[test]
    fn test_api_url_matches_docs() {
        assert!(
            API_BASE_URL.contains("babamul.caltech.edu"),
            "API_BASE_URL should point to babamul.caltech.edu"
        );
    }

    #[test]
    fn test_auth_header_format() {
        let client = BabamulClient::new("my-token").unwrap();
        assert_eq!(client.auth_header(), "token my-token");
    }
}
