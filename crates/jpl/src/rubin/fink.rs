//! Fink alert broker client.
//!
//! [Fink](https://ztf.fink-portal.org) is a French-led broker built on Apache
//! Spark with strong ML classification pipelines. It offers both a REST API
//! and Kafka streaming.
//!
//! **Authentication:** None required for REST API queries.
//! Kafka streaming requires credentials obtained by registration.
//!
//! **Documentation:** <https://fink-broker.org>

use serde::Deserialize;
use serde_json::Value;
use starfield_core::data::source_utils::{build_http_client, check_response_status};
use starfield_core::{Result, StarfieldError};

/// REST API base URL (ZTF data).
pub const API_BASE_URL: &str = "https://api.ztf.fink-portal.org";

/// REST API base URL for LSST data.
pub const LSST_API_BASE_URL: &str = "https://api.lsst.fink-portal.org";

/// Web portal URL.
pub const PORTAL_URL: &str = "https://ztf.fink-portal.org";

/// Client for the Fink broker REST API.
pub struct FinkClient {
    client: reqwest::blocking::Client,
    base_url: String,
}

/// A Fink object record.
#[derive(Debug, Deserialize)]
pub struct FinkObject {
    /// Object identifier.
    #[serde(rename = "i:objectId")]
    pub object_id: Option<String>,
    /// Right ascension (degrees).
    #[serde(rename = "i:ra")]
    pub ra: Option<f64>,
    /// Declination (degrees).
    #[serde(rename = "i:dec")]
    pub dec: Option<f64>,
    /// Fink classification.
    #[serde(rename = "v:classification")]
    pub classification: Option<String>,
}

impl FinkClient {
    /// Create a new Fink client targeting the ZTF data portal.
    pub fn new() -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            base_url: API_BASE_URL.to_string(),
        })
    }

    /// Create a new Fink client targeting the LSST data portal.
    pub fn new_lsst() -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            base_url: LSST_API_BASE_URL.to_string(),
        })
    }

    /// Query objects by name (objectId).
    pub fn get_objects(&self, object_id: &str) -> Result<Vec<Value>> {
        let url = format!("{}/api/v1/objects", self.base_url);
        let response = check_response_status(
            self.client
                .post(&url)
                .form(&[("objectId", object_id), ("output-format", "json")])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Fink request failed: {}", e)))?,
            &url,
        )?;
        response
            .json()
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse Fink response: {}", e)))
    }

    /// Cone search by RA/Dec/radius.
    ///
    /// `ra` and `dec` are in degrees, `radius` is in arcseconds.
    pub fn cone_search(&self, ra: f64, dec: f64, radius_arcsec: f64) -> Result<Vec<Value>> {
        let url = format!("{}/api/v1/conesearch", self.base_url);
        let response = check_response_status(
            self.client
                .post(&url)
                .form(&[
                    ("ra", ra.to_string()),
                    ("dec", dec.to_string()),
                    ("radius", radius_arcsec.to_string()),
                    ("output-format", "json".to_string()),
                ])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Fink request failed: {}", e)))?,
            &url,
        )?;
        response
            .json()
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse Fink response: {}", e)))
    }

    /// Get the latest alerts for a given classification class.
    pub fn get_latest(&self, class_name: &str, n: usize) -> Result<Vec<Value>> {
        let url = format!("{}/api/v1/latests", self.base_url);
        let response = check_response_status(
            self.client
                .post(&url)
                .form(&[
                    ("class", class_name.to_string()),
                    ("n", n.to_string()),
                    ("output-format", "json".to_string()),
                ])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Fink request failed: {}", e)))?,
            &url,
        )?;
        response
            .json()
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse Fink response: {}", e)))
    }

    /// Query Solar System Objects by designation or number.
    pub fn get_sso(&self, designation: &str) -> Result<Vec<Value>> {
        let url = format!("{}/api/v1/sso", self.base_url);
        let response = check_response_status(
            self.client
                .post(&url)
                .form(&[("n_or_d", designation), ("output-format", "json")])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Fink request failed: {}", e)))?,
            &url,
        )?;
        response
            .json()
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse Fink response: {}", e)))
    }

    /// List available classification classes.
    pub fn get_classes(&self) -> Result<Value> {
        let url = format!("{}/api/v1/classes", self.base_url);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Fink request failed: {}", e)))?,
            &url,
        )?;
        response
            .json()
            .map_err(|e| StarfieldError::DataError(format!("Failed to parse Fink response: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(FinkClient::new().is_ok());
    }

    #[test]
    fn test_lsst_client_creation() {
        assert!(FinkClient::new_lsst().is_ok());
    }

    #[test]
    fn test_api_url_matches_docs() {
        assert!(
            API_BASE_URL.contains("ztf.fink-portal.org"),
            "API_BASE_URL should point to the current ztf.fink-portal.org host"
        );
    }
}
