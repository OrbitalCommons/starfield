//! ALeRCE alert broker client.
//!
//! [ALeRCE](https://alerce.online) (Automatic Learning for the Rapid
//! Classification of Events) is a Chilean-led broker focused on ML
//! classification of transient and variable sources.
//!
//! **Authentication:** None required.
//!
//! **Documentation:** <https://alerceapi.readthedocs.io>

use serde::Deserialize;
use serde_json::Value;
use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::{build_http_client, check_response_status};

/// REST API base URL.
pub const API_BASE_URL: &str = "https://api.alerce.online/ztf/v1";

/// Documentation URL.
pub const DOCS_URL: &str = "https://alerceapi.readthedocs.io";

/// Client for the ALeRCE broker REST API.
pub struct AlerceClient {
    client: reqwest::blocking::Client,
}

/// Summary of an ALeRCE object.
#[derive(Debug, Deserialize)]
pub struct AlerceObject {
    /// ALeRCE object identifier (typically a ZTF object ID).
    pub oid: String,
    /// Mean right ascension (degrees).
    pub meanra: f64,
    /// Mean declination (degrees).
    pub meandec: f64,
    /// Number of detections.
    #[serde(default)]
    pub ndet: Option<i64>,
    /// First detection MJD.
    #[serde(default)]
    pub firstmjd: Option<f64>,
    /// Last detection MJD.
    #[serde(default)]
    pub lastmjd: Option<f64>,
}

/// A single photometric detection.
#[derive(Debug, Deserialize)]
pub struct AlerceDetection {
    /// Candidate ID.
    pub candid: Value,
    /// Modified Julian Date.
    pub mjd: f64,
    /// Filter/band identifier.
    pub fid: i32,
    /// Magnitude.
    #[serde(default)]
    pub mag: Option<f64>,
    /// Magnitude error.
    #[serde(default)]
    pub e_mag: Option<f64>,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
}

/// Classification probability from a classifier.
#[derive(Debug, Deserialize)]
pub struct AlerceClassification {
    /// Classifier name.
    pub classifier_name: String,
    /// Classifier version.
    pub classifier_version: String,
    /// Class name.
    pub class_name: String,
    /// Probability.
    pub probability: f64,
}

impl AlerceClient {
    /// Create a new ALeRCE client.
    pub fn new() -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self { client })
    }

    /// Get a single object by its identifier.
    pub fn get_object(&self, object_id: &str) -> Result<AlerceObject> {
        let url = format!("{}/objects/{}", API_BASE_URL, object_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ALeRCE request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ALeRCE response: {}", e))
        })
    }

    /// Get detections for an object.
    pub fn get_detections(&self, object_id: &str) -> Result<Vec<AlerceDetection>> {
        let url = format!("{}/objects/{}/detections", API_BASE_URL, object_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ALeRCE request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ALeRCE response: {}", e))
        })
    }

    /// Get classification probabilities for an object.
    pub fn get_probabilities(&self, object_id: &str) -> Result<Vec<AlerceClassification>> {
        let url = format!("{}/objects/{}/probabilities", API_BASE_URL, object_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ALeRCE request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ALeRCE response: {}", e))
        })
    }

    /// Get the full lightcurve (detections + non-detections) for an object.
    pub fn get_lightcurve(&self, object_id: &str) -> Result<Value> {
        let url = format!("{}/objects/{}/lightcurve", API_BASE_URL, object_id);
        let response = check_response_status(
            self.client
                .get(&url)
                .send()
                .map_err(|e| StarfieldError::DataError(format!("ALeRCE request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse ALeRCE response: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(AlerceClient::new().is_ok());
    }

    #[test]
    fn test_api_url_matches_docs() {
        assert!(
            API_BASE_URL.contains("alerce.online"),
            "API_BASE_URL should point to alerce.online"
        );
    }

    #[test]
    #[ignore]
    fn test_live_get_object() {
        let client = AlerceClient::new().unwrap();
        let obj = client.get_object("ZTF21aakilyd").unwrap();
        assert!(!obj.oid.is_empty());
    }
}
