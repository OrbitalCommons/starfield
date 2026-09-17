//! Lasair alert broker client.
//!
//! [Lasair](https://lasair-ztf.lsst.ac.uk) is a UK-based broker (Edinburgh,
//! Queen's Belfast, Oxford) that provides SQL-based alert filtering.
//!
//! **Authentication:** API token required.
//! Register at <https://lasair-ztf.lsst.ac.uk/login/> and find your token
//! on the "My Profile" page.
//!
//! **Rate limits:** 100 calls/hour, max 10,000 rows per query (standard).
//! Power users: 10,000 calls/hour, 1,000,000 rows (request upgrade).
//!
//! **Documentation:** <https://lasair.readthedocs.io>

use serde_json::Value;
use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::{build_http_client, check_response_status};
use std::time::{Duration, Instant};

/// REST API base URL (ZTF instance).
pub const API_BASE_URL: &str = "https://lasair-ztf.lsst.ac.uk/api";

/// REST API base URL for the upcoming LSST instance (commissioning).
pub const LSST_API_BASE_URL: &str = "https://lasair-lsst.lsst.ac.uk/api";

/// Web portal URL.
pub const PORTAL_URL: &str = "https://lasair-ztf.lsst.ac.uk";

/// Standard rate limit: 100 calls/hour = one call per 36 seconds.
const RATE_LIMIT_MS: u64 = 36_000;

/// Client for the Lasair broker REST API.
///
/// Requires an API token. Register at <https://lasair-ztf.lsst.ac.uk/login/>
/// and retrieve your token from the profile page.
pub struct LasairClient {
    client: reqwest::blocking::Client,
    token: String,
    last_request: Option<Instant>,
}

impl LasairClient {
    /// Create a new Lasair client with the given API token.
    pub fn new(token: &str) -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            token: token.to_string(),
            last_request: None,
        })
    }

    fn rate_limit(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed();
            let min_delay = Duration::from_millis(RATE_LIMIT_MS);
            if elapsed < min_delay {
                std::thread::sleep(min_delay - elapsed);
            }
        }
        self.last_request = Some(Instant::now());
    }

    /// Cone search around a position.
    ///
    /// `ra` and `dec` in degrees, `radius` in arcseconds.
    /// `request_type` is one of `"all"`, `"nearest"`, or `"count"`.
    pub fn cone_search(
        &mut self,
        ra: f64,
        dec: f64,
        radius_arcsec: f64,
        request_type: &str,
    ) -> Result<Value> {
        self.rate_limit();
        let url = format!("{}/cone/", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .query(&[
                    ("ra", ra.to_string()),
                    ("dec", dec.to_string()),
                    ("radius", radius_arcsec.to_string()),
                    ("requestType", request_type.to_string()),
                    ("token", self.token.clone()),
                ])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Lasair request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Lasair response: {}", e))
        })
    }

    /// Execute a SQL SELECT query on the Lasair database.
    ///
    /// Only SELECT statements are allowed. Example:
    /// `"SELECT objectId, ramean, decmean FROM objects LIMIT 10"`
    pub fn query(&mut self, sql: &str) -> Result<Value> {
        self.rate_limit();
        let url = format!("{}/query/", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .query(&[("selected", sql), ("token", &self.token)])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Lasair request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Lasair response: {}", e))
        })
    }

    /// Get the machine-readable object page for a given object ID.
    pub fn get_object(&mut self, object_id: &str) -> Result<Value> {
        self.rate_limit();
        let url = format!("{}/object/", API_BASE_URL);
        let response = check_response_status(
            self.client
                .get(&url)
                .query(&[("objectId", object_id), ("token", &self.token)])
                .send()
                .map_err(|e| StarfieldError::DataError(format!("Lasair request failed: {}", e)))?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse Lasair response: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(LasairClient::new("test-token").is_ok());
    }

    #[test]
    fn test_api_url_matches_docs() {
        assert!(
            API_BASE_URL.contains("lasair"),
            "API_BASE_URL should point to lasair"
        );
    }
}
