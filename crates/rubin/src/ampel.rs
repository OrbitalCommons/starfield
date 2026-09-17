//! AMPEL alert broker client.
//!
//! [AMPEL](https://ampel-ztf.zeuthen.desy.de) (Alert Management, Photometry, and
//! Evaluation of Lightcurves) is a DESY-hosted modular framework for analyzing
//! astronomical transient alerts.
//!
//! AMPEL exposes two separate API surfaces:
//!
//! - **Catalog match** — unauthenticated positional cross-match against catalogs.
//! - **ZTF archive** — access to the full ZTF alert archive. Requires a bearer
//!   token obtained through GitHub organization membership.
//!
//! **Authentication:**
//! - Catalog match: None required.
//! - Archive: Bearer token required. Join the AMPEL GitHub org and generate a
//!   token at <https://ampel-ztf.zeuthen.desy.de/live/dashboard/tokens>.
//!
//! **Documentation:**
//! - Catalog: <https://ampel-ztf.zeuthen.desy.de/api/catalogmatch/docs>
//! - Archive: <https://ampel-ztf.zeuthen.desy.de/api/ztf/archive/v3/docs>

use serde::Deserialize;
use serde_json::Value;
use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::{build_http_client, check_response_status};

/// Catalog match API URL.
pub const CATALOG_API_URL: &str = "https://ampel-ztf.zeuthen.desy.de/api/catalogmatch";

/// ZTF alert archive API URL (v3).
pub const ARCHIVE_API_URL: &str = "https://ampel-ztf.zeuthen.desy.de/api/ztf/archive/v3";

/// Catalog match API documentation.
pub const CATALOG_DOCS_URL: &str = "https://ampel-ztf.zeuthen.desy.de/api/catalogmatch/docs";

/// Archive API documentation.
pub const ARCHIVE_DOCS_URL: &str = "https://ampel-ztf.zeuthen.desy.de/api/ztf/archive/v3/docs";

/// Token management dashboard URL.
pub const TOKEN_URL: &str = "https://ampel-ztf.zeuthen.desy.de/live/dashboard/tokens";

/// A catalog cross-match result.
#[derive(Debug, Deserialize)]
pub struct CatalogMatch {
    /// Catalog name.
    pub catalog: String,
    /// Distance to match in arcseconds.
    #[serde(default)]
    pub dist_arcsec: Option<f64>,
    /// Match body (catalog-specific fields).
    #[serde(default)]
    pub body: Value,
}

/// Client for the AMPEL broker APIs.
///
/// Catalog match queries require no authentication. Archive queries require a
/// bearer token — see module docs for how to obtain one.
pub struct AmpelClient {
    client: reqwest::blocking::Client,
    archive_token: Option<String>,
}

impl AmpelClient {
    /// Create a client for catalog match queries only (no auth needed).
    pub fn new() -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            archive_token: None,
        })
    }

    /// Create a client with an archive bearer token.
    ///
    /// The token enables both catalog match and archive queries.
    /// Generate a token at <https://ampel-ztf.zeuthen.desy.de/live/dashboard/tokens>.
    pub fn with_archive_token(token: &str) -> Result<Self> {
        let client = build_http_client(30)?;
        Ok(Self {
            client,
            archive_token: Some(token.to_string()),
        })
    }

    /// Positional cross-match against AMPEL catalogs.
    ///
    /// `ra` and `dec` in degrees, `radius_arcsec` search radius.
    /// `catalogs` is a JSON object specifying which catalogs to query, e.g.
    /// `{"SDSS_spec": {"rs_arcsec": 10}, "milliquas": {"rs_arcsec": 5}}`.
    pub fn catalog_match(&self, ra: f64, dec: f64, catalogs: &Value) -> Result<Vec<CatalogMatch>> {
        let url = format!("{}/cone_search/nearest", CATALOG_API_URL);
        let body = serde_json::json!({
            "ra_deg": ra,
            "dec_deg": dec,
            "catalogs": catalogs,
        });
        let response = check_response_status(
            self.client.post(&url).json(&body).send().map_err(|e| {
                StarfieldError::DataError(format!("AMPEL catalog request failed: {}", e))
            })?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse AMPEL catalog response: {}", e))
        })
    }

    /// Get a single alert from the ZTF archive by candid.
    ///
    /// Requires an archive token (use `with_archive_token`).
    pub fn get_alert(&self, candid: i64) -> Result<Value> {
        let token = self
            .archive_token
            .as_ref()
            .ok_or_else(|| StarfieldError::DataError("Archive token required".into()))?;
        let url = format!("{}/alert/{}", ARCHIVE_API_URL, candid);
        let response = check_response_status(
            self.client
                .get(&url)
                .bearer_auth(token)
                .send()
                .map_err(|e| {
                    StarfieldError::DataError(format!("AMPEL archive request failed: {}", e))
                })?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse AMPEL archive response: {}", e))
        })
    }

    /// Get all alerts for a ZTF object from the archive.
    ///
    /// Requires an archive token (use `with_archive_token`).
    pub fn get_object_alerts(&self, ztf_name: &str) -> Result<Value> {
        let token = self
            .archive_token
            .as_ref()
            .ok_or_else(|| StarfieldError::DataError("Archive token required".into()))?;
        let url = format!("{}/object/{}/alerts", ARCHIVE_API_URL, ztf_name);
        let response = check_response_status(
            self.client
                .get(&url)
                .bearer_auth(token)
                .send()
                .map_err(|e| {
                    StarfieldError::DataError(format!("AMPEL archive request failed: {}", e))
                })?,
            &url,
        )?;
        response.json().map_err(|e| {
            StarfieldError::DataError(format!("Failed to parse AMPEL archive response: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(AmpelClient::new().is_ok());
    }

    #[test]
    fn test_client_with_token() {
        assert!(AmpelClient::with_archive_token("test-token").is_ok());
    }

    #[test]
    fn test_archive_requires_token() {
        let client = AmpelClient::new().unwrap();
        assert!(client.get_alert(123456).is_err());
    }

    #[test]
    fn test_api_urls_match_docs() {
        assert!(
            CATALOG_API_URL.contains("ampel-ztf.zeuthen.desy.de"),
            "CATALOG_API_URL should point to AMPEL"
        );
        assert!(
            ARCHIVE_API_URL.contains("ampel-ztf.zeuthen.desy.de"),
            "ARCHIVE_API_URL should point to AMPEL"
        );
        assert!(
            CATALOG_DOCS_URL.starts_with(CATALOG_API_URL),
            "Catalog docs URL should be under catalog API"
        );
        assert!(
            ARCHIVE_DOCS_URL.starts_with(ARCHIVE_API_URL),
            "Archive docs URL should be under archive API"
        );
    }
}
