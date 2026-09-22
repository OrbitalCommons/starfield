//! Pitt-Google alert broker client.
//!
//! [Pitt-Google](https://pitt-broker.readthedocs.io) is a cloud-native broker
//! hosted on Google Cloud Platform. Unlike other brokers, it does not expose a
//! REST API — data is accessed through GCP services (Pub/Sub, BigQuery,
//! Cloud Storage).
//!
//! **Authentication:** Google Cloud service account credentials.
//!
//! **Setup:**
//! 1. Create a free GCP project at <https://console.cloud.google.com/cloud-resource-manager>
//! 2. Create a service account and download its JSON key
//! 3. Set `GOOGLE_CLOUD_PROJECT` and `GOOGLE_APPLICATION_CREDENTIALS` env vars
//!
//! **GCP Project ID:** `ardent-cycling-243415`
//!
//! **Documentation:** <https://pittgoogle-client.readthedocs.io>
//!
//! **Python client:** `pip install pittgoogle-client`
//!
//! This module provides connection metadata and constants. For actual data
//! access, use the `pittgoogle-client` Python package or the GCP client
//! libraries directly.

use starfield_core::data::source_utils::build_http_client;
use starfield_core::{Result, StarfieldError};

/// Documentation URL.
pub const DOCS_URL: &str = "https://pittgoogle-client.readthedocs.io";

/// GCP project ID that hosts Pitt-Google data.
pub const GCP_PROJECT_ID: &str = "ardent-cycling-243415";

/// Pub/Sub topic for ZTF alerts.
pub const PUBSUB_TOPIC_ZTF: &str = "ztf-alerts";

/// Pub/Sub topic for simulated LSST alerts.
pub const PUBSUB_TOPIC_LSST_SIM: &str = "lsst-alerts-simulated";

/// BigQuery table for ZTF alerts.
pub const BIGQUERY_ZTF: &str = "ardent-cycling-243415.ztf.alerts_v4_02";

/// BigQuery table for LSST alerts.
pub const BIGQUERY_LSST: &str = "ardent-cycling-243415.lsst.alerts_v9_0";

/// Client for Pitt-Google broker metadata.
///
/// Pitt-Google is cloud-native and does not expose a REST API. This client
/// provides access to metadata and connection information. For data access,
/// use the GCP client libraries or the `pittgoogle-client` Python package.
pub struct PittGoogleClient {
    _client: reqwest::blocking::Client,
}

impl PittGoogleClient {
    /// Create a new Pitt-Google metadata client.
    pub fn new() -> Result<Self> {
        let client = build_http_client(15)?;
        Ok(Self { _client: client })
    }

    /// Return the GCP project ID for Pitt-Google.
    pub fn project_id(&self) -> &'static str {
        GCP_PROJECT_ID
    }

    /// Return the Pub/Sub topic for ZTF alerts.
    pub fn ztf_topic(&self) -> &'static str {
        PUBSUB_TOPIC_ZTF
    }

    /// Return the BigQuery table for ZTF alerts.
    pub fn ztf_bigquery_table(&self) -> &'static str {
        BIGQUERY_ZTF
    }

    /// Verify that the documentation site is reachable.
    pub fn check_docs_reachable(&self) -> Result<()> {
        let response = self._client.head(DOCS_URL).send().map_err(|e| {
            StarfieldError::DataError(format!("Pitt-Google docs unreachable: {}", e))
        })?;
        if response.status().is_success() || response.status().as_u16() == 405 {
            Ok(())
        } else {
            Err(StarfieldError::DataError(format!(
                "Pitt-Google docs returned HTTP {}",
                response.status()
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        assert!(PittGoogleClient::new().is_ok());
    }

    #[test]
    fn test_project_id() {
        let client = PittGoogleClient::new().unwrap();
        assert_eq!(client.project_id(), "ardent-cycling-243415");
    }
}
