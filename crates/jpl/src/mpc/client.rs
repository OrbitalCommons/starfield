//! MPC HTTP client with rate limiting
//!
//! Provides access to MPC bulk data files and the web service API.

use std::io::Read;
use std::time::{Duration, Instant};

use starfield_core::data::source_utils::{build_http_client, check_response_status};
use starfield_core::{Result, StarfieldError};

use crate::mpc::mpcorb::{parse_mpcorb, MpcOrbRecord};
use crate::mpc::observatory::{parse_observatory_codes, Observatory};

/// Base URL for MPC data files
const MPC_DATA_URL: &str = "https://www.minorplanetcenter.net/iau";

/// Minimum delay between HTTP requests (2 seconds)
const RATE_LIMIT_MS: u64 = 2000;

/// Client for accessing Minor Planet Center data
pub struct MpcClient {
    client: reqwest::blocking::Client,
    last_request: Option<Instant>,
}

impl MpcClient {
    /// Create a new MPC client with default settings
    pub fn new() -> Result<Self> {
        let client = build_http_client(120)?;
        Ok(Self {
            client,
            last_request: None,
        })
    }

    /// Enforce rate limiting between requests
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

    /// Fetch text content from a URL with rate limiting
    fn fetch_text(&mut self, url: &str) -> Result<String> {
        self.rate_limit();
        log::info!("Fetching {}", url);

        let response = check_response_status(
            self.client.get(url).send().map_err(|e| {
                StarfieldError::DataError(format!("MPC request failed for {}: {}", url, e))
            })?,
            url,
        )?;

        response
            .text()
            .map_err(|e| StarfieldError::DataError(format!("Failed to read MPC response: {}", e)))
    }

    /// Fetch raw bytes from a URL with rate limiting (for large files)
    fn fetch_bytes(&mut self, url: &str) -> Result<Vec<u8>> {
        self.rate_limit();
        log::info!("Fetching {} (bytes)", url);

        let mut response = check_response_status(
            self.client.get(url).send().map_err(|e| {
                StarfieldError::DataError(format!("MPC request failed for {}: {}", url, e))
            })?,
            url,
        )?;

        let mut bytes = Vec::new();
        response.read_to_end(&mut bytes).map_err(|e| {
            StarfieldError::DataError(format!("Failed to read MPC response: {}", e))
        })?;

        Ok(bytes)
    }

    /// Download and parse the MPC observatory code registry.
    ///
    /// Source: <https://minorplanetcenter.net/iau/lists/ObsCodes.html>
    pub fn fetch_observatory_codes(&mut self) -> Result<Vec<Observatory>> {
        let url = format!("{}/lists/ObsCodes.html", MPC_DATA_URL);
        let html = self.fetch_text(&url)?;

        // The observatory data is inside <pre> tags in the HTML
        let text = extract_pre_content(&html).ok_or_else(|| {
            StarfieldError::DataError(
                "Could not find observatory data in HTML response".to_string(),
            )
        })?;

        parse_observatory_codes(&text)
    }

    /// Download and parse the full MPCORB.DAT catalog.
    ///
    /// This is a large download (~300+ MB). Results are streamed and parsed.
    /// Source: <https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT>
    pub fn fetch_mpcorb(&mut self) -> Result<Vec<MpcOrbRecord>> {
        let url = format!("{}/MPCORB/MPCORB.DAT", MPC_DATA_URL);
        let bytes = self.fetch_bytes(&url)?;
        let text = String::from_utf8_lossy(&bytes);
        parse_mpcorb(&text)
    }

    /// Fetch a subset of MPCORB focusing on distant objects (TNOs, Centaurs).
    ///
    /// Filters records where semimajor axis exceeds `min_a_au`.
    pub fn fetch_distant_objects(&mut self, min_a_au: f64) -> Result<Vec<MpcOrbRecord>> {
        let all = self.fetch_mpcorb()?;
        Ok(all
            .into_iter()
            .filter(|r| r.semimajor_axis > min_a_au)
            .collect())
    }
}

/// Extract text content between <pre> and </pre> tags
fn extract_pre_content(html: &str) -> Option<String> {
    let lower = html.to_lowercase();
    let start = lower.find("<pre>")? + 5;
    let end = lower[start..].find("</pre>")? + start;
    Some(html[start..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_pre_content() {
        let html = "<html><body><pre>hello world</pre></body></html>";
        assert_eq!(extract_pre_content(html), Some("hello world".to_string()));
    }

    #[test]
    fn test_extract_pre_content_missing() {
        assert_eq!(extract_pre_content("<html>no pre here</html>"), None);
    }

    #[test]
    fn test_extract_pre_content_case_insensitive() {
        let html = "<PRE>data</PRE>";
        assert_eq!(extract_pre_content(html), Some("data".to_string()));
    }

    #[test]
    fn test_client_creation() {
        let client = MpcClient::new();
        assert!(client.is_ok());
    }

    #[test]
    #[ignore]
    fn test_live_observatory_codes() {
        let mut client = MpcClient::new().unwrap();
        let observatories = client.fetch_observatory_codes().unwrap();
        assert!(
            observatories.len() > 1000,
            "Expected 1000+ observatories, got {}",
            observatories.len()
        );

        // Greenwich should be present
        let greenwich = observatories.iter().find(|o| o.code == "000");
        assert!(greenwich.is_some(), "Greenwich (000) should be present");
        assert_eq!(greenwich.unwrap().name, "Greenwich");
    }
}
