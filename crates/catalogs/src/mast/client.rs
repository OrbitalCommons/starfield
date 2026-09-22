//! `MastClient` and the Mashup invoke plumbing shared by every
//! observation / catalog query.
//!
//! Mashup's request format:
//!
//! ```text
//! POST https://mast.stsci.edu/api/v0/invoke
//! Content-Type: application/x-www-form-urlencoded
//! Body: request=<URL-encoded JSON>
//! ```
//!
//! where the JSON is `{"service": "...", "params": {...}, "format": "json"}`.
//!
//! Response shape:
//!
//! ```json
//! { "status": "COMPLETE",
//!   "msg": "",
//!   "data": [ { ...row... }, ... ],
//!   "fields": [ { "name": "obsid", "type": "string" }, ... ],
//!   "paging": { "page": 1, "pageSize": 5000, "rowsTotal": 12 } }
//! ```

use std::path::PathBuf;
use std::time::Duration;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value as JsonValue;
use starfield_core::data::source_utils::ensure_cache_subdir;
use starfield_core::{Result, StarfieldError};

const MAST_INVOKE_URL: &str = "https://mast.stsci.edu/api/v0/invoke";
const DEFAULT_TIMEOUT_SECS: u64 = 60;
/// Per-request page size. MAST allows up to 50_000 rows per Mashup
/// response; 5000 is a practical balance between memory and request
/// count for HST-density fields. Wider cones may need pagination —
/// tracked as a follow-on; v1 fetches a single page and emits a warning
/// when more rows are available.
const DEFAULT_PAGESIZE: u64 = 5_000;

/// Public client for the MAST Mashup API.
///
/// Wraps a [`reqwest::blocking::Client`] with a 60 s default timeout.
/// Construction is cheap (one client per process is plenty); the client
/// is `Send + Sync` so it can be shared across threads.
pub struct MastClient {
    http: reqwest::blocking::Client,
}

impl MastClient {
    pub fn new() -> Result<Self> {
        Self::with_timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
    }

    pub fn with_timeout(timeout: Duration) -> Result<Self> {
        let http = reqwest::blocking::Client::builder()
            .timeout(timeout)
            .user_agent("starfield-mast/0.1")
            .build()
            .map_err(|e| StarfieldError::DataError(format!("build http client: {}", e)))?;
        Ok(Self { http })
    }

    /// POST a Mashup request and decode the typed response. Errors out
    /// when the HTTP request fails, the server returns a non-`COMPLETE`
    /// status, or the JSON shape doesn't match `T`.
    pub(crate) fn invoke<T>(&self, service: &str, params: JsonValue) -> Result<MashupOk<T>>
    where
        T: DeserializeOwned,
    {
        let req = MashupRequest {
            service,
            params,
            format: "json",
            page: 1,
            pagesize: DEFAULT_PAGESIZE,
            removenullcolumns: true,
        };
        let body = serde_json::to_string(&req)
            .map_err(|e| StarfieldError::DataError(format!("encode request: {}", e)))?;

        let response = self
            .http
            .post(MAST_INVOKE_URL)
            .header(
                reqwest::header::CONTENT_TYPE,
                "application/x-www-form-urlencoded",
            )
            .body(format!("request={}", urlencode(&body)))
            .send()
            .map_err(|e| StarfieldError::DataError(format!("MAST POST: {}", e)))?;

        if !response.status().is_success() {
            return Err(StarfieldError::DataError(format!(
                "MAST returned HTTP {}",
                response.status()
            )));
        }

        // Read the body once and decode in two passes — first as the
        // generic envelope to check `status`, then as the typed `data`
        // array — so we can surface an error message from the server
        // before the typed deserialization fails confusingly.
        let bytes = response
            .bytes()
            .map_err(|e| StarfieldError::DataError(format!("read MAST body: {}", e)))?;
        let envelope: MashupEnvelope = serde_json::from_slice(&bytes).map_err(|e| {
            let preview = String::from_utf8_lossy(&bytes[..bytes.len().min(200)]);
            StarfieldError::DataError(format!(
                "decode MAST envelope: {} — first 200 bytes: {}",
                e, preview
            ))
        })?;
        if envelope.status != "COMPLETE" {
            return Err(StarfieldError::DataError(format!(
                "MAST returned status={}: {}",
                envelope.status,
                envelope.msg.unwrap_or_default()
            )));
        }

        let typed: TypedMashupResponse<T> = serde_json::from_slice(&bytes).map_err(|e| {
            StarfieldError::DataError(format!("decode MAST {} rows: {}", service, e))
        })?;

        Ok(MashupOk {
            data: typed.data,
            paging: typed.paging,
        })
    }

    /// Cache directory used by [`Self::download_product`].
    /// `~/.cache/starfield/mast/`
    pub fn cache_dir(&self) -> Result<PathBuf> {
        ensure_cache_subdir("mast").map_err(StarfieldError::IoError)
    }
}

#[derive(Debug, Serialize)]
struct MashupRequest<'a> {
    service: &'a str,
    params: JsonValue,
    format: &'a str,
    page: u32,
    pagesize: u64,
    removenullcolumns: bool,
}

/// Envelope shape we use to read `status` / `msg` before committing to
/// a typed deserialization of `data`.
#[derive(Debug, Deserialize)]
struct MashupEnvelope {
    status: String,
    msg: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TypedMashupResponse<T> {
    data: Vec<T>,
    paging: Option<MashupPaging>,
}

/// Paging metadata returned alongside the `data` array.
#[derive(Debug, Clone, Deserialize)]
pub struct MashupPaging {
    pub page: u32,
    #[serde(rename = "pageSize")]
    pub page_size: u64,
    #[serde(rename = "rowsTotal")]
    pub rows_total: u64,
}

/// Successful Mashup response: typed rows + the paging metadata so
/// callers can detect truncation.
pub struct MashupOk<T> {
    pub data: Vec<T>,
    pub paging: Option<MashupPaging>,
}

impl<T> MashupOk<T> {
    /// True when the server reports more rows than this single page
    /// returned. v1 doesn't auto-paginate; callers should re-issue with
    /// a tighter cone or check this flag.
    pub fn truncated(&self) -> bool {
        self.paging
            .as_ref()
            .is_some_and(|p| p.rows_total > self.data.len() as u64)
    }
}

/// Minimal application/x-www-form-urlencoded encoder for the Mashup
/// `request=<JSON>` body. We can't reach for `urlencoding` because it
/// isn't already in the workspace; this covers what JSON throws at us
/// (curly braces, brackets, quotes, commas, colons, spaces, etc.).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::urlencode;

    #[test]
    fn urlencode_keeps_unreserved_intact() {
        assert_eq!(urlencode("abcXYZ09-_.~"), "abcXYZ09-_.~");
    }

    #[test]
    fn urlencode_percent_encodes_json_punctuation() {
        // Spot-check the characters Mashup sees in our payload.
        assert_eq!(urlencode("{}"), "%7B%7D");
        assert_eq!(urlencode("[]"), "%5B%5D");
        assert_eq!(urlencode("\""), "%22");
        assert_eq!(urlencode(":"), "%3A");
        assert_eq!(urlencode(","), "%2C");
        assert_eq!(urlencode(" "), "%20");
    }
}
