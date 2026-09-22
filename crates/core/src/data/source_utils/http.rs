//! Shared HTTP client utilities

use std::time::Duration;

use crate::{Result, StarfieldError};

/// Build a blocking HTTP client with the given timeout.
///
/// All datasource crates use the same pattern for creating reqwest clients.
/// This centralizes that boilerplate.
pub fn build_http_client(timeout_secs: u64) -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| StarfieldError::DataError(format!("Failed to create HTTP client: {}", e)))
}

/// Check that an HTTP response has a success status, returning a descriptive
/// error that includes `context` (typically the service or URL) on failure.
pub fn check_response_status(
    response: reqwest::blocking::Response,
    context: &str,
) -> Result<reqwest::blocking::Response> {
    if response.status().is_success() {
        return Ok(response);
    }
    Err(StarfieldError::DataError(format!(
        "{} returned HTTP {}",
        context,
        response.status()
    )))
}

/// Assert that an HTTP endpoint is reachable via a HEAD request.
///
/// Any HTTP response (including 403, 404, 405) proves the server is up and
/// the hostname resolves. Only connection failures and timeouts cause a panic.
///
/// Intended for `#[ignore]` integration tests that verify upstream URLs.
pub fn assert_endpoint_reachable(url: &str) {
    if let EndpointStatus::Unreachable(e) = check_endpoint_status(url) {
        panic!("HEAD request failed for {}: {}", url, e);
    }
}

/// Outcome of probing an endpoint via HEAD. Distinguishes "our URL is wrong / service
/// has vanished" from "remote infrastructure problem" so callers can decide whether
/// to treat a failure as a hard error (unresolvable/refused/gone) vs. a reportable
/// warning (e.g. the remote server has an expired TLS certificate).
#[derive(Debug)]
pub enum EndpointStatus {
    /// Any HTTP response was received — server is up and TLS (if any) succeeded.
    Ok,
    /// TLS handshake failed (expired/invalid cert, unknown CA, protocol mismatch).
    /// The remote host is reachable but its TLS configuration is broken. This is an
    /// operational issue on the remote side, not a URL-correctness problem that
    /// URL-verification tests are designed to catch.
    TlsFailure(String),
    /// DNS resolution, connection establishment, or request completion failed for
    /// a non-TLS reason. This is what URL-verification tests exist to catch.
    Unreachable(String),
}

impl EndpointStatus {
    pub fn is_ok(&self) -> bool {
        matches!(self, EndpointStatus::Ok)
    }
}

/// Probe an HTTP endpoint via HEAD and return a typed status instead of panicking.
///
/// Use this from `#[ignore]` URL-verification tests that want to collect results
/// across many URLs and report a summary rather than aborting on the first failure.
pub fn check_endpoint_status(url: &str) -> EndpointStatus {
    let client = match reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => return EndpointStatus::Unreachable(format!("client build: {}", e)),
    };

    match client.head(url).send() {
        Ok(_) => EndpointStatus::Ok,
        Err(err) => {
            if is_tls_error(&err) {
                EndpointStatus::TlsFailure(format_err_chain(&err))
            } else {
                EndpointStatus::Unreachable(format_err_chain(&err))
            }
        }
    }
}

/// Walk a `reqwest::Error`'s source chain looking for tell-tale TLS-failure signals.
/// reqwest doesn't expose a structured "this is a TLS error" predicate, so we match
/// on substrings from the concrete error types (native-tls / rustls / webpki).
fn is_tls_error(err: &reqwest::Error) -> bool {
    let chain = format_err_chain(err).to_ascii_lowercase();
    const TLS_MARKERS: &[&str] = &[
        "certificate",
        "tls handshake",
        "ssl",
        "bad certificate",
        "untrusted",
        "expired",
        "invalidcertificate",
    ];
    TLS_MARKERS.iter().any(|m| chain.contains(m))
}

fn format_err_chain<E: std::error::Error + ?Sized>(err: &E) -> String {
    use std::fmt::Write;
    let mut out = err.to_string();
    let mut current: Option<&dyn std::error::Error> = err.source();
    while let Some(src) = current {
        let _ = write!(out, ": {}", src);
        current = src.source();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_http_client() {
        let client = build_http_client(30);
        assert!(client.is_ok());
    }
}
