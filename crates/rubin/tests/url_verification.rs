//! Integration tests that verify all broker URL constants resolve.
//!
//! These tests are marked `#[ignore]` because they require network access.
//! Run with: `cargo test -p starfield-rubin --test url_verification -- --ignored`
//!
//! Behavior: each URL is probed with a HEAD request. Any HTTP response (including
//! 403/404/405) proves the server is up and the URL resolves. Results are bucketed:
//!
//! - `Ok` — accepted as a pass.
//! - `TlsFailure` (e.g. expired certificate on the remote host) — logged as a warning.
//!   This is an operational issue on the *broker's* side, not a URL-correctness
//!   issue our test is designed to catch, so we don't fail CI for it. A warning
//!   still surfaces it in logs.
//! - `Unreachable` (DNS, connection refused, timeout) — hard failure. Most likely
//!   means the broker moved/renamed their endpoint, or we mistyped a URL.

use starfield_datasource_utils::{check_endpoint_status, EndpointStatus};
use starfield_rubin::*;

fn check_all<'a>(urls: impl IntoIterator<Item = (&'a str, &'a str)>, kind: &str) {
    let mut unreachable: Vec<String> = Vec::new();
    let mut tls_warnings: Vec<String> = Vec::new();

    for (name, url) in urls {
        println!("Checking {}: {} -> {}", kind, name, url);
        match check_endpoint_status(url) {
            EndpointStatus::Ok => {}
            EndpointStatus::TlsFailure(msg) => {
                eprintln!(
                    "  WARN  TLS issue at {} ({}): {} — not failing CI, remote \
                     operational issue",
                    name, url, msg
                );
                tls_warnings.push(format!("{} ({}): {}", name, url, msg));
            }
            EndpointStatus::Unreachable(msg) => {
                eprintln!("  FAIL  unreachable at {} ({}): {}", name, url, msg);
                unreachable.push(format!("{} ({}): {}", name, url, msg));
            }
        }
    }

    if !tls_warnings.is_empty() {
        eprintln!(
            "\n{} {} URL(s) had TLS issues (treated as warnings):",
            tls_warnings.len(),
            kind
        );
        for w in &tls_warnings {
            eprintln!("  - {}", w);
        }
    }

    assert!(
        unreachable.is_empty(),
        "{} {} URL(s) are unreachable:\n  - {}",
        unreachable.len(),
        kind,
        unreachable.join("\n  - "),
    );
}

#[test]
#[ignore = "live upstream; must bypass the mirror and cache. Detects archive rot, so a warm or mirrored resolve would defeat it"]
fn test_all_broker_api_urls_reachable() {
    check_all(all_broker_api_urls(), "API");
}

#[test]
#[ignore = "live upstream; must bypass the mirror and cache. Detects archive rot, so a warm or mirrored resolve would defeat it"]
fn test_all_broker_doc_urls_reachable() {
    check_all(all_broker_doc_urls(), "docs");
}

#[test]
fn test_doc_urls_match_code_constants() {
    let api_urls = all_broker_api_urls();
    let doc_urls = all_broker_doc_urls();

    // Verify we have the expected number of entries
    assert_eq!(
        api_urls.len(),
        7,
        "Expected 7 API URL entries (6 brokers + AMPEL split)"
    );
    assert_eq!(doc_urls.len(), 10, "Expected 10 doc URL entries");

    // Verify API URLs match their module constants
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "ALeRCE" && *u == alerce::API_BASE_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "ANTARES" && *u == antares::API_BASE_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "Fink" && *u == fink::API_BASE_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "Lasair" && *u == lasair::API_BASE_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "AMPEL catalog" && *u == ampel::CATALOG_API_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "AMPEL archive" && *u == ampel::ARCHIVE_API_URL));
    assert!(api_urls
        .iter()
        .any(|(n, u)| *n == "Babamul" && *u == babamul::API_BASE_URL));

    // Verify doc URLs match their module constants
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "ALeRCE docs" && *u == alerce::DOCS_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "ANTARES portal" && *u == antares::PORTAL_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "Fink portal" && *u == fink::PORTAL_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "Lasair portal" && *u == lasair::PORTAL_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "Pitt-Google docs" && *u == pitt_google::DOCS_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "AMPEL catalog docs" && *u == ampel::CATALOG_DOCS_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "AMPEL archive docs" && *u == ampel::ARCHIVE_DOCS_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "Babamul portal" && *u == babamul::PORTAL_URL));
    assert!(doc_urls
        .iter()
        .any(|(n, u)| *n == "Babamul docs" && *u == babamul::DOCS_URL));
}

#[test]
fn test_all_urls_are_https() {
    for (name, url) in all_broker_api_urls() {
        assert!(
            url.starts_with("https://"),
            "{} API URL should use HTTPS: {}",
            name,
            url
        );
    }
    for (name, url) in all_broker_doc_urls() {
        assert!(
            url.starts_with("https://"),
            "{} doc URL should use HTTPS: {}",
            name,
            url
        );
    }
}
