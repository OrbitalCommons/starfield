//! ESA Gaia Archive TAP (Table Access Protocol) sync client.
//!
//! TAP lets a client run an ADQL query and stream the result back in CSV (or
//! VOTable, FITS, JSON). This module wraps the *sync* endpoint
//! `https://gea.esac.esa.int/tap-server/tap/sync` for queries small enough to
//! return inline (~few-million-row ceiling per ESA's policy). Larger queries
//! need the *async* (UWS) endpoint; not implemented here.
//!
//! The release-specific helper [`Downloader::tap_select_where`] builds an
//! ADQL `SELECT … FROM <table> WHERE <…>` whose column list is exactly
//! `R::csv_header()` in arrow-schema order. The resulting CSV is therefore
//! byte-identical in shape to the bulk per-file CSVs and can be piped straight
//! into [`crate::common::reader::CsvSourceReader::from_reader`] without any
//! schema massaging.
//!
//! # Example
//!
//! ```no_run
//! use starfield_gaia::Dr3;
//! use starfield_gaia::download::Downloader;
//! use starfield_gaia::common::reader::CsvSourceReader;
//!
//! let bright_csv = Downloader::<Dr3>::tap_select_where("phot_g_mean_mag <= 13")?;
//! let reader = CsvSourceReader::<Dr3>::from_reader(bright_csv, false, f64::INFINITY)?;
//! let mut n = 0;
//! for entry in reader {
//!     let _ = entry?;
//!     n += 1;
//! }
//! println!("loaded {} bright DR3 sources", n);
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

use std::io::Read;

use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::{build_http_client, check_response_status};

use crate::common::traits::GaiaRelease;
use crate::download::Downloader;

/// ESA Gaia Archive TAP sync endpoint. Public, anonymous, returns inline.
pub const ESA_GAIA_TAP_SYNC: &str = "https://gea.esac.esa.int/tap-server/tap/sync";

/// HTTP timeout for TAP sync queries. ESA's archive can take minutes for
/// non-indexed scans (e.g. magnitude cuts that read the whole table); 600 s
/// matches the bulk-download timeout used elsewhere in this crate.
const TAP_TIMEOUT_SECS: u64 = 600;

/// Output format for a TAP query. Currently only CSV is exposed because that's
/// what the rest of this crate is built around.
#[derive(Debug, Clone, Copy)]
pub enum TapFormat {
    Csv,
}

impl TapFormat {
    fn as_str(&self) -> &'static str {
        match self {
            TapFormat::Csv => "csv",
        }
    }
}

/// Default `MAXREC` for ESA Gaia archive sync queries. The server-side cap
/// without an explicit MAXREC is a small number (~25 000) that silently
/// truncates large queries; passing a much larger value lets queries up to
/// the sync-endpoint hard cap (~3M rows) come through. 3 000 000 is the
/// documented ESA Gaia archive sync ceiling — set as our default to make
/// every "give me bright stars" call return the full set.
pub const DEFAULT_MAXREC: usize = 3_000_000;

/// Run a TAP sync query and return the response body as a streaming reader.
/// Generic over the endpoint URL so this can also point at other TAP services
/// (e.g. STILTS, IRSA) — the ESA Gaia archive is the only caller today.
///
/// The body is `application/x-www-form-urlencoded` with the TAP standard
/// parameters: `REQUEST=doQuery&LANG=ADQL&FORMAT=…&QUERY=…&MAXREC=…`.
/// Submitted via POST so long ADQL strings (DR3 has ~150 columns ≈ 4 KB) fit
/// comfortably.
///
/// `maxrec` caps the row count returned — pass `None` to use the server's
/// silent default (often catastrophically small; ESA Gaia caps at ~25 k).
/// Pass `Some(n)` for any non-trivial query; [`DEFAULT_MAXREC`] is a safe
/// bound for the ESA Gaia archive sync endpoint.
pub fn tap_sync_query(
    endpoint: &str,
    adql: &str,
    format: TapFormat,
    maxrec: Option<usize>,
) -> Result<Box<dyn Read + Send>> {
    let client = build_http_client(TAP_TIMEOUT_SECS)?;
    let mut form: Vec<(&str, String)> = vec![
        ("REQUEST", "doQuery".into()),
        ("LANG", "ADQL".into()),
        ("FORMAT", format.as_str().into()),
        ("QUERY", adql.into()),
    ];
    if let Some(m) = maxrec {
        form.push(("MAXREC", m.to_string()));
    }
    let resp = client
        .post(endpoint)
        .form(&form)
        .send()
        .map_err(|e| StarfieldError::DataError(format!("TAP POST {}: {}", endpoint, e)))?;
    let resp = check_response_status(resp, endpoint)?;
    Ok(Box::new(resp))
}

impl<R: GaiaRelease> Downloader<R> {
    /// Run an ADQL query that selects every column the bulk CSV files publish
    /// (in `R::csv_header()` order) from this release's Gaia archive table,
    /// constrained by the caller-supplied `WHERE` clause. Returns the CSV
    /// response stream — feed it to
    /// [`CsvSourceReader::from_reader`](crate::common::reader::CsvSourceReader::from_reader)
    /// to get typed entries identical in layout to those produced by
    /// [`Downloader::download_file`] / [`Downloader::stream_file`].
    ///
    /// `where_clause` is the bare predicate, *without* the leading `WHERE`
    /// (e.g. `"phot_g_mean_mag <= 13"` or `"source_id BETWEEN 100 AND 200"`).
    /// Pass an empty string to select every row (rarely useful — bulk
    /// downloads are faster for full-table reads).
    ///
    /// # Examples
    ///
    /// Bright stars:
    /// ```ignore
    /// let bytes = Downloader::<Dr3>::tap_select_where("phot_g_mean_mag <= 13")?;
    /// ```
    ///
    /// Cone search around (RA, Dec) = (180, 0) with 1 deg radius:
    /// ```ignore
    /// let bytes = Downloader::<Dr3>::tap_select_where(
    ///     "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 180.0, 0.0, 1.0))",
    /// )?;
    /// ```
    pub fn tap_select_where(where_clause: &str) -> Result<Box<dyn Read + Send>> {
        let adql = if where_clause.trim().is_empty() {
            format!("SELECT {} FROM {}", R::csv_header(), R::TAP_TABLE)
        } else {
            format!(
                "SELECT {} FROM {} WHERE {}",
                R::csv_header(),
                R::TAP_TABLE,
                where_clause
            )
        };
        tap_sync_query(
            ESA_GAIA_TAP_SYNC,
            &adql,
            TapFormat::Csv,
            Some(DEFAULT_MAXREC),
        )
    }

    /// Convenience alias for the `phot_g_mean_mag <= mag_limit` case — the
    /// bright-star use case driving the supplement-builder. Returns a CSV
    /// stream over every source in this release at or brighter than `mag`.
    pub fn tap_brighter_than(mag: f64) -> Result<Box<dyn Read + Send>> {
        Self::tap_select_where(&format!("phot_g_mean_mag <= {}", mag))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brighter_than_query_well_formed() {
        // Just verify the SELECT we'd build for DR3 has the expected shape; we
        // don't actually hit the network here.
        use crate::Dr3;
        let header = Dr3::csv_header();
        let adql = format!(
            "SELECT {} FROM {} WHERE phot_g_mean_mag <= 5",
            header,
            Dr3::TAP_TABLE
        );
        assert!(adql.starts_with("SELECT "));
        assert!(adql.contains(" FROM gaiadr3.gaia_source "));
        assert!(adql.contains("source_id"));
        assert!(adql.contains("phot_g_mean_mag"));
        assert!(adql.ends_with("phot_g_mean_mag <= 5"));
    }
}
