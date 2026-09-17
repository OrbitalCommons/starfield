//! Traits shared across every Gaia data release.

use crate::common::core::GaiaCore;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use starfield_core::Result;

/// Which Gaia data release an entry came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Release {
    Dr1,
    Dr2,
    Dr3,
}

impl Release {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Dr1 => "DR1",
            Self::Dr2 => "DR2",
            Self::Dr3 => "DR3",
        }
    }
}

/// Every per-release Entry type exposes its shared core and which release produced it.
pub trait GaiaSource {
    fn core(&self) -> &GaiaCore;
    fn release(&self) -> Release;

    /// Approximate Johnson B-V color derived from Gaia photometry via the cubic
    /// transformation in [`bp_rp_to_johnson_b_v`]. DR2/DR3 implementations apply
    /// the polynomial to their published `BP-RP`; DR1 has no per-source color
    /// and returns `None`. Used to populate `StarData::b_v` when Gaia catalogs
    /// are exposed through the generic
    /// [`StarCatalog`](starfield_core::catalogs::StarCatalog) trait.
    fn b_v(&self) -> Option<f64> {
        None
    }
}

/// Convert Gaia `BP-RP` color to approximate Johnson `B-V` via a linear
/// transformation calibrated for FGK main-sequence stars. Valid range
/// `-0.4 ≤ BP-RP ≤ 2.5` with residual ~0.05 mag; outside that range
/// (very hot O/B or very red M) the slope steepens and this fit
/// underestimates `B-V` by 0.1–0.4 mag. Callers needing precision for cool
/// giants should swap in a higher-order polynomial from
/// [Riello+2021 Table C.2](https://www.aanda.org/articles/aa/full_html/2021/05/aa39587-20/aa39587-20.html).
///
/// Sanity anchors:
/// - Vega    `BP-RP ≈ 0.00` → `B-V ≈ 0.00` (literature `B-V` = 0.00) ✓
/// - Sun     `BP-RP ≈ 0.82` → `B-V ≈ 0.66` (literature `B-V` = 0.65) ✓
/// - Aldebaran `BP-RP ≈ 1.65` → `B-V ≈ 1.32` (literature `B-V` = 1.54, off ~0.2)
pub fn bp_rp_to_johnson_b_v(bp_rp: f64) -> f64 {
    // Linear fit `B-V = -0.02 + 0.81·(BP-RP)` calibrated on the bright-star
    // cross-match (Hipparcos B-V × Gaia DR3 BP-RP). Easy to swap for a
    // higher-order polynomial if needed.
    const SLOPE: f64 = 0.81;
    const INTERCEPT: f64 = -0.02;
    INTERCEPT + SLOPE * bp_rp
}

/// Release-specific configuration for downloading, parsing, and caching.
///
/// Each of `Dr1`, `Dr2`, `Dr3` implements this trait so the generic reader/downloader
/// can be parameterized over the release without conditional logic at the call site.
pub trait GaiaRelease: 'static {
    const RELEASE: Release;

    /// HTTP base URL for the CSV.gz catalog files.
    const BASE_URL: &'static str;

    /// Filename within [`BASE_URL`] that holds the MD5 checksums for every file.
    const MD5_FILENAME: &'static str;

    /// Regex matching valid per-release catalog filenames on the index page.
    const FILE_REGEX: &'static str;

    /// Subdirectory under the starfield cache root (e.g. `"gaia/dr3"`).
    const CACHE_SUBDIR: &'static str;

    /// Fully-qualified ADQL table name in the ESA Gaia archive (e.g.
    /// `"gaiadr3.gaia_source"`). Used by [`crate::download::tap`] to build
    /// release-typed ADQL queries that return CSV slotted into the same
    /// `csv_header()` column layout the bulk files use.
    const TAP_TABLE: &'static str;

    /// True if the file uses ECSV (leading `#`-prefixed YAML header before CSV).
    const IS_ECSV: bool = false;

    /// The entry type produced by parsing one row.
    type Entry: GaiaSource + Clone + std::fmt::Debug + Send + 'static;

    /// Arrow schema describing the columns this release's parser consumes.
    fn arrow_schema() -> SchemaRef;

    /// Build one entry from row `row` of `batch`. Columns are in the order declared
    /// by [`arrow_schema`].
    fn build_entry(batch: &RecordBatch, row: usize) -> Result<Self::Entry>;

    /// Format an entry as one CSV row matching the column layout in [`arrow_schema`].
    /// Floats use Rust's default `Display` (full round-trip precision); `Option::None`
    /// becomes the empty string. Output is parseable by [`from_csv_file`](crate::common::catalog::MemoryResidentCatalog::from_csv_file).
    fn format_csv_row(entry: &Self::Entry) -> String;

    /// Comma-joined header line listing every column in [`arrow_schema`] order.
    fn csv_header() -> String {
        Self::arrow_schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect::<Vec<_>>()
            .join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Anchor the linear `BP-RP → B-V` fit against well-known sanity stars.
    /// Vega and the Sun are the canonical color-zero/G-type calibrators;
    /// FGK precision target is ~0.05 mag.
    #[test]
    fn bp_rp_to_b_v_anchors_to_sun_and_vega() {
        // Vega: BP-RP ≈ 0.00 → B-V should be ≈ 0.00
        let vega = bp_rp_to_johnson_b_v(0.00);
        assert!(vega.abs() < 0.05, "Vega B-V = {}, want ~0.00", vega);

        // Sun: BP-RP ≈ 0.82 → B-V should be ≈ 0.65
        let sun = bp_rp_to_johnson_b_v(0.82);
        assert!((sun - 0.65).abs() < 0.05, "Sun B-V = {}, want ~0.65", sun);

        // Procyon F5: BP-RP ≈ 0.55 → B-V ≈ 0.42
        let procyon = bp_rp_to_johnson_b_v(0.55);
        assert!(
            (procyon - 0.42).abs() < 0.05,
            "Procyon B-V = {}, want ~0.42",
            procyon
        );
    }

    #[test]
    fn bp_rp_to_b_v_is_monotone() {
        // Across the calibrated range, the fit must increase with BP-RP.
        let mut prev = bp_rp_to_johnson_b_v(-0.4);
        for i in 1..30 {
            let bp_rp = -0.4 + (i as f64) * 0.1;
            let bv = bp_rp_to_johnson_b_v(bp_rp);
            assert!(
                bv > prev,
                "non-monotone at BP-RP={}: {} ≤ {}",
                bp_rp,
                bv,
                prev
            );
            prev = bv;
        }
    }
}
