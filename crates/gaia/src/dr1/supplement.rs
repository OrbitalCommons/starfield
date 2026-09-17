//! Hipparcos-derived bright-star supplement for Gaia DR1.
//!
//! See [`crate::common::supplement`] for the shared schema, ID-masking
//! scheme, and provenance/missing-fields documentation. This module just
//! glues the common helpers to DR1's reference epoch (J2015.0) and
//! [`Dr1Entry`] type, and embeds the DR1-specific CSV produced by the
//! `hipparcos-gaia-match` binary in `starfield-gaia-tools`.
//!
//! End-user entry point: [`Dr1Catalog::augment_missing`](crate::Dr1Catalog::augment_missing).
//!
//! # DR1-specific notes
//!
//! Most DR1 sources have only a position (no proper motion or parallax) —
//! TGAS provides 5-parameter astrometry for ~2M Hipparcos/Tycho-2 sources,
//! the rest is 2-parameter. So DR1 is *especially* reliant on this
//! supplement for bright stars: many bright Hipparcos entries with no DR1
//! match whatsoever.

use starfield_core::Result;

use crate::common::supplement::{make_supplement_core, parse_supplement_csv, SupplementRow};
use crate::dr1::entry::{AstrometricExtra, Dr1Entry, ScanDirection};

pub use crate::common::supplement::{
    decode_supplement_hip, encode_supplement_source_id, is_supplement_source_id,
    SUPPLEMENT_SOURCE_ID_BIT,
};

/// J2015.0 — DR1's reference epoch.
pub const SUPPLEMENT_REF_EPOCH: f64 = 2015.0;

const EMBEDDED_SUPPLEMENT_CSV: &str = include_str!("../../data/dr1-bright-star-supplement.csv");

/// Parse the embedded DR1 supplement.
pub fn parse_embedded_supplement() -> Result<Vec<SupplementRow>> {
    parse_supplement_csv(EMBEDDED_SUPPLEMENT_CSV)
}

/// Wrap a [`SupplementRow`] into a fully-formed [`Dr1Entry`].
pub fn supplement_row_to_entry(r: &SupplementRow) -> Dr1Entry {
    Dr1Entry {
        core: make_supplement_core(r, SUPPLEMENT_REF_EPOCH),
        astrometric_extra: AstrometricExtra::default(),
        scan_direction: ScanDirection::default(),
        tgas: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_supplement_parses() {
        let rows = parse_embedded_supplement().expect("embedded supplement parses");
        assert!(
            rows.len() > 1000,
            "expected thousands of rows, got {}",
            rows.len()
        );
        for r in &rows {
            assert!(r.hip > 0);
            assert!(r.ra >= 0.0 && r.ra <= 360.0);
            assert!(r.dec >= -90.0 && r.dec <= 90.0);
            assert!(r.fitted_g_mag.is_finite());
        }
    }

    #[test]
    fn supplement_row_to_entry_smoke() {
        let row = SupplementRow {
            hip: 12345,
            ra: 180.5,
            dec: -23.0,
            parallax_mas: Some(15.0),
            pmra_mas_yr: Some(-100.0),
            pmdec_mas_yr: Some(50.0),
            b_v: Some(0.5),
            fitted_g_mag: 6.8,
        };
        let entry = supplement_row_to_entry(&row);
        assert_eq!(entry.core.ra, 180.5);
        assert_eq!(entry.core.phot_g_mean_mag, 6.8);
        assert_eq!(entry.core.ref_epoch, SUPPLEMENT_REF_EPOCH);
        assert_eq!(decode_supplement_hip(entry.core.source_id), Some(12345));
        assert!(entry.tgas.is_none());
    }
}
