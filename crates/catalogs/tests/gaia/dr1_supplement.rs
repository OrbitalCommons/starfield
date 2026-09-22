//! End-to-end test for `Dr1Catalog::augment_missing` against the embedded
//! Hipparcos-derived bright-star supplement.
#![cfg(feature = "dr1")]

use starfield_catalogs::gaia::dr1::supplement::{
    decode_supplement_hip, is_supplement_source_id, parse_embedded_supplement, SUPPLEMENT_REF_EPOCH,
};
use starfield_catalogs::gaia::Dr1Catalog;
use starfield_core::catalogs::StarCatalog;

#[test]
fn augment_missing_inserts_thousands_of_bright_stars() {
    let mut cat = Dr1Catalog::new();
    assert_eq!(cat.len(), 0);

    let added = cat.augment_missing(f64::INFINITY).unwrap();
    assert_eq!(cat.len(), added);
    assert!(
        added > 5_000,
        "expected > 5000 supplement entries, got {}",
        added
    );
}

#[test]
fn augment_missing_respects_mag_limit() {
    let mut cat_full = Dr1Catalog::new();
    let n_full = cat_full.augment_missing(f64::INFINITY).unwrap();

    let mut cat_bright = Dr1Catalog::new();
    let n_bright = cat_bright.augment_missing(8.0).unwrap();

    assert!(n_bright < n_full);
    assert!(n_bright > 0);

    for star in cat_bright.stars() {
        assert!(star.core.phot_g_mean_mag <= 8.0);
    }
}

#[test]
fn augment_missing_is_idempotent() {
    let mut cat = Dr1Catalog::new();
    let n1 = cat.augment_missing(f64::INFINITY).unwrap();
    let n2 = cat.augment_missing(f64::INFINITY).unwrap();
    assert_eq!(n1, n2);
    assert_eq!(cat.len(), n1);
}

#[test]
fn every_inserted_entry_has_supplement_source_id_and_dr1_epoch() {
    let mut cat = Dr1Catalog::new();
    cat.augment_missing(f64::INFINITY).unwrap();

    let rows = parse_embedded_supplement().unwrap();
    let mut hips_seen = 0;
    for star in cat.stars() {
        let core = &star.core;
        assert!(is_supplement_source_id(core.source_id));
        let hip = decode_supplement_hip(core.source_id).unwrap();
        assert!(hip > 0 && hip < 200_000);
        assert_eq!(core.ref_epoch, SUPPLEMENT_REF_EPOCH);
        hips_seen += 1;
    }
    assert_eq!(hips_seen, rows.len());
}
