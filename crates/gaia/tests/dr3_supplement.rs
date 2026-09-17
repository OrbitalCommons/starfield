//! End-to-end test for `Dr3Catalog::augment_missing` against the embedded
//! Hipparcos-derived bright-star supplement.

use starfield_core::catalogs::StarCatalog;
use starfield_gaia::dr3::supplement::{
    decode_supplement_hip, is_supplement_source_id, parse_embedded_supplement, SUPPLEMENT_REF_EPOCH,
};
use starfield_gaia::Dr3Catalog;

#[test]
fn augment_missing_inserts_thousands_of_bright_stars() {
    let mut cat = Dr3Catalog::new();
    assert_eq!(cat.len(), 0);

    let added = cat.augment_missing(f64::INFINITY).unwrap();
    assert_eq!(cat.len(), added);
    // The embedded supplement is the unmatched-Hipparcos set produced by
    // the cross-match against DR3 G ≤ 12; expect thousands of entries.
    assert!(
        added > 5_000,
        "expected > 5000 supplement entries, got {}",
        added
    );
}

#[test]
fn augment_missing_respects_mag_limit() {
    let mut cat_full = Dr3Catalog::new();
    let n_full = cat_full.augment_missing(f64::INFINITY).unwrap();

    let mut cat_bright = Dr3Catalog::new();
    let n_bright = cat_bright.augment_missing(8.0).unwrap();

    assert!(
        n_bright < n_full,
        "mag-limit 8 should drop entries vs no limit"
    );
    assert!(n_bright > 0, "some Hipparcos must have G ≤ 8");

    // Every retained entry must satisfy the mag limit.
    for star in cat_bright.stars() {
        assert!(
            star.core.phot_g_mean_mag <= 8.0,
            "entry with G > 8.0 leaked through: {}",
            star.core.phot_g_mean_mag
        );
    }
}

#[test]
fn augment_missing_is_idempotent() {
    let mut cat = Dr3Catalog::new();
    let n1 = cat.augment_missing(f64::INFINITY).unwrap();
    let n2 = cat.augment_missing(f64::INFINITY).unwrap();
    // Both calls report the same insert count; the catalog size is
    // unchanged because supplement source_ids collide with themselves
    // (HashMap overwrites).
    assert_eq!(n1, n2);
    assert_eq!(cat.len(), n1);
}

#[test]
fn every_inserted_entry_has_supplement_source_id_and_dr3_epoch() {
    let mut cat = Dr3Catalog::new();
    cat.augment_missing(f64::INFINITY).unwrap();

    let rows = parse_embedded_supplement().unwrap();
    let mut hips_seen = 0;
    for star in cat.stars() {
        let core = &star.core;
        assert!(
            is_supplement_source_id(core.source_id),
            "expected supplement-marker source_id, got {}",
            core.source_id
        );
        let hip = decode_supplement_hip(core.source_id).unwrap();
        // HIP numbers are positive and Hipparcos publishes ≤ ~120 000.
        assert!(hip > 0 && hip < 200_000);
        assert_eq!(core.ref_epoch, SUPPLEMENT_REF_EPOCH);
        hips_seen += 1;
    }
    // Sanity: the catalog has exactly one entry per supplement row.
    assert_eq!(hips_seen, rows.len());
}
