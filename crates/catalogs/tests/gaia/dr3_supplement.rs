//! End-to-end tests for `Dr3Catalog::augment_missing` and
//! `Dr3Catalog::augment_missing_in_cone` against the embedded
//! Hipparcos-derived bright-star supplement.

use std::collections::HashSet;

use starfield_catalogs::gaia::dr3::supplement::{
    decode_supplement_hip, is_supplement_source_id, parse_embedded_supplement, SUPPLEMENT_REF_EPOCH,
};
use starfield_catalogs::gaia::{Cone, Dr3Catalog};
use starfield_core::catalogs::StarCatalog;

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

#[test]
fn augment_missing_in_cone_matches_filtered_supplement() {
    // Orion: dense with naked-eye stars, so a 10 deg cone holds many rows.
    let cone = Cone::from_degrees(83.8, -1.2, 10.0);
    let mag_limit = 9.0;
    let rows = parse_embedded_supplement().unwrap();
    let expected = rows
        .iter()
        .filter(|r| r.fitted_g_mag <= mag_limit && cone.contains_radec_deg(r.ra, r.dec))
        .count();

    let mut cat = Dr3Catalog::new();
    let added = cat.augment_missing_in_cone(cone, mag_limit).unwrap();
    assert_eq!(added, expected);
    assert_eq!(cat.len(), added);
    assert!(
        added > 0,
        "expected supplement stars in a 10 deg Orion cone"
    );

    for star in cat.stars() {
        assert!(cone.contains_radec_deg(star.core.ra, star.core.dec));
        assert!(star.core.phot_g_mean_mag <= mag_limit);
        assert!(is_supplement_source_id(star.core.source_id));
    }
}

#[test]
fn augment_missing_in_cone_is_a_small_subset_of_the_whole_sky() {
    let cone = Cone::from_degrees(83.8, -1.2, 2.0);
    let mut cone_cat = Dr3Catalog::new();
    let n_cone = cone_cat
        .augment_missing_in_cone(cone, f64::INFINITY)
        .unwrap();

    let mut sky_cat = Dr3Catalog::new();
    let n_sky = sky_cat.augment_missing(f64::INFINITY).unwrap();

    assert!(
        n_cone * 100 < n_sky,
        "a 2 deg cone ({n_cone} rows) should hold well under 1% of the sky ({n_sky} rows)"
    );
    let sky_ids: HashSet<u64> = sky_cat.stars().map(|s| s.core.source_id).collect();
    for star in cone_cat.stars() {
        assert!(sky_ids.contains(&star.core.source_id));
    }
}

#[test]
fn augment_missing_in_cone_whole_sky_cone_matches_augment_missing() {
    let whole_sky = Cone::from_degrees(0.0, 0.0, 180.0);
    let mut cone_cat = Dr3Catalog::new();
    let n_cone = cone_cat.augment_missing_in_cone(whole_sky, 8.0).unwrap();

    let mut sky_cat = Dr3Catalog::new();
    let n_sky = sky_cat.augment_missing(8.0).unwrap();

    assert_eq!(n_cone, n_sky);
    assert_eq!(cone_cat.len(), sky_cat.len());
}

#[test]
fn augment_missing_in_cone_includes_centred_star_and_excludes_antipode() {
    let rows = parse_embedded_supplement().unwrap();
    let brightest = rows
        .iter()
        .min_by(|a, b| a.fitted_g_mag.total_cmp(&b.fitted_g_mag))
        .unwrap();

    let mut near = Dr3Catalog::new();
    near.augment_missing_in_cone(
        Cone::from_degrees(brightest.ra, brightest.dec, 0.01),
        f64::INFINITY,
    )
    .unwrap();
    let hips: Vec<u32> = near
        .stars()
        .map(|s| decode_supplement_hip(s.core.source_id).unwrap())
        .collect();
    assert!(hips.contains(&brightest.hip));

    let mut far = Dr3Catalog::new();
    far.augment_missing_in_cone(
        Cone::from_degrees((brightest.ra + 180.0) % 360.0, -brightest.dec, 0.01),
        f64::INFINITY,
    )
    .unwrap();
    let far_hips: Vec<u32> = far
        .stars()
        .map(|s| decode_supplement_hip(s.core.source_id).unwrap())
        .collect();
    assert!(!far_hips.contains(&brightest.hip));
}

#[test]
fn augment_missing_in_cone_respects_mag_limit() {
    let cone = Cone::from_degrees(83.8, -1.2, 10.0);
    let mut faint = Dr3Catalog::new();
    let n_faint = faint.augment_missing_in_cone(cone, f64::INFINITY).unwrap();
    let mut bright = Dr3Catalog::new();
    let n_bright = bright.augment_missing_in_cone(cone, 6.0).unwrap();
    assert!(n_bright < n_faint);
    for star in bright.stars() {
        assert!(star.core.phot_g_mean_mag <= 6.0);
    }
}
