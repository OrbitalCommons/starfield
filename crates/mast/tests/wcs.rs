//! Synthetic WCS tests — build a tiny FITS header with known keywords,
//! parse it through `Wcs::read_from_cards`, and check pixel↔world,
//! footprint, and plate-scale math.

use fitsio_pure::header::Card;
use fitsio_pure::value::Value;
use starfield_mast::Wcs;

/// Build a minimal Card for a known float value.
fn float_card(keyword: &str, value: f64) -> Card {
    Card {
        keyword: pad_keyword(keyword),
        value: Some(Value::Float(value)),
        comment: None,
    }
}

fn int_card(keyword: &str, value: i64) -> Card {
    Card {
        keyword: pad_keyword(keyword),
        value: Some(Value::Integer(value)),
        comment: None,
    }
}

fn string_card(keyword: &str, value: &str) -> Card {
    Card {
        keyword: pad_keyword(keyword),
        value: Some(Value::String(value.to_string())),
        comment: None,
    }
}

fn pad_keyword(keyword: &str) -> [u8; 8] {
    let mut out = [b' '; 8];
    let bytes = keyword.as_bytes();
    out[..bytes.len()].copy_from_slice(bytes);
    out
}

/// 100×100 axis-aligned TAN WCS centred at (180°, 0°), 1″/pixel.
fn tan_test_cards() -> Vec<Card> {
    let one_arcsec_deg = 1.0 / 3600.0;
    vec![
        float_card("CRVAL1", 180.0),
        float_card("CRVAL2", 0.0),
        float_card("CRPIX1", 50.5),
        float_card("CRPIX2", 50.5),
        float_card("CD1_1", one_arcsec_deg),
        float_card("CD1_2", 0.0),
        float_card("CD2_1", 0.0),
        float_card("CD2_2", one_arcsec_deg),
        string_card("CTYPE1", "RA---TAN"),
        string_card("CTYPE2", "DEC--TAN"),
        int_card("NAXIS1", 100),
        int_card("NAXIS2", 100),
    ]
}

#[test]
fn parses_required_cards() {
    let cards = tan_test_cards();
    let wcs = Wcs::read_from_cards(&cards).unwrap();
    assert_eq!(wcs.crval1, 180.0);
    assert_eq!(wcs.crval2, 0.0);
    assert_eq!(wcs.crpix1, 50.5);
    assert_eq!(wcs.crpix2, 50.5);
    assert_eq!(wcs.naxis1, 100);
    assert_eq!(wcs.naxis2, 100);
    assert_eq!(wcs.ctype1, "RA---TAN");
    assert_eq!(wcs.ctype2, "DEC--TAN");
    assert!(wcs.is_tan());
}

#[test]
fn pixel_to_world_at_reference_point_returns_crval() {
    let wcs = Wcs::read_from_cards(&tan_test_cards()).unwrap();
    let (ra, dec) = wcs.pixel_to_world(50.5, 50.5).unwrap();
    assert!(
        (ra - 180.0).abs() < 1e-12,
        "ra at CRPIX should be CRVAL1, got {}",
        ra
    );
    assert!(
        (dec - 0.0).abs() < 1e-12,
        "dec at CRPIX should be CRVAL2, got {}",
        dec
    );
}

#[test]
fn pixel_to_world_offset_matches_small_angle_approximation() {
    // At Dec=0, RA at +1 pixel east is RA0 + (1″/3600) / cos(0) = RA0 + 1/3600 deg.
    let wcs = Wcs::read_from_cards(&tan_test_cards()).unwrap();
    let (ra, dec) = wcs.pixel_to_world(51.5, 50.5).unwrap();
    let expected_dra = 1.0 / 3600.0;
    assert!(
        (ra - (180.0 + expected_dra)).abs() < 1e-9,
        "RA off by more than 1 mas: got {}, expected {}",
        ra,
        180.0 + expected_dra
    );
    assert!(
        dec.abs() < 1e-9,
        "dec should stay at 0 along the row, got {}",
        dec
    );

    // +1 pixel north → Dec rises by exactly 1/3600 (no cos factor in Dec).
    let (_ra2, dec2) = wcs.pixel_to_world(50.5, 51.5).unwrap();
    assert!(
        (dec2 - 1.0 / 3600.0).abs() < 1e-12,
        "dec off by more than 1e-12: got {}",
        dec2
    );
}

#[test]
fn footprint_returns_four_corners_in_reasonable_bounds() {
    // 100x100 at 1″/pix → ~50″ from centre to each corner.
    let wcs = Wcs::read_from_cards(&tan_test_cards()).unwrap();
    let corners = wcs.footprint().unwrap();
    assert_eq!(corners.len(), 4);
    let half_arcsec = 50.5 / 3600.0; // distance from centre to corner
    for (i, (ra, dec)) in corners.iter().enumerate() {
        let dra = (ra - 180.0).abs();
        let ddec = dec.abs();
        assert!(
            dra <= half_arcsec * 1.5,
            "corner {} RA out of range: {}",
            i,
            ra
        );
        assert!(
            ddec <= half_arcsec * 1.5,
            "corner {} Dec out of range: {}",
            i,
            dec
        );
    }

    // Corners must be ordered (BL, BR, TR, TL) — Dec sign matches:
    // first two are bottom (negative Dec), last two are top (positive).
    assert!(corners[0].1 < 0.0, "BL should be south of centre");
    assert!(corners[1].1 < 0.0, "BR should be south of centre");
    assert!(corners[2].1 > 0.0, "TR should be north of centre");
    assert!(corners[3].1 > 0.0, "TL should be north of centre");
}

#[test]
fn pixel_scale_returns_arcsec_per_pixel() {
    let wcs = Wcs::read_from_cards(&tan_test_cards()).unwrap();
    let scale = wcs.pixel_scale_arcsec();
    assert!(
        (scale - 1.0).abs() < 1e-9,
        "axis-aligned 1″/pix CD should give scale ≈ 1.0, got {}",
        scale
    );
}

#[test]
fn pc_plus_cdelt_fallback_normalises_into_cd_matrix() {
    let one_arcsec_deg = 1.0 / 3600.0;
    let mut cards = vec![
        float_card("CRVAL1", 0.0),
        float_card("CRVAL2", 0.0),
        float_card("CRPIX1", 50.5),
        float_card("CRPIX2", 50.5),
        // No CD matrix — only PC + CDELT.
        float_card("PC1_1", 1.0),
        float_card("PC1_2", 0.0),
        float_card("PC2_1", 0.0),
        float_card("PC2_2", 1.0),
        float_card("CDELT1", one_arcsec_deg),
        float_card("CDELT2", one_arcsec_deg),
        string_card("CTYPE1", "RA---TAN"),
        string_card("CTYPE2", "DEC--TAN"),
        int_card("NAXIS1", 100),
        int_card("NAXIS2", 100),
    ];
    let wcs = Wcs::read_from_cards(&cards).unwrap();
    assert!((wcs.cd[0][0] - one_arcsec_deg).abs() < 1e-15);
    assert!((wcs.cd[1][1] - one_arcsec_deg).abs() < 1e-15);

    // Sanity: pixel_to_world still returns CRVAL at CRPIX.
    let (ra, dec) = wcs.pixel_to_world(50.5, 50.5).unwrap();
    assert!(ra.abs() < 1e-12 && dec.abs() < 1e-12);

    // Drop CDELT — should now error.
    cards.retain(|c| c.keyword_str() != "CDELT1" && c.keyword_str() != "CDELT2");
    let err = Wcs::read_from_cards(&cards).expect_err("missing CDELT should error");
    assert!(err.to_string().contains("PC+CDELT"));
}

#[test]
fn missing_required_keyword_errors() {
    let mut cards = tan_test_cards();
    cards.retain(|c| c.keyword_str() != "CRVAL1");
    let err = Wcs::read_from_cards(&cards).expect_err("missing CRVAL1 should error");
    assert!(err.to_string().contains("CRVAL1"));
}

#[test]
fn tan_sip_ctype_is_treated_as_tan() {
    // HST FLT/FLC products use CTYPE like `RA---TAN-SIP` — the SIP
    // suffix is a distortion model on top of TAN. Our linear evaluator
    // treats them as TAN (with the documented "SIP not applied" caveat),
    // so footprint() / pixel_to_world() must succeed without erroring.
    let mut cards = tan_test_cards();
    cards.retain(|c| c.keyword_str() != "CTYPE1" && c.keyword_str() != "CTYPE2");
    cards.push(string_card("CTYPE1", "RA---TAN-SIP"));
    cards.push(string_card("CTYPE2", "DEC--TAN-SIP"));
    let wcs = Wcs::read_from_cards(&cards).unwrap();
    assert!(wcs.is_tan(), "TAN-SIP should be accepted as TAN");
    let (ra, dec) = wcs.pixel_to_world(50.5, 50.5).unwrap();
    assert!((ra - 180.0).abs() < 1e-12 && dec.abs() < 1e-12);
    let _ = wcs.footprint().unwrap();
}

#[test]
fn non_tan_projection_errors_at_pixel_to_world() {
    let mut cards = tan_test_cards();
    // Replace CTYPE1 with a SIN projection so is_tan() returns false.
    cards.retain(|c| c.keyword_str() != "CTYPE1");
    cards.push(string_card("CTYPE1", "RA---SIN"));
    let wcs = Wcs::read_from_cards(&cards).unwrap();
    assert!(!wcs.is_tan());
    let err = wcs
        .pixel_to_world(50.5, 50.5)
        .expect_err("non-TAN should error");
    assert!(err.to_string().contains("TAN"));
}
