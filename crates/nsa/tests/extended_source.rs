//! Verifies `NsaEntry::sersic_profile()` returns a well-shaped
//! `starfield_core::catalogs::SersicProfile`.
//!
//! Builds a one-row NSA FITS table with known Sersic params, loads it
//! through the public API, and checks the field mapping + the upstream
//! `surface_brightness_at` evaluation.

use std::io::Write;

use fitsio_pure::bintable::{
    serialize_binary_table_hdu, BinaryColumnData, BinaryColumnDescriptor, BinaryColumnType,
};
use fitsio_pure::header::serialize_header;
use fitsio_pure::primary::build_primary_header;
use starfield_core::catalogs::{ExtendedSource, StarCatalog};
use starfield_nsa::{NsaCatalog, NsaVersion};

fn empty_primary_hdu() -> Vec<u8> {
    let cards = build_primary_header(8, &[]).unwrap();
    serialize_header(&cards)
}

fn write_minimal_nsa_fits(
    sersic_th50: f32,
    sersic_n: f32,
    sersic_ba: f32,
    sersic_phi: f32,
) -> tempfile::NamedTempFile {
    let columns = vec![
        BinaryColumnDescriptor {
            name: Some("NSAID".into()),
            repeat: 1,
            col_type: BinaryColumnType::Int,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("RA".into()),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some("DEC".into()),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some("Z".into()),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_TH50".into()),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_N".into()),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_BA".into()),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_PHI".into()),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_FLUX".into()),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
        BinaryColumnDescriptor {
            name: Some("SERSIC_FLUX_IVAR".into()),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
    ];
    let col_data = vec![
        BinaryColumnData::Int(vec![777]),
        BinaryColumnData::Double(vec![123.0]),
        BinaryColumnData::Double(vec![45.0]),
        BinaryColumnData::Float(vec![0.04]),
        BinaryColumnData::Float(vec![sersic_th50]),
        BinaryColumnData::Float(vec![sersic_n]),
        BinaryColumnData::Float(vec![sersic_ba]),
        BinaryColumnData::Float(vec![sersic_phi]),
        BinaryColumnData::Float(vec![1.0; 5]),
        BinaryColumnData::Float(vec![0.5; 5]),
    ];
    let bt = serialize_binary_table_hdu(&columns, &col_data, 1).unwrap();
    let mut fits = empty_primary_hdu();
    fits.extend_from_slice(&bt);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits).unwrap();
    tmp.as_file().sync_all().unwrap();
    tmp
}

#[test]
fn sersic_profile_field_mapping() {
    // NSA stores `sersic_ba` directly as b/a; upstream's
    // `axis_ratio` is also b/a — no flip needed (in contrast with
    // `Dr3GalaxyCandidate` and `BrightGalaxy` which store `1 - b/a`).
    let tmp = write_minimal_nsa_fits(2.5, 4.0, 0.7, 60.0);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    assert_eq!(cat.version(), NsaVersion::V0_1_2);

    let e = cat.get_star(777).unwrap();
    let p = e.sersic_profile().expect("Sersic params present");
    assert_eq!(p.theta_half_arcsec, 2.5);
    assert_eq!(p.n, 4.0);
    assert!(
        (p.axis_ratio - 0.7).abs() < 1e-6,
        "axis_ratio should pass through unchanged: got {}",
        p.axis_ratio
    );
    assert_eq!(p.position_angle_deg, 60.0);
}

#[test]
fn sersic_profile_evaluates_surface_brightness() {
    // Circular n=1 (exponential disk), R_e = 10". At r = R_e the SB
    // should be exactly I_e (i.e. surface_brightness_at returns 1.0).
    let tmp = write_minimal_nsa_fits(10.0, 1.0, 1.0, 0.0);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(777).unwrap();
    let p = e.sersic_profile().unwrap();

    // PA = 0 means major axis along +y → offset (0, 10) is at R_e.
    let sb_at_re = p.surface_brightness_at(0.0, 10.0);
    assert!(
        (sb_at_re - 1.0).abs() < 1e-9,
        "I(R_e) / I_e should be 1.0 along major axis, got {}",
        sb_at_re
    );

    // Centre is exp(b_n) > 1.
    let sb_centre = p.surface_brightness_at(0.0, 0.0);
    assert!(
        sb_centre > 1.0,
        "centre SB should be exp(b_n) > 1, got {}",
        sb_centre
    );
}

#[test]
fn sersic_profile_returns_none_on_pathological_axis_ratio() {
    // sersic_ba ≤ 0 isn't a real NSA value (the fit constrains it
    // positive), but defend against it anyway.
    let tmp = write_minimal_nsa_fits(2.5, 4.0, 0.0, 60.0);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(777).unwrap();
    assert!(
        e.sersic_profile().is_none(),
        "axis_ratio = 0 should yield None (degenerate edge-on)"
    );
}
