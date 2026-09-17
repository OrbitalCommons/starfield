//! Round-trip tests for the Part-2 surface area: extra columns + the
//! [`Photometry`], [`IsophoteSeries`], and (gated) [`RadialProfile`] trait
//! impls on [`NsaEntry`].
//!
//! Each test builds a synthetic NSA-shaped binary table HDU containing only
//! the columns under test (the shared minimum: NSAID/RA/DEC/Z/SERSIC_*) plus
//! whichever Part-2 columns the test cares about, runs it through
//! `NsaCatalog::from_fits_file`, and asserts the trait surface returns the
//! right values.

use std::io::Write;

use fitsio_pure::bintable::{
    serialize_binary_table_hdu, BinaryColumnData, BinaryColumnDescriptor, BinaryColumnType,
};
use fitsio_pure::header::serialize_header;
use fitsio_pure::primary::build_primary_header;

use starfield_core::catalogs::{Band, IsophoteSeries, Photometry, StarCatalog};
use starfield_nsa::NsaCatalog;

#[cfg(feature = "radial-profiles")]
use starfield_core::catalogs::RadialProfile;

fn empty_primary_hdu() -> Vec<u8> {
    let cards = build_primary_header(8, &[]).unwrap();
    serialize_header(&cards)
}

/// Build the always-required column descriptors and per-row data for `n_rows`
/// galaxies (NSAID = `(i+1) * 100`, RA/Dec/Z varying, dummy Sérsic fit).
fn baseline_columns_and_data(
    n_rows: usize,
) -> (Vec<BinaryColumnDescriptor>, Vec<BinaryColumnData>) {
    let cols = vec![
        BinaryColumnDescriptor {
            name: Some(String::from("NSAID")),
            repeat: 1,
            col_type: BinaryColumnType::Int,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("RA")),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("DEC")),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("Z")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_TH50")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_N")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_BA")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_PHI")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_FLUX")),
            repeat: 7,
            col_type: BinaryColumnType::Float,
            byte_width: 28,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_FLUX_IVAR")),
            repeat: 7,
            col_type: BinaryColumnType::Float,
            byte_width: 28,
        },
    ];
    let nsaid: Vec<i32> = (0..n_rows as i32).map(|i| (i + 1) * 100).collect();
    let ra: Vec<f64> = (0..n_rows).map(|i| 10.0 + i as f64).collect();
    let dec: Vec<f64> = (0..n_rows).map(|i| -5.0 + i as f64).collect();
    let z: Vec<f32> = (0..n_rows).map(|i| 0.01 * (i + 1) as f32).collect();
    let scalar: Vec<f32> = (0..n_rows).map(|_| 1.0).collect();
    let band_flat: Vec<f32> = (0..n_rows * 7).map(|i| (i + 1) as f32).collect();
    let data = vec![
        BinaryColumnData::Int(nsaid),
        BinaryColumnData::Double(ra),
        BinaryColumnData::Double(dec),
        BinaryColumnData::Float(z),
        BinaryColumnData::Float(scalar.clone()),
        BinaryColumnData::Float(scalar.clone()),
        BinaryColumnData::Float(scalar.clone()),
        BinaryColumnData::Float(scalar),
        BinaryColumnData::Float(band_flat.clone()),
        BinaryColumnData::Float(band_flat),
    ];
    (cols, data)
}

fn write_fits_to_tempfile(
    columns: &[BinaryColumnDescriptor],
    col_data: &[BinaryColumnData],
    n_rows: usize,
) -> tempfile::NamedTempFile {
    let bt_ext = serialize_binary_table_hdu(columns, col_data, n_rows).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();
    tmp
}

// ===========================================================================
// Photometry trait
// ===========================================================================

#[test]
fn photometry_returns_none_when_columns_absent() {
    let n_rows = 1;
    let (cols, data) = baseline_columns_and_data(n_rows);
    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();
    // No NMGY/EXTINCTION/KCORRECT in this fixture → every band must yield None.
    for band in [
        Band::GalexFuv,
        Band::GalexNuv,
        Band::SdssU,
        Band::SdssG,
        Band::SdssR,
        Band::SdssI,
        Band::SdssZ,
    ] {
        assert_eq!(e.flux_nmgy(band), None, "{:?} flux must be None", band);
        assert_eq!(e.extinction_mag(band), None, "{:?} ext must be None", band);
        assert_eq!(
            e.k_correction_mag(band),
            None,
            "{:?} kcorr must be None",
            band
        );
    }
}

#[test]
fn photometry_with_full_v1_0_1_columns() {
    let n_rows = 1;
    let (mut cols, mut data) = baseline_columns_and_data(n_rows);
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("NMGY")),
        repeat: 7,
        col_type: BinaryColumnType::Float,
        byte_width: 28,
    });
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("NMGY_IVAR")),
        repeat: 7,
        col_type: BinaryColumnType::Float,
        byte_width: 28,
    });
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("EXTINCTION")),
        repeat: 7,
        col_type: BinaryColumnType::Float,
        byte_width: 28,
    });
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("KCORRECT")),
        repeat: 7,
        col_type: BinaryColumnType::Float,
        byte_width: 28,
    });
    // FUV, NUV, u, g, r, i, z fluxes — distinct values per band.
    let nmgy: Vec<f32> = vec![0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0];
    let nmgy_ivar: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    let ext: Vec<f32> = vec![0.1, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02];
    let kcor: Vec<f32> = vec![0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.0];
    data.push(BinaryColumnData::Float(nmgy.clone()));
    data.push(BinaryColumnData::Float(nmgy_ivar.clone()));
    data.push(BinaryColumnData::Float(ext.clone()));
    data.push(BinaryColumnData::Float(kcor.clone()));

    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();

    // Bands NSA carries — exact match.
    let bands = [
        (Band::GalexFuv, 0),
        (Band::GalexNuv, 1),
        (Band::SdssU, 2),
        (Band::SdssG, 3),
        (Band::SdssR, 4),
        (Band::SdssI, 5),
        (Band::SdssZ, 6),
    ];
    for (band, idx) in bands {
        assert!(
            (e.flux_nmgy(band).unwrap() - nmgy[idx] as f64).abs() < 1e-6,
            "{:?} flux",
            band
        );
        assert!(
            (e.flux_ivar(band).unwrap() - nmgy_ivar[idx] as f64).abs() < 1e-5,
            "{:?} ivar",
            band
        );
        assert!(
            (e.extinction_mag(band).unwrap() - ext[idx] as f64).abs() < 1e-6,
            "{:?} ext",
            band
        );
        assert!(
            (e.k_correction_mag(band).unwrap() - kcor[idx] as f64).abs() < 1e-6,
            "{:?} kcorr",
            band
        );
    }

    // Bands NSA does NOT carry — all None.
    for band in [
        Band::TwoMassJ,
        Band::TwoMassH,
        Band::TwoMassK,
        Band::GaiaG,
        Band::GaiaBp,
        Band::GaiaRp,
        Band::HipparcosHp,
        Band::JohnsonV,
        Band::JohnsonB,
    ] {
        assert_eq!(e.flux_nmgy(band), None, "{:?}", band);
    }
}

#[test]
fn ab_magnitude_composes_extinction_and_kcorr() {
    // r-band flux = 100 nMgy → AB = 22.5 - 2.5*log10(100) = 17.5
    // ext_r = 0.5, kcorr_r = 0.2
    // dereddened: 17.5 - 0.5 = 17.0
    // dered + kcorr: 17.0 - 0.2 = 16.8
    let n_rows = 1;
    let (mut cols, mut data) = baseline_columns_and_data(n_rows);
    for name in ["NMGY", "EXTINCTION", "KCORRECT"] {
        cols.push(BinaryColumnDescriptor {
            name: Some(String::from(name)),
            repeat: 7,
            col_type: BinaryColumnType::Float,
            byte_width: 28,
        });
    }
    let nmgy: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 100.0, 1.0, 1.0];
    let ext: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0];
    let kcor: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0];
    data.push(BinaryColumnData::Float(nmgy));
    data.push(BinaryColumnData::Float(ext));
    data.push(BinaryColumnData::Float(kcor));

    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();

    let raw = e.ab_magnitude(Band::SdssR, false, false).unwrap();
    let dered = e.ab_magnitude(Band::SdssR, true, false).unwrap();
    let full = e.ab_magnitude(Band::SdssR, true, true).unwrap();
    assert!((raw - 17.5).abs() < 1e-6, "raw r mag: {}", raw);
    assert!((dered - 17.0).abs() < 1e-6, "dered: {}", dered);
    assert!((full - 16.8).abs() < 1e-6, "full: {}", full);
}

// ===========================================================================
// IsophoteSeries trait
// ===========================================================================

#[test]
fn isophote_series_none_when_scalars_missing() {
    let n_rows = 1;
    let (cols, data) = baseline_columns_and_data(n_rows);
    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();
    assert_eq!(e.isophote_samples(Band::SdssR), None);
}

#[test]
fn isophote_series_two_samples_at_50_and_90() {
    let n_rows = 1;
    let (mut cols, mut data) = baseline_columns_and_data(n_rows);
    for name in ["BA50", "PHI50", "BA90", "PHI90", "PETROTH50", "PETROTH90"] {
        cols.push(BinaryColumnDescriptor {
            name: Some(String::from(name)),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        });
    }
    data.push(BinaryColumnData::Float(vec![0.85])); // BA50
    data.push(BinaryColumnData::Float(vec![45.0])); // PHI50
    data.push(BinaryColumnData::Float(vec![0.55])); // BA90
    data.push(BinaryColumnData::Float(vec![60.0])); // PHI90
    data.push(BinaryColumnData::Float(vec![3.5])); // PETROTH50
    data.push(BinaryColumnData::Float(vec![12.0])); // PETROTH90

    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();

    // Trait ignores `band`; same series returned regardless.
    for band in [Band::SdssR, Band::SdssG, Band::GalexFuv, Band::JohnsonV] {
        let samples = e.isophote_samples(band).unwrap();
        assert_eq!(samples.len(), 2, "expected 2 samples, got {:?}", samples);

        assert!((samples[0].radius_arcsec - 3.5).abs() < 1e-9);
        assert!((samples[0].axis_ratio - 0.85).abs() < 1e-6);
        assert!((samples[0].position_angle_deg - 45.0).abs() < 1e-6);

        assert!((samples[1].radius_arcsec - 12.0).abs() < 1e-9);
        assert!((samples[1].axis_ratio - 0.55).abs() < 1e-6);
        assert!((samples[1].position_angle_deg - 60.0).abs() < 1e-6);
    }
}

// ===========================================================================
// RadialProfile trait (radial-profiles feature)
// ===========================================================================

#[cfg(feature = "radial-profiles")]
#[test]
fn radial_profile_round_trip_v1_0_1() {
    use starfield_nsa::N_PROFILE_RADII;

    let n_rows = 1;
    let (mut cols, mut data) = baseline_columns_and_data(n_rows);
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("PROFTHETA")),
        repeat: N_PROFILE_RADII,
        col_type: BinaryColumnType::Float,
        byte_width: (N_PROFILE_RADII * 4),
    });
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("PROFMEAN")),
        repeat: N_PROFILE_RADII * 7,
        col_type: BinaryColumnType::Float,
        byte_width: (N_PROFILE_RADII * 7 * 4),
    });
    cols.push(BinaryColumnDescriptor {
        name: Some(String::from("PROFMEAN_IVAR")),
        repeat: N_PROFILE_RADII * 7,
        col_type: BinaryColumnType::Float,
        byte_width: (N_PROFILE_RADII * 7 * 4),
    });

    // Radii = [0.5, 1.0, 1.5, …, 7.5]
    let radii: Vec<f32> = (0..N_PROFILE_RADII).map(|i| 0.5 + 0.5 * i as f32).collect();
    // PROFMEAN[r][b] = (r * 10) + b — distinct per cell, easy to check
    // FITS layout: row-major flat = radius slow, band fast → flat[r * 7 + b].
    let mut profmean: Vec<f32> = Vec::with_capacity(N_PROFILE_RADII * 7);
    for r in 0..N_PROFILE_RADII {
        for b in 0..7 {
            profmean.push((r * 10 + b) as f32);
        }
    }
    let profmean_ivar: Vec<f32> = profmean.iter().map(|x| 1.0 / (1.0 + x)).collect();

    data.push(BinaryColumnData::Float(radii.clone()));
    data.push(BinaryColumnData::Float(profmean.clone()));
    data.push(BinaryColumnData::Float(profmean_ivar.clone()));

    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    let e = cat.get_star(100).unwrap();

    let got_radii = e.profile_radii_arcsec().unwrap();
    assert_eq!(got_radii.len(), N_PROFILE_RADII);
    for (i, &r) in got_radii.iter().enumerate() {
        assert!((r - radii[i] as f64).abs() < 1e-6);
    }

    // Spot-check r-band (idx 4) brightness across all radii.
    let sb_r = e.profile_surface_brightness(Band::SdssR).unwrap();
    assert_eq!(sb_r.len(), N_PROFILE_RADII);
    for (r, &got) in sb_r.iter().enumerate() {
        let want = (r * 10 + 4) as f64;
        assert!(
            (got - want).abs() < 1e-6,
            "r={} got={} want={}",
            r,
            got,
            want
        );
    }

    // Bands NSA doesn't carry → None even when arrays are present.
    assert_eq!(e.profile_surface_brightness(Band::GaiaG), None);

    // ivar populated.
    let ivar_r = e.profile_surface_brightness_ivar(Band::SdssR).unwrap();
    assert_eq!(ivar_r.len(), N_PROFILE_RADII);
}

#[cfg(feature = "radial-profiles")]
#[test]
fn radial_profile_v0_1_2_remaps_5_bands_into_7_slot() {
    use starfield_nsa::{NsaVersion, N_PROFILE_RADII};

    let n_rows = 1;
    // v0_1_2 baseline: SERSIC_FLUX repeat=5, no underscore variant.
    let mut cols = vec![
        BinaryColumnDescriptor {
            name: Some(String::from("NSAID")),
            repeat: 1,
            col_type: BinaryColumnType::Int,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("RA")),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("DEC")),
            repeat: 1,
            col_type: BinaryColumnType::Double,
            byte_width: 8,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("Z")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_TH50")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_N")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_BA")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_PHI")),
            repeat: 1,
            col_type: BinaryColumnType::Float,
            byte_width: 4,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_FLUX")),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_FLUX_IVAR")),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("PROFTHETA")),
            repeat: N_PROFILE_RADII,
            col_type: BinaryColumnType::Float,
            byte_width: (N_PROFILE_RADII * 4),
        },
        BinaryColumnDescriptor {
            name: Some(String::from("PROFMEAN")),
            repeat: N_PROFILE_RADII * 5, // 5-band v0_1_2
            col_type: BinaryColumnType::Float,
            byte_width: (N_PROFILE_RADII * 5 * 4),
        },
    ];
    let _ = &mut cols; // silence unused-mut if a reviewer adds more

    // 5-band PROFMEAN: u, g, r, i, z. flat[r * 5 + b].
    let mut profmean: Vec<f32> = Vec::with_capacity(N_PROFILE_RADII * 5);
    for r in 0..N_PROFILE_RADII {
        for b in 0..5 {
            profmean.push((r * 100 + b) as f32);
        }
    }
    let radii: Vec<f32> = (0..N_PROFILE_RADII).map(|i| 0.1 * (i + 1) as f32).collect();

    let data = vec![
        BinaryColumnData::Int(vec![777]),
        BinaryColumnData::Double(vec![0.0]),
        BinaryColumnData::Double(vec![0.0]),
        BinaryColumnData::Float(vec![0.05]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![0.0]),
        BinaryColumnData::Float(vec![1.0; 5]),
        BinaryColumnData::Float(vec![1.0; 5]),
        BinaryColumnData::Float(radii),
        BinaryColumnData::Float(profmean),
    ];

    let tmp = write_fits_to_tempfile(&cols, &data, n_rows);
    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    assert_eq!(cat.version(), NsaVersion::V0_1_2);
    let e = cat.get_star(777).unwrap();

    // Bands not carried by v0_1_2: FUV/NUV — surface brightness is the
    // padded zero, but the trait still returns Some(slice). The renderer
    // should treat zero as "no signal" the same way it does for `flux_nmgy`.
    let sb_fuv = e.profile_surface_brightness(Band::GalexFuv).unwrap();
    assert!(sb_fuv.iter().all(|&x| x == 0.0));

    // u-band (v0_1_2 src idx 0 → in-memory idx 2): flat[r*5+0] = r*100
    let sb_u = e.profile_surface_brightness(Band::SdssU).unwrap();
    for (r, &got) in sb_u.iter().enumerate() {
        let want = (r * 100) as f64;
        assert!(
            (got - want).abs() < 1e-6,
            "u r={} got={} want={}",
            r,
            got,
            want
        );
    }
    // r-band (src idx 2 → mem idx 4): flat[r*5+2] = r*100+2
    let sb_r = e.profile_surface_brightness(Band::SdssR).unwrap();
    for (r, &got) in sb_r.iter().enumerate() {
        let want = (r * 100 + 2) as f64;
        assert!(
            (got - want).abs() < 1e-6,
            "r r={} got={} want={}",
            r,
            got,
            want
        );
    }
}
