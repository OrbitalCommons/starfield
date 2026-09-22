//! Synthetic FITS round-trip for [`NsaCatalog::from_fits_file`].
//!
//! Builds an NSA-shaped binary table HDU (10 columns matching the curated
//! [`NsaEntry`] surface area) using `fitsio_pure`'s serialization helpers,
//! writes it to a tempfile, loads it through the public API, and verifies
//! every field round-trips byte-for-byte.

use std::io::Write;

use fitsio_pure::bintable::{
    serialize_binary_table_hdu, BinaryColumnData, BinaryColumnDescriptor, BinaryColumnType,
};
use fitsio_pure::header::serialize_header;
use fitsio_pure::primary::build_primary_header;

use starfield_catalogs::nsa::NsaCatalog;

fn empty_primary_hdu() -> Vec<u8> {
    let cards = build_primary_header(8, &[]).unwrap();
    serialize_header(&cards)
}

#[test]
fn round_trip_minimal_nsa_table() {
    let n_rows: usize = 4;

    let columns = vec![
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

    let nsaid: Vec<i32> = vec![10, 20, 30, 40];
    let ra: Vec<f64> = vec![10.5, 175.25, 220.0, 359.875];
    let dec: Vec<f64> = vec![-30.5, 0.0, 12.125, 60.75];
    let z: Vec<f32> = vec![0.005, 0.020, 0.080, 0.150];
    let th50: Vec<f32> = vec![1.5, 3.25, 7.0, 12.5];
    let nser: Vec<f32> = vec![1.0, 2.0, 4.0, 6.0];
    let ba: Vec<f32> = vec![0.95, 0.65, 0.40, 0.10];
    let phi: Vec<f32> = vec![0.0, 45.0, 90.0, 135.0];

    // Per-galaxy 7-band fluxes; flatten to a single Vec<f32> in row-major order.
    let flux_per_row: Vec<[f32; 7]> = (0..n_rows)
        .map(|i| {
            let base = (i as f32) * 10.0;
            [
                base + 0.1,
                base + 0.2,
                base + 0.3,
                base + 0.4,
                base + 0.5,
                base + 0.6,
                base + 0.7,
            ]
        })
        .collect();
    let ivar_per_row: Vec<[f32; 7]> = (0..n_rows)
        .map(|i| {
            let base = 100.0 - (i as f32);
            [
                base + 1.0,
                base + 2.0,
                base + 3.0,
                base + 4.0,
                base + 5.0,
                base + 6.0,
                base + 7.0,
            ]
        })
        .collect();
    let flux_flat: Vec<f32> = flux_per_row.iter().flatten().copied().collect();
    let ivar_flat: Vec<f32> = ivar_per_row.iter().flatten().copied().collect();

    let col_data = vec![
        BinaryColumnData::Int(nsaid.clone()),
        BinaryColumnData::Double(ra.clone()),
        BinaryColumnData::Double(dec.clone()),
        BinaryColumnData::Float(z.clone()),
        BinaryColumnData::Float(th50.clone()),
        BinaryColumnData::Float(nser.clone()),
        BinaryColumnData::Float(ba.clone()),
        BinaryColumnData::Float(phi.clone()),
        BinaryColumnData::Float(flux_flat),
        BinaryColumnData::Float(ivar_flat),
    ];

    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, n_rows).unwrap();

    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    assert_eq!(cat.len(), n_rows);

    use starfield_core::catalogs::StarCatalog;
    for i in 0..n_rows {
        let id = nsaid[i] as u32;
        let e = cat
            .get_star(id as usize)
            .unwrap_or_else(|| panic!("missing NSAID={}", id));

        assert_eq!(e.nsaid, id);
        assert!((e.ra - ra[i]).abs() < 1e-12);
        assert!((e.dec - dec[i]).abs() < 1e-12);
        assert!((e.z - z[i]).abs() < 1e-7);
        assert!((e.sersic_th50 - th50[i]).abs() < 1e-6);
        assert!((e.sersic_n - nser[i]).abs() < 1e-6);
        assert!((e.sersic_ba - ba[i]).abs() < 1e-6);
        assert!((e.sersic_phi - phi[i]).abs() < 1e-5);
        for b in 0..7 {
            assert!(
                (e.sersic_flux[b] - flux_per_row[i][b]).abs() < 1e-5,
                "flux row {} band {}: got {} want {}",
                i,
                b,
                e.sersic_flux[b],
                flux_per_row[i][b]
            );
            assert!(
                (e.sersic_flux_ivar[b] - ivar_per_row[i][b]).abs() < 1e-4,
                "ivar row {} band {}: got {} want {}",
                i,
                b,
                e.sersic_flux_ivar[b],
                ivar_per_row[i][b]
            );
        }
    }
}

#[test]
fn ab_magnitude_handles_zero_flux() {
    let n_rows: usize = 1;
    let columns = vec![
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
    // r-band (index 4) flux is zero: ab_magnitude should return None.
    let flux: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let ivar: Vec<f32> = vec![1.0; 7];

    let col_data = vec![
        BinaryColumnData::Int(vec![42]),
        BinaryColumnData::Double(vec![100.0]),
        BinaryColumnData::Double(vec![10.0]),
        BinaryColumnData::Float(vec![0.05]),
        BinaryColumnData::Float(vec![2.0]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![0.7]),
        BinaryColumnData::Float(vec![30.0]),
        BinaryColumnData::Float(flux),
        BinaryColumnData::Float(ivar),
    ];

    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, n_rows).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    use starfield_core::catalogs::StarCatalog;
    let e = cat.get_star(42).unwrap();
    assert_eq!(
        e.sersic_ab_magnitude(4),
        None,
        "zero r-band flux must yield None"
    );
    assert!(e.sersic_ab_magnitude(0).is_some());

    let v = e.unit_vector();
    assert!((v.norm() - 1.0).abs() < 1e-12);
}

/// v0_1_2 (5-band) FITS files have SERSIC_FLUX repeat=5 with `u, g, r, i, z`
/// in indices 0..5. The loader must remap these into the canonical 7-slot
/// in-memory layout (FUV/NUV at 0/1 zeroed, ugriz at 2..7).
#[test]
fn round_trip_v0_1_2_five_band() {
    use starfield_catalogs::nsa::NsaVersion;

    let n_rows: usize = 2;
    let columns = vec![
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
    ];

    // u, g, r, i, z fluxes — distinct values per band so we can verify the
    // remap into in-memory indices 2..7.
    let flux_per_row: [[f32; 5]; 2] = [[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]];
    let ivar_per_row: [[f32; 5]; 2] = [[0.1, 0.2, 0.3, 0.4, 0.5], [1.1, 1.2, 1.3, 1.4, 1.5]];
    let flux_flat: Vec<f32> = flux_per_row.iter().flatten().copied().collect();
    let ivar_flat: Vec<f32> = ivar_per_row.iter().flatten().copied().collect();

    let col_data = vec![
        BinaryColumnData::Int(vec![100, 200]),
        BinaryColumnData::Double(vec![10.0, 20.0]),
        BinaryColumnData::Double(vec![-1.0, 30.0]),
        BinaryColumnData::Float(vec![0.01, 0.05]),
        BinaryColumnData::Float(vec![1.0, 2.0]),
        BinaryColumnData::Float(vec![1.0, 4.0]),
        BinaryColumnData::Float(vec![0.9, 0.5]),
        BinaryColumnData::Float(vec![10.0, 80.0]),
        BinaryColumnData::Float(flux_flat),
        BinaryColumnData::Float(ivar_flat),
    ];

    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, n_rows).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    assert_eq!(cat.version(), NsaVersion::V0_1_2);
    assert_eq!(cat.len(), n_rows);

    use starfield_core::catalogs::StarCatalog;
    for (row, id) in [(0usize, 100u32), (1, 200)] {
        let e = cat.get_star(id as usize).unwrap();
        // FUV/NUV slots must be zero (no GALEX in v0_1_2).
        assert_eq!(e.sersic_flux[0], 0.0, "FUV slot must be zero");
        assert_eq!(e.sersic_flux[1], 0.0, "NUV slot must be zero");
        assert_eq!(e.sersic_flux_ivar[0], 0.0);
        assert_eq!(e.sersic_flux_ivar[1], 0.0);
        // ugriz must be in indices 2..7.
        for src_band in 0..5 {
            let dst = 2 + src_band;
            assert!(
                (e.sersic_flux[dst] - flux_per_row[row][src_band]).abs() < 1e-6,
                "row {} band src={} dst={}: got {} want {}",
                row,
                src_band,
                dst,
                e.sersic_flux[dst],
                flux_per_row[row][src_band]
            );
            assert!((e.sersic_flux_ivar[dst] - ivar_per_row[row][src_band]).abs() < 1e-6);
        }
        // ab_magnitude(4) is the canonical r-band slot. For row 0, r-flux=3.0
        // → 22.5 - 2.5*log10(3) ≈ 21.30. ab_magnitude(0) (FUV) returns None
        // because the slot is zero.
        assert!(e.sersic_ab_magnitude(NsaVersion::R_BAND_IDX).is_some());
        assert_eq!(
            e.sersic_ab_magnitude(0),
            None,
            "FUV slot is zero, must be None"
        );
        assert_eq!(
            e.sersic_ab_magnitude(1),
            None,
            "NUV slot is zero, must be None"
        );
    }
}

/// A repeat-count that's neither 5 nor 7 must be a clear error.
#[test]
fn unknown_band_count_is_rejected() {
    let columns = vec![
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
            repeat: 6,
            col_type: BinaryColumnType::Float,
            byte_width: 24,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSIC_FLUX_IVAR")),
            repeat: 6,
            col_type: BinaryColumnType::Float,
            byte_width: 24,
        },
    ];

    let col_data = vec![
        BinaryColumnData::Int(vec![1]),
        BinaryColumnData::Double(vec![0.0]),
        BinaryColumnData::Double(vec![0.0]),
        BinaryColumnData::Float(vec![0.0]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![1.0]),
        BinaryColumnData::Float(vec![0.5]),
        BinaryColumnData::Float(vec![0.0]),
        BinaryColumnData::Float(vec![1.0; 6]),
        BinaryColumnData::Float(vec![1.0; 6]),
    ];

    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, 1).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let err = NsaCatalog::from_fits_file(tmp.path()).unwrap_err();
    let s = err.to_string();
    assert!(
        s.contains("repeat=6") || s.contains("neither 5 (v0_1_2) nor 7 (v1_0_1)"),
        "expected version-mismatch error, got: {}",
        s
    );
}

/// v0_1_2 in the wild uses `SERSICFLUX` / `SERSICFLUX_IVAR` (no underscore
/// between `SERSIC` and `FLUX`). v1_0_1 uses `SERSIC_FLUX` / `SERSIC_FLUX_IVAR`.
/// The loader must accept either spelling.
#[test]
fn round_trip_no_underscore_flux_column_name() {
    use starfield_catalogs::nsa::NsaVersion;

    let n_rows: usize = 1;
    let columns = vec![
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
            name: Some(String::from("SERSICFLUX")),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
        BinaryColumnDescriptor {
            name: Some(String::from("SERSICFLUX_IVAR")),
            repeat: 5,
            col_type: BinaryColumnType::Float,
            byte_width: 20,
        },
    ];

    // Distinct r-band flux so the AB mag check below is unambiguous.
    let flux: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // u, g, r, i, z
    let ivar: Vec<f32> = vec![0.5; 5];

    let col_data = vec![
        BinaryColumnData::Int(vec![777]),
        BinaryColumnData::Double(vec![123.0]),
        BinaryColumnData::Double(vec![45.0]),
        BinaryColumnData::Float(vec![0.04]),
        BinaryColumnData::Float(vec![1.5]),
        BinaryColumnData::Float(vec![2.5]),
        BinaryColumnData::Float(vec![0.7]),
        BinaryColumnData::Float(vec![60.0]),
        BinaryColumnData::Float(flux),
        BinaryColumnData::Float(ivar),
    ];

    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, n_rows).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let cat = NsaCatalog::from_fits_file(tmp.path()).unwrap();
    assert_eq!(cat.version(), NsaVersion::V0_1_2);
    use starfield_core::catalogs::StarCatalog;
    let e = cat.get_star(777).unwrap();
    // r-band src index 2 → in-memory index 4, value 3.0 nMgy → AB ≈ 21.30
    assert!((e.sersic_flux[NsaVersion::R_BAND_IDX] - 3.0).abs() < 1e-6);
    assert_eq!(e.sersic_flux[0], 0.0); // FUV padded
    assert_eq!(e.sersic_flux[1], 0.0); // NUV padded
}

/// Erroring out should clearly mention every name it tried, so an unfamiliar
/// release name doesn't produce a misleading "missing column SERSIC_FLUX" message.
#[test]
fn missing_flux_column_lists_all_candidates() {
    let columns = vec![
        BinaryColumnDescriptor {
            name: Some(String::from("NSAID")),
            repeat: 1,
            col_type: BinaryColumnType::Int,
            byte_width: 4,
        },
        // Intentionally drop both SERSIC_FLUX and SERSICFLUX.
    ];
    let col_data = vec![BinaryColumnData::Int(vec![1])];
    let bt_ext = serialize_binary_table_hdu(&columns, &col_data, 1).unwrap();
    let mut fits_bytes = empty_primary_hdu();
    fits_bytes.extend_from_slice(&bt_ext);

    let tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.as_file().write_all(&fits_bytes).unwrap();
    tmp.as_file().sync_all().unwrap();

    let err = NsaCatalog::from_fits_file(tmp.path()).unwrap_err();
    let s = err.to_string();
    assert!(s.contains("SERSIC_FLUX"), "want SERSIC_FLUX in: {}", s);
    assert!(s.contains("SERSICFLUX"), "want SERSICFLUX in: {}", s);
}
