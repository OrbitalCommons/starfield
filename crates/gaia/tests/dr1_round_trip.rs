//! DR1 smoke test — verifies the gaia_source schema loads, DR1-specific sub-structs
//! (scan direction, astrometric extras) surface, and the TGAS cross-id splice path works.
#![cfg(feature = "dr1")]

use std::collections::HashMap;
use std::io::Write;

use starfield_core::catalogs::StarCatalog;
use starfield_gaia::dr1::load_tgas_block_map;
use starfield_gaia::{Dr1Catalog, GaiaSource, Release, TgasBlock};

const HEADER: &str = "source_id,solution_id,ref_epoch,random_index,ra,ra_error,dec,dec_error,ra_dec_corr,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,l,b,ecl_lon,ecl_lat,phot_g_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_n_obs,phot_variable_flag,astrometric_n_obs_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_primary_flag,duplicated_source,matched_observations,astrometric_n_obs_ac,astrometric_n_good_obs_al,astrometric_n_good_obs_ac,astrometric_n_bad_obs_al,astrometric_n_bad_obs_ac,astrometric_delta_q,astrometric_relegation_factor,astrometric_weight_al,astrometric_weight_ac,astrometric_priors_used,scan_direction_strength_k1,scan_direction_strength_k2,scan_direction_strength_k3,scan_direction_strength_k4,scan_direction_mean_k1,scan_direction_mean_k2,scan_direction_mean_k3,scan_direction_mean_k4";

fn row_from(overrides: &HashMap<&str, &str>) -> String {
    HEADER
        .split(',')
        .map(|col| overrides.get(col).copied().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(",")
}

fn write_fixture(rows: &[String]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for row in rows {
        writeln!(f, "{}", row).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn dr1_parses_core_and_scan_direction() {
    let row = row_from(&HashMap::from([
        ("source_id", "7627862074752"),
        ("solution_id", "1635378410781933568"),
        ("ref_epoch", "2015.0"),
        ("ra", "45.03"),
        ("ra_error", "0.31"),
        ("dec", "0.235"),
        ("dec_error", "0.22"),
        ("parallax", "6.35"),
        ("parallax_error", "0.31"),
        ("pmra", "43.75"),
        ("pmdec", "-7.64"),
        ("l", "176.74"),
        ("b", "-48.71"),
        ("ecl_lon", "42.64"),
        ("ecl_lat", "-16.12"),
        ("phot_g_mean_mag", "7.99"),
        ("phot_variable_flag", "NOT_AVAILABLE"),
        ("astrometric_primary_flag", "true"),
        ("scan_direction_strength_k1", "0.38"),
        ("scan_direction_strength_k2", "0.54"),
        ("scan_direction_mean_k1", "-113.76"),
        ("scan_direction_mean_k2", "21.39"),
    ]));
    let fixture = write_fixture(&[row]);
    let catalog = Dr1Catalog::from_csv_file(fixture.path(), 20.0).expect("load fixture");

    assert_eq!(catalog.len(), 1);
    let e = catalog.get_star(7627862074752).expect("by source_id");
    assert_eq!(e.release(), Release::Dr1);
    assert!((e.core.parallax.unwrap() - 6.35).abs() < 1e-6);
    assert!((e.scan_direction.strength_k1.unwrap() - 0.38).abs() < 1e-5);
    assert!((e.scan_direction.mean_k1.unwrap() - (-113.76)).abs() < 1e-4);
    assert!(e.tgas.is_none(), "tgas only populated via attach_tgas");
}

#[test]
fn dr1_tgas_cross_ids_splice_in() {
    // First load a normal DR1 gaia_source row.
    let source_id = "7627862074752";
    let row = row_from(&HashMap::from([
        ("source_id", source_id),
        ("solution_id", "1635378410781933568"),
        ("ref_epoch", "2015.0"),
        ("ra", "45.03"),
        ("ra_error", "0.31"),
        ("dec", "0.235"),
        ("dec_error", "0.22"),
        ("l", "176.74"),
        ("b", "-48.71"),
        ("ecl_lon", "42.64"),
        ("ecl_lat", "-16.12"),
        ("phot_g_mean_mag", "7.99"),
    ]));
    let fixture = write_fixture(&[row]);
    let mut catalog = Dr1Catalog::from_csv_file(fixture.path(), 20.0).unwrap();

    // Now write a tiny TGAS CSV with hip/tycho2_id for that source_id.
    let mut tgas_file = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(tgas_file, "hip,tycho2_id,source_id").unwrap();
    writeln!(tgas_file, "13989,1001-55-1,{}", source_id).unwrap();
    tgas_file.flush().unwrap();
    let map = load_tgas_block_map(tgas_file.path()).unwrap();
    assert_eq!(map.len(), 1);
    let block: &TgasBlock = map.get(&7627862074752).unwrap();
    assert_eq!(block.hip, Some(13989));
    assert_eq!(block.tycho2_id.as_deref(), Some("1001-55-1"));

    catalog.attach_tgas(&map);
    let e = catalog.get_star(7627862074752).unwrap();
    let tgas = e.tgas.as_ref().expect("tgas attached");
    assert_eq!(tgas.hip, Some(13989));
    assert_eq!(tgas.tycho2_id.as_deref(), Some("1001-55-1"));
}
