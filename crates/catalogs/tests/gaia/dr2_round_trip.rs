//! DR2 smoke test — synthetic fixture exercising BP/RP, RV, and Apsis sub-structs.
#![cfg(feature = "dr2")]

use std::collections::HashMap;
use std::io::Write;

use starfield_catalogs::gaia::{Dr2Catalog, GaiaSource, Release};
use starfield_core::catalogs::StarCatalog;

const HEADER: &str = "source_id,solution_id,designation,ref_epoch,random_index,ra,ra_error,dec,dec_error,ra_dec_corr,parallax,parallax_error,parallax_over_error,pmra,pmra_error,pmdec,pmdec_error,l,b,ecl_lon,ecl_lat,phot_g_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_n_obs,phot_variable_flag,astrometric_n_obs_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_primary_flag,duplicated_source,matched_observations,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,astrometric_gof_al,astrometric_chi2_al,astrometric_params_solved,astrometric_weight_al,astrometric_pseudo_colour,astrometric_pseudo_colour_error,mean_varpi_factor_al,astrometric_matched_observations,visibility_periods_used,astrometric_sigma5d_max,frame_rotator_object_type,phot_bp_n_obs,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_mean_flux_over_error,phot_bp_mean_mag,phot_rp_n_obs,phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_mean_flux_over_error,phot_rp_mean_mag,phot_bp_rp_excess_factor,phot_proc_mode,bp_rp,bp_g,g_rp,radial_velocity,radial_velocity_error,rv_nb_transits,rv_template_teff,rv_template_logg,rv_template_fe_h,priam_flags,teff_val,teff_percentile_lower,teff_percentile_upper,a_g_val,a_g_percentile_lower,a_g_percentile_upper,e_bp_min_rp_val,e_bp_min_rp_percentile_lower,e_bp_min_rp_percentile_upper,flame_flags,radius_val,radius_percentile_lower,radius_percentile_upper,lum_val,lum_percentile_lower,lum_percentile_upper";

fn row_from(overrides: &HashMap<&str, &str>) -> String {
    HEADER
        .split(',')
        .map(|col| overrides.get(col).copied().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(",")
}

fn write_fixture(rows: &[String]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .expect("temp file");
    // DR2 is plain CSV, no ECSV preamble.
    writeln!(f, "{}", HEADER).unwrap();
    for row in rows {
        writeln!(f, "{}", row).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn dr2_parses_bp_rp_rv_and_apsis() {
    let row = row_from(&HashMap::from([
        ("source_id", "1000225938242805248"),
        ("solution_id", "1635721458409799680"),
        ("designation", "Gaia DR2 1000225938242805248"),
        ("ref_epoch", "2015.5"),
        ("ra", "103.447"),
        ("ra_error", "0.04"),
        ("dec", "56.022"),
        ("dec_error", "0.04"),
        ("parallax", "0.58"),
        ("parallax_error", "0.07"),
        ("pmra", "6.04"),
        ("pmdec", "5.05"),
        ("l", "160.16"),
        ("b", "22.53"),
        ("ecl_lon", "98.91"),
        ("ecl_lat", "32.99"),
        ("phot_g_mean_mag", "15.77"),
        ("phot_variable_flag", "NOT_AVAILABLE"),
        ("phot_bp_mean_mag", "16.11"),
        ("phot_rp_mean_mag", "15.27"),
        ("bp_rp", "0.836"),
        ("radial_velocity", "12.5"),
        ("radial_velocity_error", "0.3"),
        ("rv_nb_transits", "7"),
        ("rv_template_teff", "5750.0"),
        ("teff_val", "5800.0"),
        ("teff_percentile_lower", "5700.0"),
        ("teff_percentile_upper", "5900.0"),
        ("radius_val", "1.1"),
        ("lum_val", "1.2"),
    ]));
    let fixture = write_fixture(&[row]);
    let catalog = Dr2Catalog::from_csv_file(fixture.path(), 20.0).expect("load fixture");

    assert_eq!(catalog.len(), 1);
    let e = catalog.get_star(1000225938242805248).expect("by source_id");
    assert_eq!(e.release(), Release::Dr2);
    assert_eq!(
        e.designation.as_deref(),
        Some("Gaia DR2 1000225938242805248")
    );

    let bp = e.bp_rp.as_ref().expect("BP/RP present");
    assert!((bp.bp_rp.unwrap() - 0.836).abs() < 1e-5);

    let rv = e.radial_velocity.as_ref().expect("RV present");
    assert!((rv.radial_velocity.unwrap() - 12.5).abs() < 1e-9);

    let ap = e.astrophysical.as_ref().expect("Apsis present");
    assert!((ap.teff_val.unwrap() - 5800.0).abs() < 1e-3);
    assert!((ap.radius_val.unwrap() - 1.1).abs() < 1e-5);
    assert!((ap.lum_val.unwrap() - 1.2).abs() < 1e-5);

    assert!(e.b_v().is_some());
}
