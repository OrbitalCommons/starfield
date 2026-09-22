//! End-to-end parse test: write a synthetic DR3 ECSV fixture with two rows, load
//! it through the production `Dr3Catalog::from_csv_file` path, and check that the
//! nested sub-struct grouping (BP/RP, RV, GSP-Phot, etc.) surfaces the expected
//! values.
#![cfg(feature = "dr3")]

use std::collections::HashMap;
use std::io::Write;

use starfield_catalogs::gaia::{Dr3Catalog, GaiaSource, Release};
use starfield_core::catalogs::StarCatalog;

const HEADER: &str = "source_id,solution_id,ref_epoch,random_index,designation,ra,ra_error,dec,dec_error,ra_dec_corr,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error,l,b,ecl_lon,ecl_lat,phot_g_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_n_obs,phot_variable_flag,astrometric_n_obs_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_primary_flag,duplicated_source,matched_observations,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,astrometric_gof_al,astrometric_chi2_al,astrometric_params_solved,visibility_periods_used,astrometric_sigma5d_max,nu_eff_used_in_astrometry,pseudocolour,pseudocolour_error,ruwe,ipd_gof_harmonic_amplitude,ipd_gof_harmonic_phase,ipd_frac_multi_peak,ipd_frac_odd_win,phot_bp_mean_mag,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_n_obs,phot_rp_mean_mag,phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_n_obs,bp_rp,bp_g,g_rp,phot_bp_rp_excess_factor,phot_proc_mode,radial_velocity,radial_velocity_error,rv_method_used,rv_nb_transits,rv_expected_sig_to_noise,rv_amplitude_robust,rv_template_teff,rv_template_logg,rv_template_fe_h,rv_atm_param_origin,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,mh_gspphot,distance_gspphot,distance_gspphot_lower,distance_gspphot_upper,azero_gspphot,ag_gspphot,ebpminrp_gspphot,libname_gspphot,has_xp_continuous,has_xp_sampled,has_rvs,has_epoch_photometry,has_epoch_rv,has_mcmc_gspphot,has_mcmc_msc,in_qso_candidates,in_galaxy_candidates,in_andromeda_survey,non_single_star,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_star";

/// Build one row by keying field values off column names.
/// Any name not present in `overrides` gets the empty string (null for nullable columns).
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
    writeln!(f, "# %ECSV 1.0").unwrap();
    writeln!(f, "# ---").unwrap();
    writeln!(f, "# delimiter: ','").unwrap();
    writeln!(f, "# datatype:").unwrap();
    writeln!(f, "# - name: source_id").unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for row in rows {
        writeln!(f, "{}", row).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn dr3_parses_nested_substructs() {
    let bright = row_from(&HashMap::from([
        ("source_id", "1234567890123456789"),
        ("solution_id", "1635721458409799680"),
        ("ref_epoch", "2016.0"),
        ("designation", "Gaia DR3 1234567890123456789"),
        ("ra", "101.2874"),
        ("ra_error", "0.03"),
        ("dec", "-16.7161"),
        ("dec_error", "0.04"),
        ("parallax", "379.21"),
        ("parallax_error", "0.15"),
        ("pmra", "-546.05"),
        ("pmdec", "-1223.14"),
        ("l", "227.2"),
        ("b", "-8.89"),
        ("ecl_lon", "173.10"),
        ("ecl_lat", "-5.86"),
        ("phot_g_mean_mag", "-1.46"),
        ("phot_g_mean_flux", "16842868.0"),
        ("phot_variable_flag", "VARIABLE"),
        ("astrometric_primary_flag", "true"),
        ("duplicated_source", "false"),
        ("ruwe", "1.05"),
        ("phot_bp_mean_mag", "-1.43"),
        ("phot_bp_mean_flux", "11100000.0"),
        ("phot_rp_mean_mag", "-1.48"),
        ("phot_rp_mean_flux", "9800000.0"),
        ("bp_rp", "0.05"),
        ("radial_velocity", "-7.6"),
        ("radial_velocity_error", "0.8"),
        ("rv_nb_transits", "38"),
        ("rv_template_teff", "9940.0"),
        ("teff_gspphot", "9940.0"),
        ("teff_gspphot_lower", "9900.0"),
        ("teff_gspphot_upper", "9980.0"),
        ("libname_gspphot", "MARCS"),
        ("has_xp_continuous", "true"),
        ("has_rvs", "true"),
        ("classprob_dsc_combmod_star", "0.999"),
    ]));

    let faint = row_from(&HashMap::from([
        ("source_id", "9876543210987654321"),
        ("solution_id", "1635721458409799680"),
        ("ref_epoch", "2016.0"),
        ("ra", "45.0"),
        ("ra_error", "0.2"),
        ("dec", "0.1"),
        ("dec_error", "0.2"),
        ("l", "176.9"),
        ("b", "-48.9"),
        ("ecl_lon", "42.5"),
        ("ecl_lat", "-16.3"),
        ("phot_g_mean_mag", "19.8"),
        ("phot_variable_flag", "NOT_AVAILABLE"),
    ]));

    let fixture = write_fixture(&[bright, faint]);
    let catalog = Dr3Catalog::from_csv_file(fixture.path(), 20.0).expect("load fixture");

    assert_eq!(catalog.len(), 2, "both rows should parse");

    let bright = catalog
        .get_star(1234567890123456789)
        .expect("bright star by source_id");
    assert_eq!(bright.release(), Release::Dr3);
    assert!((bright.core.ra - 101.2874).abs() < 1e-9);
    assert!((bright.core.phot_g_mean_mag - (-1.46)).abs() < 1e-9);
    assert_eq!(bright.core.astrometric_primary_flag, Some(true));

    let bp_rp = bright.bp_rp.as_ref().expect("BP/RP block present");
    assert!((bp_rp.bp_rp.unwrap() - 0.05).abs() < 1e-6);
    assert!((bp_rp.phot_bp_mean_mag.unwrap() - (-1.43)).abs() < 1e-9);

    let rv = bright.radial_velocity.as_ref().expect("RV block present");
    assert!((rv.radial_velocity.unwrap() - (-7.6)).abs() < 1e-6);
    assert_eq!(rv.rv_nb_transits, Some(38));

    let gsp = bright.gspphot.as_ref().expect("GSP-Phot block present");
    assert!((gsp.teff_gspphot.unwrap() - 9940.0).abs() < 1e-3);
    assert_eq!(gsp.libname_gspphot.as_deref(), Some("MARCS"));

    assert_eq!(bright.data_links.has_rvs, Some(true));
    assert_eq!(bright.data_links.has_xp_continuous, Some(true));

    let faint = catalog
        .get_star(9876543210987654321)
        .expect("faint star by source_id");
    assert!(faint.bp_rp.is_none());
    assert!(faint.radial_velocity.is_none());
    assert!(faint.gspphot.is_none());
    assert!(faint.core.parallax.is_none());

    assert!(bright.b_v().is_some());
    assert!(faint.b_v().is_none());
}

#[test]
fn dr3_mag_limit_filters_rows() {
    let bright = row_from(&HashMap::from([
        ("source_id", "1234567890123456789"),
        ("solution_id", "1635721458409799680"),
        ("ref_epoch", "2016.0"),
        ("ra", "101.2874"),
        ("ra_error", "0.03"),
        ("dec", "-16.7161"),
        ("dec_error", "0.04"),
        ("l", "227.2"),
        ("b", "-8.89"),
        ("ecl_lon", "173.10"),
        ("ecl_lat", "-5.86"),
        ("phot_g_mean_mag", "-1.46"),
    ]));
    let faint = row_from(&HashMap::from([
        ("source_id", "9876543210987654321"),
        ("solution_id", "1635721458409799680"),
        ("ref_epoch", "2016.0"),
        ("ra", "45.0"),
        ("ra_error", "0.2"),
        ("dec", "0.1"),
        ("dec_error", "0.2"),
        ("l", "176.9"),
        ("b", "-48.9"),
        ("ecl_lon", "42.5"),
        ("ecl_lat", "-16.3"),
        ("phot_g_mean_mag", "19.8"),
    ]));

    let fixture = write_fixture(&[bright, faint]);
    let catalog = Dr3Catalog::from_csv_file(fixture.path(), 10.0).expect("load fixture");
    assert_eq!(catalog.len(), 1);
    assert!(catalog.get_star(1234567890123456789).is_some());
    assert!(catalog.get_star(9876543210987654321).is_none());
}

#[test]
fn dr3_missing_required_column_errors() {
    // Header intentionally lacks several columns declared by the schema.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "# %ECSV 1.0").unwrap();
    writeln!(f, "source_id,solution_id,ref_epoch,ra,dec").unwrap();
    writeln!(f, "1,2,2016.0,10.0,20.0").unwrap();
    f.flush().unwrap();
    let result = Dr3Catalog::from_csv_file(f.path(), 20.0);
    assert!(result.is_err(), "expected missing-column error");
    let msg = result.err().unwrap().to_string();
    assert!(
        msg.contains("missing column"),
        "error should flag missing column: {}",
        msg
    );
}
