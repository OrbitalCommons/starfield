//! `GaiaCore::proper_motion()` exercised end-to-end through the CSV parser.

#![cfg(feature = "dr3")]

use std::collections::HashMap;
use std::io::Write;

use starfield_catalogs::gaia::Dr3Catalog;
use starfield_core::ProperMotion;

const HEADER: &str = "source_id,solution_id,ref_epoch,random_index,designation,ra,ra_error,dec,dec_error,ra_dec_corr,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error,l,b,ecl_lon,ecl_lat,phot_g_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_n_obs,phot_variable_flag,astrometric_n_obs_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_primary_flag,duplicated_source,matched_observations,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,astrometric_gof_al,astrometric_chi2_al,astrometric_params_solved,visibility_periods_used,astrometric_sigma5d_max,nu_eff_used_in_astrometry,pseudocolour,pseudocolour_error,ruwe,ipd_gof_harmonic_amplitude,ipd_gof_harmonic_phase,ipd_frac_multi_peak,ipd_frac_odd_win,phot_bp_mean_mag,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_n_obs,phot_rp_mean_mag,phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_n_obs,bp_rp,bp_g,g_rp,phot_bp_rp_excess_factor,phot_proc_mode,radial_velocity,radial_velocity_error,rv_method_used,rv_nb_transits,rv_expected_sig_to_noise,rv_amplitude_robust,rv_template_teff,rv_template_logg,rv_template_fe_h,rv_atm_param_origin,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,mh_gspphot,distance_gspphot,distance_gspphot_lower,distance_gspphot_upper,azero_gspphot,ag_gspphot,ebpminrp_gspphot,libname_gspphot,has_xp_continuous,has_xp_sampled,has_rvs,has_epoch_photometry,has_epoch_rv,has_mcmc_gspphot,has_mcmc_msc,in_qso_candidates,in_galaxy_candidates,in_andromeda_survey,non_single_star,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_star";

fn row_from(overrides: &HashMap<&str, &str>) -> String {
    HEADER
        .split(',')
        .map(|c| overrides.get(c).copied().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(",")
}

fn write_fixture(rows: &[(u64, Option<f64>, Option<f64>)]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for (id, pmra, pmdec) in rows {
        let pmra_str = pmra.map(|v| v.to_string()).unwrap_or_default();
        let pmdec_str = pmdec.map(|v| v.to_string()).unwrap_or_default();
        let row = row_from(&HashMap::from([
            ("source_id", id.to_string().as_str()),
            ("solution_id", "1635721458409799680"),
            ("ref_epoch", "2016.0"),
            ("ra", "10.0"),
            ("ra_error", "0.04"),
            ("dec", "20.0"),
            ("dec_error", "0.04"),
            ("pmra", pmra_str.as_str()),
            ("pmdec", pmdec_str.as_str()),
            ("l", "0.0"),
            ("b", "0.0"),
            ("ecl_lon", "0.0"),
            ("ecl_lat", "0.0"),
            ("phot_g_mean_mag", "12.0"),
            ("phot_variable_flag", "NOT_AVAILABLE"),
        ]));
        writeln!(f, "{}", row).unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn proper_motion_returns_some_when_both_components_present() {
    let f = write_fixture(&[(1, Some(-4.6), Some(11.2))]);
    let cat = Dr3Catalog::from_csv_file(f.path(), f64::INFINITY).unwrap();
    let entry = cat.get_star(1).unwrap();
    let pm = entry
        .core
        .proper_motion()
        .expect("both components populated");
    assert_eq!(pm, ProperMotion::new(-4.6, 11.2));
    assert_eq!(pm.pmra, -4.6);
    assert_eq!(pm.pmdec, 11.2);
}

#[test]
fn proper_motion_returns_none_when_either_component_missing() {
    let f = write_fixture(&[
        (10, None, None),       // both missing → None
        (11, Some(1.5), None),  // pmdec missing → None
        (12, None, Some(-2.3)), // pmra missing → None
    ]);
    let cat = Dr3Catalog::from_csv_file(f.path(), f64::INFINITY).unwrap();
    for id in [10, 11, 12] {
        let pm = cat.get_star(id).unwrap().core.proper_motion();
        assert!(
            pm.is_none(),
            "source_id {} should have no proper_motion, got {:?}",
            id,
            pm
        );
    }
}

#[test]
fn proper_motion_zero_components_round_trip() {
    // pmra == 0 / pmdec == 0 is a valid (if unusual) Gaia value — not the
    // same as "missing". The accessor should return Some(ProperMotion::ZERO).
    let f = write_fixture(&[(99, Some(0.0), Some(0.0))]);
    let cat = Dr3Catalog::from_csv_file(f.path(), f64::INFINITY).unwrap();
    let pm = cat.get_star(99).unwrap().core.proper_motion().unwrap();
    assert_eq!(pm, ProperMotion::ZERO);
}

use starfield_core::catalogs::StarCatalog;
