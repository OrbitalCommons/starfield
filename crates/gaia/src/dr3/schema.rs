//! DR3 Arrow schema and row constructor.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use starfield_core::Result;

use crate::common::core::{GaiaCore, VarFlag};
use crate::common::format::*;
use crate::common::parse::*;
use crate::common::traits::{GaiaRelease, Release};
use crate::dr3::entry::{
    AstrometricExtra, BpRpPhotometry, Classifications, DataLinks, Dr3Entry, GspPhot, IpdQuality,
    RadialVelocityDr3,
};

/// Zero-sized release marker for Gaia DR3. Used to parameterize the generic reader,
/// downloader, and catalog types.
#[derive(Debug, Clone, Copy)]
pub struct Dr3;

impl GaiaRelease for Dr3 {
    const RELEASE: Release = Release::Dr3;
    const BASE_URL: &'static str = "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/";
    const MD5_FILENAME: &'static str = "_MD5SUM.txt";
    const FILE_REGEX: &'static str = r#"(GaiaSource_\d+-\d+\.csv\.gz)"#;
    const CACHE_SUBDIR: &'static str = "gaia/dr3";
    const TAP_TABLE: &'static str = "gaiadr3.gaia_source";
    const IS_ECSV: bool = true;

    type Entry = Dr3Entry;

    fn arrow_schema() -> SchemaRef {
        Arc::new(Schema::new(
            COLUMNS.iter().map(|c| c.field()).collect::<Vec<_>>(),
        ))
    }

    fn build_entry(batch: &RecordBatch, row: usize) -> Result<Self::Entry> {
        let c = ColIdx::new();

        let core = GaiaCore {
            source_id: req_u64(batch, c.source_id, row)?,
            solution_id: req_u64(batch, c.solution_id, row)?,
            ref_epoch: req_f64(batch, c.ref_epoch, row)?,
            random_index: opt_u64(batch, c.random_index, row)?,
            ra: req_f64(batch, c.ra, row)?,
            ra_error: req_f32(batch, c.ra_error, row)?,
            dec: req_f64(batch, c.dec, row)?,
            dec_error: req_f32(batch, c.dec_error, row)?,
            ra_dec_corr: opt_f32(batch, c.ra_dec_corr, row)?,
            parallax: opt_f64(batch, c.parallax, row)?,
            parallax_error: opt_f32(batch, c.parallax_error, row)?,
            pmra: opt_f64(batch, c.pmra, row)?,
            pmra_error: opt_f32(batch, c.pmra_error, row)?,
            pmdec: opt_f64(batch, c.pmdec, row)?,
            pmdec_error: opt_f32(batch, c.pmdec_error, row)?,
            l: req_f64(batch, c.l, row)?,
            b: req_f64(batch, c.b, row)?,
            ecl_lon: req_f64(batch, c.ecl_lon, row)?,
            ecl_lat: req_f64(batch, c.ecl_lat, row)?,
            phot_g_mean_mag: opt_f64(batch, c.phot_g_mean_mag, row)?.unwrap_or(f64::INFINITY),
            phot_g_mean_flux: opt_f64(batch, c.phot_g_mean_flux, row)?,
            phot_g_mean_flux_error: opt_f64(batch, c.phot_g_mean_flux_error, row)?,
            phot_g_n_obs: opt_u32(batch, c.phot_g_n_obs, row)?,
            phot_variable_flag: opt_str(batch, c.phot_variable_flag, row)?
                .map(VarFlag::parse)
                .unwrap_or_default(),
            astrometric_n_obs_al: opt_u32(batch, c.astrometric_n_obs_al, row)?,
            astrometric_excess_noise: opt_f64(batch, c.astrometric_excess_noise, row)?,
            astrometric_excess_noise_sig: opt_f64(batch, c.astrometric_excess_noise_sig, row)?,
            astrometric_primary_flag: opt_bool(batch, c.astrometric_primary_flag, row)?,
            duplicated_source: opt_bool(batch, c.duplicated_source, row)?,
            matched_observations: None, // DR3 dropped this column; use astrometric_matched_transits if needed
        };

        let astrometric_extra = AstrometricExtra {
            astrometric_n_good_obs_al: opt_u32(batch, c.astrometric_n_good_obs_al, row)?,
            astrometric_n_bad_obs_al: opt_u32(batch, c.astrometric_n_bad_obs_al, row)?,
            astrometric_gof_al: opt_f32(batch, c.astrometric_gof_al, row)?,
            astrometric_chi2_al: opt_f32(batch, c.astrometric_chi2_al, row)?,
            astrometric_params_solved: opt_u32(batch, c.astrometric_params_solved, row)?,
            visibility_periods_used: opt_u32(batch, c.visibility_periods_used, row)?,
            astrometric_sigma5d_max: opt_f32(batch, c.astrometric_sigma5d_max, row)?,
            nu_eff_used_in_astrometry: opt_f32(batch, c.nu_eff_used_in_astrometry, row)?,
            pseudocolour: opt_f32(batch, c.pseudocolour, row)?,
            pseudocolour_error: opt_f32(batch, c.pseudocolour_error, row)?,
        };

        let ipd = IpdQuality {
            ruwe: opt_f32(batch, c.ruwe, row)?,
            ipd_gof_harmonic_amplitude: opt_f32(batch, c.ipd_gof_harmonic_amplitude, row)?,
            ipd_gof_harmonic_phase: opt_f32(batch, c.ipd_gof_harmonic_phase, row)?,
            ipd_frac_multi_peak: opt_u32(batch, c.ipd_frac_multi_peak, row)?,
            ipd_frac_odd_win: opt_u32(batch, c.ipd_frac_odd_win, row)?,
        };

        let bp_mag = opt_f64(batch, c.phot_bp_mean_mag, row)?;
        let rp_mag = opt_f64(batch, c.phot_rp_mean_mag, row)?;
        let bp_rp = if bp_mag.is_some() || rp_mag.is_some() {
            Some(BpRpPhotometry {
                phot_bp_mean_mag: bp_mag,
                phot_bp_mean_flux: opt_f64(batch, c.phot_bp_mean_flux, row)?,
                phot_bp_mean_flux_error: opt_f64(batch, c.phot_bp_mean_flux_error, row)?,
                phot_bp_n_obs: opt_u32(batch, c.phot_bp_n_obs, row)?,
                phot_rp_mean_mag: rp_mag,
                phot_rp_mean_flux: opt_f64(batch, c.phot_rp_mean_flux, row)?,
                phot_rp_mean_flux_error: opt_f64(batch, c.phot_rp_mean_flux_error, row)?,
                phot_rp_n_obs: opt_u32(batch, c.phot_rp_n_obs, row)?,
                bp_rp: opt_f32(batch, c.bp_rp, row)?,
                bp_g: opt_f32(batch, c.bp_g, row)?,
                g_rp: opt_f32(batch, c.g_rp, row)?,
                phot_bp_rp_excess_factor: opt_f32(batch, c.phot_bp_rp_excess_factor, row)?,
                phot_proc_mode: opt_u32(batch, c.phot_proc_mode, row)?,
            })
        } else {
            None
        };

        let rv_value = opt_f64(batch, c.radial_velocity, row)?;
        let radial_velocity = if rv_value.is_some() {
            Some(RadialVelocityDr3 {
                radial_velocity: rv_value,
                radial_velocity_error: opt_f32(batch, c.radial_velocity_error, row)?,
                rv_method_used: opt_u32(batch, c.rv_method_used, row)?,
                rv_nb_transits: opt_u32(batch, c.rv_nb_transits, row)?,
                rv_expected_sig_to_noise: opt_f32(batch, c.rv_expected_sig_to_noise, row)?,
                rv_amplitude_robust: opt_f32(batch, c.rv_amplitude_robust, row)?,
                rv_template_teff: opt_f32(batch, c.rv_template_teff, row)?,
                rv_template_logg: opt_f32(batch, c.rv_template_logg, row)?,
                rv_template_fe_h: opt_f32(batch, c.rv_template_fe_h, row)?,
                rv_atm_param_origin: opt_u32(batch, c.rv_atm_param_origin, row)?,
            })
        } else {
            None
        };

        let teff = opt_f32(batch, c.teff_gspphot, row)?;
        let gspphot = if teff.is_some() {
            Some(GspPhot {
                teff_gspphot: teff,
                teff_gspphot_lower: opt_f32(batch, c.teff_gspphot_lower, row)?,
                teff_gspphot_upper: opt_f32(batch, c.teff_gspphot_upper, row)?,
                logg_gspphot: opt_f32(batch, c.logg_gspphot, row)?,
                mh_gspphot: opt_f32(batch, c.mh_gspphot, row)?,
                distance_gspphot: opt_f32(batch, c.distance_gspphot, row)?,
                distance_gspphot_lower: opt_f32(batch, c.distance_gspphot_lower, row)?,
                distance_gspphot_upper: opt_f32(batch, c.distance_gspphot_upper, row)?,
                azero_gspphot: opt_f32(batch, c.azero_gspphot, row)?,
                ag_gspphot: opt_f32(batch, c.ag_gspphot, row)?,
                ebpminrp_gspphot: opt_f32(batch, c.ebpminrp_gspphot, row)?,
                libname_gspphot: opt_str(batch, c.libname_gspphot, row)?.map(String::from),
            })
        } else {
            None
        };

        let data_links = DataLinks {
            has_xp_continuous: opt_bool(batch, c.has_xp_continuous, row)?,
            has_xp_sampled: opt_bool(batch, c.has_xp_sampled, row)?,
            has_rvs: opt_bool(batch, c.has_rvs, row)?,
            has_epoch_photometry: opt_bool(batch, c.has_epoch_photometry, row)?,
            has_epoch_rv: opt_bool(batch, c.has_epoch_rv, row)?,
            has_mcmc_gspphot: opt_bool(batch, c.has_mcmc_gspphot, row)?,
            has_mcmc_msc: opt_bool(batch, c.has_mcmc_msc, row)?,
        };

        let classifications = Classifications {
            in_qso_candidates: opt_bool(batch, c.in_qso_candidates, row)?,
            in_galaxy_candidates: opt_bool(batch, c.in_galaxy_candidates, row)?,
            in_andromeda_survey: opt_bool(batch, c.in_andromeda_survey, row)?,
            non_single_star: opt_u32(batch, c.non_single_star, row)?,
            classprob_dsc_combmod_quasar: opt_f32(batch, c.classprob_dsc_combmod_quasar, row)?,
            classprob_dsc_combmod_galaxy: opt_f32(batch, c.classprob_dsc_combmod_galaxy, row)?,
            classprob_dsc_combmod_star: opt_f32(batch, c.classprob_dsc_combmod_star, row)?,
        };

        Ok(Dr3Entry {
            core,
            designation: opt_str(batch, c.designation, row)?.map(String::from),
            pm: opt_f32(batch, c.pm, row)?,
            parallax_over_error: opt_f32(batch, c.parallax_over_error, row)?,
            astrometric_extra,
            ipd,
            bp_rp,
            radial_velocity,
            gspphot,
            data_links,
            classifications,
        })
    }

    fn format_csv_row(e: &Self::Entry) -> String {
        let c = &e.core;
        let ax = &e.astrometric_extra;
        let ip = &e.ipd;
        let bp = e.bp_rp.as_ref();
        let rv = e.radial_velocity.as_ref();
        let gsp = e.gspphot.as_ref();
        let dl = &e.data_links;
        let cl = &e.classifications;
        // Order MUST match COLUMNS below.
        [
            c.source_id.to_string(),
            c.solution_id.to_string(),
            c.ref_epoch.to_string(),
            fopt(c.random_index),
            fopt_str(e.designation.as_deref()).to_string(),
            c.ra.to_string(),
            c.ra_error.to_string(),
            c.dec.to_string(),
            c.dec_error.to_string(),
            fopt(c.ra_dec_corr),
            fopt(c.parallax),
            fopt(c.parallax_error),
            fopt(e.parallax_over_error),
            fopt(e.pm),
            fopt(c.pmra),
            fopt(c.pmra_error),
            fopt(c.pmdec),
            fopt(c.pmdec_error),
            c.l.to_string(),
            c.b.to_string(),
            c.ecl_lon.to_string(),
            c.ecl_lat.to_string(),
            c.phot_g_mean_mag.to_string(),
            fopt(c.phot_g_mean_flux),
            fopt(c.phot_g_mean_flux_error),
            fopt(c.phot_g_n_obs),
            fvar(c.phot_variable_flag).to_string(),
            fopt(c.astrometric_n_obs_al),
            fopt(c.astrometric_excess_noise),
            fopt(c.astrometric_excess_noise_sig),
            fopt_bool(c.astrometric_primary_flag).to_string(),
            fopt_bool(c.duplicated_source).to_string(),
            fopt(ax.astrometric_n_good_obs_al),
            fopt(ax.astrometric_n_bad_obs_al),
            fopt(ax.astrometric_gof_al),
            fopt(ax.astrometric_chi2_al),
            fopt(ax.astrometric_params_solved),
            fopt(ax.visibility_periods_used),
            fopt(ax.astrometric_sigma5d_max),
            fopt(ax.nu_eff_used_in_astrometry),
            fopt(ax.pseudocolour),
            fopt(ax.pseudocolour_error),
            fopt(ip.ruwe),
            fopt(ip.ipd_gof_harmonic_amplitude),
            fopt(ip.ipd_gof_harmonic_phase),
            fopt(ip.ipd_frac_multi_peak),
            fopt(ip.ipd_frac_odd_win),
            fopt(bp.and_then(|b| b.phot_bp_mean_mag)),
            fopt(bp.and_then(|b| b.phot_bp_mean_flux)),
            fopt(bp.and_then(|b| b.phot_bp_mean_flux_error)),
            fopt(bp.and_then(|b| b.phot_bp_n_obs)),
            fopt(bp.and_then(|b| b.phot_rp_mean_mag)),
            fopt(bp.and_then(|b| b.phot_rp_mean_flux)),
            fopt(bp.and_then(|b| b.phot_rp_mean_flux_error)),
            fopt(bp.and_then(|b| b.phot_rp_n_obs)),
            fopt(bp.and_then(|b| b.bp_rp)),
            fopt(bp.and_then(|b| b.bp_g)),
            fopt(bp.and_then(|b| b.g_rp)),
            fopt(bp.and_then(|b| b.phot_bp_rp_excess_factor)),
            fopt(bp.and_then(|b| b.phot_proc_mode)),
            fopt(rv.and_then(|r| r.radial_velocity)),
            fopt(rv.and_then(|r| r.radial_velocity_error)),
            fopt(rv.and_then(|r| r.rv_method_used)),
            fopt(rv.and_then(|r| r.rv_nb_transits)),
            fopt(rv.and_then(|r| r.rv_expected_sig_to_noise)),
            fopt(rv.and_then(|r| r.rv_amplitude_robust)),
            fopt(rv.and_then(|r| r.rv_template_teff)),
            fopt(rv.and_then(|r| r.rv_template_logg)),
            fopt(rv.and_then(|r| r.rv_template_fe_h)),
            fopt(rv.and_then(|r| r.rv_atm_param_origin)),
            fopt(gsp.and_then(|g| g.teff_gspphot)),
            fopt(gsp.and_then(|g| g.teff_gspphot_lower)),
            fopt(gsp.and_then(|g| g.teff_gspphot_upper)),
            fopt(gsp.and_then(|g| g.logg_gspphot)),
            fopt(gsp.and_then(|g| g.mh_gspphot)),
            fopt(gsp.and_then(|g| g.distance_gspphot)),
            fopt(gsp.and_then(|g| g.distance_gspphot_lower)),
            fopt(gsp.and_then(|g| g.distance_gspphot_upper)),
            fopt(gsp.and_then(|g| g.azero_gspphot)),
            fopt(gsp.and_then(|g| g.ag_gspphot)),
            fopt(gsp.and_then(|g| g.ebpminrp_gspphot)),
            fopt_str(gsp.and_then(|g| g.libname_gspphot.as_deref())).to_string(),
            fopt_bool(dl.has_xp_continuous).to_string(),
            fopt_bool(dl.has_xp_sampled).to_string(),
            fopt_bool(dl.has_rvs).to_string(),
            fopt_bool(dl.has_epoch_photometry).to_string(),
            fopt_bool(dl.has_epoch_rv).to_string(),
            fopt_bool(dl.has_mcmc_gspphot).to_string(),
            fopt_bool(dl.has_mcmc_msc).to_string(),
            fopt_bool(cl.in_qso_candidates).to_string(),
            fopt_bool(cl.in_galaxy_candidates).to_string(),
            fopt_bool(cl.in_andromeda_survey).to_string(),
            fopt(cl.non_single_star),
            fopt(cl.classprob_dsc_combmod_quasar),
            fopt(cl.classprob_dsc_combmod_galaxy),
            fopt(cl.classprob_dsc_combmod_star),
        ]
        .join(",")
    }
}

// ---- Column table ---------------------------------------------------------

struct ColSpec {
    name: &'static str,
    ty: DataType,
    nullable: bool,
}

impl ColSpec {
    const fn req(name: &'static str, ty: DataType) -> Self {
        Self {
            name,
            ty,
            nullable: false,
        }
    }
    const fn opt(name: &'static str, ty: DataType) -> Self {
        Self {
            name,
            ty,
            nullable: true,
        }
    }
    fn field(&self) -> Field {
        Field::new(self.name, self.ty.clone(), self.nullable)
    }
}

// Column order defines the schema; indices below must match.
static COLUMNS: &[ColSpec] = &[
    ColSpec::req("source_id", DataType::UInt64),
    ColSpec::req("solution_id", DataType::UInt64),
    ColSpec::req("ref_epoch", DataType::Float64),
    ColSpec::opt("random_index", DataType::Int64),
    ColSpec::opt("designation", DataType::Utf8),
    ColSpec::req("ra", DataType::Float64),
    ColSpec::req("ra_error", DataType::Float32),
    ColSpec::req("dec", DataType::Float64),
    ColSpec::req("dec_error", DataType::Float32),
    ColSpec::opt("ra_dec_corr", DataType::Float32),
    ColSpec::opt("parallax", DataType::Float64),
    ColSpec::opt("parallax_error", DataType::Float32),
    ColSpec::opt("parallax_over_error", DataType::Float32),
    ColSpec::opt("pm", DataType::Float32),
    ColSpec::opt("pmra", DataType::Float64),
    ColSpec::opt("pmra_error", DataType::Float32),
    ColSpec::opt("pmdec", DataType::Float64),
    ColSpec::opt("pmdec_error", DataType::Float32),
    ColSpec::req("l", DataType::Float64),
    ColSpec::req("b", DataType::Float64),
    ColSpec::req("ecl_lon", DataType::Float64),
    ColSpec::req("ecl_lat", DataType::Float64),
    ColSpec::opt("phot_g_mean_mag", DataType::Float64),
    ColSpec::opt("phot_g_mean_flux", DataType::Float64),
    ColSpec::opt("phot_g_mean_flux_error", DataType::Float64),
    ColSpec::opt("phot_g_n_obs", DataType::Int32),
    ColSpec::opt("phot_variable_flag", DataType::Utf8),
    ColSpec::opt("astrometric_n_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_excess_noise", DataType::Float64),
    ColSpec::opt("astrometric_excess_noise_sig", DataType::Float64),
    ColSpec::opt("astrometric_primary_flag", DataType::Boolean),
    ColSpec::opt("duplicated_source", DataType::Boolean),
    ColSpec::opt("astrometric_n_good_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_n_bad_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_gof_al", DataType::Float32),
    ColSpec::opt("astrometric_chi2_al", DataType::Float32),
    ColSpec::opt("astrometric_params_solved", DataType::Int32),
    ColSpec::opt("visibility_periods_used", DataType::Int32),
    ColSpec::opt("astrometric_sigma5d_max", DataType::Float32),
    ColSpec::opt("nu_eff_used_in_astrometry", DataType::Float32),
    ColSpec::opt("pseudocolour", DataType::Float32),
    ColSpec::opt("pseudocolour_error", DataType::Float32),
    ColSpec::opt("ruwe", DataType::Float32),
    ColSpec::opt("ipd_gof_harmonic_amplitude", DataType::Float32),
    ColSpec::opt("ipd_gof_harmonic_phase", DataType::Float32),
    ColSpec::opt("ipd_frac_multi_peak", DataType::Int32),
    ColSpec::opt("ipd_frac_odd_win", DataType::Int32),
    ColSpec::opt("phot_bp_mean_mag", DataType::Float64),
    ColSpec::opt("phot_bp_mean_flux", DataType::Float64),
    ColSpec::opt("phot_bp_mean_flux_error", DataType::Float64),
    ColSpec::opt("phot_bp_n_obs", DataType::Int32),
    ColSpec::opt("phot_rp_mean_mag", DataType::Float64),
    ColSpec::opt("phot_rp_mean_flux", DataType::Float64),
    ColSpec::opt("phot_rp_mean_flux_error", DataType::Float64),
    ColSpec::opt("phot_rp_n_obs", DataType::Int32),
    ColSpec::opt("bp_rp", DataType::Float32),
    ColSpec::opt("bp_g", DataType::Float32),
    ColSpec::opt("g_rp", DataType::Float32),
    ColSpec::opt("phot_bp_rp_excess_factor", DataType::Float32),
    ColSpec::opt("phot_proc_mode", DataType::Int32),
    ColSpec::opt("radial_velocity", DataType::Float64),
    ColSpec::opt("radial_velocity_error", DataType::Float32),
    ColSpec::opt("rv_method_used", DataType::Int32),
    ColSpec::opt("rv_nb_transits", DataType::Int32),
    ColSpec::opt("rv_expected_sig_to_noise", DataType::Float32),
    ColSpec::opt("rv_amplitude_robust", DataType::Float32),
    ColSpec::opt("rv_template_teff", DataType::Float32),
    ColSpec::opt("rv_template_logg", DataType::Float32),
    ColSpec::opt("rv_template_fe_h", DataType::Float32),
    ColSpec::opt("rv_atm_param_origin", DataType::Int32),
    ColSpec::opt("teff_gspphot", DataType::Float32),
    ColSpec::opt("teff_gspphot_lower", DataType::Float32),
    ColSpec::opt("teff_gspphot_upper", DataType::Float32),
    ColSpec::opt("logg_gspphot", DataType::Float32),
    ColSpec::opt("mh_gspphot", DataType::Float32),
    ColSpec::opt("distance_gspphot", DataType::Float32),
    ColSpec::opt("distance_gspphot_lower", DataType::Float32),
    ColSpec::opt("distance_gspphot_upper", DataType::Float32),
    ColSpec::opt("azero_gspphot", DataType::Float32),
    ColSpec::opt("ag_gspphot", DataType::Float32),
    ColSpec::opt("ebpminrp_gspphot", DataType::Float32),
    ColSpec::opt("libname_gspphot", DataType::Utf8),
    ColSpec::opt("has_xp_continuous", DataType::Boolean),
    ColSpec::opt("has_xp_sampled", DataType::Boolean),
    ColSpec::opt("has_rvs", DataType::Boolean),
    ColSpec::opt("has_epoch_photometry", DataType::Boolean),
    ColSpec::opt("has_epoch_rv", DataType::Boolean),
    ColSpec::opt("has_mcmc_gspphot", DataType::Boolean),
    ColSpec::opt("has_mcmc_msc", DataType::Boolean),
    ColSpec::opt("in_qso_candidates", DataType::Boolean),
    ColSpec::opt("in_galaxy_candidates", DataType::Boolean),
    ColSpec::opt("in_andromeda_survey", DataType::Boolean),
    ColSpec::opt("non_single_star", DataType::Int32),
    ColSpec::opt("classprob_dsc_combmod_quasar", DataType::Float32),
    ColSpec::opt("classprob_dsc_combmod_galaxy", DataType::Float32),
    ColSpec::opt("classprob_dsc_combmod_star", DataType::Float32),
];

struct ColIdx {
    source_id: usize,
    solution_id: usize,
    ref_epoch: usize,
    random_index: usize,
    designation: usize,
    ra: usize,
    ra_error: usize,
    dec: usize,
    dec_error: usize,
    ra_dec_corr: usize,
    parallax: usize,
    parallax_error: usize,
    parallax_over_error: usize,
    pm: usize,
    pmra: usize,
    pmra_error: usize,
    pmdec: usize,
    pmdec_error: usize,
    l: usize,
    b: usize,
    ecl_lon: usize,
    ecl_lat: usize,
    phot_g_mean_mag: usize,
    phot_g_mean_flux: usize,
    phot_g_mean_flux_error: usize,
    phot_g_n_obs: usize,
    phot_variable_flag: usize,
    astrometric_n_obs_al: usize,
    astrometric_excess_noise: usize,
    astrometric_excess_noise_sig: usize,
    astrometric_primary_flag: usize,
    duplicated_source: usize,
    astrometric_n_good_obs_al: usize,
    astrometric_n_bad_obs_al: usize,
    astrometric_gof_al: usize,
    astrometric_chi2_al: usize,
    astrometric_params_solved: usize,
    visibility_periods_used: usize,
    astrometric_sigma5d_max: usize,
    nu_eff_used_in_astrometry: usize,
    pseudocolour: usize,
    pseudocolour_error: usize,
    ruwe: usize,
    ipd_gof_harmonic_amplitude: usize,
    ipd_gof_harmonic_phase: usize,
    ipd_frac_multi_peak: usize,
    ipd_frac_odd_win: usize,
    phot_bp_mean_mag: usize,
    phot_bp_mean_flux: usize,
    phot_bp_mean_flux_error: usize,
    phot_bp_n_obs: usize,
    phot_rp_mean_mag: usize,
    phot_rp_mean_flux: usize,
    phot_rp_mean_flux_error: usize,
    phot_rp_n_obs: usize,
    bp_rp: usize,
    bp_g: usize,
    g_rp: usize,
    phot_bp_rp_excess_factor: usize,
    phot_proc_mode: usize,
    radial_velocity: usize,
    radial_velocity_error: usize,
    rv_method_used: usize,
    rv_nb_transits: usize,
    rv_expected_sig_to_noise: usize,
    rv_amplitude_robust: usize,
    rv_template_teff: usize,
    rv_template_logg: usize,
    rv_template_fe_h: usize,
    rv_atm_param_origin: usize,
    teff_gspphot: usize,
    teff_gspphot_lower: usize,
    teff_gspphot_upper: usize,
    logg_gspphot: usize,
    mh_gspphot: usize,
    distance_gspphot: usize,
    distance_gspphot_lower: usize,
    distance_gspphot_upper: usize,
    azero_gspphot: usize,
    ag_gspphot: usize,
    ebpminrp_gspphot: usize,
    libname_gspphot: usize,
    has_xp_continuous: usize,
    has_xp_sampled: usize,
    has_rvs: usize,
    has_epoch_photometry: usize,
    has_epoch_rv: usize,
    has_mcmc_gspphot: usize,
    has_mcmc_msc: usize,
    in_qso_candidates: usize,
    in_galaxy_candidates: usize,
    in_andromeda_survey: usize,
    non_single_star: usize,
    classprob_dsc_combmod_quasar: usize,
    classprob_dsc_combmod_galaxy: usize,
    classprob_dsc_combmod_star: usize,
}

impl ColIdx {
    fn new() -> Self {
        // Indices derived from COLUMNS declaration order above.
        let mut it = 0usize..;
        Self {
            source_id: it.next().unwrap(),
            solution_id: it.next().unwrap(),
            ref_epoch: it.next().unwrap(),
            random_index: it.next().unwrap(),
            designation: it.next().unwrap(),
            ra: it.next().unwrap(),
            ra_error: it.next().unwrap(),
            dec: it.next().unwrap(),
            dec_error: it.next().unwrap(),
            ra_dec_corr: it.next().unwrap(),
            parallax: it.next().unwrap(),
            parallax_error: it.next().unwrap(),
            parallax_over_error: it.next().unwrap(),
            pm: it.next().unwrap(),
            pmra: it.next().unwrap(),
            pmra_error: it.next().unwrap(),
            pmdec: it.next().unwrap(),
            pmdec_error: it.next().unwrap(),
            l: it.next().unwrap(),
            b: it.next().unwrap(),
            ecl_lon: it.next().unwrap(),
            ecl_lat: it.next().unwrap(),
            phot_g_mean_mag: it.next().unwrap(),
            phot_g_mean_flux: it.next().unwrap(),
            phot_g_mean_flux_error: it.next().unwrap(),
            phot_g_n_obs: it.next().unwrap(),
            phot_variable_flag: it.next().unwrap(),
            astrometric_n_obs_al: it.next().unwrap(),
            astrometric_excess_noise: it.next().unwrap(),
            astrometric_excess_noise_sig: it.next().unwrap(),
            astrometric_primary_flag: it.next().unwrap(),
            duplicated_source: it.next().unwrap(),
            astrometric_n_good_obs_al: it.next().unwrap(),
            astrometric_n_bad_obs_al: it.next().unwrap(),
            astrometric_gof_al: it.next().unwrap(),
            astrometric_chi2_al: it.next().unwrap(),
            astrometric_params_solved: it.next().unwrap(),
            visibility_periods_used: it.next().unwrap(),
            astrometric_sigma5d_max: it.next().unwrap(),
            nu_eff_used_in_astrometry: it.next().unwrap(),
            pseudocolour: it.next().unwrap(),
            pseudocolour_error: it.next().unwrap(),
            ruwe: it.next().unwrap(),
            ipd_gof_harmonic_amplitude: it.next().unwrap(),
            ipd_gof_harmonic_phase: it.next().unwrap(),
            ipd_frac_multi_peak: it.next().unwrap(),
            ipd_frac_odd_win: it.next().unwrap(),
            phot_bp_mean_mag: it.next().unwrap(),
            phot_bp_mean_flux: it.next().unwrap(),
            phot_bp_mean_flux_error: it.next().unwrap(),
            phot_bp_n_obs: it.next().unwrap(),
            phot_rp_mean_mag: it.next().unwrap(),
            phot_rp_mean_flux: it.next().unwrap(),
            phot_rp_mean_flux_error: it.next().unwrap(),
            phot_rp_n_obs: it.next().unwrap(),
            bp_rp: it.next().unwrap(),
            bp_g: it.next().unwrap(),
            g_rp: it.next().unwrap(),
            phot_bp_rp_excess_factor: it.next().unwrap(),
            phot_proc_mode: it.next().unwrap(),
            radial_velocity: it.next().unwrap(),
            radial_velocity_error: it.next().unwrap(),
            rv_method_used: it.next().unwrap(),
            rv_nb_transits: it.next().unwrap(),
            rv_expected_sig_to_noise: it.next().unwrap(),
            rv_amplitude_robust: it.next().unwrap(),
            rv_template_teff: it.next().unwrap(),
            rv_template_logg: it.next().unwrap(),
            rv_template_fe_h: it.next().unwrap(),
            rv_atm_param_origin: it.next().unwrap(),
            teff_gspphot: it.next().unwrap(),
            teff_gspphot_lower: it.next().unwrap(),
            teff_gspphot_upper: it.next().unwrap(),
            logg_gspphot: it.next().unwrap(),
            mh_gspphot: it.next().unwrap(),
            distance_gspphot: it.next().unwrap(),
            distance_gspphot_lower: it.next().unwrap(),
            distance_gspphot_upper: it.next().unwrap(),
            azero_gspphot: it.next().unwrap(),
            ag_gspphot: it.next().unwrap(),
            ebpminrp_gspphot: it.next().unwrap(),
            libname_gspphot: it.next().unwrap(),
            has_xp_continuous: it.next().unwrap(),
            has_xp_sampled: it.next().unwrap(),
            has_rvs: it.next().unwrap(),
            has_epoch_photometry: it.next().unwrap(),
            has_epoch_rv: it.next().unwrap(),
            has_mcmc_gspphot: it.next().unwrap(),
            has_mcmc_msc: it.next().unwrap(),
            in_qso_candidates: it.next().unwrap(),
            in_galaxy_candidates: it.next().unwrap(),
            in_andromeda_survey: it.next().unwrap(),
            non_single_star: it.next().unwrap(),
            classprob_dsc_combmod_quasar: it.next().unwrap(),
            classprob_dsc_combmod_galaxy: it.next().unwrap(),
            classprob_dsc_combmod_star: it.next().unwrap(),
        }
    }
}
