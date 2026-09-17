//! DR2 Arrow schema and row constructor.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use starfield_core::Result;

use crate::common::core::{GaiaCore, VarFlag};
use crate::common::format::*;
use crate::common::parse::*;
use crate::common::traits::{GaiaRelease, Release};
use crate::dr2::entry::{AstroParams, AstrometricExtra, BpRpPhotometry, Dr2Entry, RadialVelocity};

/// Zero-sized release marker for Gaia DR2.
#[derive(Debug, Clone, Copy)]
pub struct Dr2;

impl GaiaRelease for Dr2 {
    const RELEASE: Release = Release::Dr2;
    const BASE_URL: &'static str = "https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/";
    const MD5_FILENAME: &'static str = "MD5SUM.txt";
    const FILE_REGEX: &'static str = r#"(GaiaSource_\d+_\d+\.csv\.gz)"#;
    const CACHE_SUBDIR: &'static str = "gaia/dr2";
    const TAP_TABLE: &'static str = "gaiadr2.gaia_source";
    const IS_ECSV: bool = false;

    type Entry = Dr2Entry;

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
            matched_observations: opt_u32(batch, c.matched_observations, row)?,
        };

        let astrometric_extra = AstrometricExtra {
            astrometric_n_good_obs_al: opt_u32(batch, c.astrometric_n_good_obs_al, row)?,
            astrometric_n_bad_obs_al: opt_u32(batch, c.astrometric_n_bad_obs_al, row)?,
            astrometric_gof_al: opt_f32(batch, c.astrometric_gof_al, row)?,
            astrometric_chi2_al: opt_f32(batch, c.astrometric_chi2_al, row)?,
            astrometric_params_solved: opt_u32(batch, c.astrometric_params_solved, row)?,
            astrometric_weight_al: opt_f32(batch, c.astrometric_weight_al, row)?,
            astrometric_pseudo_colour: opt_f32(batch, c.astrometric_pseudo_colour, row)?,
            astrometric_pseudo_colour_error: opt_f32(
                batch,
                c.astrometric_pseudo_colour_error,
                row,
            )?,
            mean_varpi_factor_al: opt_f32(batch, c.mean_varpi_factor_al, row)?,
            astrometric_matched_observations: opt_u32(
                batch,
                c.astrometric_matched_observations,
                row,
            )?,
            visibility_periods_used: opt_u32(batch, c.visibility_periods_used, row)?,
            astrometric_sigma5d_max: opt_f32(batch, c.astrometric_sigma5d_max, row)?,
            frame_rotator_object_type: opt_u32(batch, c.frame_rotator_object_type, row)?,
        };

        let bp_mag = opt_f64(batch, c.phot_bp_mean_mag, row)?;
        let rp_mag = opt_f64(batch, c.phot_rp_mean_mag, row)?;
        let bp_rp = if bp_mag.is_some() || rp_mag.is_some() {
            Some(BpRpPhotometry {
                phot_bp_n_obs: opt_u32(batch, c.phot_bp_n_obs, row)?,
                phot_bp_mean_flux: opt_f64(batch, c.phot_bp_mean_flux, row)?,
                phot_bp_mean_flux_error: opt_f64(batch, c.phot_bp_mean_flux_error, row)?,
                phot_bp_mean_flux_over_error: opt_f32(batch, c.phot_bp_mean_flux_over_error, row)?,
                phot_bp_mean_mag: bp_mag,
                phot_rp_n_obs: opt_u32(batch, c.phot_rp_n_obs, row)?,
                phot_rp_mean_flux: opt_f64(batch, c.phot_rp_mean_flux, row)?,
                phot_rp_mean_flux_error: opt_f64(batch, c.phot_rp_mean_flux_error, row)?,
                phot_rp_mean_flux_over_error: opt_f32(batch, c.phot_rp_mean_flux_over_error, row)?,
                phot_rp_mean_mag: rp_mag,
                phot_bp_rp_excess_factor: opt_f32(batch, c.phot_bp_rp_excess_factor, row)?,
                phot_proc_mode: opt_u32(batch, c.phot_proc_mode, row)?,
                bp_rp: opt_f32(batch, c.bp_rp, row)?,
                bp_g: opt_f32(batch, c.bp_g, row)?,
                g_rp: opt_f32(batch, c.g_rp, row)?,
            })
        } else {
            None
        };

        let rv_value = opt_f64(batch, c.radial_velocity, row)?;
        let radial_velocity = if rv_value.is_some() {
            Some(RadialVelocity {
                radial_velocity: rv_value,
                radial_velocity_error: opt_f32(batch, c.radial_velocity_error, row)?,
                rv_nb_transits: opt_u32(batch, c.rv_nb_transits, row)?,
                rv_template_teff: opt_f32(batch, c.rv_template_teff, row)?,
                rv_template_logg: opt_f32(batch, c.rv_template_logg, row)?,
                rv_template_fe_h: opt_f32(batch, c.rv_template_fe_h, row)?,
            })
        } else {
            None
        };

        let teff = opt_f32(batch, c.teff_val, row)?;
        let astrophysical = if teff.is_some() {
            Some(AstroParams {
                priam_flags: opt_u64(batch, c.priam_flags, row)?,
                teff_val: teff,
                teff_percentile_lower: opt_f32(batch, c.teff_percentile_lower, row)?,
                teff_percentile_upper: opt_f32(batch, c.teff_percentile_upper, row)?,
                a_g_val: opt_f32(batch, c.a_g_val, row)?,
                a_g_percentile_lower: opt_f32(batch, c.a_g_percentile_lower, row)?,
                a_g_percentile_upper: opt_f32(batch, c.a_g_percentile_upper, row)?,
                e_bp_min_rp_val: opt_f32(batch, c.e_bp_min_rp_val, row)?,
                e_bp_min_rp_percentile_lower: opt_f32(batch, c.e_bp_min_rp_percentile_lower, row)?,
                e_bp_min_rp_percentile_upper: opt_f32(batch, c.e_bp_min_rp_percentile_upper, row)?,
                flame_flags: opt_u64(batch, c.flame_flags, row)?,
                radius_val: opt_f32(batch, c.radius_val, row)?,
                radius_percentile_lower: opt_f32(batch, c.radius_percentile_lower, row)?,
                radius_percentile_upper: opt_f32(batch, c.radius_percentile_upper, row)?,
                lum_val: opt_f32(batch, c.lum_val, row)?,
                lum_percentile_lower: opt_f32(batch, c.lum_percentile_lower, row)?,
                lum_percentile_upper: opt_f32(batch, c.lum_percentile_upper, row)?,
            })
        } else {
            None
        };

        Ok(Dr2Entry {
            core,
            designation: opt_str(batch, c.designation, row)?.map(String::from),
            parallax_over_error: opt_f32(batch, c.parallax_over_error, row)?,
            astrometric_extra,
            bp_rp,
            radial_velocity,
            astrophysical,
        })
    }

    fn format_csv_row(e: &Self::Entry) -> String {
        let c = &e.core;
        let ax = &e.astrometric_extra;
        let bp = e.bp_rp.as_ref();
        let rv = e.radial_velocity.as_ref();
        let ap = e.astrophysical.as_ref();
        // Order MUST match COLUMNS below.
        [
            c.source_id.to_string(),
            c.solution_id.to_string(),
            fopt_str(e.designation.as_deref()).to_string(),
            c.ref_epoch.to_string(),
            fopt(c.random_index),
            c.ra.to_string(),
            c.ra_error.to_string(),
            c.dec.to_string(),
            c.dec_error.to_string(),
            fopt(c.ra_dec_corr),
            fopt(c.parallax),
            fopt(c.parallax_error),
            fopt(e.parallax_over_error),
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
            fopt(c.matched_observations),
            fopt(ax.astrometric_n_good_obs_al),
            fopt(ax.astrometric_n_bad_obs_al),
            fopt(ax.astrometric_gof_al),
            fopt(ax.astrometric_chi2_al),
            fopt(ax.astrometric_params_solved),
            fopt(ax.astrometric_weight_al),
            fopt(ax.astrometric_pseudo_colour),
            fopt(ax.astrometric_pseudo_colour_error),
            fopt(ax.mean_varpi_factor_al),
            fopt(ax.astrometric_matched_observations),
            fopt(ax.visibility_periods_used),
            fopt(ax.astrometric_sigma5d_max),
            fopt(ax.frame_rotator_object_type),
            fopt(bp.and_then(|b| b.phot_bp_n_obs)),
            fopt(bp.and_then(|b| b.phot_bp_mean_flux)),
            fopt(bp.and_then(|b| b.phot_bp_mean_flux_error)),
            fopt(bp.and_then(|b| b.phot_bp_mean_flux_over_error)),
            fopt(bp.and_then(|b| b.phot_bp_mean_mag)),
            fopt(bp.and_then(|b| b.phot_rp_n_obs)),
            fopt(bp.and_then(|b| b.phot_rp_mean_flux)),
            fopt(bp.and_then(|b| b.phot_rp_mean_flux_error)),
            fopt(bp.and_then(|b| b.phot_rp_mean_flux_over_error)),
            fopt(bp.and_then(|b| b.phot_rp_mean_mag)),
            fopt(bp.and_then(|b| b.phot_bp_rp_excess_factor)),
            fopt(bp.and_then(|b| b.phot_proc_mode)),
            fopt(bp.and_then(|b| b.bp_rp)),
            fopt(bp.and_then(|b| b.bp_g)),
            fopt(bp.and_then(|b| b.g_rp)),
            fopt(rv.and_then(|r| r.radial_velocity)),
            fopt(rv.and_then(|r| r.radial_velocity_error)),
            fopt(rv.and_then(|r| r.rv_nb_transits)),
            fopt(rv.and_then(|r| r.rv_template_teff)),
            fopt(rv.and_then(|r| r.rv_template_logg)),
            fopt(rv.and_then(|r| r.rv_template_fe_h)),
            fopt(ap.and_then(|a| a.priam_flags)),
            fopt(ap.and_then(|a| a.teff_val)),
            fopt(ap.and_then(|a| a.teff_percentile_lower)),
            fopt(ap.and_then(|a| a.teff_percentile_upper)),
            fopt(ap.and_then(|a| a.a_g_val)),
            fopt(ap.and_then(|a| a.a_g_percentile_lower)),
            fopt(ap.and_then(|a| a.a_g_percentile_upper)),
            fopt(ap.and_then(|a| a.e_bp_min_rp_val)),
            fopt(ap.and_then(|a| a.e_bp_min_rp_percentile_lower)),
            fopt(ap.and_then(|a| a.e_bp_min_rp_percentile_upper)),
            fopt(ap.and_then(|a| a.flame_flags)),
            fopt(ap.and_then(|a| a.radius_val)),
            fopt(ap.and_then(|a| a.radius_percentile_lower)),
            fopt(ap.and_then(|a| a.radius_percentile_upper)),
            fopt(ap.and_then(|a| a.lum_val)),
            fopt(ap.and_then(|a| a.lum_percentile_lower)),
            fopt(ap.and_then(|a| a.lum_percentile_upper)),
        ]
        .join(",")
    }
}

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

static COLUMNS: &[ColSpec] = &[
    ColSpec::req("source_id", DataType::UInt64),
    ColSpec::req("solution_id", DataType::UInt64),
    ColSpec::opt("designation", DataType::Utf8),
    ColSpec::req("ref_epoch", DataType::Float64),
    ColSpec::opt("random_index", DataType::Int64),
    ColSpec::req("ra", DataType::Float64),
    ColSpec::req("ra_error", DataType::Float32),
    ColSpec::req("dec", DataType::Float64),
    ColSpec::req("dec_error", DataType::Float32),
    ColSpec::opt("ra_dec_corr", DataType::Float32),
    ColSpec::opt("parallax", DataType::Float64),
    ColSpec::opt("parallax_error", DataType::Float32),
    ColSpec::opt("parallax_over_error", DataType::Float32),
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
    ColSpec::opt("matched_observations", DataType::Int32),
    ColSpec::opt("astrometric_n_good_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_n_bad_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_gof_al", DataType::Float32),
    ColSpec::opt("astrometric_chi2_al", DataType::Float32),
    ColSpec::opt("astrometric_params_solved", DataType::Int32),
    ColSpec::opt("astrometric_weight_al", DataType::Float32),
    ColSpec::opt("astrometric_pseudo_colour", DataType::Float32),
    ColSpec::opt("astrometric_pseudo_colour_error", DataType::Float32),
    ColSpec::opt("mean_varpi_factor_al", DataType::Float32),
    ColSpec::opt("astrometric_matched_observations", DataType::Int32),
    ColSpec::opt("visibility_periods_used", DataType::Int32),
    ColSpec::opt("astrometric_sigma5d_max", DataType::Float32),
    ColSpec::opt("frame_rotator_object_type", DataType::Int32),
    ColSpec::opt("phot_bp_n_obs", DataType::Int32),
    ColSpec::opt("phot_bp_mean_flux", DataType::Float64),
    ColSpec::opt("phot_bp_mean_flux_error", DataType::Float64),
    ColSpec::opt("phot_bp_mean_flux_over_error", DataType::Float32),
    ColSpec::opt("phot_bp_mean_mag", DataType::Float64),
    ColSpec::opt("phot_rp_n_obs", DataType::Int32),
    ColSpec::opt("phot_rp_mean_flux", DataType::Float64),
    ColSpec::opt("phot_rp_mean_flux_error", DataType::Float64),
    ColSpec::opt("phot_rp_mean_flux_over_error", DataType::Float32),
    ColSpec::opt("phot_rp_mean_mag", DataType::Float64),
    ColSpec::opt("phot_bp_rp_excess_factor", DataType::Float32),
    ColSpec::opt("phot_proc_mode", DataType::Int32),
    ColSpec::opt("bp_rp", DataType::Float32),
    ColSpec::opt("bp_g", DataType::Float32),
    ColSpec::opt("g_rp", DataType::Float32),
    ColSpec::opt("radial_velocity", DataType::Float64),
    ColSpec::opt("radial_velocity_error", DataType::Float32),
    ColSpec::opt("rv_nb_transits", DataType::Int32),
    ColSpec::opt("rv_template_teff", DataType::Float32),
    ColSpec::opt("rv_template_logg", DataType::Float32),
    ColSpec::opt("rv_template_fe_h", DataType::Float32),
    ColSpec::opt("priam_flags", DataType::Int64),
    ColSpec::opt("teff_val", DataType::Float32),
    ColSpec::opt("teff_percentile_lower", DataType::Float32),
    ColSpec::opt("teff_percentile_upper", DataType::Float32),
    ColSpec::opt("a_g_val", DataType::Float32),
    ColSpec::opt("a_g_percentile_lower", DataType::Float32),
    ColSpec::opt("a_g_percentile_upper", DataType::Float32),
    ColSpec::opt("e_bp_min_rp_val", DataType::Float32),
    ColSpec::opt("e_bp_min_rp_percentile_lower", DataType::Float32),
    ColSpec::opt("e_bp_min_rp_percentile_upper", DataType::Float32),
    ColSpec::opt("flame_flags", DataType::Int64),
    ColSpec::opt("radius_val", DataType::Float32),
    ColSpec::opt("radius_percentile_lower", DataType::Float32),
    ColSpec::opt("radius_percentile_upper", DataType::Float32),
    ColSpec::opt("lum_val", DataType::Float32),
    ColSpec::opt("lum_percentile_lower", DataType::Float32),
    ColSpec::opt("lum_percentile_upper", DataType::Float32),
];

struct ColIdx {
    source_id: usize,
    solution_id: usize,
    designation: usize,
    ref_epoch: usize,
    random_index: usize,
    ra: usize,
    ra_error: usize,
    dec: usize,
    dec_error: usize,
    ra_dec_corr: usize,
    parallax: usize,
    parallax_error: usize,
    parallax_over_error: usize,
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
    matched_observations: usize,
    astrometric_n_good_obs_al: usize,
    astrometric_n_bad_obs_al: usize,
    astrometric_gof_al: usize,
    astrometric_chi2_al: usize,
    astrometric_params_solved: usize,
    astrometric_weight_al: usize,
    astrometric_pseudo_colour: usize,
    astrometric_pseudo_colour_error: usize,
    mean_varpi_factor_al: usize,
    astrometric_matched_observations: usize,
    visibility_periods_used: usize,
    astrometric_sigma5d_max: usize,
    frame_rotator_object_type: usize,
    phot_bp_n_obs: usize,
    phot_bp_mean_flux: usize,
    phot_bp_mean_flux_error: usize,
    phot_bp_mean_flux_over_error: usize,
    phot_bp_mean_mag: usize,
    phot_rp_n_obs: usize,
    phot_rp_mean_flux: usize,
    phot_rp_mean_flux_error: usize,
    phot_rp_mean_flux_over_error: usize,
    phot_rp_mean_mag: usize,
    phot_bp_rp_excess_factor: usize,
    phot_proc_mode: usize,
    bp_rp: usize,
    bp_g: usize,
    g_rp: usize,
    radial_velocity: usize,
    radial_velocity_error: usize,
    rv_nb_transits: usize,
    rv_template_teff: usize,
    rv_template_logg: usize,
    rv_template_fe_h: usize,
    priam_flags: usize,
    teff_val: usize,
    teff_percentile_lower: usize,
    teff_percentile_upper: usize,
    a_g_val: usize,
    a_g_percentile_lower: usize,
    a_g_percentile_upper: usize,
    e_bp_min_rp_val: usize,
    e_bp_min_rp_percentile_lower: usize,
    e_bp_min_rp_percentile_upper: usize,
    flame_flags: usize,
    radius_val: usize,
    radius_percentile_lower: usize,
    radius_percentile_upper: usize,
    lum_val: usize,
    lum_percentile_lower: usize,
    lum_percentile_upper: usize,
}

impl ColIdx {
    fn new() -> Self {
        let mut it = 0usize..;
        Self {
            source_id: it.next().unwrap(),
            solution_id: it.next().unwrap(),
            designation: it.next().unwrap(),
            ref_epoch: it.next().unwrap(),
            random_index: it.next().unwrap(),
            ra: it.next().unwrap(),
            ra_error: it.next().unwrap(),
            dec: it.next().unwrap(),
            dec_error: it.next().unwrap(),
            ra_dec_corr: it.next().unwrap(),
            parallax: it.next().unwrap(),
            parallax_error: it.next().unwrap(),
            parallax_over_error: it.next().unwrap(),
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
            matched_observations: it.next().unwrap(),
            astrometric_n_good_obs_al: it.next().unwrap(),
            astrometric_n_bad_obs_al: it.next().unwrap(),
            astrometric_gof_al: it.next().unwrap(),
            astrometric_chi2_al: it.next().unwrap(),
            astrometric_params_solved: it.next().unwrap(),
            astrometric_weight_al: it.next().unwrap(),
            astrometric_pseudo_colour: it.next().unwrap(),
            astrometric_pseudo_colour_error: it.next().unwrap(),
            mean_varpi_factor_al: it.next().unwrap(),
            astrometric_matched_observations: it.next().unwrap(),
            visibility_periods_used: it.next().unwrap(),
            astrometric_sigma5d_max: it.next().unwrap(),
            frame_rotator_object_type: it.next().unwrap(),
            phot_bp_n_obs: it.next().unwrap(),
            phot_bp_mean_flux: it.next().unwrap(),
            phot_bp_mean_flux_error: it.next().unwrap(),
            phot_bp_mean_flux_over_error: it.next().unwrap(),
            phot_bp_mean_mag: it.next().unwrap(),
            phot_rp_n_obs: it.next().unwrap(),
            phot_rp_mean_flux: it.next().unwrap(),
            phot_rp_mean_flux_error: it.next().unwrap(),
            phot_rp_mean_flux_over_error: it.next().unwrap(),
            phot_rp_mean_mag: it.next().unwrap(),
            phot_bp_rp_excess_factor: it.next().unwrap(),
            phot_proc_mode: it.next().unwrap(),
            bp_rp: it.next().unwrap(),
            bp_g: it.next().unwrap(),
            g_rp: it.next().unwrap(),
            radial_velocity: it.next().unwrap(),
            radial_velocity_error: it.next().unwrap(),
            rv_nb_transits: it.next().unwrap(),
            rv_template_teff: it.next().unwrap(),
            rv_template_logg: it.next().unwrap(),
            rv_template_fe_h: it.next().unwrap(),
            priam_flags: it.next().unwrap(),
            teff_val: it.next().unwrap(),
            teff_percentile_lower: it.next().unwrap(),
            teff_percentile_upper: it.next().unwrap(),
            a_g_val: it.next().unwrap(),
            a_g_percentile_lower: it.next().unwrap(),
            a_g_percentile_upper: it.next().unwrap(),
            e_bp_min_rp_val: it.next().unwrap(),
            e_bp_min_rp_percentile_lower: it.next().unwrap(),
            e_bp_min_rp_percentile_upper: it.next().unwrap(),
            flame_flags: it.next().unwrap(),
            radius_val: it.next().unwrap(),
            radius_percentile_lower: it.next().unwrap(),
            radius_percentile_upper: it.next().unwrap(),
            lum_val: it.next().unwrap(),
            lum_percentile_lower: it.next().unwrap(),
            lum_percentile_upper: it.next().unwrap(),
        }
    }
}
