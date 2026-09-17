//! DR1 Arrow schema and row constructor.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use starfield_core::Result;

use crate::common::core::{GaiaCore, VarFlag};
use crate::common::format::*;
use crate::common::parse::*;
use crate::common::traits::{GaiaRelease, Release};
use crate::dr1::entry::{AstrometricExtra, Dr1Entry, ScanDirection};

/// Zero-sized release marker for Gaia DR1 `gaia_source` files.
#[derive(Debug, Clone, Copy)]
pub struct Dr1;

impl GaiaRelease for Dr1 {
    const RELEASE: Release = Release::Dr1;
    const BASE_URL: &'static str = "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/";
    const MD5_FILENAME: &'static str = "MD5SUM.txt";
    const FILE_REGEX: &'static str = r#"(GaiaSource_\d{3}-\d{3}-\d{3}\.csv\.gz)"#;
    const CACHE_SUBDIR: &'static str = "gaia/dr1";
    const TAP_TABLE: &'static str = "gaiadr1.gaia_source";
    const IS_ECSV: bool = false;

    type Entry = Dr1Entry;

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
            // Sources without G-band photometry (rare) get INFINITY so any finite
            // `--mag-limit` filters them out.
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
            astrometric_n_obs_ac: opt_u32(batch, c.astrometric_n_obs_ac, row)?,
            astrometric_n_good_obs_al: opt_u32(batch, c.astrometric_n_good_obs_al, row)?,
            astrometric_n_good_obs_ac: opt_u32(batch, c.astrometric_n_good_obs_ac, row)?,
            astrometric_n_bad_obs_al: opt_u32(batch, c.astrometric_n_bad_obs_al, row)?,
            astrometric_n_bad_obs_ac: opt_u32(batch, c.astrometric_n_bad_obs_ac, row)?,
            astrometric_delta_q: opt_f32(batch, c.astrometric_delta_q, row)?,
            astrometric_relegation_factor: opt_f32(batch, c.astrometric_relegation_factor, row)?,
            astrometric_weight_al: opt_f32(batch, c.astrometric_weight_al, row)?,
            astrometric_weight_ac: opt_f32(batch, c.astrometric_weight_ac, row)?,
            astrometric_priors_used: opt_u32(batch, c.astrometric_priors_used, row)?,
        };

        let scan_direction = ScanDirection {
            strength_k1: opt_f32(batch, c.scan_direction_strength_k1, row)?,
            strength_k2: opt_f32(batch, c.scan_direction_strength_k2, row)?,
            strength_k3: opt_f32(batch, c.scan_direction_strength_k3, row)?,
            strength_k4: opt_f32(batch, c.scan_direction_strength_k4, row)?,
            mean_k1: opt_f32(batch, c.scan_direction_mean_k1, row)?,
            mean_k2: opt_f32(batch, c.scan_direction_mean_k2, row)?,
            mean_k3: opt_f32(batch, c.scan_direction_mean_k3, row)?,
            mean_k4: opt_f32(batch, c.scan_direction_mean_k4, row)?,
        };

        Ok(Dr1Entry {
            core,
            astrometric_extra,
            scan_direction,
            tgas: None,
        })
    }

    fn format_csv_row(e: &Self::Entry) -> String {
        let c = &e.core;
        let ax = &e.astrometric_extra;
        let sd = &e.scan_direction;
        // Order MUST match COLUMNS below.
        [
            c.source_id.to_string(),
            c.solution_id.to_string(),
            c.ref_epoch.to_string(),
            fopt(c.random_index),
            c.ra.to_string(),
            c.ra_error.to_string(),
            c.dec.to_string(),
            c.dec_error.to_string(),
            fopt(c.ra_dec_corr),
            fopt(c.parallax),
            fopt(c.parallax_error),
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
            fopt(ax.astrometric_n_obs_ac),
            fopt(ax.astrometric_n_good_obs_al),
            fopt(ax.astrometric_n_good_obs_ac),
            fopt(ax.astrometric_n_bad_obs_al),
            fopt(ax.astrometric_n_bad_obs_ac),
            fopt(ax.astrometric_delta_q),
            fopt(ax.astrometric_relegation_factor),
            fopt(ax.astrometric_weight_al),
            fopt(ax.astrometric_weight_ac),
            fopt(ax.astrometric_priors_used),
            fopt(sd.strength_k1),
            fopt(sd.strength_k2),
            fopt(sd.strength_k3),
            fopt(sd.strength_k4),
            fopt(sd.mean_k1),
            fopt(sd.mean_k2),
            fopt(sd.mean_k3),
            fopt(sd.mean_k4),
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
    ColSpec::req("ref_epoch", DataType::Float64),
    ColSpec::opt("random_index", DataType::Int64),
    ColSpec::req("ra", DataType::Float64),
    ColSpec::req("ra_error", DataType::Float32),
    ColSpec::req("dec", DataType::Float64),
    ColSpec::req("dec_error", DataType::Float32),
    ColSpec::opt("ra_dec_corr", DataType::Float32),
    ColSpec::opt("parallax", DataType::Float64),
    ColSpec::opt("parallax_error", DataType::Float32),
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
    ColSpec::opt("astrometric_n_obs_ac", DataType::Int32),
    ColSpec::opt("astrometric_n_good_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_n_good_obs_ac", DataType::Int32),
    ColSpec::opt("astrometric_n_bad_obs_al", DataType::Int32),
    ColSpec::opt("astrometric_n_bad_obs_ac", DataType::Int32),
    ColSpec::opt("astrometric_delta_q", DataType::Float32),
    ColSpec::opt("astrometric_relegation_factor", DataType::Float32),
    ColSpec::opt("astrometric_weight_al", DataType::Float32),
    ColSpec::opt("astrometric_weight_ac", DataType::Float32),
    ColSpec::opt("astrometric_priors_used", DataType::Int32),
    ColSpec::opt("scan_direction_strength_k1", DataType::Float32),
    ColSpec::opt("scan_direction_strength_k2", DataType::Float32),
    ColSpec::opt("scan_direction_strength_k3", DataType::Float32),
    ColSpec::opt("scan_direction_strength_k4", DataType::Float32),
    ColSpec::opt("scan_direction_mean_k1", DataType::Float32),
    ColSpec::opt("scan_direction_mean_k2", DataType::Float32),
    ColSpec::opt("scan_direction_mean_k3", DataType::Float32),
    ColSpec::opt("scan_direction_mean_k4", DataType::Float32),
];

struct ColIdx {
    source_id: usize,
    solution_id: usize,
    ref_epoch: usize,
    random_index: usize,
    ra: usize,
    ra_error: usize,
    dec: usize,
    dec_error: usize,
    ra_dec_corr: usize,
    parallax: usize,
    parallax_error: usize,
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
    astrometric_n_obs_ac: usize,
    astrometric_n_good_obs_al: usize,
    astrometric_n_good_obs_ac: usize,
    astrometric_n_bad_obs_al: usize,
    astrometric_n_bad_obs_ac: usize,
    astrometric_delta_q: usize,
    astrometric_relegation_factor: usize,
    astrometric_weight_al: usize,
    astrometric_weight_ac: usize,
    astrometric_priors_used: usize,
    scan_direction_strength_k1: usize,
    scan_direction_strength_k2: usize,
    scan_direction_strength_k3: usize,
    scan_direction_strength_k4: usize,
    scan_direction_mean_k1: usize,
    scan_direction_mean_k2: usize,
    scan_direction_mean_k3: usize,
    scan_direction_mean_k4: usize,
}

impl ColIdx {
    fn new() -> Self {
        let mut it = 0usize..;
        Self {
            source_id: it.next().unwrap(),
            solution_id: it.next().unwrap(),
            ref_epoch: it.next().unwrap(),
            random_index: it.next().unwrap(),
            ra: it.next().unwrap(),
            ra_error: it.next().unwrap(),
            dec: it.next().unwrap(),
            dec_error: it.next().unwrap(),
            ra_dec_corr: it.next().unwrap(),
            parallax: it.next().unwrap(),
            parallax_error: it.next().unwrap(),
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
            astrometric_n_obs_ac: it.next().unwrap(),
            astrometric_n_good_obs_al: it.next().unwrap(),
            astrometric_n_good_obs_ac: it.next().unwrap(),
            astrometric_n_bad_obs_al: it.next().unwrap(),
            astrometric_n_bad_obs_ac: it.next().unwrap(),
            astrometric_delta_q: it.next().unwrap(),
            astrometric_relegation_factor: it.next().unwrap(),
            astrometric_weight_al: it.next().unwrap(),
            astrometric_weight_ac: it.next().unwrap(),
            astrometric_priors_used: it.next().unwrap(),
            scan_direction_strength_k1: it.next().unwrap(),
            scan_direction_strength_k2: it.next().unwrap(),
            scan_direction_strength_k3: it.next().unwrap(),
            scan_direction_strength_k4: it.next().unwrap(),
            scan_direction_mean_k1: it.next().unwrap(),
            scan_direction_mean_k2: it.next().unwrap(),
            scan_direction_mean_k3: it.next().unwrap(),
            scan_direction_mean_k4: it.next().unwrap(),
        }
    }
}
