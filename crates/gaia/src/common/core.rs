//! Shared core fields present in every Gaia data release.

use nalgebra as na;
use serde::{Deserialize, Serialize};
use starfield_core::ProperMotion;

/// The variability flag reported by Gaia photometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VarFlag {
    #[default]
    NotAvailable,
    NotVariable,
    Variable,
}

impl VarFlag {
    pub fn parse(s: &str) -> Self {
        match s {
            "VARIABLE" => Self::Variable,
            "CONSTANT" | "NOT_VARIABLE" => Self::NotVariable,
            _ => Self::NotAvailable,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Variable => "VARIABLE",
            Self::NotVariable => "NOT_VARIABLE",
            Self::NotAvailable => "NOT_AVAILABLE",
        }
    }
}

/// Astrometric and photometric fields present in DR1, DR2, and DR3 gaia_source tables.
///
/// Every `Dr{N}Entry` embeds a `GaiaCore`. Fields that Gaia reports as nullable
/// (e.g. TGAS-only astrometry in DR1) are wrapped in `Option`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaiaCore {
    pub source_id: u64,
    pub solution_id: u64,
    pub ref_epoch: f64,
    pub random_index: Option<u64>,

    pub ra: f64,
    pub ra_error: f32,
    pub dec: f64,
    pub dec_error: f32,
    pub ra_dec_corr: Option<f32>,

    pub parallax: Option<f64>,
    pub parallax_error: Option<f32>,
    pub pmra: Option<f64>,
    pub pmra_error: Option<f32>,
    pub pmdec: Option<f64>,
    pub pmdec_error: Option<f32>,

    pub l: f64,
    pub b: f64,
    pub ecl_lon: f64,
    pub ecl_lat: f64,

    pub phot_g_mean_mag: f64,
    pub phot_g_mean_flux: Option<f64>,
    pub phot_g_mean_flux_error: Option<f64>,
    pub phot_g_n_obs: Option<u32>,
    pub phot_variable_flag: VarFlag,

    pub astrometric_n_obs_al: Option<u32>,
    pub astrometric_excess_noise: Option<f64>,
    pub astrometric_excess_noise_sig: Option<f64>,
    pub astrometric_primary_flag: Option<bool>,
    pub duplicated_source: Option<bool>,
    pub matched_observations: Option<u32>,
}

impl GaiaCore {
    /// Unit vector in ICRS coordinates (ra/dec interpreted as spherical angles).
    pub fn unit_vector(&self) -> na::Vector3<f64> {
        let ra = self.ra.to_radians();
        let dec = self.dec.to_radians();
        na::Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
    }

    /// Cartesian position in parsecs, when parallax is available and positive.
    pub fn cartesian_position(&self) -> Option<na::Vector3<f64>> {
        self.parallax.filter(|&p| p > 0.0).map(|plx| {
            let dist_pc = 1000.0 / plx;
            self.unit_vector() * dist_pc
        })
    }

    /// Typed proper motion in the Gaia DR3 convention (`pmra` carries
    /// the cos(dec) factor; both fields mas/yr). Returns `Some` only
    /// when both `pmra` and `pmdec` are populated — Gaia rows where
    /// the astrometric fit didn't produce a PM solution have either
    /// or both null and should fall back to the catalog's `ref_epoch`
    /// position.
    pub fn proper_motion(&self) -> Option<ProperMotion> {
        let pmra = self.pmra?;
        let pmdec = self.pmdec?;
        Some(ProperMotion::new(pmra, pmdec))
    }
}
