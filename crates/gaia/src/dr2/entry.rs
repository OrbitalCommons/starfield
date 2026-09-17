//! DR2 entry type.

use serde::{Deserialize, Serialize};

use crate::common::core::GaiaCore;
use crate::common::traits::{GaiaSource, Release};

/// One row of Gaia DR2 `gaia_source`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dr2Entry {
    pub core: GaiaCore,
    pub designation: Option<String>,
    pub parallax_over_error: Option<f32>,
    pub astrometric_extra: AstrometricExtra,
    pub bp_rp: Option<BpRpPhotometry>,
    pub radial_velocity: Option<RadialVelocity>,
    pub astrophysical: Option<AstroParams>,
}

/// DR2 astrometric quality metrics beyond the Core fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstrometricExtra {
    pub astrometric_n_good_obs_al: Option<u32>,
    pub astrometric_n_bad_obs_al: Option<u32>,
    pub astrometric_gof_al: Option<f32>,
    pub astrometric_chi2_al: Option<f32>,
    pub astrometric_params_solved: Option<u32>,
    pub astrometric_weight_al: Option<f32>,
    pub astrometric_pseudo_colour: Option<f32>,
    pub astrometric_pseudo_colour_error: Option<f32>,
    pub mean_varpi_factor_al: Option<f32>,
    pub astrometric_matched_observations: Option<u32>,
    pub visibility_periods_used: Option<u32>,
    pub astrometric_sigma5d_max: Option<f32>,
    pub frame_rotator_object_type: Option<u32>,
}

/// DR2 BP/RP integrated photometry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BpRpPhotometry {
    pub phot_bp_n_obs: Option<u32>,
    pub phot_bp_mean_flux: Option<f64>,
    pub phot_bp_mean_flux_error: Option<f64>,
    pub phot_bp_mean_flux_over_error: Option<f32>,
    pub phot_bp_mean_mag: Option<f64>,
    pub phot_rp_n_obs: Option<u32>,
    pub phot_rp_mean_flux: Option<f64>,
    pub phot_rp_mean_flux_error: Option<f64>,
    pub phot_rp_mean_flux_over_error: Option<f32>,
    pub phot_rp_mean_mag: Option<f64>,
    pub phot_bp_rp_excess_factor: Option<f32>,
    pub phot_proc_mode: Option<u32>,
    pub bp_rp: Option<f32>,
    pub bp_g: Option<f32>,
    pub g_rp: Option<f32>,
}

/// DR2 radial-velocity solution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RadialVelocity {
    pub radial_velocity: Option<f64>,
    pub radial_velocity_error: Option<f32>,
    pub rv_nb_transits: Option<u32>,
    pub rv_template_teff: Option<f32>,
    pub rv_template_logg: Option<f32>,
    pub rv_template_fe_h: Option<f32>,
}

/// DR2 astrophysical parameters produced by Apsis (priam and flame modules).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstroParams {
    pub priam_flags: Option<u64>,
    pub teff_val: Option<f32>,
    pub teff_percentile_lower: Option<f32>,
    pub teff_percentile_upper: Option<f32>,
    pub a_g_val: Option<f32>,
    pub a_g_percentile_lower: Option<f32>,
    pub a_g_percentile_upper: Option<f32>,
    pub e_bp_min_rp_val: Option<f32>,
    pub e_bp_min_rp_percentile_lower: Option<f32>,
    pub e_bp_min_rp_percentile_upper: Option<f32>,
    pub flame_flags: Option<u64>,
    pub radius_val: Option<f32>,
    pub radius_percentile_lower: Option<f32>,
    pub radius_percentile_upper: Option<f32>,
    pub lum_val: Option<f32>,
    pub lum_percentile_lower: Option<f32>,
    pub lum_percentile_upper: Option<f32>,
}

impl GaiaSource for Dr2Entry {
    fn core(&self) -> &GaiaCore {
        &self.core
    }
    fn release(&self) -> Release {
        Release::Dr2
    }
    fn b_v(&self) -> Option<f64> {
        self.bp_rp
            .as_ref()
            .and_then(|p| p.bp_rp)
            .map(|v| crate::common::traits::bp_rp_to_johnson_b_v(v as f64))
    }
}
