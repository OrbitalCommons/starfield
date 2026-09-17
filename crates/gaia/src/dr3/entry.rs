//! DR3 entry type with every published field grouped into coherent sub-structs.

use serde::{Deserialize, Serialize};

use crate::common::core::GaiaCore;
use crate::common::traits::{GaiaSource, Release};

/// One row of Gaia DR3 `gaia_source`.
///
/// Fields are organized into sub-structs by topic. Sub-structs that are published
/// for a subset of sources (e.g. radial velocity, GSP-Phot astrophysical parameters)
/// are wrapped in `Option` so absence is explicit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dr3Entry {
    pub core: GaiaCore,
    pub designation: Option<String>,
    pub pm: Option<f32>,
    pub parallax_over_error: Option<f32>,
    pub astrometric_extra: AstrometricExtra,
    pub ipd: IpdQuality,
    pub bp_rp: Option<BpRpPhotometry>,
    pub radial_velocity: Option<RadialVelocityDr3>,
    pub gspphot: Option<GspPhot>,
    pub data_links: DataLinks,
    pub classifications: Classifications,
}

/// DR3 astrometric quality metrics beyond the Core fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstrometricExtra {
    pub astrometric_n_good_obs_al: Option<u32>,
    pub astrometric_n_bad_obs_al: Option<u32>,
    pub astrometric_gof_al: Option<f32>,
    pub astrometric_chi2_al: Option<f32>,
    pub astrometric_params_solved: Option<u32>,
    pub visibility_periods_used: Option<u32>,
    pub astrometric_sigma5d_max: Option<f32>,
    pub nu_eff_used_in_astrometry: Option<f32>,
    pub pseudocolour: Option<f32>,
    pub pseudocolour_error: Option<f32>,
}

/// Image Parameter Determination quality metrics — new in DR3.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IpdQuality {
    pub ruwe: Option<f32>,
    pub ipd_gof_harmonic_amplitude: Option<f32>,
    pub ipd_gof_harmonic_phase: Option<f32>,
    pub ipd_frac_multi_peak: Option<u32>,
    pub ipd_frac_odd_win: Option<u32>,
}

/// BP/RP integrated photometry.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BpRpPhotometry {
    pub phot_bp_mean_mag: Option<f64>,
    pub phot_bp_mean_flux: Option<f64>,
    pub phot_bp_mean_flux_error: Option<f64>,
    pub phot_bp_n_obs: Option<u32>,
    pub phot_rp_mean_mag: Option<f64>,
    pub phot_rp_mean_flux: Option<f64>,
    pub phot_rp_mean_flux_error: Option<f64>,
    pub phot_rp_n_obs: Option<u32>,
    pub bp_rp: Option<f32>,
    pub bp_g: Option<f32>,
    pub g_rp: Option<f32>,
    pub phot_bp_rp_excess_factor: Option<f32>,
    pub phot_proc_mode: Option<u32>,
}

/// DR3 radial-velocity solution, expanded from DR2 with method + template parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RadialVelocityDr3 {
    pub radial_velocity: Option<f64>,
    pub radial_velocity_error: Option<f32>,
    pub rv_method_used: Option<u32>,
    pub rv_nb_transits: Option<u32>,
    pub rv_expected_sig_to_noise: Option<f32>,
    pub rv_amplitude_robust: Option<f32>,
    pub rv_template_teff: Option<f32>,
    pub rv_template_logg: Option<f32>,
    pub rv_template_fe_h: Option<f32>,
    pub rv_atm_param_origin: Option<u32>,
}

/// GSP-Phot astrophysical parameters (best estimate + 16/84 percentile bounds).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GspPhot {
    pub teff_gspphot: Option<f32>,
    pub teff_gspphot_lower: Option<f32>,
    pub teff_gspphot_upper: Option<f32>,
    pub logg_gspphot: Option<f32>,
    pub mh_gspphot: Option<f32>,
    pub distance_gspphot: Option<f32>,
    pub distance_gspphot_lower: Option<f32>,
    pub distance_gspphot_upper: Option<f32>,
    pub azero_gspphot: Option<f32>,
    pub ag_gspphot: Option<f32>,
    pub ebpminrp_gspphot: Option<f32>,
    pub libname_gspphot: Option<String>,
}

/// Boolean availability flags for supplementary DR3 datalink products.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataLinks {
    pub has_xp_continuous: Option<bool>,
    pub has_xp_sampled: Option<bool>,
    pub has_rvs: Option<bool>,
    pub has_epoch_photometry: Option<bool>,
    pub has_epoch_rv: Option<bool>,
    pub has_mcmc_gspphot: Option<bool>,
    pub has_mcmc_msc: Option<bool>,
}

/// Source classification flags and DSC-combmod probabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Classifications {
    pub in_qso_candidates: Option<bool>,
    pub in_galaxy_candidates: Option<bool>,
    pub in_andromeda_survey: Option<bool>,
    pub non_single_star: Option<u32>,
    pub classprob_dsc_combmod_quasar: Option<f32>,
    pub classprob_dsc_combmod_galaxy: Option<f32>,
    pub classprob_dsc_combmod_star: Option<f32>,
}

impl GaiaSource for Dr3Entry {
    fn core(&self) -> &GaiaCore {
        &self.core
    }
    fn release(&self) -> Release {
        Release::Dr3
    }
    fn b_v(&self) -> Option<f64> {
        self.bp_rp
            .as_ref()
            .and_then(|p| p.bp_rp)
            .map(|v| crate::common::traits::bp_rp_to_johnson_b_v(v as f64))
    }
}
