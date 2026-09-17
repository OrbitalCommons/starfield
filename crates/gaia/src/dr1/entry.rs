//! DR1 entry type.

use serde::{Deserialize, Serialize};

use crate::common::core::GaiaCore;
use crate::common::traits::{GaiaSource, Release};

/// One row of Gaia DR1 `gaia_source`.
///
/// In DR1, astrometric parameters (parallax, proper motion) are only populated for
/// the ~2M sources in the TGAS subset; everything else has them null in [`GaiaCore`].
/// The Hipparcos/Tycho cross-identifiers come from the separate `tgas_source.csv`
/// catalog — load them with [`load_tgas_block_map`](crate::dr1::catalog::load_tgas_block_map)
/// and splice in with [`Dr1Catalog::attach_tgas`](crate::dr1::catalog::Dr1Catalog::attach_tgas).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dr1Entry {
    pub core: GaiaCore,
    pub astrometric_extra: AstrometricExtra,
    pub scan_direction: ScanDirection,
    /// Hipparcos / Tycho-2 cross-ids when this source is in the TGAS subset.
    pub tgas: Option<TgasBlock>,
}

/// DR1-specific astrometric quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstrometricExtra {
    pub astrometric_n_obs_ac: Option<u32>,
    pub astrometric_n_good_obs_al: Option<u32>,
    pub astrometric_n_good_obs_ac: Option<u32>,
    pub astrometric_n_bad_obs_al: Option<u32>,
    pub astrometric_n_bad_obs_ac: Option<u32>,
    pub astrometric_delta_q: Option<f32>,
    pub astrometric_relegation_factor: Option<f32>,
    pub astrometric_weight_al: Option<f32>,
    pub astrometric_weight_ac: Option<f32>,
    pub astrometric_priors_used: Option<u32>,
}

/// Per-axis scan direction statistics (k=1..4). Published in DR1 only.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScanDirection {
    pub strength_k1: Option<f32>,
    pub strength_k2: Option<f32>,
    pub strength_k3: Option<f32>,
    pub strength_k4: Option<f32>,
    pub mean_k1: Option<f32>,
    pub mean_k2: Option<f32>,
    pub mean_k3: Option<f32>,
    pub mean_k4: Option<f32>,
}

/// Hipparcos / Tycho-2 cross-identifiers provided for the TGAS subset.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TgasBlock {
    pub hip: Option<u32>,
    pub tycho2_id: Option<String>,
}

impl GaiaSource for Dr1Entry {
    fn core(&self) -> &GaiaCore {
        &self.core
    }
    fn release(&self) -> Release {
        Release::Dr1
    }
}
