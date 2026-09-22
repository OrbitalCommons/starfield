//! CAOM (Common Archive Observation Model) cone-search.
//!
//! Returns `Vec<CaomObservation>` for any region of sky. The HST-only
//! convenience method post-filters by `obs_collection == "HST"` so the
//! caller doesn't have to pull JWST / GALEX / TESS / IUE / etc. just to
//! find Hubble pointings.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use starfield_catalogs::gaia::Cone;
use starfield_core::Result;

use crate::mast::client::MastClient;

/// One row of the CAOM observation table. Many of the ~30 standard CAOM
/// columns are optional in practice (per-collection differences); the
/// fields we model directly cover the common-case Hubble use; everything
/// else is preserved in `extras` so callers can dig into less-common
/// columns (proposal_pi, target_name, dataRights, etc.) without us
/// having to commit to the full schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaomObservation {
    /// Public observation id (`obs_id`). Use this to fetch data
    /// products via [`MastClient::data_products`](crate::mast::MastClient::data_products).
    #[serde(rename = "obs_id")]
    pub obs_id: String,

    /// Internal MAST id (`obsid`) — required by
    /// [`MastClient::data_products`](crate::mast::MastClient::data_products).
    /// MAST returns this as a JSON number for HST/JWST and as a string
    /// for some other collections; we keep the raw `JsonValue` and
    /// expose [`Self::obsid_string`] for the common conversion.
    #[serde(rename = "obsid", default)]
    pub obsid_internal: Option<JsonValue>,

    /// Mission / archive grouping (`"HST"`, `"JWST"`, `"GALEX"`, etc.).
    #[serde(rename = "obs_collection")]
    pub obs_collection: String,

    /// Instrument (e.g. `"ACS/WFC"`, `"WFC3/UVIS"`).
    #[serde(rename = "instrument_name", default)]
    pub instrument_name: Option<String>,

    /// Filter combination (e.g. `"F814W"`, `"F606W;CLEAR2L"`).
    #[serde(rename = "filters", default)]
    pub filters: Option<String>,

    /// Centre RA in degrees (J2000).
    #[serde(rename = "s_ra", default)]
    pub s_ra: Option<f64>,
    /// Centre Dec in degrees (J2000).
    #[serde(rename = "s_dec", default)]
    pub s_dec: Option<f64>,

    /// Total exposure time, seconds.
    #[serde(rename = "t_exptime", default)]
    pub t_exptime: Option<f64>,
    /// Observation start, MJD.
    #[serde(rename = "t_min", default)]
    pub t_min: Option<f64>,
    /// Observation end, MJD.
    #[serde(rename = "t_max", default)]
    pub t_max: Option<f64>,

    /// Proposal id.
    #[serde(rename = "proposal_id", default)]
    pub proposal_id: Option<String>,
    /// Proposal PI.
    #[serde(rename = "proposal_pi", default)]
    pub proposal_pi: Option<String>,
    /// Target name as recorded by the proposal.
    #[serde(rename = "target_name", default)]
    pub target_name: Option<String>,

    /// Direct URL to the primary data product (a FITS file for HST
    /// imaging). Often `None` here; the products list returns more
    /// detail.
    #[serde(rename = "dataURL", default)]
    pub data_url: Option<String>,

    /// `"PUBLIC"` for unrestricted data, `"PROPRIETARY"` while still
    /// under a proprietary period.
    #[serde(rename = "dataRights", default)]
    pub data_rights: Option<String>,

    /// STC-S footprint string (polygon in world coordinates). Useful for
    /// downstream geometry but tedious to parse — preserved as a string
    /// and left to callers.
    #[serde(rename = "s_region", default)]
    pub s_region: Option<String>,

    /// Anything else MAST returned that we didn't model above.
    #[serde(flatten)]
    pub extras: BTreeMap<String, JsonValue>,
}

impl CaomObservation {
    /// String form of the internal numeric `obsid`, ready to pass to
    /// [`MastClient::data_products`](crate::mast::MastClient::data_products).
    pub fn obsid_string(&self) -> Option<String> {
        match self.obsid_internal.as_ref()? {
            JsonValue::Number(n) => Some(n.to_string()),
            JsonValue::String(s) => Some(s.clone()),
            _ => None,
        }
    }
}

/// Common values of `obs_collection`. These aren't exhaustive; CAOM
/// admits any collection string. Use [`Self::as_str`] to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsCollection {
    Hst,
    Jwst,
    Galex,
    Iue,
    Kepler,
    Tess,
}

impl ObsCollection {
    pub fn as_str(self) -> &'static str {
        match self {
            ObsCollection::Hst => "HST",
            ObsCollection::Jwst => "JWST",
            ObsCollection::Galex => "GALEX",
            ObsCollection::Iue => "IUE",
            ObsCollection::Kepler => "K2",
            ObsCollection::Tess => "TESS",
        }
    }
}

impl MastClient {
    /// Cone-search the CAOM observation table. Returns observations
    /// from every collection MAST knows about; combine with a client-
    /// side filter on `obs_collection` for a single mission.
    ///
    /// Cone radius is taken from the [`Cone`] in degrees (its native
    /// unit). MAST's `Mast.Caom.Cone` service expects degrees too, so
    /// no conversion is needed.
    pub fn observations_in_cone(&self, cone: &Cone) -> Result<Vec<CaomObservation>> {
        let params = json!({
            "ra": cone.ra_rad.to_degrees(),
            "dec": cone.dec_rad.to_degrees(),
            "radius": cone.radius_rad.to_degrees(),
        });
        let response = self.invoke::<CaomObservation>("Mast.Caom.Cone", params)?;
        if response.truncated() {
            log::warn!(
                "MAST cone returned a truncated page; total rows = {} but only {} fetched. \
                 Tighten the cone or implement pagination.",
                response.paging.as_ref().map(|p| p.rows_total).unwrap_or(0),
                response.data.len()
            );
        }
        Ok(response.data)
    }

    /// HST-only cone-search. Equivalent to
    /// [`Self::observations_in_cone`] post-filtered to
    /// `obs_collection == "HST"`. Server-side filtering would be
    /// fractionally cheaper but requires a more complex Mashup service
    /// (`Mast.Caom.Filtered.Position`); deferred to a follow-on if cone
    /// sizes get large.
    pub fn hst_observations_in_cone(&self, cone: &Cone) -> Result<Vec<CaomObservation>> {
        let mut all = self.observations_in_cone(cone)?;
        all.retain(|o| o.obs_collection == ObsCollection::Hst.as_str());
        Ok(all)
    }
}
