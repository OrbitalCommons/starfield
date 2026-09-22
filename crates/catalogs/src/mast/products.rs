//! Per-observation data-product listing + FITS download.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as JsonValue};
use starfield_core::data::source_utils::datastore::{
    Artifact, ArtifactKey, ContentCheck, Datastore, Source,
};
use starfield_core::data::source_utils::{artifact_from_url, datastore_error, resolve_artifact};
use starfield_core::{Result, StarfieldError};

use crate::mast::client::MastClient;

/// One artifact produced by an observation. CAOM groups artifacts by
/// `productType` (`SCIENCE`, `PREVIEW`, `INFO`, `AUXILIARY`,
/// `THUMBNAIL`); for HST the headline science product is typically a
/// `_drz.fits` (drizzled mosaic) or `_flt.fits` (flatfielded per-
/// exposure).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProduct {
    /// Bare filename, e.g. `j8c0a1011_drz.fits`.
    #[serde(rename = "productFilename")]
    pub filename: String,

    /// CAOM productType. Use [`Self::product_type`] for the parsed
    /// enum.
    #[serde(rename = "productType", default)]
    pub product_type: Option<String>,

    /// MIME type / file class string (`image/fits`, etc.).
    #[serde(rename = "productSubGroupDescription", default)]
    pub product_subgroup: Option<String>,

    /// Public download URL. Often a `mast:` URI; we resolve it via
    /// [`MastClient::download_product`].
    #[serde(rename = "dataURI", default)]
    pub data_uri: Option<String>,

    /// Direct HTTPS download URL (when MAST publishes one). Preferred
    /// over `data_uri` when present.
    #[serde(rename = "dataURL", default)]
    pub data_url: Option<String>,

    /// File size in bytes (when MAST reports it).
    #[serde(rename = "size", default)]
    pub size_bytes: Option<u64>,

    /// Parent observation id this product belongs to.
    #[serde(rename = "obs_id", default)]
    pub obs_id: Option<String>,

    /// Description text MAST attaches to the product (rarely useful).
    #[serde(rename = "description", default)]
    pub description: Option<String>,

    /// Anything else MAST returned that we didn't model above.
    #[serde(flatten)]
    pub extras: BTreeMap<String, JsonValue>,
}

/// Parsed CAOM productType. Use [`DataProduct::product_type`] to get
/// the enum form.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProductType {
    Science,
    Preview,
    Info,
    Auxiliary,
    Thumbnail,
    Other,
}

impl ProductType {
    /// Parse a CAOM `productType` string. Unknown values map to
    /// [`ProductType::Other`]. Named `parse_caom` (rather than `from_str`)
    /// to avoid shadowing the `std::str::FromStr` trait — implementing
    /// the trait would force a `Result` return that no caller wants.
    pub fn parse_caom(s: &str) -> Self {
        match s {
            "SCIENCE" => Self::Science,
            "PREVIEW" => Self::Preview,
            "INFO" => Self::Info,
            "AUXILIARY" => Self::Auxiliary,
            "THUMBNAIL" => Self::Thumbnail,
            _ => Self::Other,
        }
    }
}

impl DataProduct {
    /// Stable archive artifact identity. MAST URIs distinguish observations even
    /// when two products have the same filename. URL-only products use a digest
    /// of the complete URL, so query-selected products cannot collide.
    pub fn artifact(&self) -> Result<Artifact> {
        if self.filename.is_empty()
            || self.filename.contains(['/', '\\'])
            || self.filename == "."
            || self.filename == ".."
        {
            return Err(StarfieldError::DataError(
                "MAST product filename must be a bare filename".into(),
            ));
        }
        let url = self
            .resolve_url()
            .ok_or_else(|| StarfieldError::DataError("MAST product has no download URL".into()))?;
        let mut artifact = artifact_from_url(&url)?;
        if let Some(uri) = self
            .data_uri
            .as_deref()
            .and_then(|uri| uri.strip_prefix("mast:"))
        {
            // Keep opaque URIs outside the key alphabet under the URL digest;
            // replacing punctuation would let two different products collide.
            if let Ok(key) = ArtifactKey::new(format!("mast/{uri}")) {
                artifact.key = key;
            }
        }
        artifact.sources = vec![Source::new(url)];
        artifact.check = if self.filename.to_ascii_lowercase().ends_with(".fits") {
            ContentCheck::All(vec![
                ContentCheck::default_binary(),
                ContentCheck::magic(vec![b"SIMPLE  =".to_vec()], false),
            ])
        } else {
            ContentCheck::default_binary()
        };
        artifact.expected_bytes = self.size_bytes.filter(|bytes| *bytes > 0);
        artifact.provenance.description = self
            .description
            .clone()
            .unwrap_or_else(|| format!("MAST {}", self.filename));
        // Product access/redistribution can vary; an operator must populate the
        // license before adding this dynamic artifact to a mirror manifest.
        Ok(artifact)
    }

    pub fn product_type_enum(&self) -> ProductType {
        ProductType::parse_caom(self.product_type.as_deref().unwrap_or(""))
    }

    /// True for `productType=SCIENCE` *and* a `.fits` filename.
    pub fn is_science_fits(&self) -> bool {
        self.product_type_enum() == ProductType::Science
            && self.filename.to_ascii_lowercase().ends_with(".fits")
    }

    /// Resolve the best available download URL: prefer the explicit
    /// `dataURL`, then translate `mast:` URIs to the standard MAST
    /// download endpoint.
    pub fn resolve_url(&self) -> Option<String> {
        if let Some(url) = self.data_url.as_ref() {
            return Some(url.clone());
        }
        if let Some(uri) = self.data_uri.as_ref() {
            if uri.starts_with("mast:") {
                let mut url = reqwest::Url::parse("https://mast.stsci.edu/api/v0.1/Download/file")
                    .expect("constant MAST endpoint");
                url.query_pairs_mut().append_pair("uri", uri);
                return Some(url.into());
            }
            return Some(uri.clone());
        }
        None
    }
}

impl MastClient {
    /// List the data products for a given observation. Backed by the
    /// `Mast.Caom.Products` Mashup service.
    ///
    /// `obsid` is the **internal numeric** id (the `obsid` column in
    /// CAOM, not the human-readable `obs_id`). Use
    /// [`CaomObservation::obsid_string`] to extract it from a row, or
    /// pass the numeric value directly.
    pub fn data_products(&self, obsid: &str) -> Result<Vec<DataProduct>> {
        let params = json!({ "obsid": obsid });
        let response = self.invoke::<DataProduct>("Mast.Caom.Products", params)?;
        Ok(response.data)
    }

    /// Resolve `product` through the configured datastore. Valid legacy files
    /// from the per-crate cache are adopted before resolution.
    ///
    /// Errors out when the product has no resolvable download URL.
    pub fn download_product(&self, product: &DataProduct) -> Result<PathBuf> {
        let artifact = product.artifact()?;
        let cache = self.cache_dir()?;
        let dest = cache.join(&product.filename);
        resolve_artifact(&artifact, Some(&dest))
    }

    /// Resolve against only `store`, without reading the global legacy cache.
    pub fn download_product_with(
        &self,
        store: &Datastore,
        product: &DataProduct,
    ) -> Result<PathBuf> {
        store.get(&product.artifact()?).map_err(datastore_error)
    }
}
