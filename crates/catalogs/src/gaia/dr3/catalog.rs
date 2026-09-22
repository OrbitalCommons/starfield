//! DR3 catalog newtype — thin wrapper over [`MemoryResidentCatalog`].

use std::path::{Path, PathBuf};

use starfield_core::Result;

use crate::gaia::common::catalog::{GaiaCatalog, MemoryResidentCatalog};
use crate::gaia::common::cone::Cone;
use crate::gaia::common::lazy::LazyLoadingCatalog;
use crate::gaia::download::Downloader;
use crate::gaia::dr3::entry::Dr3Entry;
use crate::gaia::dr3::schema::Dr3;

/// In-memory Gaia DR3 catalog, keyed by `source_id`.
#[derive(Debug)]
pub struct Dr3Catalog(pub MemoryResidentCatalog<Dr3>);

impl Dr3Catalog {
    pub fn new() -> Self {
        Self(MemoryResidentCatalog::new())
    }

    /// Load a single `.csv` or `.csv.gz` file; discards stars fainter than `mag_limit`.
    pub fn from_csv_file(path: impl AsRef<Path>, mag_limit: f64) -> Result<Self> {
        Ok(Self(MemoryResidentCatalog::<Dr3>::from_csv_file(
            path, mag_limit,
        )?))
    }

    /// Load every DR3 entry intersecting `cone` from a HEALPix-sharded
    /// excerpt directory (one produced by
    /// `gaia-excerpt --shard-by healpix`).
    ///
    /// Convenience over [`LazyLoadingCatalog::open`] +
    /// [`GaiaCatalog::materialize_cone`]; reach for the lazy variant
    /// directly if you want streaming access or plan to issue more than
    /// one cone query against the same directory.
    pub fn from_excerpt_dir_for_cone(
        excerpt_dir: impl AsRef<Path>,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Self> {
        let lazy = LazyLoadingCatalog::<Dr3>::open(excerpt_dir)?;
        Ok(Self(lazy.materialize_cone(cone, mag_limit)?))
    }

    pub fn inner(&self) -> &MemoryResidentCatalog<Dr3> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut MemoryResidentCatalog<Dr3> {
        &mut self.0
    }

    pub fn into_inner(self) -> MemoryResidentCatalog<Dr3> {
        self.0
    }

    pub fn insert(&mut self, entry: Dr3Entry) {
        self.0.insert(entry);
    }

    pub fn merge(&mut self, other: Self) {
        self.0.merge(other.0);
    }

    /// Insert the embedded Hipparcos-derived bright-star supplement into this
    /// catalog. Returns the number of supplement entries actually added (rows
    /// with `phot_g_mean_mag <= mag_limit`).
    ///
    /// See [`crate::gaia::dr3::supplement`] for the data provenance, the synthetic
    /// `source_id` masking scheme, and the list of fields supplement entries
    /// **don't** populate (no BP/RP, no RUWE/IPD, no RV, no GSP-Phot, no
    /// classifications).
    ///
    /// `mag_limit` is applied per-row using the fitted Gaia G magnitude in
    /// the embedded CSV; pass `f64::INFINITY` to take everything. Calling
    /// this twice with the same `mag_limit` is idempotent — supplement
    /// `source_id`s collide with themselves and the underlying HashMap
    /// overwrites.
    pub fn augment_missing(&mut self, mag_limit: f64) -> Result<usize> {
        let rows = crate::gaia::dr3::supplement::parse_embedded_supplement()?;
        let mut added = 0;
        for row in &rows {
            if row.fitted_g_mag > mag_limit {
                continue;
            }
            self.0
                .insert(crate::gaia::dr3::supplement::supplement_row_to_entry(row));
            added += 1;
        }
        Ok(added)
    }
}

impl Default for Dr3Catalog {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for Dr3Catalog {
    type Target = MemoryResidentCatalog<Dr3>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GaiaCatalog<Dr3> for Dr3Catalog {
    fn entries_in_cone<'a>(
        &'a self,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<Dr3Entry>> + 'a>> {
        self.0.entries_in_cone(cone, mag_limit)
    }

    fn len(&self) -> u64 {
        <_ as GaiaCatalog<Dr3>>::len(&self.0)
    }
}

// -- Download conveniences --------------------------------------------------

/// Download a specific DR3 `GaiaSource_*.csv.gz` file from ESA's CDN.
pub fn download_file(filename: &str) -> Result<PathBuf> {
    Downloader::<Dr3>::download_file(filename)
}

/// Download every DR3 `GaiaSource_*.csv.gz` (optionally capped at `max_files`).
pub fn download_all(max_files: Option<usize>) -> Result<Vec<PathBuf>> {
    Downloader::<Dr3>::download_all(max_files)
}

/// Cached DR3 files on disk, if any.
pub fn list_cached() -> Result<Vec<PathBuf>> {
    Downloader::<Dr3>::list_cached()
}
