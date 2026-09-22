//! DR2 catalog newtype.

use std::path::{Path, PathBuf};

use starfield_core::Result;

use crate::gaia::common::catalog::{GaiaCatalog, MemoryResidentCatalog};
use crate::gaia::common::cone::Cone;
use crate::gaia::common::lazy::LazyLoadingCatalog;
use crate::gaia::download::Downloader;
use crate::gaia::dr2::entry::Dr2Entry;
use crate::gaia::dr2::schema::Dr2;

#[derive(Debug)]
pub struct Dr2Catalog(pub MemoryResidentCatalog<Dr2>);

impl Dr2Catalog {
    pub fn new() -> Self {
        Self(MemoryResidentCatalog::new())
    }

    pub fn from_csv_file(path: impl AsRef<Path>, mag_limit: f64) -> Result<Self> {
        Ok(Self(MemoryResidentCatalog::<Dr2>::from_csv_file(
            path, mag_limit,
        )?))
    }

    /// Load every DR2 entry intersecting `cone` from a HEALPix-sharded
    /// excerpt directory. Convenience over [`LazyLoadingCatalog::open`] +
    /// [`GaiaCatalog::materialize_cone`].
    pub fn from_excerpt_dir_for_cone(
        excerpt_dir: impl AsRef<Path>,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Self> {
        let lazy = LazyLoadingCatalog::<Dr2>::open(excerpt_dir)?;
        Ok(Self(lazy.materialize_cone(cone, mag_limit)?))
    }

    pub fn inner(&self) -> &MemoryResidentCatalog<Dr2> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut MemoryResidentCatalog<Dr2> {
        &mut self.0
    }

    pub fn into_inner(self) -> MemoryResidentCatalog<Dr2> {
        self.0
    }

    pub fn insert(&mut self, entry: Dr2Entry) {
        self.0.insert(entry);
    }

    pub fn merge(&mut self, other: Self) {
        self.0.merge(other.0);
    }

    /// Insert the embedded Hipparcos-derived bright-star supplement into this
    /// catalog. Returns the number of supplement entries actually added (rows
    /// with `phot_g_mean_mag <= mag_limit`).
    ///
    /// See [`crate::gaia::dr2::supplement`] for the data provenance, the synthetic
    /// `source_id` masking scheme, and the list of fields supplement entries
    /// **don't** populate (no BP/RP, no RV, no astrophysical params, etc.).
    pub fn augment_missing(&mut self, mag_limit: f64) -> Result<usize> {
        let rows = crate::gaia::dr2::supplement::parse_embedded_supplement()?;
        let mut added = 0;
        for row in &rows {
            if row.fitted_g_mag > mag_limit {
                continue;
            }
            self.0
                .insert(crate::gaia::dr2::supplement::supplement_row_to_entry(row));
            added += 1;
        }
        Ok(added)
    }
}

impl Default for Dr2Catalog {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for Dr2Catalog {
    type Target = MemoryResidentCatalog<Dr2>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GaiaCatalog<Dr2> for Dr2Catalog {
    fn entries_in_cone<'a>(
        &'a self,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<Dr2Entry>> + 'a>> {
        self.0.entries_in_cone(cone, mag_limit)
    }

    fn len(&self) -> u64 {
        <_ as GaiaCatalog<Dr2>>::len(&self.0)
    }
}

pub fn download_file(filename: &str) -> Result<PathBuf> {
    Downloader::<Dr2>::download_file(filename)
}

pub fn download_all(max_files: Option<usize>) -> Result<Vec<PathBuf>> {
    Downloader::<Dr2>::download_all(max_files)
}

pub fn list_cached() -> Result<Vec<PathBuf>> {
    Downloader::<Dr2>::list_cached()
}
