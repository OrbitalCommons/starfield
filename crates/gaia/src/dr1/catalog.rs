//! DR1 catalog newtype + optional TGAS cross-id loader.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::MultiGzDecoder;
use starfield_core::{Result, StarfieldError};

use crate::common::catalog::{GaiaCatalog, MemoryResidentCatalog};
use crate::common::cone::Cone;
use crate::common::lazy::LazyLoadingCatalog;
use crate::download::Downloader;
use crate::dr1::entry::{Dr1Entry, TgasBlock};
use crate::dr1::schema::Dr1;

/// In-memory Gaia DR1 catalog, keyed by `source_id`.
#[derive(Debug)]
pub struct Dr1Catalog(pub MemoryResidentCatalog<Dr1>);

impl Dr1Catalog {
    pub fn new() -> Self {
        Self(MemoryResidentCatalog::new())
    }

    /// Load a DR1 `GaiaSource_*.csv.gz` file.
    pub fn from_csv_file(path: impl AsRef<Path>, mag_limit: f64) -> Result<Self> {
        Ok(Self(MemoryResidentCatalog::<Dr1>::from_csv_file(
            path, mag_limit,
        )?))
    }

    /// Load every DR1 entry intersecting `cone` from a HEALPix-sharded
    /// excerpt directory. Convenience over [`LazyLoadingCatalog::open`] +
    /// [`GaiaCatalog::materialize_cone`].
    pub fn from_excerpt_dir_for_cone(
        excerpt_dir: impl AsRef<Path>,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Self> {
        let lazy = LazyLoadingCatalog::<Dr1>::open(excerpt_dir)?;
        Ok(Self(lazy.materialize_cone(cone, mag_limit)?))
    }

    pub fn inner(&self) -> &MemoryResidentCatalog<Dr1> {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut MemoryResidentCatalog<Dr1> {
        &mut self.0
    }

    pub fn into_inner(self) -> MemoryResidentCatalog<Dr1> {
        self.0
    }

    pub fn insert(&mut self, entry: Dr1Entry) {
        self.0.insert(entry);
    }

    pub fn merge(&mut self, other: Self) {
        self.0.merge(other.0);
    }

    /// Splice Hipparcos / Tycho-2 cross-ids from a TGAS map into matching sources.
    /// Entries whose `source_id` doesn't appear in the map are left untouched.
    pub fn attach_tgas(&mut self, tgas: &HashMap<u64, TgasBlock>) {
        for entry in self.0.stars_iter_mut() {
            if let Some(block) = tgas.get(&entry.core.source_id) {
                entry.tgas = Some(block.clone());
            }
        }
    }

    /// Insert the embedded Hipparcos-derived bright-star supplement into this
    /// catalog. Returns the number of supplement entries actually added (rows
    /// with `phot_g_mean_mag <= mag_limit`).
    ///
    /// See [`crate::dr1::supplement`] for the data provenance, the synthetic
    /// `source_id` masking scheme, and the list of fields supplement entries
    /// **don't** populate. DR1 specifically lacks BP/RP, so the supplement
    /// fills only the astrometric core; the `tgas` block is `None` even
    /// though the original Hipparcos cross-id is recoverable via
    /// [`crate::dr1::supplement::decode_supplement_hip`].
    pub fn augment_missing(&mut self, mag_limit: f64) -> Result<usize> {
        let rows = crate::dr1::supplement::parse_embedded_supplement()?;
        let mut added = 0;
        for row in &rows {
            if row.fitted_g_mag > mag_limit {
                continue;
            }
            self.0
                .insert(crate::dr1::supplement::supplement_row_to_entry(row));
            added += 1;
        }
        Ok(added)
    }
}

impl Default for Dr1Catalog {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for Dr1Catalog {
    type Target = MemoryResidentCatalog<Dr1>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl GaiaCatalog<Dr1> for Dr1Catalog {
    fn entries_in_cone<'a>(
        &'a self,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<Dr1Entry>> + 'a>> {
        self.0.entries_in_cone(cone, mag_limit)
    }

    fn len(&self) -> u64 {
        <_ as GaiaCatalog<Dr1>>::len(&self.0)
    }
}

pub fn download_file(filename: &str) -> Result<PathBuf> {
    Downloader::<Dr1>::download_file(filename)
}

pub fn download_all(max_files: Option<usize>) -> Result<Vec<PathBuf>> {
    Downloader::<Dr1>::download_all(max_files)
}

pub fn list_cached() -> Result<Vec<PathBuf>> {
    Downloader::<Dr1>::list_cached()
}

/// Load a DR1 `TgasSource_*.csv.gz` and return a `source_id` → `TgasBlock` map.
///
/// TGAS files share almost every column with gaia_source but additionally publish
/// `hip` and `tycho2_id` at the start of each row. We parse only those three columns
/// (plus `source_id`) via plain CSV parsing since the values are small and the file
/// structure is uniform.
pub fn load_tgas_block_map(path: impl AsRef<Path>) -> Result<HashMap<u64, TgasBlock>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(StarfieldError::IoError)?;
    let reader: Box<dyn BufRead> = if path.extension().is_some_and(|e| e == "gz") {
        Box::new(BufReader::new(MultiGzDecoder::new(BufReader::new(file))))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| StarfieldError::DataError(format!("empty tgas file: {}", path.display())))?
        .map_err(StarfieldError::IoError)?;
    let cols: Vec<&str> = header.trim_end().split(',').collect();
    let idx = |name: &str| {
        cols.iter()
            .position(|c| *c == name)
            .ok_or_else(|| StarfieldError::DataError(format!("tgas missing column {}", name)))
    };
    let hip_idx = idx("hip")?;
    let tycho_idx = idx("tycho2_id")?;
    let source_id_idx = idx("source_id")?;

    let mut map = HashMap::new();
    for line in lines {
        let line = line.map_err(StarfieldError::IoError)?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.trim_end().split(',').collect();
        if fields.len() <= source_id_idx {
            continue;
        }
        let source_id = match fields[source_id_idx].parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hip = fields.get(hip_idx).and_then(|s| {
            if s.is_empty() {
                None
            } else {
                s.parse::<u32>().ok()
            }
        });
        let tycho2_id = fields.get(tycho_idx).and_then(|s| {
            if s.is_empty() {
                None
            } else {
                Some(s.to_string())
            }
        });
        map.insert(source_id, TgasBlock { hip, tycho2_id });
    }
    Ok(map)
}
