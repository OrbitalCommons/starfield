//! Generic in-memory catalog shared by all releases, plus the
//! [`GaiaCatalog`] trait that abstracts over storage backends.
//!
//! Two implementations live in the crate:
//!
//! - [`MemoryResidentCatalog<R>`] — the historical in-memory `HashMap` keyed
//!   by `source_id`. Cone queries scan the whole map and clone matching
//!   entries.
//! - [`LazyLoadingCatalog<R>`](crate::gaia::common::lazy::LazyLoadingCatalog) —
//!   reads only the HEALPix shards a cone touches, on every query, with no
//!   in-memory cache. Use when most of the excerpt directory is irrelevant
//!   to your queries (e.g. a cone covering 0.1% of the sky).

use std::collections::HashMap;
use std::path::Path;

use starfield_core::catalogs::{StarCatalog, StarData};
use starfield_core::Result;

use crate::gaia::common::cone::Cone;
use crate::gaia::common::reader::CsvSourceReader;
use crate::gaia::common::traits::{GaiaRelease, GaiaSource, Release};

/// Storage-agnostic cone-query interface. Implementors decide whether the
/// rows live in memory (cheap repeat queries, eager up-front load) or on
/// disk (no up-front cost, every query re-reads the relevant shards).
///
/// `entries_in_cone` is the only required method; `materialize_cone` drains
/// the iterator into a [`MemoryResidentCatalog`] for callers that want to
/// reuse the result.
pub trait GaiaCatalog<R: GaiaRelease> {
    /// Stream entries within `cone` whose `phot_g_mean_mag <= mag_limit`.
    /// The iterator yields `Result` per row so backends that read from disk
    /// can surface I/O errors mid-stream without panicking.
    fn entries_in_cone<'a>(
        &'a self,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<R::Entry>> + 'a>>;

    /// Total row count this catalog can serve, regardless of cone or
    /// magnitude. Lazy backends answer this from the on-disk manifest
    /// without touching shard files.
    ///
    /// Returns `u64` (not `usize`) so the value is consistent across
    /// 32- and 64-bit targets and matches the manifest's storage type;
    /// callers that also have `StarCatalog` in scope must disambiguate
    /// `cat.len()` via UFCS, e.g.
    /// `<_ as GaiaCatalog<Dr3>>::len(&cat)`.
    fn len(&self) -> u64;

    /// True when [`len`](Self::len) is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain the cone into a memory-resident catalog.
    fn materialize_cone(&self, cone: Cone, mag_limit: f64) -> Result<MemoryResidentCatalog<R>> {
        let mut catalog = MemoryResidentCatalog::with_mag_limit(mag_limit);
        for entry in self.entries_in_cone(cone, mag_limit)? {
            catalog.insert(entry?);
        }
        Ok(catalog)
    }
}

/// In-memory Gaia catalog keyed by `source_id`, parameterized over a release marker.
///
/// Per-release newtypes (`Dr1Catalog`, `Dr2Catalog`, `Dr3Catalog`) wrap this so users
/// don't have to write turbofish at call sites.
#[derive(Debug)]
pub struct MemoryResidentCatalog<R: GaiaRelease> {
    stars: HashMap<u64, R::Entry>,
    mag_limit: f64,
}

impl<R: GaiaRelease> MemoryResidentCatalog<R> {
    pub fn new() -> Self {
        Self::with_mag_limit(f64::INFINITY)
    }

    /// Build an empty catalog with a recorded `mag_limit`. The cutoff is
    /// metadata only — callers must still gate the entries they insert.
    pub fn with_mag_limit(mag_limit: f64) -> Self {
        Self {
            stars: HashMap::new(),
            mag_limit,
        }
    }

    /// Load a single `.csv` or `.csv.gz` file. Entries with `phot_g_mean_mag > mag_limit`
    /// are discarded as they stream past.
    pub fn from_csv_file(path: impl AsRef<Path>, mag_limit: f64) -> Result<Self> {
        let mut catalog = Self::with_mag_limit(mag_limit);
        let reader = CsvSourceReader::<R>::open(path, mag_limit)?;
        for entry in reader {
            let entry = entry?;
            let id = entry.core().source_id;
            catalog.stars.insert(id, entry);
        }
        Ok(catalog)
    }

    /// The magnitude cutoff used when populating this catalog.
    pub fn mag_limit(&self) -> f64 {
        self.mag_limit
    }

    /// Which Gaia release this catalog holds.
    pub fn release(&self) -> Release {
        R::RELEASE
    }

    /// Stars strictly brighter (smaller mag) than or equal to `magnitude`, by reference.
    pub fn brighter_than_ref(&self, magnitude: f64) -> Vec<&R::Entry> {
        self.stars
            .values()
            .filter(|e| e.core().phot_g_mean_mag <= magnitude)
            .collect()
    }

    /// Merge another catalog into this one; existing entries win on `source_id` collision.
    pub fn merge(&mut self, other: Self) {
        for (id, star) in other.stars {
            self.stars.entry(id).or_insert(star);
        }
        self.mag_limit = self.mag_limit.min(other.mag_limit);
    }

    /// Insert (mostly useful in tests and synthetic catalogs).
    pub fn insert(&mut self, entry: R::Entry) {
        self.stars.insert(entry.core().source_id, entry);
    }

    /// Mutable iterator over every entry. Used by per-release catalog helpers that
    /// splice in supplementary data (e.g. DR1 TGAS cross-ids).
    pub fn stars_iter_mut(&mut self) -> impl Iterator<Item = &mut R::Entry> {
        self.stars.values_mut()
    }
}

impl<R: GaiaRelease> Default for MemoryResidentCatalog<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: GaiaRelease> GaiaCatalog<R> for MemoryResidentCatalog<R> {
    fn entries_in_cone<'a>(
        &'a self,
        cone: Cone,
        mag_limit: f64,
    ) -> Result<Box<dyn Iterator<Item = Result<R::Entry>> + 'a>> {
        Ok(Box::new(self.stars.values().filter_map(move |e| {
            let c = e.core();
            if c.phot_g_mean_mag > mag_limit {
                return None;
            }
            if !cone.contains_unit_vec(&c.unit_vector()) {
                return None;
            }
            Some(Ok(e.clone()))
        })))
    }

    fn len(&self) -> u64 {
        self.stars.len() as u64
    }
}

// `MemoryResidentCatalog` already implements `StarCatalog::len(&self) -> usize`;
// `GaiaCatalog<R>::len(&self) -> u64` shadows the name. Inherent methods take
// precedence and don't exist for `len`, so direct `cat.len()` is ambiguous when
// both traits are in scope — callers disambiguate via UFCS, see the trait doc.

impl<R: GaiaRelease> StarCatalog for MemoryResidentCatalog<R> {
    type Star = R::Entry;

    fn get_star(&self, id: usize) -> Option<&Self::Star> {
        self.stars.get(&(id as u64))
    }

    fn stars(&self) -> impl Iterator<Item = &Self::Star> {
        self.stars.values()
    }

    fn len(&self) -> usize {
        self.stars.len()
    }

    fn filter<F>(&self, predicate: F) -> Vec<&Self::Star>
    where
        F: Fn(&Self::Star) -> bool,
    {
        self.stars.values().filter(|s| predicate(s)).collect()
    }

    fn star_data(&self) -> impl Iterator<Item = StarData> + '_ {
        self.stars.values().map(|e| {
            let c = e.core();
            StarData::new(c.source_id, c.ra, c.dec, c.phot_g_mean_mag, e.b_v())
        })
    }

    fn filter_star_data<F>(&self, predicate: F) -> Vec<StarData>
    where
        F: Fn(&StarData) -> bool,
    {
        self.star_data().filter(|s| predicate(s)).collect()
    }

    fn brighter_than(&self, magnitude: f64) -> Vec<StarData> {
        self.filter_star_data(|s| s.magnitude <= magnitude)
    }

    fn stars_in_field(&self, ra_deg: f64, dec_deg: f64, fov_deg: f64) -> Vec<StarData> {
        let center = unit_vec(ra_deg, dec_deg);
        let cos_fov = (fov_deg.to_radians() / 2.0).cos();
        self.filter_star_data(|s| unit_vec(s.ra_deg(), s.dec_deg()).dot(&center) >= cos_fov)
    }
}

fn unit_vec(ra_deg: f64, dec_deg: f64) -> nalgebra::Vector3<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    nalgebra::Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}
