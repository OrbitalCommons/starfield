//! An MPCORB catalog in memory, screened for propagatable orbits
//!
//! [`MpcorbCatalog`] turns MPCORB.DAT rows into [`MinorPlanet`]s — each a
//! parsed [`MpcOrbRecord`] paired with its heliocentric ICRF
//! [`KeplerOrbit`] — and drops the rows that cannot be propagated: an
//! unparseable packed epoch, or elements that describe no bound two-body
//! orbit (`e ≥ 1` with `a > 0`, `a ≤ 0`, a NaN field), screened with
//! [`KeplerOrbit::is_finite`]. Every orbit the catalog holds therefore
//! propagates to a finite state.
//!
//! The full catalog is about 1.5 million rows (~300 MB). With the
//! `parallel` feature the rows are parsed and converted on the rayon thread
//! pool; without it, sequentially.
//!
//! # Where the file comes from
//!
//! [`MpcorbCatalog::load_default`] resolves the catalog through
//! `starfield-datastore` under the key [`MPCORB_ARTIFACT_KEY`], so it follows
//! the same `local cache -> STARFIELD_MIRROR -> upstream` order, offline
//! switch (`STARFIELD_OFFLINE`) and upstream opt-in
//! (`STARFIELD_ALLOW_UPSTREAM=1`) as every other starfield data file, and
//! concurrent loaders share one locked, atomically published copy. A file
//! already at [`legacy_cache_path`](MpcorbCatalog::legacy_cache_path) is
//! validated and adopted on first use.
//!
//! The MPC regenerates MPCORB.DAT daily, but the datastore treats the cached
//! copy as an immutable snapshot: it is used until it is replaced explicitly.
//! To take a newer snapshot, remove the key and resolve again, or import a
//! file you fetched yourself:
//!
//! ```no_run
//! use starfield_datastore::Datastore;
//! use starfield_jpl::mpc::catalog::mpcorb_artifact;
//!
//! let store = Datastore::from_env().unwrap();
//! let artifact = mpcorb_artifact().unwrap();
//! store.remove(&artifact.key).unwrap();
//! // ...then `MpcorbCatalog::load_default()`, or
//! // `store.import(&artifact, path)` with a locally fetched file.
//! ```
//!
//! A mirror fill needs the server's manifest to carry the same key; a
//! mirror that does not know `mpc/MPCORB/MPCORB.DAT` answers 404 and
//! resolution falls through to upstream (if allowed).
//!
//! ```no_run
//! use starfield_jpl::mpc::MpcorbCatalog;
//!
//! let catalog = MpcorbCatalog::load_default().unwrap();
//! let ceres = catalog.find("Ceres").unwrap();
//! println!("{} H={:?}", ceres.name(), ceres.h());
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use starfield_core::data::source_utils::{cache_dir, datastore_error, resolve_artifact};
use starfield_core::keplerlib::KeplerOrbit;
use starfield_core::positions::Position;
use starfield_core::time::{Time, Timescale};
use starfield_core::{Result, StarfieldError};

use starfield_datastore::{Artifact, ArtifactKey, ContentCheck, Datastore, Provenance, Source};

use crate::mpc::mpcorb::{parse_mpcorb_line, MpcOrbRecord};

/// Where the MPC publishes the full MPCORB catalog.
pub const MPCORB_URL: &str = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT";

/// Datastore key of the cached MPCORB snapshot. A mirror server must list
/// this key in its manifest to serve it.
pub const MPCORB_ARTIFACT_KEY: &str = "mpc/MPCORB/MPCORB.DAT";

/// The datastore artifact for MPCORB.DAT.
///
/// Its content check rejects HTML and short bodies, then requires at least
/// one row that parses as an MPCORB record anywhere in the file, so a header
/// longer than the datastore's prefix window does not matter. The check runs
/// on the complete staged file before it is published to the cache.
pub fn mpcorb_artifact() -> Result<Artifact> {
    let key = ArtifactKey::new(MPCORB_ARTIFACT_KEY).map_err(datastore_error)?;
    Ok(Artifact::new(key, vec![Source::new(MPCORB_URL)])
        .with_check(ContentCheck::All(vec![
            ContentCheck::NotHtml,
            ContentCheck::MinBytes(MPCORB_MIN_BYTES),
            ContentCheck::custom(Arc::new(|bytes: &[u8]| {
                if contains_mpcorb_row(bytes) {
                    Ok(())
                } else {
                    Err("no parseable MPCORB row".to_string())
                }
            })),
        ]))
        .with_provenance(Provenance {
            description: "MPCORB.DAT orbital elements of the minor planets".into(),
            license: "Minor Planet Center; see https://minorplanetcenter.net/iau/MPCORB.html"
                .into(),
            citation: None,
        }))
}

/// Shortest body accepted as MPCORB: one 160-byte record plus a newline.
const MPCORB_MIN_BYTES: u64 = 161;

fn contains_mpcorb_row(bytes: &[u8]) -> bool {
    bytes
        .split(|&b| b == b'\n')
        .filter_map(|line| std::str::from_utf8(line).ok())
        .any(|line| parse_mpcorb_line(line.trim_end_matches('\r')).is_some())
}

/// One MPCORB row with an orbit known to propagate.
#[derive(Debug, Clone)]
pub struct MinorPlanet {
    record: MpcOrbRecord,
    orbit: KeplerOrbit,
}

impl MinorPlanet {
    /// Pair a record with its orbit, or `None` when the packed epoch does
    /// not unpack or the elements give a non-finite state.
    pub fn from_record(record: MpcOrbRecord, ts: &Timescale) -> Option<Self> {
        let orbit = record.to_kepler_orbit(ts)?;
        if !orbit.is_finite() {
            return None;
        }
        Some(Self { record, orbit })
    }

    /// Packed MPC designation (`00001`, `K14A00A`, ...).
    pub fn designation(&self) -> &str {
        &self.record.designation
    }

    /// Readable designation (`(1) Ceres`, `2014 AA`).
    pub fn name(&self) -> &str {
        &self.record.readable_designation
    }

    /// Absolute magnitude H, if the catalog has one.
    pub fn h(&self) -> Option<f64> {
        self.record.h_magnitude
    }

    /// Slope parameter G, if the catalog has one.
    pub fn g(&self) -> Option<f64> {
        self.record.g_slope
    }

    /// Osculating epoch of the elements, TT Julian date.
    pub fn epoch_tt(&self) -> f64 {
        self.orbit.epoch_tt
    }

    /// The full catalog row (arc, RMS residual, observation counts, ...).
    pub fn record(&self) -> &MpcOrbRecord {
        &self.record
    }

    /// The heliocentric two-body orbit, ecliptic elements rotated to the ICRF.
    pub fn orbit(&self) -> &KeplerOrbit {
        &self.orbit
    }

    /// Heliocentric ICRF position (AU) and velocity (AU/day) at `time`.
    ///
    /// The vectors are relative to the Sun even though
    /// [`KeplerOrbit::at`] labels the [`Position`] `Barycentric`; add the
    /// Sun's barycentric state for a true barycentric position.
    pub fn heliocentric_at(&self, time: &Time) -> Position {
        self.orbit.at(time)
    }

    /// Whether `query` names this body: the packed designation
    /// (case-insensitive), the readable designation (case-insensitive), or
    /// the name part of a numbered readable designation, so `ceres` matches
    /// `(1) Ceres`.
    pub fn matches(&self, query: &str) -> bool {
        let query = query.trim();
        if query.is_empty() {
            return false;
        }
        if self.record.designation.eq_ignore_ascii_case(query)
            || self.record.readable_designation.eq_ignore_ascii_case(query)
        {
            return true;
        }
        match self.record.readable_designation.split_once(") ") {
            Some((number, name)) if number.starts_with('(') => name.eq_ignore_ascii_case(query),
            _ => false,
        }
    }
}

/// An MPCORB catalog in memory.
#[derive(Debug, Clone)]
pub struct MpcorbCatalog {
    bodies: Vec<MinorPlanet>,
    skipped_unusable: usize,
    path: Option<PathBuf>,
}

impl MpcorbCatalog {
    /// `~/.cache/starfield/mpcorb/MPCORB.DAT`, the flat-file location some
    /// consumers populated before the datastore. [`load_default`](Self::load_default)
    /// validates and adopts a file found here.
    pub fn legacy_cache_path() -> PathBuf {
        cache_dir().join("mpcorb").join("MPCORB.DAT")
    }

    /// Resolve MPCORB.DAT through the environment-configured datastore,
    /// adopting a valid file at [`legacy_cache_path`](Self::legacy_cache_path),
    /// and load it. See the [module documentation](self) for the offline,
    /// mirror and refresh policy.
    pub fn load_default() -> Result<Self> {
        let path = resolve_artifact(&mpcorb_artifact()?, Some(&Self::legacy_cache_path()))?;
        Self::from_file(&path)
    }

    /// Resolve MPCORB.DAT through a caller-configured datastore and load it.
    /// Never consults the legacy cache path.
    pub fn load_from_store(store: &Datastore) -> Result<Self> {
        let path = store.get(&mpcorb_artifact()?).map_err(datastore_error)?;
        Self::from_file(&path)
    }

    /// Parse an MPCORB-format file: the full catalog or any subset of rows,
    /// with or without the header block.
    ///
    /// Fails if the file cannot be read or holds no usable record.
    pub fn from_file(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let text = String::from_utf8_lossy(&bytes);
        let mut catalog = Self::from_text(&text);
        if catalog.is_empty() {
            return Err(StarfieldError::DataError(format!(
                "{} contains no usable MPCORB records",
                path.display()
            )));
        }
        catalog.path = Some(path.to_path_buf());
        Ok(catalog)
    }

    /// Parse MPCORB-format text. Header, blank and malformed lines are
    /// ignored; rows without a propagatable orbit are counted in
    /// [`skipped_unusable`](Self::skipped_unusable).
    pub fn from_text(text: &str) -> Self {
        let ts = Timescale::default();
        let lines: Vec<&str> = text.lines().collect();
        #[cfg(feature = "parallel")]
        let parsed: Vec<Option<MinorPlanet>> = lines
            .par_iter()
            .filter_map(|line| parse_mpcorb_line(line))
            .map(|record| MinorPlanet::from_record(record, &ts))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let parsed: Vec<Option<MinorPlanet>> = lines
            .iter()
            .filter_map(|line| parse_mpcorb_line(line))
            .map(|record| MinorPlanet::from_record(record, &ts))
            .collect();
        Self::from_parsed(parsed)
    }

    /// A catalog from records already in hand, such as the output of
    /// [`MpcClient::fetch_mpcorb`](crate::mpc::MpcClient::fetch_mpcorb) or a hand-picked set.
    pub fn from_records(records: Vec<MpcOrbRecord>) -> Self {
        let ts = Timescale::default();
        let parsed = records
            .into_iter()
            .map(|record| MinorPlanet::from_record(record, &ts))
            .collect();
        Self::from_parsed(parsed)
    }

    fn from_parsed(parsed: Vec<Option<MinorPlanet>>) -> Self {
        let skipped_unusable = parsed.iter().filter(|b| b.is_none()).count();
        Self {
            bodies: parsed.into_iter().flatten().collect(),
            skipped_unusable,
            path: None,
        }
    }

    /// Number of bodies with usable orbits.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True when no body has a usable orbit.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Rows that parsed as MPCORB records but were dropped for an
    /// unparseable epoch or elements that describe no bound orbit.
    pub fn skipped_unusable(&self) -> usize {
        self.skipped_unusable
    }

    /// File the catalog was read from, if any.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// All bodies, in file order.
    pub fn bodies(&self) -> &[MinorPlanet] {
        &self.bodies
    }

    /// Iterate over all bodies.
    pub fn iter(&self) -> std::slice::Iter<'_, MinorPlanet> {
        self.bodies.iter()
    }

    /// First body matching `query` by packed designation, readable
    /// designation, or name (see [`MinorPlanet::matches`]).
    pub fn find(&self, query: &str) -> Option<&MinorPlanet> {
        self.bodies.iter().find(|b| b.matches(query))
    }

    /// Earliest and latest osculating epochs present, TT Julian dates.
    pub fn epoch_range_tt(&self) -> Option<(f64, f64)> {
        let mut epochs = self.bodies.iter().map(MinorPlanet::epoch_tt);
        let first = epochs.next()?;
        Some(epochs.fold((first, first), |(lo, hi), t| (lo.min(t), hi.max(t))))
    }
}

impl<'a> IntoIterator for &'a MpcorbCatalog {
    type Item = &'a MinorPlanet;
    type IntoIter = std::slice::Iter<'a, MinorPlanet>;

    fn into_iter(self) -> Self::IntoIter {
        self.bodies.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// (1) Ceres, epoch 2026-06-09 (K2669).
    const CERES: &str = "00001    3.34  0.15 K2669 274.41935   73.29420   80.24863   10.58803  0.0796923  0.21430445   2.7655526  0 MPO980521  7297 126 1801-2026 0.83 M-v 30k Veres      4000      (1) Ceres              20260103";
    const VESTA: &str = "00004    3.25  0.15 K25BL  26.80969  151.53711  103.70232    7.14406  0.0901676  0.27158812   2.3615413  0 MPO964264  7603 112 1821-2025 0.69 M-p 18k MPCORBFIT  4000      (4) Vesta              20250624";
    /// Vesta's row re-labelled with a provisional designation and no H.
    fn provisional() -> String {
        let mut row = VESTA.to_string();
        row.replace_range(0..7, "K14A00A");
        row.replace_range(8..13, "     ");
        row.replace_range(166..194, &format!("{:<28}", "2014 AA"));
        row
    }

    /// A row whose `e = 1.0` with `a > 0` describes no bound orbit.
    fn unbound() -> String {
        CERES.replacen("0.0796923", "1.0000000", 1)
    }

    /// A row whose packed epoch does not unpack.
    fn bad_epoch() -> String {
        CERES.replacen("K2669", "Z2669", 1)
    }

    #[test]
    fn parses_text_with_header_and_screens_unusable_rows() {
        let text = format!(
            "MINOR PLANET CENTER ORBIT DATABASE (MPCORB)\n\n{}\n{}\n{}\n{}\n{}\n",
            "-".repeat(160),
            CERES,
            unbound(),
            bad_epoch(),
            VESTA
        );
        let catalog = MpcorbCatalog::from_text(&text);
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.skipped_unusable(), 2);
        assert!(catalog.path().is_none());
        assert!(catalog.iter().all(|b| b.orbit().is_finite()));
    }

    #[test]
    fn finds_by_packed_readable_and_bare_name() {
        let catalog = MpcorbCatalog::from_text(&format!("{CERES}\n{VESTA}\n{}", provisional()));
        assert_eq!(catalog.find("Ceres").unwrap().designation(), "00001");
        assert_eq!(catalog.find("ceres").unwrap().designation(), "00001");
        assert_eq!(catalog.find("(4) Vesta").unwrap().designation(), "00004");
        assert_eq!(catalog.find("00004").unwrap().name(), "(4) Vesta");
        assert_eq!(catalog.find("k14a00a").unwrap().name(), "2014 AA");
        assert_eq!(catalog.find("2014 aa").unwrap().designation(), "K14A00A");
        assert!(catalog.find("Pallas").is_none());
        assert!(catalog.find("").is_none());
        assert!(catalog.find("4").is_none());
    }

    #[test]
    fn exposes_photometric_parameters_and_epoch() {
        let catalog = MpcorbCatalog::from_text(&format!("{CERES}\n{}", provisional()));
        let ceres = catalog.find("Ceres").unwrap();
        assert_abs_diff_eq!(ceres.h().unwrap(), 3.34, epsilon = 1e-12);
        assert_abs_diff_eq!(ceres.g().unwrap(), 0.15, epsilon = 1e-12);
        assert_abs_diff_eq!(ceres.epoch_tt(), 2_461_200.5, epsilon = 1e-6);
        assert_eq!(ceres.record().arc, "1801-2026");
        assert!(catalog.find("2014 AA").unwrap().h().is_none());
        let (lo, hi) = catalog.epoch_range_tt().unwrap();
        assert!(lo < hi);
        assert_abs_diff_eq!(hi, 2_461_200.5, epsilon = 1e-6);
    }

    #[test]
    fn heliocentric_state_is_in_the_belt() {
        let catalog = MpcorbCatalog::from_text(CERES);
        let ceres = &catalog.bodies()[0];
        let ts = Timescale::default();
        let state = ceres.heliocentric_at(&ts.tt_jd(ceres.epoch_tt() + 100.0, None));
        let r = state.position.norm();
        assert!((2.5..3.0).contains(&r), "r = {r} AU");
        // Circular speed at 2.77 AU is ~0.0103 AU/day.
        assert_abs_diff_eq!(state.velocity.norm(), 0.0103, epsilon = 0.001);
    }

    #[test]
    fn from_records_matches_from_text() {
        let records = vec![
            parse_mpcorb_line(CERES).unwrap(),
            parse_mpcorb_line(&unbound()).unwrap(),
        ];
        let catalog = MpcorbCatalog::from_records(records);
        assert_eq!(catalog.len(), 1);
        assert_eq!(catalog.skipped_unusable(), 1);
        assert_eq!((&catalog).into_iter().count(), 1);
    }

    #[test]
    fn from_file_records_path_and_rejects_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("MPCORB.DAT");
        std::fs::write(&good, format!("header\n{CERES}\n{VESTA}\n")).unwrap();
        let catalog = MpcorbCatalog::from_file(&good).unwrap();
        assert_eq!(catalog.len(), 2);
        assert_eq!(catalog.path(), Some(good.as_path()));

        let empty = dir.path().join("empty.dat");
        std::fs::write(&empty, format!("header only\n{}\n", unbound())).unwrap();
        assert!(MpcorbCatalog::from_file(&empty).is_err());
        assert!(MpcorbCatalog::from_file(&dir.path().join("missing.dat")).is_err());
    }

    #[test]
    fn legacy_cache_path_is_under_the_starfield_cache() {
        let path = MpcorbCatalog::legacy_cache_path();
        assert!(path.starts_with(cache_dir()));
        assert!(path.ends_with("mpcorb/MPCORB.DAT"));
    }

    fn offline_store(root: &Path) -> Datastore {
        Datastore::builder()
            .cache_root(root.to_path_buf())
            .without_mirror()
            .offline(true)
            .progress(false)
            .build()
            .unwrap()
    }

    fn write_mpcorb(dir: &Path, header_bytes: usize) -> PathBuf {
        let path = dir.join("MPCORB.DAT");
        let header = "MINOR PLANET CENTER ORBIT DATABASE (MPCORB)\n".repeat(header_bytes / 44 + 1);
        std::fs::write(&path, format!("{header}{CERES}\n{VESTA}\n")).unwrap();
        path
    }

    #[test]
    fn content_check_finds_a_row_after_a_long_header() {
        let check = mpcorb_artifact().unwrap().check;
        let dir = tempfile::tempdir().unwrap();
        // Well past the datastore's 8 KiB prefix window.
        let path = write_mpcorb(dir.path(), 64 * 1024);
        assert!(check.check_file(&path).is_ok());
    }

    #[test]
    fn content_check_rejects_html_and_header_only_files() {
        let check = mpcorb_artifact().unwrap().check;
        let html = format!("<!DOCTYPE html><html>{}</html>", " ".repeat(400));
        assert!(check.check(html.as_bytes()).is_err());
        let header_only = format!(
            "{}\n{}\n",
            "MPCORB header ".repeat(40),
            unbound().replace('.', "x")
        );
        assert!(check.check(header_only.as_bytes()).is_err());
        assert!(check.check(format!("{CERES}\n").as_bytes()).is_ok());
    }

    #[test]
    fn offline_store_without_the_snapshot_fails_without_network() {
        let root = tempfile::tempdir().unwrap();
        let store = offline_store(root.path());
        assert!(MpcorbCatalog::load_from_store(&store).is_err());
    }

    #[test]
    fn offline_store_loads_an_imported_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let store = offline_store(root.path());
        let file = write_mpcorb(src.path(), 0);
        store.import(&mpcorb_artifact().unwrap(), &file).unwrap();
        let catalog = MpcorbCatalog::load_from_store(&store).unwrap();
        assert_eq!(catalog.len(), 2);
        assert!(catalog.find("Ceres").is_some());
    }

    #[test]
    fn offline_store_refuses_to_import_an_invalid_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let store = offline_store(root.path());
        let bad = src.path().join("MPCORB.DAT");
        std::fs::write(&bad, "not an orbit file\n".repeat(20)).unwrap();
        assert!(store.import(&mpcorb_artifact().unwrap(), &bad).is_err());
        assert!(!store.contains(&mpcorb_artifact().unwrap().key));
    }

    #[test]
    fn concurrent_legacy_adoption_publishes_one_valid_snapshot() {
        let root = tempfile::tempdir().unwrap();
        let src = tempfile::tempdir().unwrap();
        let legacy = write_mpcorb(src.path(), 0);
        let artifact = mpcorb_artifact().unwrap();
        let paths: Vec<PathBuf> = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let (root, legacy, artifact) = (root.path(), &legacy, &artifact);
                    scope.spawn(move || {
                        let store = offline_store(root);
                        starfield_core::data::source_utils::adopt_legacy(&store, artifact, legacy)
                            .unwrap();
                        store.get(artifact).unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(paths.windows(2).all(|w| w[0] == w[1]));
        let catalog = MpcorbCatalog::from_file(&paths[0]).unwrap();
        assert_eq!(catalog.len(), 2);
    }

    /// Resolves the full catalog (~300 MB) through the environment's
    /// datastore; needs a warm cache, a mirror, or `STARFIELD_ALLOW_UPSTREAM=1`.
    #[test]
    #[ignore]
    fn live_full_catalog_loads() {
        let catalog = MpcorbCatalog::load_default().unwrap();
        assert!(catalog.len() > 1_000_000, "{} bodies", catalog.len());
        assert!(catalog.find("Ceres").is_some());
    }
}
