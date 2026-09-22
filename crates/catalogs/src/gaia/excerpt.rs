//! Sharded, resumable, crash-safe excerpt writer.
//!
//! Reads any Gaia CSV(.gz) catalog, applies a caller-supplied predicate, and
//! writes the kept entries to one of N shard files chosen by a [`ShardKey`].
//! Each shard is a **multi-stream gzip** of the source release's CSV layout
//! (header line at file creation, then one fresh gz stream per input file
//! appended afterward). This means:
//!
//! - **Round-trip** through `Dr{1,2,3}Catalog::from_csv_file` works because
//!   the reader uses `MultiGzDecoder` and treats the header line as a
//!   one-time CSV header.
//! - **Crash safety** is built in: each input file's contribution is buffered
//!   in memory until [`ShardedCsvWriter::commit_file`] flushes everything,
//!   fsyncs, and atomically updates the on-disk manifest. Mid-extract crashes
//!   never leave partial rows in shards.
//! - **Resume** is automatic: [`ShardedCsvWriter::new_or_resume`] reads the
//!   manifest, truncates each shard to its last-committed byte length (cheap
//!   safety net in case fsync was incomplete), and reports which input files
//!   have already been processed so the caller can skip them.
//!
//! # Example (multi-file, resumable)
//!
//! ```no_run
//! use starfield_catalogs::gaia::excerpt::{ShardedCsvWriter, HealpixShard, excerpt_csv_file_into};
//! use starfield_catalogs::gaia::Dr3;
//!
//! let mut writer = ShardedCsvWriter::<Dr3, _>::new_or_resume(
//!     "out/", HealpixShard { level: 5 }, 20.0,
//! )?;
//! let already = writer.processed_files().clone();
//! for path in std::fs::read_dir("inputs/")? {
//!     let path = path?.path();
//!     let name = path.file_name().unwrap().to_string_lossy().into_owned();
//!     if already.contains(&name) { continue; }
//!     excerpt_csv_file_into::<Dr3, _, _>(&path, 20.0, &mut writer, |_| true)?;
//! }
//! let summary = writer.finish()?;
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use starfield_core::{Result, StarfieldError};

use crate::gaia::common::reader::CsvSourceReader;
use crate::gaia::common::traits::{GaiaRelease, GaiaSource};

const MANIFEST_FILE: &str = ".gaia-excerpt-manifest.json";
const MANIFEST_VERSION: u32 = 1;

/// JSON has no representation for `INFINITY` / `NaN`; encode "no magnitude
/// limit" as `None` and finite limits as `Some(value)`.
fn encode_mag_limit(m: f64) -> Option<f64> {
    if m.is_finite() {
        Some(m)
    } else {
        None
    }
}

// =====================================================================================
// Sharding trait + built-in embodiments
// =====================================================================================

/// Maps an entry to a shard index in `0..num_shards()`.
///
/// Built-in embodiments cover the common cases (hash-of-id, id-range banding,
/// HEALPix cell). Implement this trait directly to shard by anything else
/// (brightness, date, custom hash, etc.). User-defined sharders should
/// override [`describe`](Self::describe) so the manifest can validate that
/// a resume run is using the same partitioning function.
pub trait ShardKey<E> {
    /// Number of shards. Returned values from [`shard_of`](Self::shard_of) must lie in `0..num_shards()`.
    fn num_shards(&self) -> u32;

    /// The shard index for one entry. Must be `< num_shards()`.
    fn shard_of(&self, entry: &E) -> u32;

    /// A serializable summary stored in the manifest. On resume, the new run's
    /// description must equal the recorded one or the writer refuses to attach.
    /// Default returns `kind="custom"` with no sub-config — fine for ad-hoc
    /// closures, but means resume can't distinguish two different "custom"
    /// sharders. Override for production use.
    fn describe(&self) -> ShardKeyDescription {
        ShardKeyDescription {
            kind: "custom".into(),
            num_shards: self.num_shards(),
            healpix_level: None,
        }
    }
}

/// Persistable identity of a sharder, recorded in the manifest. `kind` is one
/// of `"hash"` / `"id-range"` / `"healpix"` / `"custom"`; `healpix_level` is
/// only set for the healpix sharder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardKeyDescription {
    pub kind: String,
    pub num_shards: u32,
    pub healpix_level: Option<u8>,
}

impl<E, S: ?Sized + ShardKey<E>> ShardKey<E> for Box<S> {
    fn num_shards(&self) -> u32 {
        (**self).num_shards()
    }
    fn shard_of(&self, entry: &E) -> u32 {
        (**self).shard_of(entry)
    }
    fn describe(&self) -> ShardKeyDescription {
        (**self).describe()
    }
}

/// Shard by `hash(source_id) % num_shards`. Roughly even distribution; loses
/// any spatial / brightness locality. Use when you want shards of similar
/// size for parallel processing.
#[derive(Debug, Clone, Copy)]
pub struct HashIdShard {
    pub num_shards: u32,
}

impl<E: GaiaSource> ShardKey<E> for HashIdShard {
    fn num_shards(&self) -> u32 {
        self.num_shards
    }
    fn shard_of(&self, entry: &E) -> u32 {
        // SplitMix64 (see `mix_id`) is deterministic across runs, so the same
        // source_id always lands in the same shard — important for incremental
        // / resumable workflows. `std::hash`-based hashers are randomized per
        // process and would make this non-reproducible.
        let mixed = mix_id(entry.core().source_id);
        (mixed % self.num_shards as u64) as u32
    }
    fn describe(&self) -> ShardKeyDescription {
        ShardKeyDescription {
            kind: "hash".into(),
            num_shards: self.num_shards,
            healpix_level: None,
        }
    }
}

fn mix_id(x: u64) -> u64 {
    // SplitMix64 finalizer — deterministic, well-distributed across input bits.
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Shard by source_id range — divide the `u64` ID space into `num_shards`
/// equal bands and slot each entry by its `source_id`. For DR2/DR3 source_ids
/// are HEALPix-12-prefixed, so this gives weak spatial coherence (entries in
/// the same band share a sky region). For DR1 the encoding differs and
/// locality is weaker.
#[derive(Debug, Clone, Copy)]
pub struct IdRangeShard {
    pub num_shards: u32,
}

impl<E: GaiaSource> ShardKey<E> for IdRangeShard {
    fn num_shards(&self) -> u32 {
        self.num_shards
    }
    fn shard_of(&self, entry: &E) -> u32 {
        let id = entry.core().source_id;
        let bucket_size = (u64::MAX / self.num_shards as u64).saturating_add(1);
        ((id / bucket_size) as u32).min(self.num_shards - 1)
    }
    fn describe(&self) -> ShardKeyDescription {
        ShardKeyDescription {
            kind: "id-range".into(),
            num_shards: self.num_shards,
            healpix_level: None,
        }
    }
}

/// Shard by HEALPix cell at the given depth — exactly one output file per
/// cell. Computes the pixel from the entry's RA/Dec via
/// [`cdshealpix::nested::hash`] and uses it as the shard index directly, so
/// `shard_NNNN.csv.gz` corresponds to HEALPix cell `NNNN` and is spatially
/// coherent: a cone-search loader can compute the cells the cone covers and
/// open only those files (instead of scanning all shards).
///
/// The number of shards is determined by `level`:
///
/// | Level | Cells (12·4ⁿ) | ~Cell side |
/// |-------|---------------|------------|
/// | 5     | 12,288        | 1.83°      |
/// | 6     | 49,152        | 0.92°      |
/// | 7     | 196,608       | 0.46°      |
///
/// File counts scale fast — keep `level` modest unless you really need fine
/// spatial resolution. Empty cells leave no file behind (the writer opens
/// shard files lazily on first row).
#[derive(Debug, Clone, Copy)]
pub struct HealpixShard {
    /// HEALPix depth (0..=29). Higher = finer pixels.
    pub level: u8,
}

impl HealpixShard {
    /// Number of HEALPix cells at this depth (`12 · 4^level`).
    pub const fn cell_count(level: u8) -> u32 {
        12u32 << (2 * level as u32)
    }
}

impl<E: GaiaSource> ShardKey<E> for HealpixShard {
    fn num_shards(&self) -> u32 {
        HealpixShard::cell_count(self.level)
    }
    fn shard_of(&self, entry: &E) -> u32 {
        let c = entry.core();
        cdshealpix::nested::hash(self.level, c.ra.to_radians(), c.dec.to_radians()) as u32
    }
    fn describe(&self) -> ShardKeyDescription {
        ShardKeyDescription {
            kind: "healpix".into(),
            num_shards: HealpixShard::cell_count(self.level),
            healpix_level: Some(self.level),
        }
    }
}

// =====================================================================================
// Manifest — durable record of per-shard sizes + processed input files
// =====================================================================================

/// On-disk manifest tracking per-shard byte length, row count, and the set
/// of input files whose rows are committed. Lives at
/// `<output_dir>/.gaia-excerpt-manifest.json` and is rewritten atomically
/// (tmp + fsync + rename) on every successful per-file commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub version: u32,
    pub release: String,
    /// `None` = no magnitude limit. We avoid storing `f64::INFINITY` directly
    /// because JSON has no representation for non-finite floats.
    pub mag_limit: Option<f64>,
    pub sharder: ShardKeyDescription,
    /// Byte length of each shard file as of the most recent successful commit.
    /// `truncate(file, shard_sizes[i])` discards anything written after.
    pub shard_sizes: Vec<u64>,
    /// Row count per shard, as of the most recent successful commit.
    pub shard_rows: Vec<u64>,
    pub processed_files: BTreeSet<String>,
    pub kept_rows: u64,
}

impl Manifest {
    fn fresh<R: GaiaRelease, S: ShardKey<R::Entry>>(sharder: &S, mag_limit: f64) -> Self {
        let n = sharder.num_shards() as usize;
        Self {
            version: MANIFEST_VERSION,
            release: format!("{:?}", R::RELEASE),
            mag_limit: encode_mag_limit(mag_limit),
            sharder: sharder.describe(),
            shard_sizes: vec![0; n],
            shard_rows: vec![0; n],
            processed_files: BTreeSet::new(),
            kept_rows: 0,
        }
    }

    fn validate_against<R: GaiaRelease, S: ShardKey<R::Entry>>(
        &self,
        sharder: &S,
        mag_limit: f64,
    ) -> Result<()> {
        let want_release = format!("{:?}", R::RELEASE);
        if self.release != want_release {
            return Err(StarfieldError::DataError(format!(
                "manifest release mismatch: on-disk={}, requested={}",
                self.release, want_release
            )));
        }
        let want = encode_mag_limit(mag_limit);
        if self.mag_limit != want {
            return Err(StarfieldError::DataError(format!(
                "manifest mag_limit mismatch: on-disk={:?}, requested={:?}",
                self.mag_limit, want
            )));
        }
        let want_desc = sharder.describe();
        if self.sharder != want_desc {
            return Err(StarfieldError::DataError(format!(
                "manifest sharder mismatch: on-disk={:?}, requested={:?}",
                self.sharder, want_desc
            )));
        }
        if self.shard_sizes.len() != sharder.num_shards() as usize {
            return Err(StarfieldError::DataError(format!(
                "manifest shard count mismatch: on-disk={}, requested={}",
                self.shard_sizes.len(),
                sharder.num_shards()
            )));
        }
        Ok(())
    }

    fn write_atomic(&self, dir: &Path) -> Result<()> {
        let final_path = dir.join(MANIFEST_FILE);
        let tmp_path = dir.join(format!("{}.tmp", MANIFEST_FILE));
        let json = serde_json::to_vec_pretty(self)
            .map_err(|e| StarfieldError::DataError(format!("manifest encode: {}", e)))?;
        let mut f = File::create(&tmp_path).map_err(StarfieldError::IoError)?;
        f.write_all(&json).map_err(StarfieldError::IoError)?;
        f.sync_all().map_err(StarfieldError::IoError)?;
        drop(f);
        fs::rename(&tmp_path, &final_path).map_err(StarfieldError::IoError)?;
        Ok(())
    }

    fn try_load(dir: &Path) -> Result<Option<Self>> {
        let p = dir.join(MANIFEST_FILE);
        if !p.exists() {
            return Ok(None);
        }
        let bytes = fs::read(&p).map_err(StarfieldError::IoError)?;
        let m: Manifest = serde_json::from_slice(&bytes)
            .map_err(|e| StarfieldError::DataError(format!("manifest decode: {}", e)))?;
        if m.version != MANIFEST_VERSION {
            return Err(StarfieldError::DataError(format!(
                "manifest version {} unsupported (expected {})",
                m.version, MANIFEST_VERSION
            )));
        }
        Ok(Some(m))
    }
}

// =====================================================================================
// Sharded gzipped CSV writer (resumable + crash-safe)
// =====================================================================================

/// Per-shard, multi-stream gzip CSV writer with manifest-tracked checkpoints.
///
/// Workflow per input file: [`begin_file`](Self::begin_file), [`write`](Self::write)
/// repeatedly, then [`commit_file`](Self::commit_file). All rows for one file are
/// buffered in memory between begin and commit; on commit the buffered rows are
/// gz-encoded as fresh streams (one per shard touched), appended to each shard
/// file, fsynced, and the manifest is atomically updated. A crash anywhere
/// before commit completes leaves the on-disk state unchanged from the previous
/// commit — re-running picks up by skipping `processed_files`.
pub struct ShardedCsvWriter<R: GaiaRelease, S: ShardKey<R::Entry>> {
    output_dir: PathBuf,
    shard_key: S,

    /// Per-shard final on-disk paths (always set; file may not yet exist).
    paths: Vec<PathBuf>,

    /// In-memory buffer of formatted CSV rows for the currently-open file,
    /// keyed by shard index.
    pending: Vec<Vec<String>>,

    /// On-disk manifest (in-sync with files between commits).
    manifest: Manifest,

    _marker: PhantomData<R>,
}

impl<R: GaiaRelease, S: ShardKey<R::Entry>> ShardedCsvWriter<R, S> {
    /// Open `output_dir`. If a manifest exists there:
    /// - validate it matches `sharder` / `mag_limit` / release
    /// - truncate each shard file to its recorded length (paranoia)
    /// - return a writer that knows which input files have already been processed
    ///
    /// If no manifest exists: create the dir + manifest fresh.
    pub fn new_or_resume(
        output_dir: impl AsRef<Path>,
        shard_key: S,
        mag_limit: f64,
    ) -> Result<Self> {
        let output_dir = output_dir.as_ref().to_path_buf();
        fs::create_dir_all(&output_dir).map_err(StarfieldError::IoError)?;
        let n = shard_key.num_shards();
        if n == 0 {
            return Err(StarfieldError::DataError(
                "ShardedCsvWriter: num_shards must be > 0".into(),
            ));
        }
        let width = digits_for(n);
        let paths: Vec<PathBuf> = (0..n)
            .map(|i| output_dir.join(format!("shard_{:0width$}.csv.gz", i, width = width)))
            .collect();

        let manifest = match Manifest::try_load(&output_dir)? {
            Some(m) => {
                m.validate_against::<R, S>(&shard_key, mag_limit)?;
                // Defensive truncate-to-recorded-length on every shard. If the
                // last commit's fsync was complete, this is a no-op. If a crash
                // left bytes after the recorded boundary, this discards them.
                for (i, path) in paths.iter().enumerate() {
                    if path.exists() {
                        let f = OpenOptions::new()
                            .write(true)
                            .open(path)
                            .map_err(StarfieldError::IoError)?;
                        f.set_len(m.shard_sizes[i])
                            .map_err(StarfieldError::IoError)?;
                        f.sync_all().map_err(StarfieldError::IoError)?;
                    }
                }
                m
            }
            None => {
                let m = Manifest::fresh::<R, S>(&shard_key, mag_limit);
                m.write_atomic(&output_dir)?;
                m
            }
        };

        Ok(Self {
            output_dir,
            shard_key,
            paths,
            pending: vec![Vec::new(); n as usize],
            manifest,
            _marker: PhantomData,
        })
    }

    /// Files whose rows are already committed to shards. CLI should skip these
    /// when iterating its input list.
    pub fn processed_files(&self) -> &BTreeSet<String> {
        &self.manifest.processed_files
    }

    /// Rows committed across all input files so far (sum across shards from
    /// the manifest). Persists across restarts.
    pub fn kept_so_far(&self) -> u64 {
        self.manifest.kept_rows
    }

    /// Begin processing one input file. Clears the in-memory pending buffer.
    pub fn begin_file(&mut self, _filename: &str) {
        for buf in &mut self.pending {
            buf.clear();
        }
    }

    /// Buffer a row in memory, routed to its shard. Does not touch disk.
    pub fn write(&mut self, entry: &R::Entry) -> Result<()> {
        let shard = self.shard_key.shard_of(entry);
        if shard >= self.pending.len() as u32 {
            return Err(StarfieldError::DataError(format!(
                "ShardKey returned {} for num_shards={}",
                shard,
                self.pending.len()
            )));
        }
        self.pending[shard as usize].push(R::format_csv_row(entry));
        Ok(())
    }

    /// Atomically commit the in-memory buffer for one input file:
    /// 1. For each shard with pending rows: open the shard file in append
    ///    mode, gz-encode the rows as a fresh stream, finish, fsync.
    ///    (If the file doesn't exist yet, write the CSV header line in its
    ///    own first gz stream before the row stream.)
    /// 2. Update manifest with new shard_sizes + processed_files += filename
    ///    + kept_rows. Atomically rewrite via `.tmp` + `rename`.
    /// 3. Clear the pending buffer.
    ///
    /// If any step errors out, the manifest is unchanged — the caller can
    /// retry the same input file safely (modulo any download caching they
    /// might do).
    pub fn commit_file(&mut self, filename: &str) -> Result<()> {
        if self.manifest.processed_files.contains(filename) {
            // Already done in a prior run / call. No-op.
            for buf in &mut self.pending {
                buf.clear();
            }
            return Ok(());
        }

        let mut new_sizes = self.manifest.shard_sizes.clone();
        let mut new_rows = self.manifest.shard_rows.clone();
        let mut added = 0u64;

        for (idx, rows) in self.pending.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            let path = &self.paths[idx];
            let need_header = !path.exists() || new_sizes[idx] == 0;
            let mut f = OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(StarfieldError::IoError)?;

            if need_header {
                let mut gz = GzEncoder::new(BufWriter::new(&mut f), Compression::default());
                writeln!(gz, "{}", R::csv_header()).map_err(StarfieldError::IoError)?;
                let buf = gz.finish().map_err(StarfieldError::IoError)?;
                buf.into_inner()
                    .map_err(|e| StarfieldError::IoError(e.into_error()))?
                    .flush()
                    .map_err(StarfieldError::IoError)?;
            }

            // One gz stream per file's contribution; concatenated into the file.
            let mut gz = GzEncoder::new(BufWriter::new(&mut f), Compression::default());
            for row in rows {
                writeln!(gz, "{}", row).map_err(StarfieldError::IoError)?;
            }
            let buf = gz.finish().map_err(StarfieldError::IoError)?;
            buf.into_inner()
                .map_err(|e| StarfieldError::IoError(e.into_error()))?
                .flush()
                .map_err(StarfieldError::IoError)?;

            f.sync_all().map_err(StarfieldError::IoError)?;
            new_sizes[idx] = f.metadata().map_err(StarfieldError::IoError)?.len();
            new_rows[idx] += rows.len() as u64;
            added += rows.len() as u64;
        }

        let mut new_manifest = self.manifest.clone();
        new_manifest.shard_sizes = new_sizes;
        new_manifest.shard_rows = new_rows;
        new_manifest.processed_files.insert(filename.to_string());
        new_manifest.kept_rows += added;
        new_manifest.write_atomic(&self.output_dir)?;
        self.manifest = new_manifest;

        for buf in &mut self.pending {
            buf.clear();
        }
        Ok(())
    }

    /// Path to the manifest file (callers who want to inspect / archive).
    pub fn manifest_path(&self) -> PathBuf {
        self.output_dir.join(MANIFEST_FILE)
    }

    /// Snapshot the current manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Finalize: returns a summary built from the on-disk manifest. The
    /// writer can also be dropped without calling this; the manifest is
    /// always up-to-date because every commit is atomic.
    pub fn finish(self) -> Result<ExcerptSummary> {
        let mut shard_files = Vec::with_capacity(self.paths.len());
        for (i, p) in self.paths.iter().enumerate() {
            if self.manifest.shard_sizes[i] > 0 {
                shard_files.push(Some(p.clone()));
            } else {
                shard_files.push(None);
            }
        }
        Ok(ExcerptSummary {
            input_rows: 0,
            kept_rows: self.manifest.kept_rows,
            per_shard_counts: self.manifest.shard_rows.clone(),
            shard_files,
        })
    }
}

fn digits_for(n: u32) -> usize {
    // Number of digits needed to represent `n - 1`, with a minimum of 4 so file
    // listings sort sensibly even for small shard counts.
    let max_idx = n.saturating_sub(1);
    let nd = if max_idx == 0 {
        1
    } else {
        (max_idx as f64).log10().floor() as usize + 1
    };
    nd.max(4)
}

// =====================================================================================
// Summary + convenience drive functions
// =====================================================================================

/// What the excerpt run produced. `shard_files[i]` is `Some(path)` iff shard `i`
/// received at least one entry.
#[derive(Debug, Clone)]
pub struct ExcerptSummary {
    pub input_rows: usize,
    pub kept_rows: u64,
    pub per_shard_counts: Vec<u64>,
    pub shard_files: Vec<Option<PathBuf>>,
}

impl ExcerptSummary {
    /// Paths of every shard file that was actually written (Some entries only).
    pub fn written_paths(&self) -> impl Iterator<Item = &PathBuf> {
        self.shard_files.iter().flatten()
    }
}

/// One-shot: open `output_dir` (resuming if a manifest exists), process
/// `input_path` if not already processed, return a summary.
pub fn excerpt_csv_file<R, S, F>(
    input_path: impl AsRef<Path>,
    mag_limit: f64,
    output_dir: impl AsRef<Path>,
    sharder: S,
    predicate: F,
) -> Result<ExcerptSummary>
where
    R: GaiaRelease,
    S: ShardKey<R::Entry>,
    F: FnMut(&R::Entry) -> bool,
{
    let path = input_path.as_ref();
    let filename = filename_of(path)?;
    let mut writer = ShardedCsvWriter::<R, S>::new_or_resume(output_dir, sharder, mag_limit)?;
    let input_rows = if writer.processed_files().contains(&filename) {
        0
    } else {
        excerpt_csv_file_into::<R, S, F>(path, mag_limit, &mut writer, predicate)?
    };
    let mut summary = writer.finish()?;
    summary.input_rows = input_rows;
    Ok(summary)
}

/// One-shot variant that reads from any byte source. Caller supplies
/// `filename` (used as the manifest key for skip-on-resume).
pub fn excerpt_csv_reader<R, S, F>(
    reader: Box<dyn std::io::Read>,
    is_gz: bool,
    mag_limit: f64,
    output_dir: impl AsRef<Path>,
    sharder: S,
    filename: &str,
    predicate: F,
) -> Result<ExcerptSummary>
where
    R: GaiaRelease,
    S: ShardKey<R::Entry>,
    F: FnMut(&R::Entry) -> bool,
{
    let mut writer = ShardedCsvWriter::<R, S>::new_or_resume(output_dir, sharder, mag_limit)?;
    let input_rows = if writer.processed_files().contains(filename) {
        0
    } else {
        excerpt_csv_reader_into::<R, S, F>(
            reader,
            is_gz,
            mag_limit,
            &mut writer,
            filename,
            predicate,
        )?
    };
    let mut summary = writer.finish()?;
    summary.input_rows = input_rows;
    Ok(summary)
}

/// Process one input file into an existing writer. Filename is derived from
/// the path's last component and used as the manifest key.
pub fn excerpt_csv_file_into<R, S, F>(
    input_path: impl AsRef<Path>,
    mag_limit: f64,
    writer: &mut ShardedCsvWriter<R, S>,
    predicate: F,
) -> Result<usize>
where
    R: GaiaRelease,
    S: ShardKey<R::Entry>,
    F: FnMut(&R::Entry) -> bool,
{
    let path = input_path.as_ref();
    let filename = filename_of(path)?;
    let reader = CsvSourceReader::<R>::open(path, mag_limit)?;
    drive(reader, writer, &filename, predicate)
}

/// Process one input reader into an existing writer. Caller supplies the
/// manifest key.
pub fn excerpt_csv_reader_into<R, S, F>(
    reader: Box<dyn std::io::Read>,
    is_gz: bool,
    mag_limit: f64,
    writer: &mut ShardedCsvWriter<R, S>,
    filename: &str,
    predicate: F,
) -> Result<usize>
where
    R: GaiaRelease,
    S: ShardKey<R::Entry>,
    F: FnMut(&R::Entry) -> bool,
{
    let reader = CsvSourceReader::<R>::from_reader(reader, is_gz, mag_limit)?;
    drive(reader, writer, filename, predicate)
}

/// Stream `reader` through `predicate` into `writer`, framed by
/// `begin_file` + `commit_file` so the per-file commit is atomic.
pub fn drive<R, S, F>(
    reader: CsvSourceReader<R>,
    writer: &mut ShardedCsvWriter<R, S>,
    filename: &str,
    mut predicate: F,
) -> Result<usize>
where
    R: GaiaRelease,
    S: ShardKey<R::Entry>,
    F: FnMut(&R::Entry) -> bool,
{
    writer.begin_file(filename);
    let mut input_rows = 0usize;
    for entry in reader {
        let entry = entry?;
        input_rows += 1;
        if predicate(&entry) {
            writer.write(&entry)?;
        }
    }
    writer.commit_file(filename)?;
    Ok(input_rows)
}

fn filename_of(path: &Path) -> Result<String> {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            StarfieldError::DataError(format!(
                "input path has no UTF-8 file name: {}",
                path.display()
            ))
        })
}

// =====================================================================================
// Excerpt directory reader (companion to ShardedCsvWriter)
// =====================================================================================

/// Read-side handle to a directory written by [`ShardedCsvWriter`].
///
/// Pairs the on-disk [`Manifest`] with knowledge of the shard filename
/// scheme so callers can map shard indices ↔ files without hardcoding
/// the `shard_NNNNN.csv.gz` pattern.
#[derive(Debug, Clone)]
pub struct ExcerptDir {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

impl ExcerptDir {
    /// Open an excerpt directory. Errors if no manifest is found.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let manifest = Manifest::try_load(&dir)?.ok_or_else(|| {
            StarfieldError::DataError(format!(
                "no excerpt manifest at {} (directory may not be a gaia-excerpt output)",
                dir.display()
            ))
        })?;
        Ok(Self { dir, manifest })
    }

    /// HEALPix depth used when the directory was sharded by HEALPix; `None`
    /// for hash / id-range / custom sharders.
    pub fn healpix_level(&self) -> Option<u8> {
        if self.manifest.sharder.kind == "healpix" {
            self.manifest.sharder.healpix_level
        } else {
            None
        }
    }

    /// Total number of shard buckets recorded in the manifest. For HEALPix this
    /// is `12 · 4^level`; not every shard necessarily has a file on disk.
    pub fn num_shards(&self) -> u32 {
        self.manifest.sharder.num_shards
    }

    /// Path to shard file `index`, regardless of whether it exists on disk
    /// (empty cells leave no file behind).
    pub fn shard_path(&self, index: u32) -> PathBuf {
        let width = digits_for(self.num_shards());
        self.dir
            .join(format!("shard_{:0width$}.csv.gz", index, width = width))
    }

    /// Like [`shard_path`](Self::shard_path) but returns `None` when the file
    /// does not exist on disk (i.e. that bucket has no rows).
    pub fn existing_shard_path(&self, index: u32) -> Option<PathBuf> {
        let p = self.shard_path(index);
        p.exists().then_some(p)
    }

    /// Existing shard files (in shard-index order). Empty buckets are skipped.
    pub fn existing_shard_paths(&self) -> Vec<PathBuf> {
        (0..self.num_shards())
            .filter_map(|i| self.existing_shard_path(i))
            .collect()
    }
}
