//! `gaia-excerpt` — apply a predicate to one or more Gaia CSV(.gz) files
//! (local paths or streamed from ESA's CDN) and write the kept entries to
//! sharded gzipped CSVs that round-trip through `Dr{1,2,3}Catalog::from_csv_file`.
//!
//! Crash-safe and resumable: on each invocation, the writer reads the
//! `.gaia-excerpt-manifest.json` in the output directory and skips any input
//! files already committed. Re-running the exact same command after a crash
//! (or sibling-OOM, or reboot) picks up cleanly.

fn main() {
    if let Err(e) = run() {
        eprintln!("gaia-excerpt: {}", e);
        std::process::exit(1);
    }
}

mod cli;
mod sort_step;

use cli::{Cli, ReleaseChoice, Sharder};
use indicatif::{ProgressBar, ProgressStyle};
use starfield_catalogs::gaia::download::Downloader;
use starfield_catalogs::gaia::excerpt::{
    excerpt_csv_file_into, excerpt_csv_reader_into, HashIdShard, HealpixShard, IdRangeShard,
    ShardKey, ShardedCsvWriter,
};
use starfield_catalogs::gaia::{Dr1, Dr2, Dr3, GaiaRelease, GaiaSource};
use starfield_core::{Result, StarfieldError};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;

const MAX_CONNECT_RETRIES: u32 = 4; // total 5 attempts

fn run() -> Result<()> {
    use clap::Parser;
    let args = Cli::parse();
    args.validate()?;
    let release = args.effective_release()?;

    // Dispatch on release once so the writer is monomorphized.
    match release {
        ReleaseChoice::Dr1 => run_release::<Dr1>(&args),
        ReleaseChoice::Dr2 => run_release::<Dr2>(&args),
        ReleaseChoice::Dr3 => run_release::<Dr3>(&args),
    }
}

fn run_release<R: GaiaRelease>(args: &Cli) -> Result<()> {
    let mag_limit = args.mag_limit.unwrap_or(f64::INFINITY);

    let sharder = make_sharder::<R::Entry>(args);
    let mut writer = ShardedCsvWriter::<R, Box<dyn ShardKey<R::Entry>>>::new_or_resume(
        &args.output_dir,
        sharder,
        mag_limit,
    )?;
    let mut predicate = build_predicate::<R::Entry>(args)?;

    let already_processed = writer.processed_files().clone();
    if !already_processed.is_empty() {
        eprintln!(
            "resume: manifest at {} lists {} already-processed files; skipping those",
            writer.manifest_path().display(),
            already_processed.len()
        );
    }

    let multi = indicatif::MultiProgress::new();
    let kept_pb = multi.add(ProgressBar::new_spinner());
    kept_pb.set_style(
        ProgressStyle::with_template("  kept: {pos} stars (across all shards)").unwrap(),
    );
    kept_pb.set_position(writer.kept_so_far());

    let mut input_rows_total: usize = 0;
    let mut skipped_failures: Vec<(String, String)> = Vec::new();

    if !args.input.is_empty() {
        let file_pb = multi.add(ProgressBar::new(args.input.len() as u64));
        file_pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} input files ({eta})",
            )
            .unwrap(),
        );
        for input in &args.input {
            let name = input
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("input")
                .to_string();
            if already_processed.contains(&name) {
                if args.clean_after_excerpt {
                    let _ = std::fs::remove_file(input);
                }
                file_pb.inc(1);
                continue;
            }
            let rows =
                excerpt_csv_file_into::<R, _, _>(input, mag_limit, &mut writer, &mut predicate)?;
            input_rows_total += rows;
            if args.clean_after_excerpt {
                std::fs::remove_file(input).map_err(StarfieldError::IoError)?;
            }
            kept_pb.set_position(writer.kept_so_far());
            file_pb.inc(1);
        }
        file_pb.finish_with_message("done");
    } else {
        // CDN mode: enumerate, then per-file streamed-or-cached with retry.
        let names = Downloader::<R>::list_remote()?;
        let total_files = args.max_files.unwrap_or(names.len()).min(names.len());
        let parallel = args.cache_raw && args.download_workers > 1;
        eprintln!(
            "fetching {} of {} {} files from CDN ({}{})",
            total_files,
            names.len(),
            release_label::<R>(),
            if args.cache_raw {
                "writing raw to cache"
            } else {
                "validated temporary download, removed after reading"
            },
            if parallel {
                format!(", {} download workers", args.download_workers)
            } else {
                String::new()
            },
        );

        // Pre-filter: skip already-processed files, optionally evicting any
        // stale raw cache entries left over from a prior run.
        let mut work: Vec<String> = Vec::with_capacity(total_files);
        let mut skipped_resume = 0usize;
        for name in names.iter().take(total_files) {
            if already_processed.contains(name) {
                if args.cache_raw && args.clean_after_excerpt {
                    Downloader::<R>::remove_cached(name)?;
                }
                skipped_resume += 1;
            } else {
                work.push(name.clone());
            }
        }

        let file_pb = multi.add(ProgressBar::new(total_files as u64));
        file_pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} CDN files ({eta})",
            )
            .unwrap(),
        );
        file_pb.set_position(skipped_resume as u64);

        if parallel {
            run_parallel_cdn::<R>(
                &work,
                args,
                mag_limit,
                &mut writer,
                &mut predicate,
                &file_pb,
                &kept_pb,
                &mut input_rows_total,
                &mut skipped_failures,
            )?;
        } else {
            for (idx, name) in work.iter().enumerate() {
                match retry_one_file::<R>(
                    idx,
                    work.len(),
                    name,
                    args,
                    mag_limit,
                    &mut writer,
                    &mut predicate,
                ) {
                    FileOutcome::Ok(rows) => {
                        input_rows_total += rows;
                    }
                    FileOutcome::GivingUp { attempts, err } => {
                        eprintln!(
                            "[{}/{}] {}: giving up after {} attempts ({})",
                            idx + 1,
                            work.len(),
                            name,
                            attempts,
                            err
                        );
                        skipped_failures.push((name.clone(), err));
                    }
                }
                kept_pb.set_position(writer.kept_so_far());
                file_pb.inc(1);
            }
        }
        file_pb.finish_with_message("done");

        if !skipped_failures.is_empty() {
            eprintln!(
                "\n{} files exhausted retries — re-run the same command to retry them:",
                skipped_failures.len()
            );
            for (p, err) in &skipped_failures {
                eprintln!("  - {}  ({})", p, err);
            }
        }
    }
    kept_pb.finish_and_clear();

    let summary = writer.finish()?;
    let written: Vec<_> = summary.written_paths().cloned().collect();
    println!(
        "read {} stars (this run); kept total {} across {} shard files",
        input_rows_total,
        summary.kept_rows,
        written.len(),
    );

    if args.sort {
        eprintln!(
            "\nsorting {} shard files by {:?} ...",
            written.len(),
            args.sort_by
        );
        let sort_pb = ProgressBar::new(written.len() as u64);
        sort_pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} sorted ({eta})",
            )
            .unwrap(),
        );
        for path in &written {
            sort_step::sort_shard::<R>(path, args.sort_by)?;
            sort_pb.inc(1);
        }
        sort_pb.finish_with_message("sorted");
    }

    Ok(())
}

fn release_label<R: GaiaRelease>() -> &'static str {
    match R::RELEASE {
        starfield_catalogs::gaia::Release::Dr1 => "DR1",
        starfield_catalogs::gaia::Release::Dr2 => "DR2",
        starfield_catalogs::gaia::Release::Dr3 => "DR3",
    }
}

fn make_sharder<E: GaiaSource + 'static>(args: &Cli) -> Box<dyn ShardKey<E>> {
    match args.shard_by {
        Sharder::Hash => Box::new(HashIdShard {
            num_shards: args.effective_bucket_count(),
        }),
        Sharder::IdRange => Box::new(IdRangeShard {
            num_shards: args.effective_bucket_count(),
        }),
        Sharder::Healpix => Box::new(HealpixShard {
            level: args.healpix_level,
        }),
    }
}

/// Outcome of one retried CDN file. Per-file commit is atomic in the new
/// `ShardedCsvWriter`, so a failure during read or commit leaves both the
/// manifest and the shards unchanged — retries are always safe.
enum FileOutcome {
    Ok(usize),
    GivingUp { attempts: u32, err: String },
}

fn retry_one_file<R>(
    idx: usize,
    total: usize,
    name: &str,
    args: &Cli,
    mag_limit: f64,
    writer: &mut ShardedCsvWriter<R, Box<dyn ShardKey<R::Entry>>>,
    predicate: &mut BoxedPredicate<R::Entry>,
) -> FileOutcome
where
    R: GaiaRelease,
{
    for attempt in 0..=MAX_CONNECT_RETRIES {
        let res: Result<usize> = if args.cache_raw {
            match Downloader::<R>::download_file(name) {
                Ok(path) => {
                    let extract =
                        excerpt_csv_file_into::<R, _, _>(&path, mag_limit, writer, &mut *predicate);
                    // Evict cached raw only after a successful extract+commit.
                    if extract.is_ok() && args.clean_after_excerpt {
                        if let Err(e) = Downloader::<R>::remove_cached(name) {
                            eprintln!(
                                "[{}/{}] {}: warn: extract ok but failed to evict {}: {}",
                                idx + 1,
                                total,
                                name,
                                path.display(),
                                e
                            );
                        }
                    }
                    extract
                }
                Err(e) => Err(e),
            }
        } else {
            match Downloader::<R>::stream_file(name) {
                Ok(stream) => excerpt_csv_reader_into::<R, _, _>(
                    stream,
                    true,
                    mag_limit,
                    writer,
                    name,
                    &mut *predicate,
                ),
                Err(e) => Err(e),
            }
        };
        match res {
            Ok(rows) => return FileOutcome::Ok(rows),
            Err(e) => {
                let err_str = e.to_string();
                if attempt == MAX_CONNECT_RETRIES {
                    return FileOutcome::GivingUp {
                        attempts: attempt + 1,
                        err: err_str,
                    };
                }
                let wait = 1u64 << attempt; // 1, 2, 4, 8, 16 seconds
                eprintln!(
                    "[{}/{}] {}: attempt {}/{} failed ({}); retrying in {}s",
                    idx + 1,
                    total,
                    name,
                    attempt + 1,
                    MAX_CONNECT_RETRIES + 1,
                    err_str,
                    wait,
                );
                std::thread::sleep(std::time::Duration::from_secs(wait));
            }
        }
    }
    FileOutcome::Ok(0)
}

/// Result of a single CDN download attempt, sent from worker threads to the
/// extract thread. The raw bytes are already on disk at `path` when `res` is
/// `Ok`; on `Err`, the file was never produced (or a partial tmp was cleaned
/// up by the datastore's validate-before-publication guarantee).
struct DownloadOutcome {
    name: String,
    res: std::result::Result<PathBuf, String>,
}

/// Spawn a download worker pool and drain results into the single-threaded
/// extract phase. Workers retry their own download up to `MAX_CONNECT_RETRIES`
/// before reporting failure. The bounded channel limits how many staged raw
/// files sit on disk at once: `download_workers` files × ~5 MB/file ≈ tens
/// of MB at peak.
#[allow(clippy::too_many_arguments)]
fn run_parallel_cdn<R: GaiaRelease>(
    work: &[String],
    args: &Cli,
    mag_limit: f64,
    writer: &mut ShardedCsvWriter<R, Box<dyn ShardKey<R::Entry>>>,
    predicate: &mut BoxedPredicate<R::Entry>,
    file_pb: &ProgressBar,
    kept_pb: &ProgressBar,
    input_rows_total: &mut usize,
    skipped_failures: &mut Vec<(String, String)>,
) -> Result<()> {
    let workers = args.download_workers.max(1) as usize;
    let names: Arc<Vec<String>> = Arc::new(work.to_vec());
    let next = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = sync_channel::<DownloadOutcome>(workers);

    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let names = Arc::clone(&names);
        let next = Arc::clone(&next);
        let tx = tx.clone();
        handles.push(std::thread::spawn(move || loop {
            let i = next.fetch_add(1, Ordering::SeqCst);
            if i >= names.len() {
                break;
            }
            let name = names[i].clone();
            let res = retry_download::<R>(&name);
            if tx.send(DownloadOutcome { name, res }).is_err() {
                break;
            }
        }));
    }
    drop(tx);

    while let Ok(item) = rx.recv() {
        match item.res {
            Ok(path) => {
                let extract =
                    excerpt_csv_file_into::<R, _, _>(&path, mag_limit, writer, &mut *predicate);
                match extract {
                    Ok(rows) => {
                        *input_rows_total += rows;
                        if args.clean_after_excerpt {
                            if let Err(e) = Downloader::<R>::remove_cached(&item.name) {
                                eprintln!(
                                    "{}: warn: extract ok but failed to evict {}: {}",
                                    item.name,
                                    path.display(),
                                    e
                                );
                            }
                        }
                    }
                    Err(e) => {
                        let err_str = e.to_string();
                        eprintln!("{}: extract failed: {}", item.name, err_str);
                        skipped_failures.push((item.name, err_str));
                    }
                }
            }
            Err(e) => {
                skipped_failures.push((item.name, e));
            }
        }
        kept_pb.set_position(writer.kept_so_far());
        file_pb.inc(1);
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// Download one file with bounded retries, returning a stringified error on
/// final failure (so it can cross thread boundaries without `Send` worries).
fn retry_download<R: GaiaRelease>(name: &str) -> std::result::Result<PathBuf, String> {
    for attempt in 0..=MAX_CONNECT_RETRIES {
        match Downloader::<R>::download_file(name) {
            Ok(p) => return Ok(p),
            Err(e) => {
                let s = e.to_string();
                if attempt == MAX_CONNECT_RETRIES {
                    return Err(format!("after {} attempts: {}", MAX_CONNECT_RETRIES + 1, s));
                }
                let wait = 1u64 << attempt; // 1, 2, 4, 8 seconds
                eprintln!(
                    "{}: download attempt {}/{} failed ({}); retrying in {}s",
                    name,
                    attempt + 1,
                    MAX_CONNECT_RETRIES + 1,
                    s,
                    wait,
                );
                std::thread::sleep(std::time::Duration::from_secs(wait));
            }
        }
    }
    unreachable!("retry_download loop should have returned by now")
}

type BoxedPredicate<E> = Box<dyn FnMut(&E) -> bool + Send>;

/// Compose mag/cone/id-range filters into a single closure. Cone filtering
/// is done with great-circle distance via core unit vectors.
fn build_predicate<E: GaiaSource + 'static>(args: &Cli) -> Result<BoxedPredicate<E>> {
    let id_range = args.id_range;
    let cone = args
        .cone
        .as_ref()
        .map(|c| starfield_catalogs::gaia::Cone::from_degrees(c.ra_deg, c.dec_deg, c.radius_deg));

    Ok(Box::new(move |e: &E| {
        let c = e.core();
        if let Some((lo, hi)) = id_range {
            if c.source_id < lo || c.source_id > hi {
                return false;
            }
        }
        if let Some(cone) = cone.as_ref() {
            if !cone.contains_unit_vec(&c.unit_vector()) {
                return false;
            }
        }
        true
    }))
}
