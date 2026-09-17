//! `hipparcos-gaia-match` — verify our cross-match strategy before building
//! the bright-star supplement loader.
//!
//! Pipeline:
//!
//! 1. Pull a bright Gaia subset for the chosen release (`--release dr1|dr2|dr3`)
//!    via the ESA TAP service: a single ADQL query selecting every column the
//!    bulk CSV files publish, filtered by `phot_g_mean_mag <= --gaia-mag-limit`
//!    (default 13). The CSV response is cached at
//!    `~/.cache/starfield/gaia-bright/<release>-mag<limit>.csv`.
//! 2. Load the Hipparcos catalog (auto-downloaded), filtered to
//!    `--hipparcos-mag-limit` (default Hp ≤ 12).
//! 3. Build a HEALPix-keyed spatial index over the Gaia bright subset, then
//!    for each Hipparcos entry: propagate its J1991.25 position to the chosen
//!    DR's reference epoch using the Hipparcos proper motion, query the index
//!    for any Gaia source within `--match-radius-arcsec`, and record the
//!    nearest match.
//! 4. Write a matches CSV with `(hip_id, gaia_source_id, sep_arcsec, hp_mag,
//!    b_v, gaia_g, gaia_bp, gaia_rp, gaia_bp_rp)` (some columns blank for
//!    DR1, which lacks BP/RP).
//! 5. Fit `G ≈ a + b·Hp + c·(B-V)` by ordinary least squares over the matches
//!    that have all three quantities, print the coefficients and RMS.
//! 6. Emit a PNG scatter plot of `G` vs `Hp` colored by `B-V`, with the
//!    regression line overlaid at the mean color.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra as na;
use plotters::prelude::*;
use starfield_core::catalogs::StarCatalog;
use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::cache_dir;
use starfield_gaia::common::reader::CsvSourceReader;
use starfield_gaia::download::Downloader;
use starfield_gaia::{Dr1, Dr2, Dr3, GaiaRelease, GaiaSource};
use starfield_hipparcos::{download_hipparcos, HipparcosCatalog};

/// Hipparcos catalog reference epoch (Julian years).
const HIPPARCOS_EPOCH: f64 = 1991.25;
/// HEALPix depth used for the spatial index. Level 12 ≈ 51 arcsec/cell;
/// querying the cell + its 8 neighbors covers a ~150 arcsec window which
/// comfortably bounds the default 5 arcsec match radius even at the cell
/// boundary.
const HEALPIX_DEPTH: u8 = 12;

#[derive(Parser, Debug)]
#[command(
    name = "hipparcos-gaia-match",
    version,
    about = "Cross-match Hipparcos to a Gaia DR's bright end via TAP and fit G(Hp, B-V)"
)]
struct Args {
    /// Which Gaia release to query against.
    #[arg(long, value_enum)]
    release: ReleaseChoice,

    /// Gaia G-band magnitude limit (TAP-side filter `phot_g_mean_mag <=`).
    /// Hipparcos overlap caps out around G ≈ 12; the default 13 gives a small
    /// buffer for the regression.
    #[arg(long, default_value_t = 13.0)]
    gaia_mag_limit: f64,

    /// Hipparcos Hp-magnitude limit. Default 12 is essentially the whole
    /// catalog (Hipparcos reaches ~11.5 with degraded precision past ~10).
    #[arg(long, default_value_t = 12.0)]
    hipparcos_mag_limit: f64,

    /// Cross-match radius in arcsec. 5" is generous: typical Hipparcos
    /// astrometric precision is ~1 mas; the dominant residual at the bright
    /// end is unmodelled non-linear motion (binaries) plus Gaia's degraded
    /// astrometry for G < 6.
    #[arg(long, default_value_t = 5.0)]
    match_radius_arcsec: f64,

    /// Re-fetch the Gaia bright CSV from TAP even if it exists in the cache.
    #[arg(long, default_value_t = false)]
    refresh: bool,

    /// Number of source_id-range batches to issue against TAP. ESA's sync
    /// endpoint silently caps responses around 25 k rows even with a large
    /// MAXREC; pagination by `source_id BETWEEN` works around that. ESA also
    /// has a per-query server-side time budget — too few batches → some
    /// batches HTTP 408 even with retries. Default 128 keeps each batch
    /// well under both caps for any G ≤ 13 query.
    #[arg(long, default_value_t = 128)]
    batches: u32,

    /// Output CSV of matched (hip, gaia) pairs. Defaults to
    /// `hipparcos-<release>-matches.csv` in CWD.
    #[arg(long)]
    output_csv: Option<PathBuf>,

    /// Output PNG of the G-vs-Hp scatter + regression overlay. Defaults to
    /// `hipparcos-<release>-mag-regression.png` in CWD.
    #[arg(long)]
    output_plot: Option<PathBuf>,

    /// Optional path to write a slim supplement CSV containing one row per
    /// Hipparcos entry with no Gaia match — these are the bright stars Gaia
    /// either saturated or never observed cleanly. Each row has
    /// `hip, ra_j2016, dec_j2016, parallax_mas, pmra_mas_yr, pmdec_mas_yr,
    /// b_v, fitted_g_mag` with the position propagated to the DR's reference
    /// epoch and the G mag derived from the regression
    /// `G ≈ a + b·Hp + c·(B-V)` fit on this run.
    #[arg(long)]
    supplement: Option<PathBuf>,
}

#[derive(ValueEnum, Debug, Clone, Copy, PartialEq, Eq)]
enum ReleaseChoice {
    Dr1,
    Dr2,
    Dr3,
}

impl ReleaseChoice {
    fn as_str(&self) -> &'static str {
        match self {
            ReleaseChoice::Dr1 => "dr1",
            ReleaseChoice::Dr2 => "dr2",
            ReleaseChoice::Dr3 => "dr3",
        }
    }
    /// Reference epoch for source positions in this DR (Julian years).
    fn ref_epoch(&self) -> f64 {
        match self {
            ReleaseChoice::Dr1 => 2015.0,
            ReleaseChoice::Dr2 => 2015.5,
            ReleaseChoice::Dr3 => 2016.0,
        }
    }
}

/// Compact Gaia bright-subset row carried in memory during cross-matching.
/// `bp` / `rp` / `bp_rp` are NaN for DR1 (BP/RP didn't exist before DR2).
#[derive(Debug, Clone, Copy)]
struct GaiaBright {
    source_id: u64,
    ra_deg: f64,
    dec_deg: f64,
    g: f64,
    bp: f64,
    rp: f64,
    bp_rp: f64,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hipparcos-gaia-match: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    let release = args.release;

    // ---------------------------------------------------------------- Hipparcos
    eprintln!("loading Hipparcos…");
    let hip_path = download_hipparcos()?;
    let hip = HipparcosCatalog::from_dat_file(&hip_path, args.hipparcos_mag_limit)?;
    eprintln!(
        "  {} entries (Hp ≤ {})",
        hip.len(),
        args.hipparcos_mag_limit
    );

    // ---------------------------------------------------------------- Gaia bright via TAP
    let cache_path = bright_csv_cache_path(release, args.gaia_mag_limit)?;
    if args.refresh && cache_path.exists() {
        let _ = fs::remove_file(&cache_path);
    }
    if !cache_path.exists() {
        eprintln!(
            "fetching {} bright subset (G ≤ {}) via TAP in {} batches → {}",
            release.as_str(),
            args.gaia_mag_limit,
            args.batches,
            cache_path.display()
        );
        fetch_paginated(
            release,
            args.gaia_mag_limit,
            args.batches as usize,
            &cache_path,
        )?;
    } else {
        eprintln!(
            "using cached {} bright CSV at {}",
            release.as_str(),
            cache_path.display()
        );
    }

    // The TAP CSV's column layout is exactly R::csv_header() (we asked for it),
    // so it goes straight through the existing CsvSourceReader.
    eprintln!("parsing bright subset…");
    let bright = match release {
        ReleaseChoice::Dr1 => load_bright::<Dr1>(&cache_path, args.gaia_mag_limit)?,
        ReleaseChoice::Dr2 => load_bright::<Dr2>(&cache_path, args.gaia_mag_limit)?,
        ReleaseChoice::Dr3 => load_bright::<Dr3>(&cache_path, args.gaia_mag_limit)?,
    };
    eprintln!(
        "  {} Gaia stars at G ≤ {}",
        bright.len(),
        args.gaia_mag_limit
    );

    // --------------------------------------------------- Spatial index over bright
    eprintln!("building HEALPix(depth={}) index…", HEALPIX_DEPTH);
    let mut by_cell: HashMap<u64, Vec<usize>> = HashMap::with_capacity(bright.len());
    for (i, g) in bright.iter().enumerate() {
        let cell =
            cdshealpix::nested::hash(HEALPIX_DEPTH, g.ra_deg.to_radians(), g.dec_deg.to_radians());
        by_cell.entry(cell).or_default().push(i);
    }
    eprintln!("  {} occupied cells", by_cell.len());

    // ------------------------------------------------------- Cross-match
    let dt = release.ref_epoch() - HIPPARCOS_EPOCH;
    let radius_rad = (args.match_radius_arcsec / 3600.0).to_radians();
    let cos_threshold = radius_rad.cos();
    eprintln!(
        "matching {} Hipparcos → {} (radius {:.1}\", Δt = {:.2} yr)…",
        hip.len(),
        release.as_str(),
        args.match_radius_arcsec,
        dt
    );

    let pb = ProgressBar::new(hip.len() as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.green/blue} {pos}/{len} hip queries",
        )
        .unwrap(),
    );

    let mut matches: Vec<(u64, usize, f64)> = Vec::with_capacity(hip.len());
    let mut unmatched_hip: Vec<u64> = Vec::new();

    for h in hip.stars() {
        pb.inc(1);
        let pm_ra = h.pm_ra.unwrap_or(0.0);
        let pm_dec = h.pm_dec.unwrap_or(0.0);
        let dec_rad = h.dec.to_radians();
        let cos_dec = dec_rad.cos().max(1e-9);
        let dra_deg = pm_ra * dt / 3.6e6 / cos_dec;
        let ddec_deg = pm_dec * dt / 3.6e6;
        let ra_t = wrap_ra_deg(h.ra + dra_deg);
        let dec_t = (h.dec + ddec_deg).clamp(-90.0, 90.0);

        let h_uv = unit_vec(ra_t, dec_t);
        let cell = cdshealpix::nested::hash(HEALPIX_DEPTH, ra_t.to_radians(), dec_t.to_radians());
        let neighbours = cdshealpix::nested::neighbours(HEALPIX_DEPTH, cell, true);

        let mut best: Option<(usize, f64)> = None;
        for cell_id in neighbours.values_vec() {
            if let Some(idx_list) = by_cell.get(&cell_id) {
                for &gi in idx_list {
                    let g = &bright[gi];
                    let g_uv = unit_vec(g.ra_deg, g.dec_deg);
                    let cd = h_uv.dot(&g_uv);
                    if cd > cos_threshold && best.is_none_or(|(_, b)| cd > b) {
                        best = Some((gi, cd));
                    }
                }
            }
        }
        if let Some((gi, cd)) = best {
            let sep_arcsec = cd.clamp(-1.0, 1.0).acos().to_degrees() * 3600.0;
            matches.push((h.hip as u64, gi, sep_arcsec));
        } else {
            unmatched_hip.push(h.hip as u64);
        }
    }
    pb.finish();
    eprintln!(
        "  matched {} of {} Hipparcos ({:.2}% — {} unmatched)",
        matches.len(),
        hip.len(),
        100.0 * matches.len() as f64 / hip.len() as f64,
        unmatched_hip.len()
    );

    // ------------------------------------------------------- CSV
    let output_csv = args
        .output_csv
        .clone()
        .unwrap_or_else(|| PathBuf::from(format!("hipparcos-{}-matches.csv", release.as_str())));
    eprintln!("writing matches CSV → {}", output_csv.display());
    let mut csv = BufWriter::new(File::create(&output_csv).map_err(StarfieldError::IoError)?);
    writeln!(
        csv,
        "hip,gaia_source_id,sep_arcsec,hp_mag,b_v,gaia_g,gaia_bp,gaia_rp,gaia_bp_rp"
    )
    .map_err(StarfieldError::IoError)?;
    for &(hip_id, gi, sep) in &matches {
        let h = hip
            .get_star(hip_id as usize)
            .expect("hip id from iter must be present");
        let g = &bright[gi];
        writeln!(
            csv,
            "{},{},{:.4},{:.4},{},{:.4},{},{},{}",
            h.hip,
            g.source_id,
            sep,
            h.mag,
            opt_f(h.b_v),
            g.g,
            opt_or_nan(g.bp),
            opt_or_nan(g.rp),
            opt_or_nan(g.bp_rp),
        )
        .map_err(StarfieldError::IoError)?;
    }
    csv.flush().map_err(StarfieldError::IoError)?;

    // ------------------------------------------------------- Regression
    let mut data: Vec<(f64, f64, f64)> = Vec::with_capacity(matches.len());
    for &(hip_id, gi, _sep) in &matches {
        let h = hip.get_star(hip_id as usize).unwrap();
        let g = &bright[gi];
        if let Some(bv) = h.b_v {
            if bv.is_finite() && g.g.is_finite() && h.mag.is_finite() {
                data.push((h.mag, bv, g.g));
            }
        }
    }
    eprintln!("regression on {} pairs with valid (Hp, B-V, G)", data.len());
    let (a, b, c, rms, n) = regress_g_hp_bv(&data);
    eprintln!(
        "  G ≈ {:+.4} {:+.4}*Hp {:+.4}*(B-V)   RMS = {:.3} mag   N = {}",
        a, b, c, rms, n
    );

    // Fallback regression for entries without a B-V color (Hipparcos sometimes
    // doesn't publish one). Uses just Hp.
    let data_no_bv: Vec<(f64, f64)> = data.iter().map(|&(hp, _bv, g)| (hp, g)).collect();
    let (a1, b1, rms1, n1) = regress_g_hp(&data_no_bv);
    eprintln!(
        "  fallback (no B-V): G ≈ {:+.4} {:+.4}*Hp   RMS = {:.3} mag   N = {}",
        a1, b1, rms1, n1
    );

    // ----------------------------------------------------- Supplement CSV
    if let Some(path) = &args.supplement {
        eprintln!(
            "writing supplement CSV ({} unmatched Hipparcos) → {}",
            unmatched_hip.len(),
            path.display()
        );
        write_supplement(
            path,
            &hip,
            &unmatched_hip,
            release.ref_epoch(),
            (a, b, c),
            (a1, b1),
        )?;
    }

    // ------------------------------------------------------- Plot
    let output_plot = args.output_plot.clone().unwrap_or_else(|| {
        PathBuf::from(format!("hipparcos-{}-mag-regression.png", release.as_str()))
    });
    eprintln!("rendering plot → {}", output_plot.display());
    plot_g_vs_hp(&data, &output_plot, (a, b, c), rms, release)?;

    Ok(())
}

/// Fetch a bright subset from TAP in `batches` source_id-range slices and
/// write the concatenated CSV (one shared header + every batch's body) to
/// `cache_path`. ESA's sync endpoint silently caps responses around ~25 k
/// rows even with `MAXREC` set high, so we partition `[0, 2^63]` into N
/// equal-width source_id intervals and issue one TAP query per interval.
///
/// Each batch is retried up to [`MAX_TAP_RETRIES`] times with exponential
/// backoff against transient errors (HTTP 408 Request Timeout, 500-class,
/// connection reset). The full response is buffered to memory before
/// touching the sink, so a mid-batch failure leaves no partial data on disk.
fn fetch_paginated(
    release: ReleaseChoice,
    mag_limit: f64,
    batches: usize,
    cache_path: &Path,
) -> Result<()> {
    let pb = ProgressBar::new(batches as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} TAP batches • {msg}",
        )
        .unwrap(),
    );
    let span = (i64::MAX as u64) / batches as u64;
    let tmp = cache_path.with_extension("csv.tmp");
    let mut sink = BufWriter::new(File::create(&tmp).map_err(StarfieldError::IoError)?);
    let mut wrote_header = false;
    let mut rows_total: u64 = 0;

    let mut skipped: Vec<usize> = Vec::new();
    for b in 0..batches {
        let lo = (b as u64).saturating_mul(span);
        let hi = if b == batches - 1 {
            i64::MAX as u64
        } else {
            (b as u64 + 1).saturating_mul(span) - 1
        };
        let where_clause = format!(
            "phot_g_mean_mag <= {} AND source_id BETWEEN {} AND {}",
            mag_limit, lo, hi
        );
        match fetch_batch_buffered_with_retry(release, &where_clause, b, batches) {
            Ok(buf) => {
                let batch_rows = append_batch(&buf[..], &mut sink, &mut wrote_header)?;
                rows_total += batch_rows as u64;
                pb.set_message(format!(
                    "{} rows so far (last batch {})",
                    rows_total, batch_rows
                ));
            }
            Err(e) => {
                eprintln!(
                    "  batch {}/{}: giving up after retries ({})",
                    b + 1,
                    batches,
                    e
                );
                skipped.push(b);
            }
        }
        pb.inc(1);
    }
    sink.flush().map_err(StarfieldError::IoError)?;
    drop(sink);
    fs::rename(&tmp, cache_path).map_err(StarfieldError::IoError)?;
    if !skipped.is_empty() {
        eprintln!(
            "WARNING: {} of {} batches were skipped due to repeated TAP errors:",
            skipped.len(),
            batches
        );
        for b in &skipped {
            let lo = (*b as u64).saturating_mul(span);
            let hi = if *b == batches - 1 {
                i64::MAX as u64
            } else {
                (*b as u64 + 1).saturating_mul(span) - 1
            };
            eprintln!("  batch {} → source_id BETWEEN {} AND {}", b + 1, lo, hi);
        }
        eprintln!("the bright cache is incomplete; cross-match results below are a partial view.");
    }
    pb.finish_with_message(format!(
        "{} rows total ({} batches OK, {} skipped)",
        rows_total,
        batches - skipped.len(),
        skipped.len()
    ));
    Ok(())
}

/// Maximum retry attempts per TAP batch. Backoff schedule is 1, 2, 4, 8, 16 s
/// — total worst-case wait is ~31 s before giving up on a batch.
const MAX_TAP_RETRIES: u32 = 4; // 5 attempts total (initial + 4 retries)

/// Run a single TAP batch with exponential-backoff retries, buffering the
/// full response into memory. Returning `Ok(buf)` means the entire batch
/// arrived successfully; the buffer can then be appended to the running
/// cache without partial-data risk.
fn fetch_batch_buffered_with_retry(
    release: ReleaseChoice,
    where_clause: &str,
    batch_idx: usize,
    n_batches: usize,
) -> Result<Vec<u8>> {
    for attempt in 0..=MAX_TAP_RETRIES {
        let res = (|| -> Result<Vec<u8>> {
            let mut stream = match release {
                ReleaseChoice::Dr1 => Downloader::<Dr1>::tap_select_where(where_clause)?,
                ReleaseChoice::Dr2 => Downloader::<Dr2>::tap_select_where(where_clause)?,
                ReleaseChoice::Dr3 => Downloader::<Dr3>::tap_select_where(where_clause)?,
            };
            let mut buf = Vec::new();
            std::io::copy(&mut stream, &mut buf).map_err(StarfieldError::IoError)?;
            Ok(buf)
        })();
        match res {
            Ok(buf) => return Ok(buf),
            Err(e) => {
                if attempt == MAX_TAP_RETRIES {
                    return Err(e);
                }
                let wait = 1u64 << attempt; // 1, 2, 4, 8, 16 s
                eprintln!(
                    "  batch {}/{}: attempt {}/{} failed ({}); retrying in {}s",
                    batch_idx + 1,
                    n_batches,
                    attempt + 1,
                    MAX_TAP_RETRIES + 1,
                    e,
                    wait
                );
                std::thread::sleep(std::time::Duration::from_secs(wait));
            }
        }
    }
    unreachable!("retry loop exits via Ok or final-attempt Err");
}

/// Append one TAP batch's CSV body to `sink`, sharing the header across
/// batches (write it once on the first batch, skip it thereafter). Returns
/// the row count seen in this batch.
fn append_batch<R: std::io::Read, W: std::io::Write>(
    src: R,
    sink: &mut W,
    wrote_header: &mut bool,
) -> Result<usize> {
    let reader = BufReader::new(src);
    let mut row_count = 0;
    for (i, line) in std::io::BufRead::lines(reader).enumerate() {
        let line = line.map_err(StarfieldError::IoError)?;
        if i == 0 {
            // Header. Write it on the first batch only.
            if !*wrote_header {
                sink.write_all(line.as_bytes())
                    .map_err(StarfieldError::IoError)?;
                sink.write_all(b"\n").map_err(StarfieldError::IoError)?;
                *wrote_header = true;
            }
            continue;
        }
        sink.write_all(line.as_bytes())
            .map_err(StarfieldError::IoError)?;
        sink.write_all(b"\n").map_err(StarfieldError::IoError)?;
        row_count += 1;
    }
    Ok(row_count)
}

/// Read a TAP-fetched bright CSV through the per-release `CsvSourceReader`,
/// extracting only what the cross-match needs.
fn load_bright<R: GaiaRelease>(path: &Path, mag_limit: f64) -> Result<Vec<GaiaBright>> {
    let f = File::open(path).map_err(StarfieldError::IoError)?;
    let reader = CsvSourceReader::<R>::from_reader(Box::new(BufReader::new(f)), false, mag_limit)?;
    let mut out = Vec::new();
    for entry_res in reader {
        let e = entry_res?;
        let core = e.core();
        let (bp, rp, bp_rp) = bp_rp_of::<R>(&e);
        out.push(GaiaBright {
            source_id: core.source_id,
            ra_deg: core.ra,
            dec_deg: core.dec,
            g: core.phot_g_mean_mag,
            bp,
            rp,
            bp_rp,
        });
    }
    Ok(out)
}

/// Extract BP / RP / BP-RP from any release's entry. DR1 has none → all NaN;
/// DR2 / DR3 carry them in slightly different sub-structs but with the same
/// triple shape via the [`bp_rp_of_dr2`] / [`bp_rp_of_dr3`] helpers.
fn bp_rp_of<R: GaiaRelease>(_entry: &R::Entry) -> (f64, f64, f64) {
    // We can't write generic field access on R::Entry; the per-release shims
    // below specialize via TypeId. Compiler monomorphizes one branch live.
    use std::any::Any;
    let any: &dyn Any = _entry;
    if let Some(e) = any.downcast_ref::<starfield_gaia::Dr3Entry>() {
        return match e.bp_rp.as_ref() {
            Some(b) => (
                b.phot_bp_mean_mag.unwrap_or(f64::NAN),
                b.phot_rp_mean_mag.unwrap_or(f64::NAN),
                b.bp_rp.map(|x| x as f64).unwrap_or(f64::NAN),
            ),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
    }
    if let Some(e) = any.downcast_ref::<starfield_gaia::Dr2Entry>() {
        return match e.bp_rp.as_ref() {
            Some(b) => (
                b.phot_bp_mean_mag.unwrap_or(f64::NAN),
                b.phot_rp_mean_mag.unwrap_or(f64::NAN),
                b.bp_rp.map(|x| x as f64).unwrap_or(f64::NAN),
            ),
            None => (f64::NAN, f64::NAN, f64::NAN),
        };
    }
    // Dr1: no BP/RP.
    (f64::NAN, f64::NAN, f64::NAN)
}

fn bright_csv_cache_path(release: ReleaseChoice, mag_limit: f64) -> Result<PathBuf> {
    let dir = cache_dir().join("gaia-bright");
    fs::create_dir_all(&dir).map_err(StarfieldError::IoError)?;
    Ok(dir.join(format!("{}-mag{}.csv", release.as_str(), mag_limit)))
}

fn wrap_ra_deg(ra: f64) -> f64 {
    let mut r = ra % 360.0;
    if r < 0.0 {
        r += 360.0;
    }
    r
}

fn unit_vec(ra_deg: f64, dec_deg: f64) -> na::Vector3<f64> {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    na::Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

fn opt_f(v: Option<f64>) -> String {
    v.map(|x| format!("{:.4}", x)).unwrap_or_default()
}

fn opt_or_nan(x: f64) -> String {
    if x.is_finite() {
        format!("{:.4}", x)
    } else {
        String::new()
    }
}

/// OLS for `G = a + b·Hp + c·(B-V)` via the 3×3 normal equations. Returns
/// `(a, b, c, rms_residual, n_used)`.
fn regress_g_hp_bv(data: &[(f64, f64, f64)]) -> (f64, f64, f64, f64, usize) {
    let n = data.len();
    if n < 3 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, n);
    }
    let mut s = [0.0_f64; 6]; // 1, hp, bv, hp*hp, hp*bv, bv*bv
    let mut sg = 0.0;
    let mut shg = 0.0;
    let mut sbg = 0.0;
    for &(hp, bv, g) in data {
        s[0] += 1.0;
        s[1] += hp;
        s[2] += bv;
        s[3] += hp * hp;
        s[4] += hp * bv;
        s[5] += bv * bv;
        sg += g;
        shg += hp * g;
        sbg += bv * g;
    }
    let m = na::Matrix3::new(
        s[0], s[1], s[2], //
        s[1], s[3], s[4], //
        s[2], s[4], s[5],
    );
    let v = na::Vector3::new(sg, shg, sbg);
    let inv = match m.try_inverse() {
        Some(x) => x,
        None => return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, n),
    };
    let coef = inv * v;
    let (a, b, c) = (coef[0], coef[1], coef[2]);
    let mut sq = 0.0;
    for &(hp, bv, g) in data {
        sq += (g - (a + b * hp + c * bv)).powi(2);
    }
    (a, b, c, (sq / n as f64).sqrt(), n)
}

/// 1-D OLS for `G = a + b·Hp` (used as a fallback when B-V is unknown).
/// Returns `(a, b, rms_residual, n_used)`.
fn regress_g_hp(data: &[(f64, f64)]) -> (f64, f64, f64, usize) {
    let n = data.len();
    if n < 2 {
        return (f64::NAN, f64::NAN, f64::NAN, n);
    }
    let mut s_hp = 0.0;
    let mut s_g = 0.0;
    let mut s_hp_hp = 0.0;
    let mut s_hp_g = 0.0;
    for &(hp, g) in data {
        s_hp += hp;
        s_g += g;
        s_hp_hp += hp * hp;
        s_hp_g += hp * g;
    }
    let nf = n as f64;
    let denom = nf * s_hp_hp - s_hp * s_hp;
    if denom.abs() < 1e-12 {
        return (f64::NAN, f64::NAN, f64::NAN, n);
    }
    let b = (nf * s_hp_g - s_hp * s_g) / denom;
    let a = (s_g - b * s_hp) / nf;
    let mut sq = 0.0;
    for &(hp, g) in data {
        sq += (g - (a + b * hp)).powi(2);
    }
    (a, b, (sq / nf).sqrt(), n)
}

/// Write a slim supplement CSV: one row per unmatched Hipparcos entry, with
/// position propagated to the DR's reference epoch and a fitted Gaia G mag.
/// The downstream catalog loader (`Dr3Catalog::augment_missing`) parses this
/// shape, fills in computed l/b/ecl, encodes a synthetic source_id, and
/// inserts each row as a `Dr3Entry`.
fn write_supplement(
    path: &Path,
    hip: &HipparcosCatalog,
    unmatched: &[u64],
    ref_epoch: f64,
    coef_with_bv: (f64, f64, f64),
    coef_no_bv: (f64, f64),
) -> Result<()> {
    let dt = ref_epoch - HIPPARCOS_EPOCH;
    let mut out = BufWriter::new(File::create(path).map_err(StarfieldError::IoError)?);
    writeln!(
        out,
        "# Hipparcos bright-star supplement — generated by hipparcos-gaia-match"
    )
    .map_err(StarfieldError::IoError)?;
    writeln!(
        out,
        "# Position propagated from Hipparcos epoch (J{:.2}) to ref_epoch J{:.2} via Hipparcos PM.",
        HIPPARCOS_EPOCH, ref_epoch
    )
    .map_err(StarfieldError::IoError)?;
    writeln!(
        out,
        "# fitted_g_mag uses G ≈ {:+.4} {:+.4}*Hp {:+.4}*(B-V) when B-V is present,",
        coef_with_bv.0, coef_with_bv.1, coef_with_bv.2
    )
    .map_err(StarfieldError::IoError)?;
    writeln!(
        out,
        "# else the fallback G ≈ {:+.4} {:+.4}*Hp.",
        coef_no_bv.0, coef_no_bv.1
    )
    .map_err(StarfieldError::IoError)?;
    writeln!(
        out,
        "hip,ra,dec,parallax_mas,pmra_mas_yr,pmdec_mas_yr,b_v,fitted_g_mag"
    )
    .map_err(StarfieldError::IoError)?;
    for &hip_id in unmatched {
        let h = match hip.get_star(hip_id as usize) {
            Some(h) => h,
            None => continue,
        };
        let pm_ra = h.pm_ra.unwrap_or(0.0);
        let pm_dec = h.pm_dec.unwrap_or(0.0);
        let cos_dec = h.dec.to_radians().cos().max(1e-9);
        let dra_deg = pm_ra * dt / 3.6e6 / cos_dec;
        let ddec_deg = pm_dec * dt / 3.6e6;
        let ra_t = wrap_ra_deg(h.ra + dra_deg);
        let dec_t = (h.dec + ddec_deg).clamp(-90.0, 90.0);

        let fitted_g = match h.b_v {
            Some(bv) if bv.is_finite() => {
                coef_with_bv.0 + coef_with_bv.1 * h.mag + coef_with_bv.2 * bv
            }
            _ => coef_no_bv.0 + coef_no_bv.1 * h.mag,
        };

        writeln!(
            out,
            "{},{:.6},{:.6},{},{},{},{},{:.4}",
            h.hip,
            ra_t,
            dec_t,
            opt_f(h.parallax),
            opt_f(h.pm_ra),
            opt_f(h.pm_dec),
            opt_f(h.b_v),
            fitted_g
        )
        .map_err(StarfieldError::IoError)?;
    }
    out.flush().map_err(StarfieldError::IoError)?;
    Ok(())
}

fn plot_g_vs_hp(
    data: &[(f64, f64, f64)],
    out_path: &Path,
    coef: (f64, f64, f64),
    rms: f64,
    release: ReleaseChoice,
) -> Result<()> {
    let root = BitMapBackend::new(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE).map_err(plotter_err)?;

    let (hp_min, hp_max, g_min, g_max, bv_min, bv_max) = data_bounds(data);
    let pad = |a: f64, b: f64| {
        let r = b - a;
        (a - 0.05 * r, b + 0.05 * r)
    };
    let (hp_lo, hp_hi) = pad(hp_min, hp_max);
    let (g_lo, g_hi) = pad(g_min, g_max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Hipparcos × Gaia {}: G vs Hp (N={}, RMS={:.3})",
                release.as_str().to_uppercase(),
                data.len(),
                rms
            ),
            ("sans-serif", 28),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(hp_lo..hp_hi, g_lo..g_hi)
        .map_err(plotter_err)?;
    chart
        .configure_mesh()
        .x_desc("Hipparcos Hp magnitude")
        .y_desc(format!(
            "Gaia {} G magnitude",
            release.as_str().to_uppercase()
        ))
        .draw()
        .map_err(plotter_err)?;

    chart
        .draw_series(data.iter().map(|&(hp, bv, g)| {
            let t = ((bv - bv_min) / (bv_max - bv_min).max(1e-9)).clamp(0.0, 1.0);
            let color = HSLColor(0.7 - 0.7 * t, 0.65, 0.45); // blue → red
            Circle::new((hp, g), 2, color.filled())
        }))
        .map_err(plotter_err)?;

    let mean_bv: f64 = data.iter().map(|&(_, b, _)| b).sum::<f64>() / data.len() as f64;
    let line_pts: Vec<(f64, f64)> = (0..200)
        .map(|i| {
            let t = i as f64 / 199.0;
            let hp = hp_lo + t * (hp_hi - hp_lo);
            let g = coef.0 + coef.1 * hp + coef.2 * mean_bv;
            (hp, g)
        })
        .collect();
    chart
        .draw_series(LineSeries::new(line_pts, BLACK.stroke_width(2)))
        .map_err(plotter_err)?
        .label(format!(
            "G = {:+.3} {:+.3}·Hp {:+.3}·(B-V) @ <B-V>={:.2}",
            coef.0, coef.1, coef.2, mean_bv
        ))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(2)));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .draw()
        .map_err(plotter_err)?;

    root.present().map_err(plotter_err)?;
    Ok(())
}

fn data_bounds(data: &[(f64, f64, f64)]) -> (f64, f64, f64, f64, f64, f64) {
    let mut hp_min = f64::INFINITY;
    let mut hp_max = f64::NEG_INFINITY;
    let mut g_min = f64::INFINITY;
    let mut g_max = f64::NEG_INFINITY;
    let mut bv_min = f64::INFINITY;
    let mut bv_max = f64::NEG_INFINITY;
    for &(hp, bv, g) in data {
        if hp < hp_min {
            hp_min = hp;
        }
        if hp > hp_max {
            hp_max = hp;
        }
        if g < g_min {
            g_min = g;
        }
        if g > g_max {
            g_max = g;
        }
        if bv < bv_min {
            bv_min = bv;
        }
        if bv > bv_max {
            bv_max = bv;
        }
    }
    (hp_min, hp_max, g_min, g_max, bv_min, bv_max)
}

fn plotter_err<E: std::fmt::Display>(e: E) -> StarfieldError {
    StarfieldError::DataError(format!("plotters: {}", e))
}
