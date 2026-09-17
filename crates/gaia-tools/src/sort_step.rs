//! Optional post-processing step: sort each shard file by a chosen field.
//!
//! Reads the shard back through the release's `Dr{N}Catalog::from_csv_file`
//! path (so the data is parsed into typed entries with the same projection
//! the loader uses), sorts by the requested field, and rewrites the file
//! atomically via a tempfile + rename. In-memory per shard.

use crate::cli::SortField;
use flate2::{write::GzEncoder, Compression};
use starfield_core::{Result, StarfieldError};
use starfield_gaia::common::reader::CsvSourceReader;
use starfield_gaia::{GaiaRelease, GaiaSource};
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub fn sort_shard<R: GaiaRelease>(path: &Path, by: SortField) -> Result<()> {
    // Read every entry from the shard (no mag filtering — keep everything).
    let reader = CsvSourceReader::<R>::open(path, f64::INFINITY)?;
    let mut entries: Vec<R::Entry> = Vec::new();
    for e in reader {
        entries.push(e?);
    }

    entries.sort_by(|a, b| compare(a, b, by));

    // Write back to a sibling tmp file, then rename over.
    let tmp_path = path.with_extension("csv.gz.sorting");
    {
        let file = File::create(&tmp_path).map_err(StarfieldError::IoError)?;
        let gz = GzEncoder::new(file, Compression::default());
        let mut buf = BufWriter::new(gz);
        writeln!(buf, "{}", R::csv_header()).map_err(StarfieldError::IoError)?;
        for e in &entries {
            writeln!(buf, "{}", R::format_csv_row(e)).map_err(StarfieldError::IoError)?;
        }
        let gz = buf
            .into_inner()
            .map_err(|e: std::io::IntoInnerError<_>| StarfieldError::IoError(e.into_error()))?;
        gz.finish().map_err(StarfieldError::IoError)?;
    }
    fs::rename(&tmp_path, path).map_err(StarfieldError::IoError)?;
    Ok(())
}

fn compare<E: GaiaSource>(a: &E, b: &E, by: SortField) -> Ordering {
    let ca = a.core();
    let cb = b.core();
    match by {
        SortField::SourceId => ca.source_id.cmp(&cb.source_id),
        SortField::PhotGMeanMag => ca
            .phot_g_mean_mag
            .partial_cmp(&cb.phot_g_mean_mag)
            .unwrap_or(Ordering::Equal),
        SortField::Ra => ca.ra.partial_cmp(&cb.ra).unwrap_or(Ordering::Equal),
        SortField::RandomIndex => match (ca.random_index, cb.random_index) {
            (Some(x), Some(y)) => x.cmp(&y),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        },
    }
}
