//! Streaming CSV reader shared by every release.
//!
//! Handles plain `.csv`, gzipped `.csv.gz`, and DR3's ECSV format (long `#`-prefixed
//! YAML header before the CSV header row). Projection is computed from the actual
//! CSV header so missing/renamed columns surface as typed errors before any data
//! is parsed.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::marker::PhantomData;
use std::path::Path;
use std::sync::Arc;

use arrow::csv::reader::Format;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use flate2::read::MultiGzDecoder;

use crate::common::traits::GaiaRelease;
use starfield_core::{Result, StarfieldError};

const BATCH_SIZE: usize = 8192;

/// Row-by-row iterator over a Gaia CSV file, typed to the release's [`Entry`](GaiaRelease::Entry).
///
/// Applies a magnitude limit on [`GaiaCore::phot_g_mean_mag`](crate::common::core::GaiaCore::phot_g_mean_mag)
/// per row; entries fainter than the limit are skipped before materialization.
pub struct CsvSourceReader<R: GaiaRelease> {
    inner: arrow::csv::Reader<Box<dyn Read>>,
    current: Option<RecordBatch>,
    row: usize,
    mag_limit: f64,
    _marker: PhantomData<R>,
}

impl<R: GaiaRelease> CsvSourceReader<R> {
    /// Open a `.csv` or `.csv.gz` file. Stars with `phot_g_mean_mag > mag_limit` are skipped.
    pub fn open(path: impl AsRef<Path>, mag_limit: f64) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(StarfieldError::IoError)?;
        let metadata = file.metadata().map_err(StarfieldError::IoError)?;
        if metadata.len() == 0 {
            return Err(StarfieldError::DataError(format!(
                "gaia csv file is empty: {}",
                path.display()
            )));
        }
        let is_gz = path.extension().is_some_and(|e| e == "gz");
        Self::from_reader(Box::new(file) as Box<dyn Read>, is_gz, mag_limit).map_err(|e| match e {
            StarfieldError::DataError(msg) => {
                StarfieldError::DataError(format!("{}: {}", path.display(), msg))
            }
            other => other,
        })
    }

    /// Wire the reader to an arbitrary byte source — typically a streamed HTTP
    /// response, so callers can excerpt without ever writing the raw catalog
    /// to disk.
    ///
    /// Set `is_gz` true for gzipped streams (this wraps in `flate2::MultiGzDecoder`,
    /// which transparently handles multi-member gzip files written by the
    /// `excerpt` writer).
    /// `mag_limit` is applied per-row at the same point as in [`open`](Self::open).
    pub fn from_reader(reader: Box<dyn Read>, is_gz: bool, mag_limit: f64) -> Result<Self> {
        let raw: Box<dyn Read> = if is_gz {
            Box::new(BufReader::new(MultiGzDecoder::new(BufReader::new(reader))))
        } else {
            Box::new(BufReader::new(reader))
        };
        let mut buf = BufReader::new(raw);

        if R::IS_ECSV {
            strip_ecsv_header(&mut buf)?;
        }

        let header = read_csv_header(&mut buf)?;
        let csv_columns: Vec<&str> = header.trim_end().split(',').collect();

        // Arrow CSV's projection requires the schema to describe *every* column
        // in the source file. We build a full-width schema by overlaying our
        // typed schema (per-release `arrow_schema`) onto the CSV header order;
        // unknown columns get DataType::Utf8 and are projected out.
        let typed = R::arrow_schema();
        let mut typed_by_name: std::collections::HashMap<&str, &Field> = Default::default();
        for f in typed.fields() {
            typed_by_name.insert(f.name().as_str(), f);
        }
        let full_fields: Vec<Field> = csv_columns
            .iter()
            .map(|name| match typed_by_name.get(name) {
                Some(f) => Field::new(*name, f.data_type().clone(), f.is_nullable()),
                None => Field::new(*name, DataType::Utf8, true),
            })
            .collect();
        let full_schema = Arc::new(Schema::new(full_fields));

        // Projection picks the columns we actually want (in our schema's order
        // so `build_entry` can index batches by `ColIdx`).
        let projection: Vec<usize> = typed
            .fields()
            .iter()
            .map(|f| {
                csv_columns
                    .iter()
                    .position(|c| *c == f.name())
                    .ok_or_else(|| {
                        StarfieldError::DataError(format!(
                            "missing column `{}` (have {} columns)",
                            f.name(),
                            csv_columns.len()
                        ))
                    })
            })
            .collect::<Result<_>>()?;

        // Empty string OR literal "null" → arrow-csv treats as null. DR1/DR2
        // CSVs use empty cells; DR3 ECSV uses the literal "null" token.
        let null_re = regex::Regex::new(r"^(null)?$")
            .map_err(|e| StarfieldError::DataError(format!("compile null regex: {}", e)))?;
        let format = Format::default()
            .with_header(false)
            .with_null_regex(null_re);
        let reader = ReaderBuilder::new(full_schema)
            .with_format(format)
            .with_batch_size(BATCH_SIZE)
            .with_projection(projection)
            .build(Box::new(buf) as Box<dyn Read>)
            .map_err(|e| StarfieldError::DataError(format!("gaia csv builder failed: {}", e)))?;

        Ok(Self {
            inner: reader,
            current: None,
            row: 0,
            mag_limit,
            _marker: PhantomData,
        })
    }

    fn advance(&mut self) -> Result<Option<&RecordBatch>> {
        if self.current.is_none() {
            self.current = match self.inner.next() {
                Some(Ok(b)) => Some(b),
                Some(Err(e)) => {
                    return Err(StarfieldError::DataError(format!(
                        "gaia csv read error: {}",
                        e
                    )))
                }
                None => None,
            };
            self.row = 0;
        }
        Ok(self.current.as_ref())
    }
}

impl<R: GaiaRelease> Iterator for CsvSourceReader<R> {
    type Item = Result<R::Entry>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.advance() {
                Ok(Some(_)) => {}
                Ok(None) => return None,
                Err(e) => return Some(Err(e)),
            }
            let batch = self.current.as_ref().unwrap();
            if self.row >= batch.num_rows() {
                self.current = None;
                continue;
            }
            let row = self.row;
            self.row += 1;

            let entry = match R::build_entry(batch, row) {
                Ok(e) => e,
                Err(e) => return Some(Err(e)),
            };
            use crate::common::traits::GaiaSource;
            if entry.core().phot_g_mean_mag > self.mag_limit {
                continue;
            }
            return Some(Ok(entry));
        }
    }
}

/// Consume `#`-prefixed comment lines (ECSV YAML metadata) from the reader until
/// the first non-comment line. That line is the CSV header — it's left in the buffer.
fn strip_ecsv_header<B: BufRead>(buf: &mut B) -> Result<()> {
    loop {
        let peek = buf.fill_buf().map_err(StarfieldError::IoError)?;
        if peek.is_empty() {
            return Err(StarfieldError::DataError(
                "gaia ecsv: file ended before CSV header".into(),
            ));
        }
        if peek[0] != b'#' {
            return Ok(());
        }
        let mut line = String::new();
        buf.read_line(&mut line).map_err(StarfieldError::IoError)?;
    }
}

fn read_csv_header<B: BufRead>(buf: &mut B) -> Result<String> {
    let mut header = String::new();
    let n = buf
        .read_line(&mut header)
        .map_err(StarfieldError::IoError)?;
    if n == 0 {
        return Err(StarfieldError::DataError(
            "gaia csv: file has no header line".into(),
        ));
    }
    Ok(header)
}
