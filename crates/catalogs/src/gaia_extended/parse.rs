//! Header-driven CSV parser shared by both extended-source loaders.
//!
//! The published DR3 extended-source CSVs have ~50–80 columns each, of which
//! we only model the morphology / classification subset most useful for
//! rendering. To stay tolerant to schema drift we look up columns by header
//! name and pass back `None` for anything missing — only `source_id`, `ra`,
//! and `dec` are mandatory (signaled by [`ColumnIndex::require`]).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use flate2::read::MultiGzDecoder;
use starfield_core::{Result, StarfieldError};

/// Header → column-index lookup over the first line of a CSV file.
///
/// Use [`require`](Self::require) for columns that must be present and
/// [`optional`](Self::optional) for columns that may be absent. Returned
/// indices are positions in the comma-split row.
pub struct ColumnIndex<'a> {
    by_name: HashMap<&'a str, usize>,
    pub source: String,
}

impl<'a> ColumnIndex<'a> {
    pub fn from_header(header: &'a str, source: impl Into<String>) -> Self {
        let by_name = header
            .trim_end()
            .split(',')
            .enumerate()
            .map(|(i, name)| (name, i))
            .collect();
        Self {
            by_name,
            source: source.into(),
        }
    }

    /// Index of `name`, or an error naming the source file.
    pub fn require(&self, name: &str) -> Result<usize> {
        self.by_name.get(name).copied().ok_or_else(|| {
            StarfieldError::DataError(format!(
                "{}: required column {:?} missing from header",
                self.source, name
            ))
        })
    }

    /// `Some(index)` if `name` is in the header, else `None`.
    pub fn optional(&self, name: &str) -> Option<usize> {
        self.by_name.get(name).copied()
    }
}

/// Read field at `idx` from `fields`, returning `Ok(None)` for empty cells.
pub fn parse_opt_f64(fields: &[&str], idx: Option<usize>) -> Result<Option<f64>> {
    let Some(i) = idx else { return Ok(None) };
    let s = fields.get(i).copied().unwrap_or("").trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return Ok(None);
    }
    s.parse::<f64>().map(Some).map_err(|e| {
        StarfieldError::DataError(format!(
            "column at index {} not f64: {} (got {:?})",
            i, e, s
        ))
    })
}

pub fn parse_opt_f32(fields: &[&str], idx: Option<usize>) -> Result<Option<f32>> {
    parse_opt_f64(fields, idx).map(|opt| opt.map(|v| v as f32))
}

pub fn parse_opt_u64(fields: &[&str], idx: Option<usize>) -> Result<Option<u64>> {
    let Some(i) = idx else { return Ok(None) };
    let s = fields.get(i).copied().unwrap_or("").trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return Ok(None);
    }
    s.parse::<u64>().map(Some).map_err(|e| {
        StarfieldError::DataError(format!(
            "column at index {} not u64: {} (got {:?})",
            i, e, s
        ))
    })
}

pub fn parse_opt_bool(fields: &[&str], idx: Option<usize>) -> Result<Option<bool>> {
    let Some(i) = idx else { return Ok(None) };
    let s = fields.get(i).copied().unwrap_or("").trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        return Ok(None);
    }
    match s.to_ascii_lowercase().as_str() {
        "true" | "t" | "1" => Ok(Some(true)),
        "false" | "f" | "0" => Ok(Some(false)),
        _ => Err(StarfieldError::DataError(format!(
            "column at index {} not bool (got {:?})",
            i, s
        ))),
    }
}

pub fn parse_opt_string(fields: &[&str], idx: Option<usize>) -> Option<String> {
    let i = idx?;
    let s = fields.get(i).copied().unwrap_or("").trim();
    if s.is_empty() || s.eq_ignore_ascii_case("null") {
        None
    } else {
        Some(s.to_string())
    }
}

pub fn require_f64(fields: &[&str], idx: usize, label: &str, source: &str) -> Result<f64> {
    let s = fields.get(idx).copied().unwrap_or("").trim();
    if s.is_empty() {
        return Err(StarfieldError::DataError(format!(
            "{}: required column {} (idx {}) is empty",
            source, label, idx
        )));
    }
    s.parse::<f64>().map_err(|e| {
        StarfieldError::DataError(format!(
            "{}: column {} (idx {}) not f64: {} (got {:?})",
            source, label, idx, e, s
        ))
    })
}

pub fn require_u64(fields: &[&str], idx: usize, label: &str, source: &str) -> Result<u64> {
    let s = fields.get(idx).copied().unwrap_or("").trim();
    if s.is_empty() {
        return Err(StarfieldError::DataError(format!(
            "{}: required column {} (idx {}) is empty",
            source, label, idx
        )));
    }
    s.parse::<u64>().map_err(|e| {
        StarfieldError::DataError(format!(
            "{}: column {} (idx {}) not u64: {} (got {:?})",
            source, label, idx, e, s
        ))
    })
}

/// Open a `.csv` or `.csv.gz` file as a line iterator.
pub fn open_csv(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).map_err(StarfieldError::IoError)?;
    let is_gz = path.extension().is_some_and(|e| e == "gz");
    if is_gz {
        Ok(Box::new(BufReader::new(MultiGzDecoder::new(
            BufReader::new(file),
        ))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}
