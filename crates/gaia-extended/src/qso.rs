//! `gaiadr3.qso_candidates` loader.
//!
//! ~6.6 M quasar candidates with DSC + variability classifications, photometric
//! redshift estimates from the QSOC pipeline, and ICRF cross-matches.
//! No `phot_g_mean_mag` in this table either — join via `source_id` against
//! `gaia_source` to get photometry.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use starfield_core::{Result, StarfieldError};

use crate::parse::{
    open_csv, parse_opt_bool, parse_opt_f32, parse_opt_string, require_f64, require_u64,
    ColumnIndex,
};
use starfield_gaia::Cone;

/// One QSO-candidate entry.
///
/// Fields beyond the mandatory `source_id` / `ra` / `dec` are `Option`
/// so an unfit/unclassified row degrades gracefully rather than failing
/// the load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dr3QsoCandidate {
    /// Foreign key to `gaia_source.source_id`.
    pub source_id: u64,
    pub ra: f64,
    pub dec: f64,

    // ------------------------------------------------------------------
    // DSC outputs
    // ------------------------------------------------------------------
    pub classlabel_dsc: Option<String>,
    pub classlabel_dsc_joint: Option<String>,
    /// DSC combined-mod quasar probability (0..=1).
    pub classprob_dsc_combmod_quasar: Option<f32>,
    /// DSC combined-mod galaxy probability (0..=1).
    pub classprob_dsc_combmod_galaxy: Option<f32>,

    // ------------------------------------------------------------------
    // QSOC photometric redshift
    // ------------------------------------------------------------------
    /// QSOC pipeline photo-z point estimate.
    pub redshift_qsoc: Option<f32>,
    /// 1-sigma lower bound on `redshift_qsoc`.
    pub redshift_qsoc_lower: Option<f32>,
    /// 1-sigma upper bound on `redshift_qsoc`.
    pub redshift_qsoc_upper: Option<f32>,

    // ------------------------------------------------------------------
    // Reference-frame cross-match
    // ------------------------------------------------------------------
    /// True when the source is part of Gaia-CRF3 (the reference QSO frame).
    pub gaia_crf_source: Option<bool>,
    /// True when host-galaxy contamination is flagged.
    pub host_galaxy_flag: Option<bool>,

    // ------------------------------------------------------------------
    // Variability classifier
    // ------------------------------------------------------------------
    pub vari_best_class_name: Option<String>,
    pub vari_best_class_score: Option<f32>,
}

/// In-memory catalog of [`Dr3QsoCandidate`] keyed by `source_id`.
#[derive(Debug, Default)]
pub struct Dr3QsoCatalog {
    entries: HashMap<u64, Dr3QsoCandidate>,
}

impl Dr3QsoCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every row from a `.csv` or `.csv.gz` file.
    ///
    /// `source_id`, `ra`, and `dec` are required; missing or empty cells
    /// in any other column degrade to `None`.
    pub fn from_csv_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = open_csv(path)?;
        let mut header_line = String::new();
        let n = reader
            .read_line(&mut header_line)
            .map_err(StarfieldError::IoError)?;
        if n == 0 {
            return Err(StarfieldError::DataError(format!(
                "{}: file is empty",
                path.display()
            )));
        }
        let cols = ColumnIndex::from_header(&header_line, path.display().to_string());

        let i_source_id = cols.require("source_id")?;
        let i_ra = cols.require("ra")?;
        let i_dec = cols.require("dec")?;

        let i_classlabel_dsc = cols.optional("classlabel_dsc");
        let i_classlabel_dsc_joint = cols.optional("classlabel_dsc_joint");
        let i_classprob_dsc_quasar = cols.optional("classprob_dsc_combmod_quasar");
        let i_classprob_dsc_galaxy = cols.optional("classprob_dsc_combmod_galaxy");

        let i_redshift_qsoc = cols.optional("redshift_qsoc");
        let i_redshift_qsoc_lower = cols.optional("redshift_qsoc_lower");
        let i_redshift_qsoc_upper = cols.optional("redshift_qsoc_upper");

        let i_gaia_crf_source = cols.optional("gaia_crf_source");
        let i_host_galaxy_flag = cols.optional("host_galaxy_flag");

        let i_vari_best_class_name = cols.optional("vari_best_class_name");
        let i_vari_best_class_score = cols.optional("vari_best_class_score");

        let mut entries = HashMap::new();
        let mut line = String::new();
        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .map_err(StarfieldError::IoError)?;
            if n == 0 {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.trim_end().split(',').collect();
            let source_id = require_u64(&fields, i_source_id, "source_id", &cols.source)?;
            let ra = require_f64(&fields, i_ra, "ra", &cols.source)?;
            let dec = require_f64(&fields, i_dec, "dec", &cols.source)?;

            let entry = Dr3QsoCandidate {
                source_id,
                ra,
                dec,
                classlabel_dsc: parse_opt_string(&fields, i_classlabel_dsc),
                classlabel_dsc_joint: parse_opt_string(&fields, i_classlabel_dsc_joint),
                classprob_dsc_combmod_quasar: parse_opt_f32(&fields, i_classprob_dsc_quasar)?,
                classprob_dsc_combmod_galaxy: parse_opt_f32(&fields, i_classprob_dsc_galaxy)?,
                redshift_qsoc: parse_opt_f32(&fields, i_redshift_qsoc)?,
                redshift_qsoc_lower: parse_opt_f32(&fields, i_redshift_qsoc_lower)?,
                redshift_qsoc_upper: parse_opt_f32(&fields, i_redshift_qsoc_upper)?,
                gaia_crf_source: parse_opt_bool(&fields, i_gaia_crf_source)?,
                host_galaxy_flag: parse_opt_bool(&fields, i_host_galaxy_flag)?,
                vari_best_class_name: parse_opt_string(&fields, i_vari_best_class_name),
                vari_best_class_score: parse_opt_f32(&fields, i_vari_best_class_score)?,
            };
            entries.insert(source_id, entry);
        }
        Ok(Self { entries })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Dr3QsoCandidate> {
        self.entries.values()
    }

    pub fn get(&self, source_id: u64) -> Option<&Dr3QsoCandidate> {
        self.entries.get(&source_id)
    }

    pub fn insert(&mut self, entry: Dr3QsoCandidate) {
        self.entries.insert(entry.source_id, entry);
    }

    /// QSOs whose `(ra, dec)` falls inside `cone`. Linear scan; use when
    /// the catalog comfortably fits in memory (today's ~6.6 M rows do).
    pub fn in_cone(&self, cone: &Cone) -> Vec<&Dr3QsoCandidate> {
        self.entries
            .values()
            .filter(|q| cone.contains_radec_deg(q.ra, q.dec))
            .collect()
    }
}
