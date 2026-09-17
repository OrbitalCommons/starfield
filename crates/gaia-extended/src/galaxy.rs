//! `gaiadr3.galaxy_candidates` loader.
//!
//! The published table records DSC class probabilities and Sersic morphology
//! fits for ~6.6 M extragalactic candidates. The catalog has no `phot_g_mean_mag`
//! of its own — to render with photometry, join `source_id` against the
//! corresponding `gaia_source` row.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use starfield_core::catalogs::{ExtendedSource, SersicProfile};
use starfield_core::{Result, StarfieldError};

use crate::parse::{
    open_csv, parse_opt_f32, parse_opt_string, require_f64, require_u64, ColumnIndex,
};
use starfield_gaia::Cone;

/// One galaxy-candidate entry.
///
/// Fields beyond the mandatory `source_id` / `ra` / `dec` are `Option`
/// because the published CSV may omit columns or leave individual cells
/// blank. Rendering consumers should fall back to a default Sersic when a
/// fit is missing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dr3GalaxyCandidate {
    /// Foreign key to `gaia_source.source_id`. Use this to fetch
    /// photometry / parallax / proper motion when needed.
    pub source_id: u64,
    pub ra: f64,
    pub dec: f64,

    // ------------------------------------------------------------------
    // DSC (Discrete Source Classifier) outputs
    // ------------------------------------------------------------------
    /// DSC label (e.g. `"galaxy"`) when present.
    pub classlabel_dsc: Option<String>,
    /// Joint label combining DSC + further classifiers.
    pub classlabel_dsc_joint: Option<String>,
    /// DSC combined-mod galaxy class probability (0..=1).
    pub classprob_dsc_combmod_galaxy: Option<f32>,
    /// DSC combined-mod quasar class probability (0..=1) — present here
    /// because the same DSC gates both `galaxy_candidates` and
    /// `qso_candidates`.
    pub classprob_dsc_combmod_quasar: Option<f32>,

    // ------------------------------------------------------------------
    // Sersic morphology fit
    // ------------------------------------------------------------------
    /// Sersic effective radius (arcsec).
    pub radius_sersic: Option<f32>,
    /// Sersic index n (1.0 = exponential disk, 4.0 = de Vaucouleurs bulge).
    pub n_sersic: Option<f32>,
    /// Ellipticity e = 1 - b/a (0 = circular).
    pub ellipticity_sersic: Option<f32>,
    /// Position angle of major axis, degrees east of north.
    pub pa_sersic: Option<f32>,
    /// Goodness-of-fit statistic from the galaxy morphology fit.
    pub gof_galaxy: Option<f32>,

    // ------------------------------------------------------------------
    // Variability classifier
    // ------------------------------------------------------------------
    /// Best-class label from the variability pipeline.
    pub vari_best_class_name: Option<String>,
    /// Score for the best variability class (0..=1).
    pub vari_best_class_score: Option<f32>,
}

impl ExtendedSource for Dr3GalaxyCandidate {
    /// Returns `Some(SersicProfile)` only when the four shape fields
    /// (`radius_sersic`, `n_sersic`, `ellipticity_sersic`, `pa_sersic`)
    /// are all populated. The DR3 morphology pipeline can fail per
    /// source — rows with any field missing are unfit and a renderer
    /// should fall back to a point source.
    fn sersic_profile(&self) -> Option<SersicProfile> {
        let radius = self.radius_sersic?;
        let n = self.n_sersic?;
        let ellipticity = self.ellipticity_sersic?;
        let pa = self.pa_sersic?;
        // Convert our (e = 1 - b/a) to upstream's `axis_ratio` (b/a).
        let axis_ratio = (1.0 - ellipticity as f64).clamp(0.0, 1.0);
        Some(SersicProfile {
            theta_half_arcsec: radius as f64,
            n: n as f64,
            axis_ratio,
            position_angle_deg: pa as f64,
        })
    }
}

/// In-memory catalog of [`Dr3GalaxyCandidate`] keyed by `source_id`.
#[derive(Debug, Default)]
pub struct Dr3GalaxyCatalog {
    entries: HashMap<u64, Dr3GalaxyCandidate>,
}

impl Dr3GalaxyCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load every row from a `.csv` or `.csv.gz` file.
    ///
    /// Errors out when the file doesn't have at least the
    /// `source_id`, `ra`, and `dec` columns. Other columns are looked up
    /// by name and stored as `None` if missing.
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
        let i_classprob_dsc_galaxy = cols.optional("classprob_dsc_combmod_galaxy");
        let i_classprob_dsc_quasar = cols.optional("classprob_dsc_combmod_quasar");

        let i_radius_sersic = cols.optional("radius_sersic");
        let i_n_sersic = cols.optional("n_sersic");
        let i_ellipticity_sersic = cols.optional("ellipticity_sersic");
        let i_pa_sersic = cols.optional("pa_sersic");
        let i_gof_galaxy = cols.optional("gof_galaxy");

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

            let entry = Dr3GalaxyCandidate {
                source_id,
                ra,
                dec,
                classlabel_dsc: parse_opt_string(&fields, i_classlabel_dsc),
                classlabel_dsc_joint: parse_opt_string(&fields, i_classlabel_dsc_joint),
                classprob_dsc_combmod_galaxy: parse_opt_f32(&fields, i_classprob_dsc_galaxy)?,
                classprob_dsc_combmod_quasar: parse_opt_f32(&fields, i_classprob_dsc_quasar)?,
                radius_sersic: parse_opt_f32(&fields, i_radius_sersic)?,
                n_sersic: parse_opt_f32(&fields, i_n_sersic)?,
                ellipticity_sersic: parse_opt_f32(&fields, i_ellipticity_sersic)?,
                pa_sersic: parse_opt_f32(&fields, i_pa_sersic)?,
                gof_galaxy: parse_opt_f32(&fields, i_gof_galaxy)?,
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

    pub fn iter(&self) -> impl Iterator<Item = &Dr3GalaxyCandidate> {
        self.entries.values()
    }

    pub fn get(&self, source_id: u64) -> Option<&Dr3GalaxyCandidate> {
        self.entries.get(&source_id)
    }

    pub fn insert(&mut self, entry: Dr3GalaxyCandidate) {
        self.entries.insert(entry.source_id, entry);
    }

    /// Galaxies whose `(ra, dec)` falls inside `cone`. Linear scan; use
    /// when the catalog comfortably fits in memory (today's ~6.6 M rows
    /// do).
    pub fn in_cone(&self, cone: &Cone) -> Vec<&Dr3GalaxyCandidate> {
        self.entries
            .values()
            .filter(|g| cone.contains_radec_deg(g.ra, g.dec))
            .collect()
    }
}
