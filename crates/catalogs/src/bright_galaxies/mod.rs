//! Hand-curated supplement of bright nearby galaxies.
//!
//! Both `starfield-nsa` (SDSS-derived) and `starfield-gaia-extended`
//! (Gaia DR3 `galaxy_candidates`) systematically miss the naked-eye and
//! wide-FOV bright extended objects — M31, M33, the Magellanic Clouds,
//! M51, M81/82, M101, M104, NGC 253, Centaurus A, Fornax A, the Virgo
//! cluster headliners, etc. NSA explicitly excluded them because the
//! SDSS photometric pipeline can't fit objects that span many image
//! fields; Gaia's morphology pipeline can't fit them either because each
//! `galaxy_candidates` row assumes a roughly point-like-to-few-arcsec
//! source.
//!
//! This crate fills that gap with a small (< 50 entries) hand-curated
//! list of nearby bright galaxies, parameterized as Sersic profiles so
//! they slot into the same renderer path as either of the larger
//! catalogs above.
//!
//! # Caveats
//!
//! - **Sersic is an approximation** for many entries. Ellipticals and
//!   bulge-dominated systems (n=4) are fine; late-type disks (n≈1) are
//!   reasonable; LMC/SMC and other strongly irregular systems use Sersic
//!   only as a rough surface-brightness envelope. The `notes` field
//!   flags entries where the approximation is loose.
//! - **Sizes** (`radius_sersic_arcsec`) are half-light radii drawn from
//!   literature Sersic fits where available, otherwise scaled from RC3
//!   D25 isophotal diameters by a morphology-dependent factor (~0.3–0.4
//!   of the semi-major D25 axis). Trust to ~30%.
//! - **`mag_v`** is the integrated Johnson V from RC3 / NED; trust to
//!   ~0.1 mag.
//!
//! Use this catalog as a *visual* supplement, not for photometric
//! calibration.
//!
//! # Example
//!
//! ```no_run
//! use starfield_catalogs::bright_galaxies::BrightGalaxyCatalog;
//! use starfield_catalogs::gaia::Cone;
//!
//! let cat = BrightGalaxyCatalog::load_embedded()?;
//! let m31 = cat.get("M31").expect("M31 is in the supplement");
//! let virgo = Cone::from_degrees(187.7, 12.4, 5.0);
//! let nearby = cat.in_cone(&virgo);
//! println!("{} bright galaxies in the Virgo cone", nearby.len());
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use starfield_core::catalogs::{ExtendedSource, SersicProfile};
use starfield_core::{Result, StarfieldError};

use starfield_catalogs::gaia::Cone;

const EMBEDDED_CSV: &str = include_str!("../../data/bright-galaxies/bright-galaxies.csv");

/// One bright-galaxy supplement entry.
///
/// Sersic parameters are fit (or approximated — see `notes`) so the
/// entry can be rendered with the same `SersicSplat`-style depositor a
/// consumer uses for `Dr3GalaxyCandidate` or `NsaEntry`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrightGalaxy {
    /// Common Messier / NGC / IC / colloquial name (`"M31"`, `"NGC253"`,
    /// `"LMC"`, etc.). Used as the catalog key — must be unique.
    pub name: String,
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// RC3-style morphological type (`"Sb"`, `"E0"`, `"SBcd"`, etc.).
    pub morph_type: String,
    /// Integrated Johnson V magnitude.
    pub mag_v: f32,
    /// Sersic effective (half-light) radius, arcsec.
    pub radius_sersic_arcsec: f32,
    /// Sersic index n. 1.0 = exponential disk, 4.0 = de Vaucouleurs.
    pub n_sersic: f32,
    /// Ellipticity e = 1 - b/a (0 = circular).
    pub ellipticity_sersic: f32,
    /// Position angle of major axis, degrees east of north.
    pub pa_sersic_deg: f32,
    /// Free-form provenance / approximation notes.
    pub notes: String,
}

impl ExtendedSource for BrightGalaxy {
    fn sersic_profile(&self) -> Option<SersicProfile> {
        // Convert our (e = 1 - b/a) to upstream's `axis_ratio` (b/a).
        let axis_ratio = 1.0 - self.ellipticity_sersic as f64;
        Some(SersicProfile {
            theta_half_arcsec: self.radius_sersic_arcsec as f64,
            n: self.n_sersic as f64,
            axis_ratio,
            position_angle_deg: self.pa_sersic_deg as f64,
        })
    }
}

/// In-memory catalog of [`BrightGalaxy`] keyed by `name`.
#[derive(Debug, Default)]
pub struct BrightGalaxyCatalog {
    by_name: HashMap<String, BrightGalaxy>,
}

impl BrightGalaxyCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse the embedded supplement CSV bundled with this crate.
    pub fn load_embedded() -> Result<Self> {
        Self::from_csv_str(EMBEDDED_CSV, "<embedded>")
    }

    /// Parse a `.csv` file off disk. The schema must match the header
    /// of the embedded CSV (same columns, same order).
    pub fn from_csv_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path).map_err(StarfieldError::IoError)?;
        Self::from_csv_str(&text, &path.display().to_string())
    }

    fn from_csv_str(text: &str, source: &str) -> Result<Self> {
        let mut by_name = HashMap::new();
        let mut lines = text.lines();
        let header = lines.next().ok_or_else(|| {
            StarfieldError::DataError(format!("{}: empty CSV (no header)", source))
        })?;
        let cols: Vec<&str> = header.trim().split(',').collect();
        let expected = [
            "name",
            "ra_deg",
            "dec_deg",
            "morph_type",
            "mag_v",
            "radius_sersic_arcsec",
            "n_sersic",
            "ellipticity_sersic",
            "pa_sersic_deg",
            "notes",
        ];
        if cols.len() < expected.len() || cols[..expected.len()] != expected {
            return Err(StarfieldError::DataError(format!(
                "{}: header columns must be {:?}, got {:?}",
                source, expected, cols
            )));
        }

        for (lineno, raw) in lines.enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Notes is the trailing field; split into 10 pieces only so
            // any internal commas (none today, but be safe) don't break
            // the row.
            let fields: Vec<&str> = line.splitn(expected.len(), ',').collect();
            if fields.len() != expected.len() {
                return Err(StarfieldError::DataError(format!(
                    "{}: line {} has {} fields, expected {}",
                    source,
                    lineno + 2,
                    fields.len(),
                    expected.len()
                )));
            }
            let name = fields[0].trim().to_string();
            let entry = BrightGalaxy {
                name: name.clone(),
                ra_deg: parse_f64(fields[1], "ra_deg", source, lineno + 2)?,
                dec_deg: parse_f64(fields[2], "dec_deg", source, lineno + 2)?,
                morph_type: fields[3].trim().to_string(),
                mag_v: parse_f32(fields[4], "mag_v", source, lineno + 2)?,
                radius_sersic_arcsec: parse_f32(
                    fields[5],
                    "radius_sersic_arcsec",
                    source,
                    lineno + 2,
                )?,
                n_sersic: parse_f32(fields[6], "n_sersic", source, lineno + 2)?,
                ellipticity_sersic: parse_f32(fields[7], "ellipticity_sersic", source, lineno + 2)?,
                pa_sersic_deg: parse_f32(fields[8], "pa_sersic_deg", source, lineno + 2)?,
                notes: fields[9].trim().to_string(),
            };
            if by_name.insert(name.clone(), entry).is_some() {
                return Err(StarfieldError::DataError(format!(
                    "{}: duplicate entry name {:?} (line {})",
                    source,
                    name,
                    lineno + 2
                )));
            }
        }
        Ok(Self { by_name })
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &BrightGalaxy> {
        self.by_name.values()
    }

    pub fn get(&self, name: &str) -> Option<&BrightGalaxy> {
        self.by_name.get(name)
    }

    pub fn insert(&mut self, entry: BrightGalaxy) {
        self.by_name.insert(entry.name.clone(), entry);
    }

    /// Galaxies whose `(ra, dec)` falls inside `cone`.
    pub fn in_cone(&self, cone: &Cone) -> Vec<&BrightGalaxy> {
        self.by_name
            .values()
            .filter(|g| cone.contains_radec_deg(g.ra_deg, g.dec_deg))
            .collect()
    }
}

fn parse_f64(s: &str, label: &str, source: &str, lineno: usize) -> Result<f64> {
    s.trim().parse::<f64>().map_err(|e| {
        StarfieldError::DataError(format!(
            "{}: line {}: column {} not f64: {} (got {:?})",
            source, lineno, label, e, s
        ))
    })
}

fn parse_f32(s: &str, label: &str, source: &str, lineno: usize) -> Result<f32> {
    s.trim().parse::<f32>().map_err(|e| {
        StarfieldError::DataError(format!(
            "{}: line {}: column {} not f32: {} (got {:?})",
            source, lineno, label, e, s
        ))
    })
}
