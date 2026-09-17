//! ESA Gaia DR3 extended-source catalogs.
//!
//! DR3 publishes two morphology-aware catalogs of extended sources, in
//! addition to the main `gaia_source` table that `starfield-gaia` covers:
//!
//! - **`gaiadr3.galaxy_candidates`** (~6.6 M rows) — galaxy candidates
//!   with Sersic morphology fits and DSC class probabilities.
//! - **`gaiadr3.qso_candidates`** (~6.6 M rows) — QSO candidates with
//!   photometric redshifts and ICRF cross-matches.
//!
//! Neither table carries `phot_g_mean_mag` directly — they reference the
//! `gaia_source` row via `source_id`. To attach photometry, join with
//! the matching DR3 row in `starfield-gaia` (e.g. via
//! [`Dr3Catalog`](starfield_gaia::Dr3Catalog)).
//!
//! # Data acquisition
//!
//! The two CSVs ship from ESA's CDN at:
//!
//! - `https://cdn.gea.esac.esa.int/Gaia/gdr3/Misc/galaxy_candidates/`
//! - `https://cdn.gea.esac.esa.int/Gaia/gdr3/Misc/qso_candidates/`
//!
//! For now this crate just parses the published CSV files (`.csv` or
//! `.csv.gz`). A bulk downloader is left as future work — the row counts
//! are small enough (~6.6 M each) that one-shot manual downloads are
//! tolerable.
//!
//! # Schema flexibility
//!
//! The parsers look up columns by **header name**, not position. Unknown
//! columns are ignored. Required columns (`source_id`, `ra`, `dec`)
//! produce a typed error if missing; everything else is optional and
//! degrades to `None` when the column isn't present in the file.
//!
//! # Example
//!
//! ```no_run
//! use starfield_gaia_extended::{Dr3GalaxyCatalog, Dr3QsoCatalog};
//! use starfield_gaia::Cone;
//!
//! let galaxies = Dr3GalaxyCatalog::from_csv_file("galaxy_candidates.csv.gz")?;
//! let quasars = Dr3QsoCatalog::from_csv_file("qso_candidates.csv.gz")?;
//!
//! let cone = Cone::from_degrees(266.4, -29.0, 5.0);
//! let nearby_galaxies = galaxies.in_cone(&cone);
//! let nearby_quasars = quasars.in_cone(&cone);
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

pub mod galaxy;
pub mod parse;
pub mod qso;

pub use galaxy::{Dr3GalaxyCandidate, Dr3GalaxyCatalog};
pub use qso::{Dr3QsoCandidate, Dr3QsoCatalog};
