//! Frozen-snapshot element tables and live-SBDB drift detection.
//!
//! A common data-hygiene pattern for small-body work: a crate carries a
//! *frozen snapshot* of orbital elements (provenance-commented constants
//! that regression tests pin against, so builds and tests stay
//! deterministic and offline), and a separate, network-gated check diffs
//! that snapshot against live SBDB lookups with per-element tolerances.
//! Frozen tables are usually transcribed by hand, and hand transcription
//! rots silently — in one downstream workspace a ~30 m near-Earth asteroid
//! ended up listed as a 110 AU trans-Neptunian object, and a scrambled
//! semi-major-axis column displaced a whole resonance analysis by tens of
//! AU. This module is the offline diff core that catches both.
//!
//! # The frozen-snapshot + drift-allowlist pattern
//!
//! Orbit solutions are not constants: JPL refits as observed arcs
//! lengthen and republishes elements at new epochs. For distant
//! high-eccentricity objects the perihelion distance `q`, inclination `i`,
//! and node `Ω` are well constrained, while `a` and `e` are strongly
//! correlated and drift together along their fit degeneracy — fractional
//! shifts in `a` of a few ×0.1% are routine between solution epochs, and
//! the longest-period objects (e.g. Sedna, 506 → 544 AU between solutions)
//! can move by several percent or more while `q`, `i`, `Ω`, and `H` stay
//! put. A snapshot that deliberately pins *paper-epoch* solutions will
//! therefore accumulate legitimate `a`/`e` diffs over time.
//!
//! An out-of-tolerance diff means one of two things, both of which a human
//! should look at:
//!
//! 1. **Transcription error** — fix the snapshot.
//! 2. **Genuine solution drift** — either update the snapshot (re-pinning
//!    every downstream regression value that depends on it) or, if the
//!    snapshot intentionally pins a historical solution, record the known
//!    drift in the snapshot's provenance comment (a drift allowlist) so
//!    the next reviewer does not "fix" it back.
//!
//! # Offline vs network
//!
//! [`diff_elements`] and everything it touches are pure and unit-tested
//! offline. Only [`diff_against_live`] performs network lookups, and its
//! test is `#[ignore]`d like the other live SBDB tests.
//!
//! # Example
//!
//! ```
//! use starfield::sbdb::snapshot::{diff_elements, Element, ElementSnapshot, LiveElements, Tolerances};
//!
//! let snapshot = ElementSnapshot {
//!     name: "Sedna".to_string(),
//!     designation: "Sedna".to_string(),
//!     a: Some(506.0),
//!     e: Some(0.85),
//!     i_deg: Some(11.9),
//!     omega_deg: Some(311.5),
//!     omega_big_deg: Some(144.5),
//!     h_mag: Some(1.6),
//! };
//!
//! // A later JPL solution has drifted along the a-e degeneracy.
//! let live = LiveElements {
//!     a: Some(544.0),
//!     e: Some(0.855),
//!     i_deg: Some(11.9),
//!     omega_deg: Some(311.4),
//!     omega_big_deg: Some(144.4),
//!     h_mag: Some(1.5),
//!     ..Default::default()
//! };
//!
//! let diffs = diff_elements(&snapshot, &live, &Tolerances::published_table());
//! assert_eq!(diffs.len(), 1);
//! assert_eq!(diffs[0].element, Element::SemiMajorAxis);
//! ```

use crate::data::sbdb::{SbdbClient, SbdbLookupResponse};
use crate::{Result, StarfieldError};

/// Which orbital element a diff refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    /// Semi-major axis a (AU)
    SemiMajorAxis,
    /// Eccentricity e
    Eccentricity,
    /// Inclination i (degrees)
    Inclination,
    /// Argument of perihelion ω (degrees)
    ArgPerihelion,
    /// Longitude of ascending node Ω (degrees)
    AscendingNode,
    /// Absolute magnitude H
    AbsoluteMagnitude,
}

impl Element {
    /// Short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Element::SemiMajorAxis => "a [AU]",
            Element::Eccentricity => "e",
            Element::Inclination => "i [deg]",
            Element::ArgPerihelion => "omega [deg]",
            Element::AscendingNode => "Omega [deg]",
            Element::AbsoluteMagnitude => "H [mag]",
        }
    }

    /// Whether the element is an angle in degrees (delta is wrap-aware).
    pub fn is_angle(&self) -> bool {
        matches!(
            self,
            Element::Inclination | Element::ArgPerihelion | Element::AscendingNode
        )
    }
}

/// One row of a frozen table, in the form the differ compares. Elements
/// the table does not carry are `None` and are skipped.
#[derive(Debug, Clone, Default)]
pub struct ElementSnapshot {
    /// Human-readable name used in diff reports.
    pub name: String,
    /// SBDB search string (`sstr`) used for the live lookup.
    pub designation: String,
    /// Semi-major axis (AU)
    pub a: Option<f64>,
    /// Eccentricity
    pub e: Option<f64>,
    /// Inclination (degrees)
    pub i_deg: Option<f64>,
    /// Argument of perihelion (degrees)
    pub omega_deg: Option<f64>,
    /// Longitude of ascending node (degrees)
    pub omega_big_deg: Option<f64>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
}

/// Osculating elements from a live SBDB lookup, decoupled from the
/// response types so the diff logic can be unit-tested offline.
#[derive(Debug, Clone, Default)]
pub struct LiveElements {
    /// Semi-major axis (AU)
    pub a: Option<f64>,
    /// Eccentricity
    pub e: Option<f64>,
    /// Inclination (degrees)
    pub i_deg: Option<f64>,
    /// Argument of perihelion (degrees)
    pub omega_deg: Option<f64>,
    /// Longitude of ascending node (degrees)
    pub omega_big_deg: Option<f64>,
    /// Absolute magnitude H
    pub h_mag: Option<f64>,
    /// Epoch of osculation (JD TDB)
    pub epoch_jd: Option<f64>,
    /// JPL orbit solution ID (provenance for reports)
    pub orbit_id: Option<String>,
}

impl LiveElements {
    /// Flatten an SBDB lookup response into the fields the differ uses.
    ///
    /// Errors if the response carries no orbit block — for a snapshot
    /// table a missing orbit is a table bug (bad designation), not drift.
    pub fn from_lookup(response: &SbdbLookupResponse) -> Result<Self> {
        let orbit = response.orbit.as_ref().ok_or_else(|| {
            StarfieldError::DataError(format!(
                "SBDB returned no orbit for {:?}",
                response.object.designation
            ))
        })?;
        Ok(LiveElements {
            a: orbit.semi_major_axis,
            e: orbit.eccentricity,
            i_deg: orbit.inclination,
            omega_deg: orbit.arg_perihelion,
            omega_big_deg: orbit.long_asc_node,
            h_mag: response.phys_par.as_ref().and_then(|p| p.abs_magnitude_h),
            epoch_jd: orbit.epoch_jd,
            orbit_id: orbit.orbit_id.clone(),
        })
    }
}

/// Per-element comparison tolerances. Each is the *total* allowed
/// |live − snapshot|: an allowance for orbit-solution drift plus the
/// rounding half-width of the snapshot itself.
#[derive(Debug, Clone, Copy)]
pub struct Tolerances {
    /// Fractional tolerance on a (relative to the snapshot value).
    pub a_frac: f64,
    /// Absolute floor on the a tolerance (AU) — covers table rounding.
    pub a_abs: f64,
    /// Absolute tolerance on e.
    pub e_abs: f64,
    /// Absolute tolerance on i, ω, Ω (degrees, wrap-aware).
    pub angle_deg: f64,
    /// Absolute tolerance on H (mag).
    pub h_mag: f64,
}

impl Tolerances {
    /// Tolerances for tables transcribed at typical *published-table*
    /// precision: a rounded to the AU (±0.5), e to 0.01 (±0.005), angles
    /// to 0.1° (±0.05), H to 0.1 (±0.05). On top of the rounding:
    ///
    /// - a: 1% fractional drift. For distant high-eccentricity orbits a is
    ///   the loosest-constrained element (strongly correlated with e); a
    ///   few ×0.1% is routine between solution epochs and ~1% is reached
    ///   by the longest-period objects. Beyond 1% is worth a human look.
    /// - e: 0.01 — drift in e tracks a at the few ×0.001 level,
    ///   comparable to the table's own rounding.
    /// - angles: 0.5° — i, ω, Ω are typically stable to ≲0.1° once an
    ///   orbit is multi-opposition; 0.5° flags real solution changes.
    /// - H: 0.5 mag — SBDB absolute magnitudes are re-derived photometric
    ///   fits and routinely move by a few tenths, independent of
    ///   astrometry.
    pub fn published_table() -> Self {
        Tolerances {
            a_frac: 0.01,
            a_abs: 1.0,
            e_abs: 0.01,
            angle_deg: 0.5,
            h_mag: 0.5,
        }
    }

    /// Tolerances for tables transcribed at (near) full SBDB precision
    /// (a to 0.1 AU, e to 1e-4, angles to 0.01°). The same drift physics
    /// as [`Tolerances::published_table`] but tighter, appropriate for a
    /// recently frozen, unrounded snapshot: 0.5% in a, 0.005 in e, 0.1°
    /// in angles.
    pub fn full_precision() -> Self {
        Tolerances {
            a_frac: 0.005,
            a_abs: 0.2,
            e_abs: 0.005,
            angle_deg: 0.1,
            h_mag: 0.5,
        }
    }

    fn for_element(&self, element: Element, snapshot_value: f64) -> f64 {
        match element {
            Element::SemiMajorAxis => (self.a_frac * snapshot_value.abs()).max(self.a_abs),
            Element::Eccentricity => self.e_abs,
            Element::Inclination | Element::ArgPerihelion | Element::AscendingNode => {
                self.angle_deg
            }
            Element::AbsoluteMagnitude => self.h_mag,
        }
    }
}

/// One out-of-tolerance element: which object, which element, what the
/// frozen table says, what JPL says now, and by how much they disagree.
#[derive(Debug, Clone)]
pub struct ElementDiff {
    /// Object name (from the snapshot table).
    pub object: String,
    /// Which element drifted.
    pub element: Element,
    /// Value in the frozen table.
    pub snapshot: f64,
    /// Live SBDB value.
    pub live: f64,
    /// live − snapshot (wrap-aware for angles, in the element's units).
    pub delta: f64,
    /// The tolerance that was exceeded (same units).
    pub tolerance: f64,
    /// JPL orbit solution ID of the live lookup, if reported.
    pub orbit_id: Option<String>,
}

impl std::fmt::Display for ElementDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} snapshot {} vs live {} (delta {:+.4}, tolerance {:.4}{})",
            self.object,
            self.element.label(),
            self.snapshot,
            self.live,
            self.delta,
            self.tolerance,
            match &self.orbit_id {
                Some(id) => format!(", JPL soln {id}"),
                None => String::new(),
            }
        )
    }
}

/// Signed angle difference `to − from` in degrees, wrapped to (−180°, 180°].
///
/// ```
/// use starfield::sbdb::snapshot::angle_delta_deg;
/// assert!((angle_delta_deg(359.9, 0.3) - 0.4).abs() < 1e-9);
/// assert!((angle_delta_deg(0.3, 359.9) + 0.4).abs() < 1e-9);
/// ```
pub fn angle_delta_deg(from: f64, to: f64) -> f64 {
    let mut d = (to - from).rem_euclid(360.0);
    if d > 180.0 {
        d -= 360.0;
    }
    d
}

/// Diff one snapshot row against live elements. Returns only the elements
/// whose |delta| exceeds tolerance; elements missing on either side are
/// skipped (snapshot tables only carry `Some` for vetted columns, and
/// SBDB always reports the Keplerian set for real objects).
pub fn diff_elements(
    snapshot: &ElementSnapshot,
    live: &LiveElements,
    tolerances: &Tolerances,
) -> Vec<ElementDiff> {
    let pairs = [
        (Element::SemiMajorAxis, snapshot.a, live.a),
        (Element::Eccentricity, snapshot.e, live.e),
        (Element::Inclination, snapshot.i_deg, live.i_deg),
        (Element::ArgPerihelion, snapshot.omega_deg, live.omega_deg),
        (
            Element::AscendingNode,
            snapshot.omega_big_deg,
            live.omega_big_deg,
        ),
        (Element::AbsoluteMagnitude, snapshot.h_mag, live.h_mag),
    ];

    let mut diffs = Vec::new();
    for (element, snapshot_value, live_value) in pairs {
        let (Some(s), Some(l)) = (snapshot_value, live_value) else {
            continue;
        };
        let delta = if element.is_angle() {
            angle_delta_deg(s, l)
        } else {
            l - s
        };
        let tolerance = tolerances.for_element(element, s);
        if delta.abs() > tolerance {
            diffs.push(ElementDiff {
                object: snapshot.name.clone(),
                element,
                snapshot: s,
                live: l,
                delta,
                tolerance,
                orbit_id: live.orbit_id.clone(),
            });
        }
    }
    diffs
}

/// Diff a snapshot table against live SBDB lookups (network).
///
/// Errors on the first failed lookup or orbit-less response — for a
/// snapshot table a missing object is a table bug, not drift. Otherwise
/// returns every out-of-tolerance element across the whole table; an
/// empty result means the table still matches JPL to within normal
/// solution drift.
pub fn diff_against_live(
    client: &SbdbClient,
    snapshots: &[ElementSnapshot],
    tolerances: &Tolerances,
) -> Result<Vec<ElementDiff>> {
    let mut diffs = Vec::new();
    for snapshot in snapshots {
        let response = client.lookup(&snapshot.designation)?;
        let live = LiveElements::from_lookup(&response)?;
        diffs.extend(diff_elements(snapshot, &live, tolerances));
    }
    Ok(diffs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> ElementSnapshot {
        ElementSnapshot {
            name: "Testna".to_string(),
            designation: "Testna".to_string(),
            a: Some(500.0),
            e: Some(0.85),
            i_deg: Some(11.9),
            omega_deg: Some(311.5),
            omega_big_deg: Some(144.5),
            h_mag: Some(1.6),
        }
    }

    fn matching_live() -> LiveElements {
        LiveElements {
            a: Some(500.0),
            e: Some(0.85),
            i_deg: Some(11.9),
            omega_deg: Some(311.5),
            omega_big_deg: Some(144.5),
            h_mag: Some(1.6),
            ..Default::default()
        }
    }

    #[test]
    fn test_identical_elements_no_diff() {
        let diffs = diff_elements(
            &snapshot(),
            &matching_live(),
            &Tolerances::published_table(),
        );
        assert!(diffs.is_empty(), "{diffs:?}");
    }

    #[test]
    fn test_drift_within_tolerance_no_diff() {
        // 0.4% in a, half the e tolerance, 0.3 deg in omega, 0.2 mag in H:
        // all within the documented published-table allowances.
        let live = LiveElements {
            a: Some(502.0),
            e: Some(0.855),
            i_deg: Some(11.9),
            omega_deg: Some(311.8),
            omega_big_deg: Some(144.5),
            h_mag: Some(1.8),
            ..Default::default()
        };
        let diffs = diff_elements(&snapshot(), &live, &Tolerances::published_table());
        assert!(diffs.is_empty(), "{diffs:?}");
    }

    #[test]
    fn test_semi_major_axis_drift_flagged() {
        let live = LiveElements {
            a: Some(540.0), // 8% — way beyond the 1% allowance
            ..matching_live()
        };
        let diffs = diff_elements(&snapshot(), &live, &Tolerances::published_table());
        assert_eq!(diffs.len(), 1);
        let d = &diffs[0];
        assert_eq!(d.element, Element::SemiMajorAxis);
        assert_eq!(d.object, "Testna");
        assert_eq!(d.snapshot, 500.0);
        assert_eq!(d.live, 540.0);
        assert!((d.delta - 40.0).abs() < 1e-12);
        assert!((d.tolerance - 5.0).abs() < 1e-12); // 1% of 500
    }

    #[test]
    fn test_multiple_elements_flagged_independently() {
        let live = LiveElements {
            e: Some(0.88),     // |delta e| = 0.03 > 0.01
            i_deg: Some(13.0), // 1.1 deg > 0.5
            ..matching_live()
        };
        let diffs = diff_elements(&snapshot(), &live, &Tolerances::published_table());
        let elements: Vec<Element> = diffs.iter().map(|d| d.element).collect();
        assert_eq!(
            elements,
            vec![Element::Eccentricity, Element::Inclination],
            "{diffs:?}"
        );
    }

    #[test]
    fn test_angle_delta_is_wrap_aware() {
        // 359.9 vs 0.3 is a 0.4 deg difference, not 359.6.
        assert!((angle_delta_deg(359.9, 0.3) - 0.4).abs() < 1e-9);
        assert!((angle_delta_deg(0.3, 359.9) + 0.4).abs() < 1e-9);

        let snap = ElementSnapshot {
            omega_deg: Some(359.9),
            ..snapshot()
        };
        let live = LiveElements {
            omega_deg: Some(0.3),
            ..matching_live()
        };
        let diffs = diff_elements(&snap, &live, &Tolerances::published_table());
        assert!(
            diffs.is_empty(),
            "wrap-around must not be flagged: {diffs:?}"
        );
    }

    #[test]
    fn test_missing_elements_skipped() {
        // A snapshot that only carries a, e, i ignores live omega/Omega/H.
        let snap = ElementSnapshot {
            omega_deg: None,
            omega_big_deg: None,
            h_mag: None,
            ..snapshot()
        };
        let live = LiveElements {
            omega_deg: Some(100.0),    // wildly different but not compared
            omega_big_deg: Some(10.0), // ditto
            h_mag: None,
            ..matching_live()
        };
        let diffs = diff_elements(&snap, &live, &Tolerances::published_table());
        assert!(diffs.is_empty(), "{diffs:?}");
    }

    #[test]
    fn test_full_precision_is_tighter() {
        // 0.7% in a passes published_table (1%) but fails full_precision (0.5%).
        let live = LiveElements {
            a: Some(503.5),
            ..matching_live()
        };
        assert!(diff_elements(&snapshot(), &live, &Tolerances::published_table()).is_empty());
        let diffs = diff_elements(&snapshot(), &live, &Tolerances::full_precision());
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].element, Element::SemiMajorAxis);
    }

    #[test]
    fn test_diff_display_names_object_and_element() {
        let live = LiveElements {
            a: Some(540.0),
            orbit_id: Some("12".to_string()),
            ..matching_live()
        };
        let diffs = diff_elements(&snapshot(), &live, &Tolerances::published_table());
        let msg = diffs[0].to_string();
        assert!(msg.contains("Testna"), "{msg}");
        assert!(msg.contains("a [AU]"), "{msg}");
        assert!(msg.contains("540"), "{msg}");
        assert!(msg.contains("JPL soln 12"), "{msg}");
    }

    /// Live-network check: a published-precision Sedna snapshot diffed
    /// against the current JPL solution. q, i, and Omega are stable across
    /// solution epochs; a and e drift along their fit degeneracy, so this
    /// only asserts on the stable elements.
    #[test]
    #[ignore]
    fn test_diff_against_live_sedna() {
        let snapshots = vec![ElementSnapshot {
            name: "Sedna".to_string(),
            designation: "Sedna".to_string(),
            // a and e omitted on purpose: they drift along the a-e
            // degeneracy between JPL solutions (506 -> 544 AU already
            // observed) and would make this test rot.
            a: None,
            e: None,
            i_deg: Some(11.9),
            omega_deg: Some(311.5),
            omega_big_deg: Some(144.5),
            h_mag: Some(1.6),
        }];
        let client = SbdbClient::new().unwrap();
        let diffs = diff_against_live(&client, &snapshots, &Tolerances::published_table()).unwrap();
        // omega can move a few tenths of a degree between solutions; the
        // others should hold. Print rather than fail on omega-only drift.
        for d in &diffs {
            println!("{d}");
            assert!(
                d.element == Element::ArgPerihelion || d.element == Element::AbsoluteMagnitude,
                "unexpected drift: {d}"
            );
        }
    }
}
