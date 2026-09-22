//! Shared infrastructure for the Hipparcos-derived bright-star supplements
//! that fill in stars Gaia's CCDs saturate / never measure cleanly.
//!
//! Each Gaia data release (DR1, DR2, DR3) has its own slim supplement CSV
//! living under `crates/catalogs/data/gaia/dr{1,2,3}-bright-star-supplement.csv`.
//! The CSV row shape is identical across releases — what differs is:
//!
//! 1. The reference epoch (J2015.0 / J2015.5 / J2016.0) at which the
//!    Hipparcos position has been propagated.
//! 2. The fitted G-band coefficients used for `phot_g_mean_mag` (each
//!    release has its own G-vs-Hp regression vs the bright Hipparcos pairs).
//! 3. The per-release `Entry` type the supplement row is materialized into,
//!    which lives in `dr{1,2,3}/supplement.rs`.
//!
//! This module owns the [`SupplementRow`] schema, the CSV parser, the
//! synthetic-source-id masking helpers, the J2000 coordinate transforms
//! (galactic + ecliptic), and a [`make_supplement_core`] builder that
//! returns a fully populated [`GaiaCore`] — the only piece shared 1:1
//! across releases. Per-release modules wrap that core into their own
//! `Entry` struct.
//!
//! See `dr{1,2,3}::supplement` for the per-release docs and `Catalog::
//! augment_missing` entry points.
//!
//! # ID masking
//!
//! Real Gaia source IDs encode `(HEALPix12 << 35) | run_id`; HEALPix12 has
//! ~2·10⁸ cells, so the maximum legitimate `source_id` is roughly 6.92·10¹⁸,
//! comfortably below 2⁶³ = 9.22·10¹⁸. Supplement entries set bit 63 and
//! place the original Hipparcos `HIP` number in the low 31 bits:
//!
//! ```text
//! synthetic_source_id = 0x8000_0000_0000_0000 | (hip & 0x7FFF_FFFF)
//! ```
//!
//! Supplement IDs are disjoint from any real Gaia ID (in any DR), so a
//! HashMap keyed by `source_id` never collides between the two populations.

use serde::{Deserialize, Serialize};
use starfield_core::{Result, StarfieldError};

use crate::gaia::common::core::{GaiaCore, VarFlag};

/// Sentinel bit set on every supplement-entry `source_id`. Real Gaia IDs cap
/// out around 6.92·10¹⁸ (well below 2⁶³), so any `source_id ≥ 1<<63` is a
/// supplement entry and never collides with a real one.
pub const SUPPLEMENT_SOURCE_ID_BIT: u64 = 1u64 << 63;

/// Encode a Hipparcos `HIP` number into a synthetic Gaia `source_id`. The
/// high bit (63) is set; the low 31 bits hold `hip` (HIP numbers max at
/// ~120000, so 31 bits is plenty).
pub const fn encode_supplement_source_id(hip: u32) -> u64 {
    SUPPLEMENT_SOURCE_ID_BIT | ((hip as u64) & 0x7FFF_FFFF)
}

/// Whether `source_id` came from a supplement (vs. a real Gaia source).
pub const fn is_supplement_source_id(source_id: u64) -> bool {
    source_id & SUPPLEMENT_SOURCE_ID_BIT != 0
}

/// Recover the Hipparcos `HIP` number from a supplement-entry `source_id`,
/// or `None` if it isn't a supplement entry.
pub const fn decode_supplement_hip(source_id: u64) -> Option<u32> {
    if is_supplement_source_id(source_id) {
        Some((source_id & 0x7FFF_FFFF) as u32)
    } else {
        None
    }
}

/// One row of the slim supplement CSV. Field names match the CSV header,
/// which is the same across DR1 / DR2 / DR3 (positions are propagated to
/// each release's own reference epoch — the column name is just `ra_jXXXX`
/// per release).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SupplementRow {
    pub hip: u32,
    /// Hipparcos RA propagated to this release's `ref_epoch`, in degrees.
    pub ra: f64,
    /// Hipparcos Dec propagated to this release's `ref_epoch`, in degrees.
    pub dec: f64,
    pub parallax_mas: Option<f64>,
    pub pmra_mas_yr: Option<f64>,
    pub pmdec_mas_yr: Option<f64>,
    pub b_v: Option<f64>,
    pub fitted_g_mag: f64,
}

/// Parse a supplement CSV string. The header is the same across releases
/// (`hip,ra,dec,…`) — the per-release reference epoch is documented in the
/// file's leading `#` comment block and applied via the per-release
/// `SUPPLEMENT_REF_EPOCH` constant when materializing the entry.
pub fn parse_supplement_csv(csv: &str) -> Result<Vec<SupplementRow>> {
    const EXPECTED_HEADER: &str =
        "hip,ra,dec,parallax_mas,pmra_mas_yr,pmdec_mas_yr,b_v,fitted_g_mag";
    let expected_header: &str = EXPECTED_HEADER;
    let mut out = Vec::with_capacity(16_000);
    let mut header_seen = false;
    for (lineno, line) in csv.lines().enumerate() {
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        if !header_seen {
            if line.trim() != expected_header {
                return Err(StarfieldError::DataError(format!(
                    "supplement CSV header mismatch at line {}: got {:?}, want {:?}",
                    lineno + 1,
                    line,
                    expected_header
                )));
            }
            header_seen = true;
            continue;
        }
        out.push(parse_row(line, lineno + 1)?);
    }
    if !header_seen {
        return Err(StarfieldError::DataError(
            "supplement CSV had no header row".into(),
        ));
    }
    Ok(out)
}

fn parse_row(line: &str, lineno: usize) -> Result<SupplementRow> {
    let parts: Vec<&str> = line.split(',').collect();
    if parts.len() != 8 {
        return Err(StarfieldError::DataError(format!(
            "supplement CSV line {}: expected 8 fields, got {}",
            lineno,
            parts.len()
        )));
    }
    let parse_req_u32 = |s: &str| {
        s.parse::<u32>()
            .map_err(|e| StarfieldError::DataError(format!("hip parse line {}: {}", lineno, e)))
    };
    let parse_req_f64 = |s: &str, field: &str| {
        s.parse::<f64>().map_err(|e| {
            StarfieldError::DataError(format!("{} parse line {}: {}", field, lineno, e))
        })
    };
    let parse_opt_f64 = |s: &str, field: &str| {
        if s.is_empty() {
            Ok(None)
        } else {
            s.parse::<f64>().map(Some).map_err(|e| {
                StarfieldError::DataError(format!("{} parse line {}: {}", field, lineno, e))
            })
        }
    };
    Ok(SupplementRow {
        hip: parse_req_u32(parts[0])?,
        ra: parse_req_f64(parts[1], "ra")?,
        dec: parse_req_f64(parts[2], "dec")?,
        parallax_mas: parse_opt_f64(parts[3], "parallax_mas")?,
        pmra_mas_yr: parse_opt_f64(parts[4], "pmra_mas_yr")?,
        pmdec_mas_yr: parse_opt_f64(parts[5], "pmdec_mas_yr")?,
        b_v: parse_opt_f64(parts[6], "b_v")?,
        fitted_g_mag: parse_req_f64(parts[7], "fitted_g_mag")?,
    })
}

/// Build a fully-populated [`GaiaCore`] from a [`SupplementRow`] and the
/// per-release reference epoch. Per-release supplement modules wrap the
/// returned `core` into their own `Entry` type.
pub fn make_supplement_core(row: &SupplementRow, ref_epoch: f64) -> GaiaCore {
    let (l, b) = equatorial_to_galactic(row.ra, row.dec);
    let (ecl_lon, ecl_lat) = equatorial_to_ecliptic(row.ra, row.dec);

    GaiaCore {
        source_id: encode_supplement_source_id(row.hip),
        solution_id: 0,
        ref_epoch,
        random_index: None,
        ra: row.ra,
        ra_error: 0.0,
        dec: row.dec,
        dec_error: 0.0,
        ra_dec_corr: None,
        parallax: row.parallax_mas,
        parallax_error: None,
        pmra: row.pmra_mas_yr,
        pmra_error: None,
        pmdec: row.pmdec_mas_yr,
        pmdec_error: None,
        l,
        b,
        ecl_lon,
        ecl_lat,
        phot_g_mean_mag: row.fitted_g_mag,
        phot_g_mean_flux: None,
        phot_g_mean_flux_error: None,
        phot_g_n_obs: None,
        phot_variable_flag: VarFlag::default(),
        astrometric_n_obs_al: None,
        astrometric_excess_noise: None,
        astrometric_excess_noise_sig: None,
        astrometric_primary_flag: None,
        duplicated_source: None,
        matched_observations: None,
    }
}

// ----------------------------------------------------------------- coordinate helpers

/// J2000 obliquity of the ecliptic, degrees (IAU 2006 nominal).
const OBLIQUITY_DEG: f64 = 23.4392911;

/// J2000 north galactic pole and origin, IAU 1958 (degrees).
const RA_NGP_DEG: f64 = 192.85948;
const DEC_NGP_DEG: f64 = 27.12825;
const L_NCP_DEG: f64 = 122.93192;

/// Convert J2000 equatorial (RA, Dec) to galactic (l, b), in degrees.
pub(crate) fn equatorial_to_galactic(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let ra_ngp = RA_NGP_DEG.to_radians();
    let dec_ngp = DEC_NGP_DEG.to_radians();
    let l_ncp = L_NCP_DEG.to_radians();

    let sin_b = dec.sin() * dec_ngp.sin() + dec.cos() * dec_ngp.cos() * (ra - ra_ngp).cos();
    let b = sin_b.clamp(-1.0, 1.0).asin();

    let y = dec.cos() * (ra - ra_ngp).sin();
    let x = dec.sin() * dec_ngp.cos() - dec.cos() * dec_ngp.sin() * (ra - ra_ngp).cos();
    let l = l_ncp - y.atan2(x);
    let mut l_deg = l.to_degrees();
    l_deg = l_deg.rem_euclid(360.0);
    (l_deg, b.to_degrees())
}

/// Convert J2000 equatorial (RA, Dec) to ecliptic (lon, lat), in degrees.
pub(crate) fn equatorial_to_ecliptic(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let eps = OBLIQUITY_DEG.to_radians();

    let sin_lat = dec.sin() * eps.cos() - dec.cos() * eps.sin() * ra.sin();
    let lat = sin_lat.clamp(-1.0, 1.0).asin();
    let y = ra.sin() * eps.cos() + dec.tan() * eps.sin();
    let x = ra.cos();
    let mut lon_deg = y.atan2(x).to_degrees();
    lon_deg = lon_deg.rem_euclid(360.0);
    (lon_deg, lat.to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_mask_round_trip() {
        let id = encode_supplement_source_id(42);
        assert!(is_supplement_source_id(id));
        assert_eq!(decode_supplement_hip(id), Some(42));
    }

    #[test]
    fn id_mask_disjoint_from_real_gaia_range() {
        let pseudo_real = 6_900_000_000_000_000_000u64;
        assert!(!is_supplement_source_id(pseudo_real));
        assert_eq!(decode_supplement_hip(pseudo_real), None);
    }

    /// Spot-check the equatorial→galactic transform against the known
    /// galactic center: (RA, Dec) ≈ (266.405, -28.936) → (l, b) ≈ (0, 0).
    #[test]
    fn equatorial_to_galactic_galactic_center() {
        let (l, b) = equatorial_to_galactic(266.40499, -28.93617);
        assert!(l.abs() < 0.1 || (l - 360.0).abs() < 0.1, "l = {}", l);
        assert!(b.abs() < 0.1, "b = {}", b);
    }

    #[test]
    fn equatorial_to_ecliptic_origin() {
        let (lon, lat) = equatorial_to_ecliptic(0.0, 0.0);
        assert!(lon.abs() < 0.01 || (lon - 360.0).abs() < 0.01);
        assert!(lat.abs() < 0.01);
    }

    #[test]
    fn make_supplement_core_populates_required_fields() {
        let row = SupplementRow {
            hip: 12345,
            ra: 180.5,
            dec: -23.0,
            parallax_mas: Some(15.0),
            pmra_mas_yr: Some(-100.0),
            pmdec_mas_yr: Some(50.0),
            b_v: Some(0.5),
            fitted_g_mag: 6.8,
        };
        let core = make_supplement_core(&row, 2016.0);
        assert_eq!(core.ra, 180.5);
        assert_eq!(core.dec, -23.0);
        assert_eq!(core.parallax, Some(15.0));
        assert_eq!(core.phot_g_mean_mag, 6.8);
        assert_eq!(core.ref_epoch, 2016.0);
        assert_eq!(decode_supplement_hip(core.source_id), Some(12345));
        assert!(core.l >= 0.0 && core.l <= 360.0);
        assert!(core.b >= -90.0 && core.b <= 90.0);
        assert!(core.ecl_lon >= 0.0 && core.ecl_lon <= 360.0);
        assert!(core.ecl_lat >= -90.0 && core.ecl_lat <= 90.0);
    }

    /// Check parse round-trip with a synthetic supplement CSV.
    #[test]
    fn parse_supplement_csv_round_trip() {
        let csv = "# header comment\n\
                   # another\n\
                   hip,ra,dec,parallax_mas,pmra_mas_yr,pmdec_mas_yr,b_v,fitted_g_mag\n\
                   1,180.5,-23.0,15.0,-100.0,50.0,0.5,6.8\n\
                   2,90.0,30.0,,,,,7.5\n";
        let rows = parse_supplement_csv(csv).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].hip, 1);
        assert_eq!(rows[0].fitted_g_mag, 6.8);
        assert_eq!(rows[0].b_v, Some(0.5));
        assert_eq!(rows[1].hip, 2);
        assert_eq!(rows[1].b_v, None);
        assert_eq!(rows[1].pmra_mas_yr, None);
    }
}
