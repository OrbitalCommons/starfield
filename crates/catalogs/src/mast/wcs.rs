//! Minimal FITS WCS parser + TAN-projection helpers.
//!
//! Reads the keys we care about (CRVAL1/2, CRPIX1/2, CD or PC+CDELT,
//! CTYPE1/2, NAXIS1/2) from a FITS header via `fitsio_pure`. Provides
//! pixel↔world for TAN projections plus a 4-corner footprint so callers
//! can answer "where is this image pointing and what does it cover?"
//!
//! HST products use either:
//!
//! - DRZ / DRC drizzled mosaics — distortion-corrected upstream, linear
//!   CD-matrix WCS is exact.
//! - FLT / FLC flatfielded per-exposure files — carry SIP polynomial
//!   distortion that this crate does **not** apply. The footprint is
//!   accurate to ~arcsec (fine for "what region is covered" but not for
//!   sub-pixel astrometry).
//!
//! Only TAN (gnomonic) projection is supported. Anything else
//! (`SIN`, `ARC`, `ZEA`, `STG`, `CAR`, `HPX`, …) errors out at
//! `pixel_to_world` time so consumers don't silently get bad
//! coordinates.

use std::fs;
use std::path::Path;

use fitsio_pure::hdu::{parse_fits, HduInfo};
use fitsio_pure::header::Card;
use fitsio_pure::value::Value;
use starfield_core::{Result, StarfieldError};

/// Linear FITS WCS, populated from a FITS header.
///
/// Field names match the FITS keyword conventions: `crval1` is the
/// reference-point RA in degrees, `crval2` is the reference-point Dec
/// in degrees, etc. `cd` is the 2x2 CD matrix in degrees per pixel;
/// older PC + CDELT layouts are normalised into a CD matrix at parse
/// time. `naxis1` / `naxis2` are the image dimensions in pixels.
#[derive(Debug, Clone)]
pub struct Wcs {
    pub crval1: f64,
    pub crval2: f64,
    pub crpix1: f64,
    pub crpix2: f64,
    pub cd: [[f64; 2]; 2],
    pub ctype1: String,
    pub ctype2: String,
    pub naxis1: u32,
    pub naxis2: u32,
}

impl Wcs {
    /// Read WCS from a FITS file on disk.
    ///
    /// HST imaging products typically put the WCS in a `SCI`
    /// extension, not the primary HDU; this scans the primary first
    /// then every extension and returns the first HDU that carries a
    /// usable WCS (CRVAL1 + CRVAL2 + CRPIX1 + CRPIX2 + a CD or
    /// PC+CDELT matrix + CTYPE1/2 + NAXIS1/2).
    pub fn read_from_fits<P: AsRef<Path>>(path: P) -> Result<Self> {
        let bytes = fs::read(path.as_ref()).map_err(StarfieldError::IoError)?;
        Self::read_from_bytes(&bytes)
    }

    /// Read WCS from in-memory FITS bytes.
    pub fn read_from_bytes(bytes: &[u8]) -> Result<Self> {
        let fits = parse_fits(bytes)
            .map_err(|e| StarfieldError::DataError(format!("FITS parse: {:?}", e)))?;
        for hdu in fits.iter() {
            if !matches!(hdu.info, HduInfo::Primary { .. } | HduInfo::Image { .. }) {
                continue;
            }
            if let Ok(wcs) = Self::read_from_cards(&hdu.cards) {
                return Ok(wcs);
            }
        }
        Err(StarfieldError::DataError(
            "no HDU in this FITS file carried a complete WCS".into(),
        ))
    }

    /// Parse WCS from a slice of FITS header cards.
    pub fn read_from_cards(cards: &[Card]) -> Result<Self> {
        let crval1 = require_float(cards, "CRVAL1")?;
        let crval2 = require_float(cards, "CRVAL2")?;
        let crpix1 = require_float(cards, "CRPIX1")?;
        let crpix2 = require_float(cards, "CRPIX2")?;
        let ctype1 = require_string(cards, "CTYPE1")?;
        let ctype2 = require_string(cards, "CTYPE2")?;
        let naxis1 = require_int(cards, "NAXIS1")? as u32;
        let naxis2 = require_int(cards, "NAXIS2")? as u32;
        let cd = read_cd_matrix(cards)?;
        Ok(Wcs {
            crval1,
            crval2,
            crpix1,
            crpix2,
            cd,
            ctype1,
            ctype2,
            naxis1,
            naxis2,
        })
    }

    /// True iff both axes use the TAN projection. Also matches
    /// `RA---TAN-SIP` / `DEC--TAN-SIP` — SIP is a distortion suffix on
    /// top of the TAN base projection, and our linear-only evaluator
    /// treats it as plain TAN with the documented "SIP not applied"
    /// caveat (see crate docs).
    pub fn is_tan(&self) -> bool {
        ctype_projection(&self.ctype1) == Some("TAN")
            && ctype_projection(&self.ctype2) == Some("TAN")
    }

    /// Convert pixel coordinates `(x, y)` (FITS-style, 1-indexed,
    /// where (1.0, 1.0) is the centre of the first pixel) to world
    /// coordinates `(ra_deg, dec_deg)`.
    ///
    /// Errors out for non-TAN projections.
    pub fn pixel_to_world(&self, x: f64, y: f64) -> Result<(f64, f64)> {
        if !self.is_tan() {
            return Err(StarfieldError::DataError(format!(
                "pixel_to_world: only TAN projection supported, got CTYPE1={:?} / CTYPE2={:?}",
                self.ctype1, self.ctype2
            )));
        }
        let dx = x - self.crpix1;
        let dy = y - self.crpix2;
        // Standard / native (xi, eta), in degrees, then converted to
        // radians for the inverse-gnomonic formulae below.
        let xi_deg = self.cd[0][0] * dx + self.cd[0][1] * dy;
        let eta_deg = self.cd[1][0] * dx + self.cd[1][1] * dy;
        let xi = xi_deg.to_radians();
        let eta = eta_deg.to_radians();

        let alpha0 = self.crval1.to_radians();
        let delta0 = self.crval2.to_radians();

        let rho = (xi * xi + eta * eta).sqrt();
        if rho == 0.0 {
            return Ok((self.crval1, self.crval2));
        }
        let c = rho.atan();
        let sin_c = c.sin();
        let cos_c = c.cos();
        let sin_d0 = delta0.sin();
        let cos_d0 = delta0.cos();

        let delta = (cos_c * sin_d0 + (eta * sin_c * cos_d0) / rho).asin();
        let alpha = alpha0 + (xi * sin_c).atan2(rho * cos_d0 * cos_c - eta * sin_d0 * sin_c);
        // Wrap RA into [0, 360).
        let mut ra_deg = alpha.to_degrees();
        ra_deg = ra_deg.rem_euclid(360.0);
        Ok((ra_deg, delta.to_degrees()))
    }

    /// Returns the four image corners in world coordinates `(ra_deg,
    /// dec_deg)`, ordered (BL, BR, TR, TL) where bottom-left is pixel
    /// (0.5, 0.5) — the pixel-grid edge, not the centre of the corner
    /// pixel.
    ///
    /// Errors out for non-TAN projections (consistent with
    /// [`Self::pixel_to_world`]).
    pub fn footprint(&self) -> Result<[(f64, f64); 4]> {
        let nx = self.naxis1 as f64;
        let ny = self.naxis2 as f64;
        Ok([
            self.pixel_to_world(0.5, 0.5)?,
            self.pixel_to_world(nx + 0.5, 0.5)?,
            self.pixel_to_world(nx + 0.5, ny + 0.5)?,
            self.pixel_to_world(0.5, ny + 0.5)?,
        ])
    }

    /// Approximate plate scale at the reference point, arcsec/pixel.
    /// Computed as `sqrt(|det(CD)|) * 3600`, which matches the standard
    /// astropy `wcs.proj_plane_pixel_scales` average to within a few
    /// percent for typical small-FOV instruments.
    pub fn pixel_scale_arcsec(&self) -> f64 {
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        det.abs().sqrt() * 3600.0
    }
}

/// Extract the base 3-character projection code from a CTYPE string
/// like `"RA---TAN"` / `"DEC--TAN"` / `"RA---SIN"` / `"RA---TAN-SIP"`.
///
/// FITS WCS CTYPE format: an axis name (`RA`, `DEC`, `GLON`, etc.)
/// padded with `-` to 4 chars, then `-`, then a 3-char projection code,
/// optionally followed by a distortion suffix like `-SIP`. We look for
/// the projection code by splitting on `-`, dropping empty tokens, and
/// taking the **second** token (first = axis name, second = projection).
fn ctype_projection(ctype: &str) -> Option<&str> {
    let mut tokens = ctype.trim().split('-').filter(|s| !s.is_empty());
    let _axis = tokens.next()?;
    tokens.next()
}

fn read_cd_matrix(cards: &[Card]) -> Result<[[f64; 2]; 2]> {
    if let (Some(a), Some(b), Some(c), Some(d)) = (
        find_float(cards, "CD1_1"),
        find_float(cards, "CD1_2"),
        find_float(cards, "CD2_1"),
        find_float(cards, "CD2_2"),
    ) {
        return Ok([[a, b], [c, d]]);
    }
    // PC + CDELT fallback (older WCS layout).
    let pc11 = find_float(cards, "PC1_1").unwrap_or(1.0);
    let pc12 = find_float(cards, "PC1_2").unwrap_or(0.0);
    let pc21 = find_float(cards, "PC2_1").unwrap_or(0.0);
    let pc22 = find_float(cards, "PC2_2").unwrap_or(1.0);
    let cdelt1 = find_float(cards, "CDELT1");
    let cdelt2 = find_float(cards, "CDELT2");
    match (cdelt1, cdelt2) {
        (Some(d1), Some(d2)) => Ok([[d1 * pc11, d1 * pc12], [d2 * pc21, d2 * pc22]]),
        _ => Err(StarfieldError::DataError(
            "no CD matrix and no PC+CDELT keys; can't derive WCS rotation".into(),
        )),
    }
}

fn find_card<'a>(cards: &'a [Card], keyword: &str) -> Option<&'a Card> {
    cards.iter().find(|c| c.keyword_str() == keyword)
}

fn find_float(cards: &[Card], keyword: &str) -> Option<f64> {
    let card = find_card(cards, keyword)?;
    match card.value.as_ref()? {
        Value::Float(f) => Some(*f),
        Value::Integer(i) => Some(*i as f64),
        _ => None,
    }
}

fn require_float(cards: &[Card], keyword: &str) -> Result<f64> {
    find_float(cards, keyword).ok_or_else(|| {
        StarfieldError::DataError(format!(
            "WCS: required FITS keyword {} missing or not numeric",
            keyword
        ))
    })
}

fn require_int(cards: &[Card], keyword: &str) -> Result<i64> {
    let card = find_card(cards, keyword).ok_or_else(|| {
        StarfieldError::DataError(format!("WCS: required FITS keyword {} missing", keyword))
    })?;
    match card.value.as_ref() {
        Some(Value::Integer(i)) => Ok(*i),
        Some(Value::Float(f)) => Ok(*f as i64),
        _ => Err(StarfieldError::DataError(format!(
            "WCS: FITS keyword {} not an integer",
            keyword
        ))),
    }
}

fn require_string(cards: &[Card], keyword: &str) -> Result<String> {
    let card = find_card(cards, keyword).ok_or_else(|| {
        StarfieldError::DataError(format!("WCS: required FITS keyword {} missing", keyword))
    })?;
    match card.value.as_ref() {
        Some(Value::String(s)) => Ok(s.trim().to_string()),
        _ => Err(StarfieldError::DataError(format!(
            "WCS: FITS keyword {} not a string",
            keyword
        ))),
    }
}
