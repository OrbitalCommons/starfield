//! Kernel-free equatorial <-> horizontal coordinates, and the sexagesimal formats
//! around them.
//!
//! Everything in this module is a pure scalar function: no [`Time`](crate::time::Time),
//! no [`Position`](crate::positions::Position), no SPICE kernel, no download, no
//! delta-T table. That is the whole point. [`GeographicPosition::altaz`] is the
//! full-accuracy entry point, but it requires a `PositionKind::Apparent` position,
//! which can only be produced by `observe(...).apparent(...)` against a loaded
//! ephemeris kernel -- so a caller who already knows the hour angle and declination
//! of their target has no way in.
//!
//! The rotation itself was already written and correct; it was
//! [`GeographicPosition::itrs_to_horizon`], which is `pub(crate)`. These functions
//! build the ITRS direction that rotation wants and call it, so there is exactly one
//! implementation of the geometry in this crate.

use std::f64::consts::PI;

use super::WGS84;

const HOURS_TO_RAD: f64 = PI / 12.0;
const RAD_TO_DEG: f64 = 180.0 / PI;
const DEG_TO_RAD: f64 = PI / 180.0;

/// Altitude and azimuth of a target at a given hour angle and declination.
///
/// * `hour_angle_hours` — local hour angle, in hours, increasing westward.
/// * `dec_degrees` — declination, in degrees.
/// * `latitude_degrees` — observer geodetic latitude, in degrees.
///
/// Returns `(altitude_degrees, azimuth_degrees)`, azimuth measured clockwise from
/// north (0 = N, 90 = E), in `[0, 360)`.
///
/// This is the geometric transform only: no refraction, no parallax, no aberration.
/// Add refraction with [`super::GeographicPosition::refract`] if you want observed altitude.
pub fn altaz_from_ha_dec(
    hour_angle_hours: f64,
    dec_degrees: f64,
    latitude_degrees: f64,
) -> (f64, f64) {
    let ha = hour_angle_hours * HOURS_TO_RAD;
    let dec = dec_degrees * DEG_TO_RAD;

    // ITRS unit vector of the target, for an observer on the prime meridian. Hour
    // angle increases westward, so the target sits at ITRS longitude -ha. Using
    // longitude 0 here is not an approximation: the observer's own longitude cancels
    // out of the hour-angle form, which is exactly why this entry point needs no time.
    let (sd, cd) = dec.sin_cos();
    let (sh, ch) = ha.sin_cos();
    let itrs = nalgebra::Vector3::new(cd * ch, -cd * sh, sd);

    let observer = WGS84.latlon(latitude_degrees, 0.0, 0.0);
    let (alt, az) = observer.itrs_to_horizon(&itrs);
    (alt * RAD_TO_DEG, az * RAD_TO_DEG)
}

/// The inverse of [`altaz_from_ha_dec`]: hour angle and declination of a target at a
/// given altitude and azimuth.
///
/// Returns `(hour_angle_hours, dec_degrees)`, with the hour angle wrapped to
/// `[-12, 12)` — the same branch ERFA's `eraAe2hd` returns.
pub fn ha_dec_from_alt_az(
    altitude_degrees: f64,
    azimuth_degrees: f64,
    latitude_degrees: f64,
) -> (f64, f64) {
    let alt = altitude_degrees * DEG_TO_RAD;
    let az = azimuth_degrees * DEG_TO_RAD;
    let lat = latitude_degrees * DEG_TO_RAD;

    let (sa, ca) = alt.sin_cos();
    let (sz, cz) = az.sin_cos();
    let (sp, cp) = lat.sin_cos();

    // Local horizon frame (south, east, up), the same basis `itrs_to_horizon`
    // produces, then rotated back to equatorial.
    let south = -ca * cz;
    let east = ca * sz;
    let up = sa;

    // dec is the component along the celestial pole; the hour-angle plane is spanned
    // by the remaining two. atan2 rather than asin so the tropical case where the
    // target transits north of the zenith does not fold.
    // The horizon basis (south, east, up) is an orthogonal rotation of ITRS, so the
    // inverse is its transpose:
    //     [south]   [ sp   0  -cp ] [x]
    //     [east ] = [  0   1    0 ] [y]
    //     [up   ]   [ cp   0   sp ] [z]
    let x = sp * south + cp * up;
    let y = east;
    let z = -cp * south + sp * up;

    let dec = z.atan2((x * x + y * y).sqrt());
    let mut ha = (-y).atan2(x) / HOURS_TO_RAD;
    if ha >= 12.0 {
        ha -= 24.0;
    } else if ha < -12.0 {
        ha += 24.0;
    }
    (ha, dec * RAD_TO_DEG)
}

/// Hour angle of a target, in hours, from local sidereal time and right ascension.
///
/// Both arguments are in hours. The result is wrapped to `[-12, 12)`.
pub fn hour_angle(lst_hours: f64, ra_hours: f64) -> f64 {
    let mut ha = (lst_hours - ra_hours).rem_euclid(24.0);
    if ha >= 12.0 {
        ha -= 24.0;
    }
    ha
}

/// Format a signed angle in degrees as `[-]dd:mm:ss.sss`.
pub fn format_dms(degrees: f64, decimals: usize) -> String {
    format_sexagesimal(degrees, decimals, 2)
}

/// Format an angle in hours as `hh:mm:ss.sss`, wrapped to `[0, 24)`.
pub fn format_hms(hours: f64, decimals: usize) -> String {
    format_sexagesimal(hours.rem_euclid(24.0), decimals, 2)
}

fn format_sexagesimal(value: f64, decimals: usize, unit_width: usize) -> String {
    let sign = if value < 0.0 { "-" } else { "" };
    let v = value.abs();
    let mut whole = v.trunc();
    let mut minutes = (v - whole) * 60.0;
    let mut seconds = (minutes - minutes.trunc()) * 60.0;
    minutes = minutes.trunc();

    // Carry, so a value that rounds up at the requested precision does not print
    // `:60.000`. Done on the ROUNDED seconds, because that is what gets printed.
    let scale = 10f64.powi(decimals as i32);
    seconds = (seconds * scale).round() / scale;
    if seconds >= 60.0 {
        seconds -= 60.0;
        minutes += 1.0;
    }
    if minutes >= 60.0 {
        minutes -= 60.0;
        whole += 1.0;
    }
    format!(
        "{sign}{whole:0unit_width$.0}:{minutes:02.0}:{seconds:0width$.decimals$}",
        width = if decimals > 0 { decimals + 3 } else { 2 },
    )
}

/// Parse `[-]dd:mm:ss(.sss)` (or space/`d m s`-separated) into signed degrees.
///
/// A leading `-` applies to the whole angle, not just the degrees field — the usual
/// trap, since `-00:30:00` must come out negative even though the degrees field is 0.
pub fn parse_dms(text: &str) -> Option<f64> {
    parse_sexagesimal(text)
}

/// Parse `hh:mm:ss(.sss)` into hours.
pub fn parse_hms(text: &str) -> Option<f64> {
    parse_sexagesimal(text)
}

fn parse_sexagesimal(text: &str) -> Option<f64> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }
    let negative = trimmed.starts_with('-');
    let body = trimmed.trim_start_matches(['-', '+']);
    let fields: Vec<&str> = body
        .split(|c: char| !(c.is_ascii_digit() || c == '.'))
        .filter(|s| !s.is_empty())
        .collect();
    if fields.is_empty() || fields.len() > 3 {
        return None;
    }
    let mut total = 0.0;
    for (i, field) in fields.iter().enumerate() {
        let parsed: f64 = field.parse().ok()?;
        if i > 0 && parsed >= 60.0 {
            return None;
        }
        total += parsed / 60f64.powi(i as i32);
    }
    Some(if negative { -total } else { total })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dms_round_trips_through_text() {
        for deg in [0.0, 1.5, -1.5, 89.9999, -89.9999, 23.43929, -0.5, 180.0] {
            let text = format_dms(deg, 4);
            let back = parse_dms(&text).expect("parses");
            assert!((back - deg).abs() < 1e-7, "{deg} -> {text} -> {back}");
        }
    }

    #[test]
    fn negative_sign_applies_to_the_whole_angle_not_just_degrees() {
        // The classic trap: the degrees field is zero, so a per-field sign is lost.
        assert_eq!(parse_dms("-00:30:00"), Some(-0.5));
        assert_eq!(format_dms(-0.5, 1), "-00:30:00.0");
    }

    #[test]
    fn seconds_that_round_up_carry_instead_of_printing_sixty() {
        // 12:59:59.9999 at 3 decimals must not become "12:59:60.000".
        let text = format_dms(12.0 + 59.0 / 60.0 + 59.9999 / 3600.0, 3);
        assert_eq!(text, "13:00:00.000");
    }

    #[test]
    fn hms_wraps_and_parses() {
        assert_eq!(format_hms(24.5, 2), "00:30:00.00");
        assert_eq!(
            parse_hms("12 34 56.5"),
            Some(12.0 + 34.0 / 60.0 + 56.5 / 3600.0)
        );
    }

    #[test]
    fn out_of_range_and_malformed_fields_are_rejected() {
        assert_eq!(parse_dms("10:60:00"), None);
        assert_eq!(parse_dms("10:00:61"), None);
        assert_eq!(parse_dms(""), None);
        assert_eq!(parse_dms("1:2:3:4"), None);
    }

    #[test]
    fn hour_angle_wraps_to_the_half_open_branch() {
        assert!((hour_angle(1.0, 23.0) - 2.0).abs() < 1e-12);
        assert!((hour_angle(23.0, 1.0) - -2.0).abs() < 1e-12);
        assert!((hour_angle(0.0, 12.0) - -12.0).abs() < 1e-12);
    }
}
