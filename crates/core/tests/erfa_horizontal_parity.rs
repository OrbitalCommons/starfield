//! Acceptance for the kernel-free horizontal-coordinate entry points, against ERFA.
//!
//! Oracle: pyerfa 2.0.1.5 (`erfa.hd2ae` / `erfa.ae2hd`), the Python binding of the
//! IAU SOFA-derived C library. The grid was pre-registered before this code was
//! written: latitude spanning +/-89, declination spanning +/-89, a full hour-angle
//! circle, and 135 rows in the northern tropics where the target transits NORTH of
//! the zenith -- the configuration where an `asin` form of the inverse folds.

use starfield_core::toposlib::{altaz_from_ha_dec, ha_dec_from_alt_az};

const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
const TOL_RAD: f64 = 1e-9;
const TOL_DEG_ROUNDTRIP: f64 = 1e-9;

struct Row {
    lat_deg: f64,
    dec_deg: f64,
    ha_hours: f64,
    az_rad: f64,
    alt_rad: f64,
    ha_back_rad: f64,
    dec_back_rad: f64,
}

fn grid() -> Vec<Row> {
    let raw = include_str!("erfa_hadec_grid.csv");
    raw.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<f64> = l.split(',').map(|x| x.parse().unwrap()).collect();
            Row {
                lat_deg: f[0],
                dec_deg: f[1],
                ha_hours: f[2],
                az_rad: f[3],
                alt_rad: f[4],
                ha_back_rad: f[5],
                dec_back_rad: f[6],
            }
        })
        .collect()
}

/// Smallest signed difference between two angles, in radians.
fn ang_diff(a: f64, b: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let d = (a - b).rem_euclid(two_pi);
    if d > std::f64::consts::PI {
        d - two_pi
    } else {
        d
    }
    .abs()
}

#[test]
fn altaz_from_ha_dec_matches_erfa_hd2ae() {
    let rows = grid();
    assert_eq!(rows.len(), 2160, "fixture size changed");

    let mut worst_alt: f64 = 0.0;
    let mut worst_az: f64 = 0.0;
    let mut az_degenerate = 0usize;
    let mut tropical_checked = 0usize;

    for r in &rows {
        let (alt_deg, az_deg) = altaz_from_ha_dec(r.ha_hours, r.dec_deg, r.lat_deg);
        let d_alt = (alt_deg * DEG_TO_RAD - r.alt_rad).abs();
        worst_alt = worst_alt.max(d_alt);

        // Azimuth is genuinely undefined at the zenith and the nadir; count those
        // rather than quietly dropping them.
        if (r.alt_rad.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-12 {
            az_degenerate += 1;
        } else {
            worst_az = worst_az.max(ang_diff(az_deg * DEG_TO_RAD, r.az_rad));
        }
        if r.lat_deg > 0.0 && r.lat_deg < 23.5 && r.dec_deg > r.lat_deg {
            tropical_checked += 1;
        }
    }

    println!("rows                     : {}", rows.len());
    println!("tropical, N of zenith    : {tropical_checked}");
    println!("azimuth-degenerate rows  : {az_degenerate} (zenith/nadir, az undefined)");
    println!("worst |d altitude| (rad) : {worst_alt:.3e}");
    println!("worst |d azimuth|  (rad) : {worst_az:.3e}");

    assert_eq!(tropical_checked, 135, "tropical arm of the grid is missing");
    assert!(
        worst_alt < TOL_RAD,
        "altitude worst {worst_alt:.3e} >= {TOL_RAD:e}"
    );
    assert!(
        worst_az < TOL_RAD,
        "azimuth worst {worst_az:.3e} >= {TOL_RAD:e}"
    );
}

#[test]
fn ha_dec_from_alt_az_matches_erfa_ae2hd() {
    let rows = grid();
    let mut worst_ha: f64 = 0.0;
    let mut worst_dec: f64 = 0.0;
    let mut skipped = 0usize;

    for r in &rows {
        // Feed ERFA's own alt/az back in, so this arm tests the inverse alone and
        // not the forward transform's error.
        if (r.alt_rad.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-12 {
            skipped += 1; // az undefined going in; nothing to test
            continue;
        }
        let (ha_hours, dec_deg) =
            ha_dec_from_alt_az(r.alt_rad / DEG_TO_RAD, r.az_rad / DEG_TO_RAD, r.lat_deg);
        worst_dec = worst_dec.max((dec_deg * DEG_TO_RAD - r.dec_back_rad).abs());
        worst_ha = worst_ha.max(ang_diff(
            ha_hours * std::f64::consts::PI / 12.0,
            r.ha_back_rad,
        ));
    }

    println!("rows tested              : {}", rows.len() - skipped);
    println!("skipped (az undefined)   : {skipped}");
    println!("worst |d hour angle| rad : {worst_ha:.3e}");
    println!("worst |d declination| rad: {worst_dec:.3e}");

    assert!(worst_ha < TOL_RAD, "hour angle worst {worst_ha:.3e}");
    assert!(worst_dec < TOL_RAD, "declination worst {worst_dec:.3e}");
}

#[test]
fn roundtrip_altaz_hadec_altaz() {
    let rows = grid();
    let mut worst: f64 = 0.0;
    for r in &rows {
        let (alt, az) = altaz_from_ha_dec(r.ha_hours, r.dec_deg, r.lat_deg);
        if (alt.abs() - 90.0).abs() < 1e-9 {
            continue; // az undefined at the zenith; the round trip is not meaningful
        }
        let (ha2, dec2) = ha_dec_from_alt_az(alt, az, r.lat_deg);
        let (alt2, az2) = altaz_from_ha_dec(ha2, dec2, r.lat_deg);
        worst = worst.max((alt2 - alt).abs());
        worst = worst.max(ang_diff(az2 * DEG_TO_RAD, az * DEG_TO_RAD) / DEG_TO_RAD);
    }
    println!("worst round-trip error (deg): {worst:.3e}");
    assert!(worst < TOL_DEG_ROUNDTRIP, "round trip worst {worst:.3e}");
}
