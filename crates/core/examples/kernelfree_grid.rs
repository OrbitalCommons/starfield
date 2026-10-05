//! Emit kernel-free alt/az for a grid read on stdin, for comparison against pyerfa.
//!
//! Input: one JSON object per line with
//!   {"ra_h":..,"dec_deg":..,"lat":..,"lon":..,"y":..,"mo":..,"d":..,"h":..,"mi":..,"s":..}
//! Output: one JSON object per line with alt/az from `altaz_from_icrs`, the same
//! without aberration (the deliberately-failing control), and starfield's own
//! UT1-UTC so the oracle can be run on a matched clock.

use serde::Deserialize;
use starfield_core::time::Timescale;
use starfield_core::toposlib::WGS84;

#[derive(Deserialize)]
struct Row {
    ra_h: f64,
    dec_deg: f64,
    lat: f64,
    lon: f64,
    y: i32,
    mo: u32,
    d: u32,
    h: u32,
    mi: u32,
    s: f64,
}

fn main() {
    let ts = Timescale::default();
    let stdin = std::io::stdin();
    let mut line = String::new();
    loop {
        line.clear();
        if std::io::BufRead::read_line(&mut stdin.lock(), &mut line).unwrap() == 0 {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let r: Row = serde_json::from_str(t).unwrap();
        let time = ts.utc((r.y, r.mo, r.d, r.h, r.mi, r.s));
        let site = WGS84.latlon(r.lat, r.lon, 0.0);

        let (alt, az) = site.altaz_from_icrs(r.ra_h * 15.0, r.dec_deg, &time);
        let (alt0, az0) = altaz_no_aberration(&site, r.ra_h * 15.0, r.dec_deg, &time);

        // starfield's own UT1 - UTC in seconds, so the oracle can match the clock.
        let dut1 = time.dut1();

        // starfield's own refraction model, at the fixture's conditions.
        let alt_ref = starfield_core::earthlib::refract(alt, 10.0, 1013.25);

        println!(
            "{{\"alt\":{alt:.12},\"az\":{az:.12},\"alt_noab\":{alt0:.12},\"az_noab\":{az0:.12},\"alt_refracted\":{alt_ref:.12},\"dut1\":{dut1:.9}}}"
        );
    }
}

/// The control: identical chain with stellar aberration removed.
fn altaz_no_aberration(
    site: &starfield_core::toposlib::GeographicPosition,
    ra_degrees: f64,
    dec_degrees: f64,
    time: &starfield_core::time::Time,
) -> (f64, f64) {
    use nalgebra::Vector3;
    use std::f64::consts::PI;
    let ra = ra_degrees * PI / 180.0;
    let dec = dec_degrees * PI / 180.0;
    let direction = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
    let itrs = time.c_matrix() * direction;

    // itrs_to_horizon is pub(crate); the example repeats it so the control
    // shares exactly one line of difference with the real path.
    let (slat, clat) = site.latitude.sin_cos();
    let (slon, clon) = site.longitude.sin_cos();
    let south = slat * clon * itrs.x + slat * slon * itrs.y - clat * itrs.z;
    let east = -slon * itrs.x + clon * itrs.y;
    let up = clat * clon * itrs.x + clat * slon * itrs.y + slat * itrs.z;
    let alt = up.atan2((south * south + east * east).sqrt());
    let mut az = east.atan2(-south);
    if az < 0.0 {
        az += 2.0 * PI;
    }
    (alt * 180.0 / PI, az * 180.0 / PI)
}
