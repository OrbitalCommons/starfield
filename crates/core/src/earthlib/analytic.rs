//! Kernel-free analytic heliocentric state of the Earth–Moon barycentre.
//!
//! Everything else in the apparent-place chain — nutation, precession, GAST,
//! the ITRS rotation, refraction — is already kernel-free. The single input
//! that forces a `SpiceKernel` into [`Position::apparent`] for a *star* is the
//! observer's barycentric **velocity**, which stellar aberration needs.
//!
//! This module supplies an approximation to that velocity from a two-body
//! Keplerian orbit of the Earth–Moon barycentre (EMB) about the Sun, using the
//! mean elements of Standish, *Keplerian Elements for Approximate Positions of
//! the Major Planets* (JPL SSD), valid 1800–2050.
//!
//! [`Position::apparent`]: crate::positions::Position::apparent
//!
//! # What it is and is not
//!
//! [`emb_heliocentric_two_body_state`] returns the **heliocentric** state of
//! the **EMB**, not the barycentric state of the Earth. Measured against DE421
//! twice a month over 1990–2040:
//!
//! * against the DE421 heliocentric EMB: at most 16,500 km (0.055 light-s) in
//!   position and 2.2 m/s in velocity — the two-body truncation error;
//! * against the DE421 **barycentric Earth**: at most 0.0092 AU (**4.6
//!   light-s**) in position and 30.5 m/s in velocity. The position gap is the
//!   Sun's offset from the solar-system barycentre, which a heliocentric orbit
//!   cannot see.
//!
//! The velocity is good enough for stellar aberration: 30.5 m/s costs
//! `dv/c` = 0.021 arcsec, against the 0.36 arcsec (1e-4 deg, 523 m/s) bar
//! that `ivonnyssen/rusty-photon` pins in `tests/reference_values.rs`.
//!
//! The position is **not** a barycentric position. Do not use it for
//! light-time, Rømer delay or BJD_TDB (it is seconds off), for parallax of
//! solar-system bodies, or for anything else that needs Earth's place relative
//! to the barycentre; load a SPICE kernel for those.

use nalgebra::{Matrix3, Vector3};

use crate::constants::DEG2RAD;
use crate::time::Time;

/// Obliquity of the ecliptic at J2000.0, in degrees (IAU 1976, 84381.448").
const OBLIQUITY_J2000_DEG: f64 = 84_381.448 / 3600.0;

/// Julian days per Julian century.
const DAYS_PER_CENTURY: f64 = 36_525.0;

/// J2000.0 as a Julian date.
const J2000: f64 = 2_451_545.0;

// Standish mean elements for the Earth–Moon barycentre, 1800–2050.
// (a AU, e, I deg, L deg, long. periapsis deg, long. node deg) and their
// per-century rates.
const A0: f64 = 1.000_002_61;
const A_DOT: f64 = 0.000_005_62;
const E0: f64 = 0.016_711_23;
const E_DOT: f64 = -0.000_043_92;
const I0: f64 = -0.000_015_31;
const I_DOT: f64 = -0.012_946_68;
const L0: f64 = 100.464_571_66;
const L_DOT: f64 = 35_999.372_449_81;
const PERI0: f64 = 102.937_681_93;
const PERI_DOT: f64 = 0.323_273_64;
const NODE0: f64 = 0.0;
const NODE_DOT: f64 = 0.0;

/// Rotation from the J2000 ecliptic frame to the equatorial (ICRS) frame.
fn ecliptic_to_equatorial() -> Matrix3<f64> {
    let eps = OBLIQUITY_J2000_DEG * DEG2RAD;
    let (s, c) = eps.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
}

/// Solve Kepler's equation `M = E - e sin E` for the eccentric anomaly.
fn solve_kepler(mean_anomaly: f64, e: f64) -> f64 {
    let mut ecc = mean_anomaly + e * mean_anomaly.sin();
    for _ in 0..64 {
        let delta = (ecc - e * ecc.sin() - mean_anomaly) / (1.0 - e * ecc.cos());
        ecc -= delta;
        if delta.abs() < 1.0e-14 {
            break;
        }
    }
    ecc
}

/// Heliocentric position (AU) and velocity (AU/day) of the Earth–Moon
/// barycentre at `time`, from a two-body orbit, in the ICRS axes.
///
/// This is **not** Earth's barycentric state: the position is up to 4.6
/// light-seconds from DE421's barycentric Earth, the velocity up to 30.5 m/s.
/// The velocity is sized for stellar aberration (0.021 arcsec); the position
/// is unfit for light-time or BJD work. See the [module docs](self) for the
/// measurements. Evaluated at TDB.
pub fn emb_heliocentric_two_body_state(time: &Time) -> (Vector3<f64>, Vector3<f64>) {
    let t = (time.tdb() - J2000) / DAYS_PER_CENTURY;

    let a = A0 + A_DOT * t;
    let e = E0 + E_DOT * t;
    let inc = (I0 + I_DOT * t) * DEG2RAD;
    let mean_longitude = (L0 + L_DOT * t) * DEG2RAD;
    let peri = (PERI0 + PERI_DOT * t) * DEG2RAD;
    let node = (NODE0 + NODE_DOT * t) * DEG2RAD;

    // Argument of periapsis, and the mean anomaly wrapped to +/-pi.
    let arg_peri = peri - node;
    let mean_anomaly = (mean_longitude - peri).rem_euclid(2.0 * std::f64::consts::PI);

    let ecc = solve_kepler(mean_anomaly, e);
    let (sin_e, cos_e) = ecc.sin_cos();
    let beta = (1.0 - e * e).sqrt();

    // In-plane coordinates, periapsis along +x.
    let x_p = a * (cos_e - e);
    let y_p = a * beta * sin_e;

    // dE/dt from the mean-motion, in rad/day; n is taken from L_DOT so the
    // velocity stays consistent with the element set rather than with a
    // separately-chosen GM.
    let n = L_DOT * DEG2RAD / DAYS_PER_CENTURY;
    let edot = n / (1.0 - e * cos_e);
    let vx_p = -a * sin_e * edot;
    let vy_p = a * beta * cos_e * edot;

    // Rotate the orbital plane into the J2000 ecliptic frame.
    let (sw, cw) = arg_peri.sin_cos();
    let (so, co) = node.sin_cos();
    let (si, ci) = inc.sin_cos();

    let rotate = |x: f64, y: f64| {
        let x_ecl = (cw * co - sw * so * ci) * x + (-sw * co - cw * so * ci) * y;
        let y_ecl = (cw * so + sw * co * ci) * x + (-sw * so + cw * co * ci) * y;
        let z_ecl = (sw * si) * x + (cw * si) * y;
        Vector3::new(x_ecl, y_ecl, z_ecl)
    };

    let r = ecliptic_to_equatorial();
    (r * rotate(x_p, y_p), r * rotate(vx_p, vy_p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Timescale;

    #[test]
    fn earth_speed_is_about_29_8_km_s() {
        let ts = Timescale::default();
        // AU/day -> km/s
        let scale = 149_597_870.7 / 86_400.0;
        for month in 1..=12 {
            let t = ts.utc((2024, month, 15, 0, 0, 0.0));
            let (pos, vel) = emb_heliocentric_two_body_state(&t);
            let speed = vel.norm() * scale;
            assert!(
                (29.0..=30.6).contains(&speed),
                "month {month}: speed {speed} km/s out of range"
            );
            let r = pos.norm();
            assert!((0.98..=1.02).contains(&r), "month {month}: r {r} AU");
        }
    }

    /// Pins the module-doc numbers: close to DE421's heliocentric EMB, and
    /// seconds of light-time away from its barycentric Earth.
    #[test]
    fn matches_de421_emb_and_is_not_barycentric_earth() {
        let mut kernel = crate::jplephem::kernel::SpiceKernel::open("test_data/de421.bsp")
            .expect("Failed to open DE421");
        let ts = Timescale::default();
        let au_km = 149_597_870.7;
        let au_day_to_m_s = au_km * 1000.0 / 86_400.0;
        let light_s_per_au = au_km / 299_792.458;

        let (mut emb_dr, mut emb_dv, mut earth_dr, mut earth_dv) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for year in (1990..=2040).step_by(2) {
            for month in 1..=12 {
                let t = ts.utc((year, month, 1, 0, 0, 0.0));
                let jd = t.tdb();
                let (pos, vel) = emb_heliocentric_two_body_state(&t);
                let emb = kernel.compute_at_jd("earth barycenter", jd).unwrap();
                let sun = kernel.compute_at_jd("sun", jd).unwrap();
                let earth = kernel.compute_at_jd("earth", jd).unwrap();

                emb_dr = emb_dr.max((pos - (emb.position - sun.position)).norm());
                emb_dv = emb_dv.max((vel - (emb.velocity - sun.velocity)).norm());
                earth_dr = earth_dr.max((pos - earth.position.coords).norm());
                earth_dv = earth_dv.max((vel - earth.velocity).norm());
            }
        }

        assert!(
            emb_dr * au_km < 20_000.0,
            "EMB position off by {} km",
            emb_dr * au_km
        );
        assert!(
            emb_dv * au_day_to_m_s < 3.0,
            "EMB velocity off by {} m/s",
            emb_dv * au_day_to_m_s
        );
        assert!(
            earth_dv * au_day_to_m_s < 35.0,
            "Earth velocity off by {} m/s",
            earth_dv * au_day_to_m_s
        );
        let earth_light_s = earth_dr * light_s_per_au;
        assert!(
            (3.0..5.0).contains(&earth_light_s),
            "barycentric Earth position gap {earth_light_s} light-s"
        );
    }

    #[test]
    fn perihelion_is_early_january() {
        let ts = Timescale::default();
        let mut best = (f64::INFINITY, 0u32);
        for day in 1..=40u32 {
            let t = ts.utc((2024, 1, day.min(31), 0, 0, 0.0));
            if day > 31 {
                continue;
            }
            let (pos, _) = emb_heliocentric_two_body_state(&t);
            if pos.norm() < best.0 {
                best = (pos.norm(), day);
            }
        }
        assert!(best.1 <= 8, "perihelion landed on Jan {}", best.1);
    }
}
