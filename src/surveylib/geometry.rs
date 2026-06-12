//! Per-epoch apparent geometry: heliocentric ecliptic states to apparent
//! geocentric equatorial positions.
//!
//! The chain is deliberately small and *typed*: target and Earth positions
//! are heliocentric **ecliptic J2000** vectors in AU; the geocentric
//! direction is their difference; and the rotation to the equatorial frame
//! is the ECLIPJ2000 matrix from [`crate::framelib`]. Declination only
//! ever comes out of that rotation — never substitute ecliptic latitude
//! for declination (an `i = 0` orbit has β ≈ 0 everywhere while its
//! declination swings ±23.4°, which misclassifies exactly the
//! survey-boundary sky).
//!
//! The geocentric difference captures the two effects that move a distant
//! object across a footprint boundary during a survey: the annual
//! parallactic oscillation (~`asin(1 AU / r)`, i.e. ±0.14° at 400 AU) and
//! the slow orbital drift over the survey baseline.
//! [`apparent_equatorial_light_time`] additionally iterates the
//! light-time retardation (days at hundreds of AU). Stellar aberration
//! (≤ 20.5″) is *not* applied — it is far below the footprint scales this
//! module serves; use the full `positions` observe chain when arcsecond
//! astrometry matters.
//!
//! Earth positions come from starfield's own ephemeris
//! ([`earth_positions_from_ephemeris`], e.g. DE421 via
//! [`crate::planetlib::Ephemeris`]) or from any caller-supplied model
//! ([`earth_positions_with`] — analytic circular Earths are often ample
//! for footprint membership and keep inner Monte-Carlo loops kernel-free).

use nalgebra::Vector3;

use crate::constants::C_AUDAY;
use crate::framelib::ECLIPJ2000_MATRIX_INV;
use crate::planetlib::{Body, Ephemeris};
use crate::time::Time;
use crate::{Result, StarfieldError};

/// An apparent geocentric equatorial position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ApparentPosition {
    /// Right ascension in degrees, in `[0°, 360°)`.
    pub ra_deg: f64,
    /// Declination in degrees.
    pub dec_deg: f64,
    /// Geocentric distance in AU.
    pub delta_au: f64,
}

/// Geometric (instantaneous) apparent geocentric equatorial position of a
/// target, from its heliocentric **ecliptic J2000** position and the
/// Earth's heliocentric ecliptic J2000 position (both AU).
///
/// # Example
///
/// ```
/// use nalgebra::Vector3;
/// use starfield::surveylib::geometry::apparent_equatorial;
///
/// // A target toward the north ecliptic pole sits at dec = 90° − ε.
/// let pos = apparent_equatorial(&Vector3::new(0.0, 0.0, 400.0), &Vector3::zeros());
/// assert!((pos.dec_deg - (90.0 - 23.4393)).abs() < 1e-3);
/// ```
pub fn apparent_equatorial(
    target_helio_ecliptic_au: &Vector3<f64>,
    earth_helio_ecliptic_au: &Vector3<f64>,
) -> ApparentPosition {
    let geocentric_ecliptic = target_helio_ecliptic_au - earth_helio_ecliptic_au;
    let geocentric_equatorial = *ECLIPJ2000_MATRIX_INV * geocentric_ecliptic;
    let delta_au = geocentric_equatorial.norm();
    let ra_deg = geocentric_equatorial
        .y
        .atan2(geocentric_equatorial.x)
        .to_degrees()
        .rem_euclid(360.0);
    let dec_deg = (geocentric_equatorial.z / delta_au).asin().to_degrees();
    ApparentPosition {
        ra_deg,
        dec_deg,
        delta_au,
    }
}

/// Light-time-corrected apparent position: where the target *appears*,
/// i.e. where it was `Δ/c` before the observation epoch.
///
/// `target_at(dt_days)` must return the target's heliocentric ecliptic
/// J2000 position (AU) at `dt_days` relative to the observation epoch
/// (`dt_days <= 0` during the iteration). The retardation is solved by
/// fixed-point iteration `dt = −Δ(dt)/c`, which converges in two or three
/// rounds for solar-system geometries; at hundreds of AU the light time
/// is days and the correction shifts the apparent position by the
/// object's sky motion over that interval.
pub fn apparent_equatorial_light_time(
    target_at: impl Fn(f64) -> Vector3<f64>,
    earth_helio_ecliptic_au: &Vector3<f64>,
) -> ApparentPosition {
    let mut dt_days = 0.0;
    let mut position = apparent_equatorial(&target_at(dt_days), earth_helio_ecliptic_au);
    for _ in 0..10 {
        let next_dt = -position.delta_au / C_AUDAY;
        if (next_dt - dt_days).abs() < 1e-9 {
            break;
        }
        dt_days = next_dt;
        position = apparent_equatorial(&target_at(dt_days), earth_helio_ecliptic_au);
    }
    position
}

/// Heliocentric ecliptic J2000 Earth positions (AU) at the given times,
/// from a real ephemeris kernel (e.g. DE421 loaded into
/// [`crate::planetlib::Ephemeris`]).
///
/// Precompute these once per survey epoch grid and share them across
/// every orbit in a population run — the grid is orbit-independent, so
/// the kernel is never queried inside per-orbit loops.
pub fn earth_positions_from_ephemeris(
    ephemeris: &mut Ephemeris,
    times: &[Time],
) -> Result<Vec<Vector3<f64>>> {
    times
        .iter()
        .map(|t| {
            ephemeris
                .ecliptic_state(Body::Earth, t)
                .map(|state| state.position.to_vector3())
                .map_err(|e| StarfieldError::CalculationError(format!("Earth ephemeris: {e}")))
        })
        .collect()
}

/// Heliocentric ecliptic J2000 Earth positions (AU) at the given times,
/// from a caller-supplied model (analytic circular orbit, cached table,
/// test stub, ...). The counterpart of [`earth_positions_from_ephemeris`]
/// for kernel-free callers.
pub fn earth_positions_with(
    mut earth_at: impl FnMut(&Time) -> Vector3<f64>,
    times: &[Time],
) -> Vec<Vector3<f64>> {
    times.iter().map(&mut earth_at).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::GM_SUN;
    use crate::jplephem::SpiceKernel;
    use crate::keplerlib::KeplerOrbit;
    use crate::time::Timescale;

    /// Heliocentric *ecliptic* position of a Keplerian orbit at `t`:
    /// `from_mean_anomaly` without the ecliptic rotation keeps the state
    /// in the frame the elements were given in.
    fn ecliptic_orbit(a: f64, e: f64, i_deg: f64, m_deg: f64, epoch: &Time) -> KeplerOrbit {
        KeplerOrbit::from_mean_anomaly(
            a * (1.0 - e * e),
            e,
            i_deg,
            0.0,
            0.0,
            m_deg,
            epoch,
            GM_SUN,
            Some(10),
            None,
        )
    }

    #[test]
    fn ecliptic_pole_maps_to_obliquity_complement() {
        let pos = apparent_equatorial(&Vector3::new(0.0, 0.0, 1.0), &Vector3::zeros());
        assert!(
            (pos.dec_deg - (90.0 - 23.439_291)).abs() < 1e-4,
            "dec = {}",
            pos.dec_deg
        );
    }

    #[test]
    fn equinox_direction_unchanged() {
        // The vernal-equinox direction is shared by both frames.
        let pos = apparent_equatorial(&Vector3::new(1.0, 0.0, 0.0), &Vector3::zeros());
        assert!(pos.ra_deg.abs() < 1e-9 || (pos.ra_deg - 360.0).abs() < 1e-9);
        assert!(pos.dec_deg.abs() < 1e-9);
    }

    #[test]
    fn ecliptic_plane_orbit_declination_swings_by_obliquity() {
        // The beta-for-delta regression: an i = 0 orbit has ecliptic
        // latitude ~0 everywhere, but its *declination* must swing by
        // +/- the obliquity (~23.4 deg) around the orbit. A conflated
        // implementation reports ~0 everywhere.
        let ts = Timescale::default();
        let epoch = ts.tt_jd(2_451_545.0, None);
        let mut min_dec = f64::INFINITY;
        let mut max_dec = f64::NEG_INFINITY;
        for k in 0..72 {
            // e is tiny but nonzero: the universal-variable Kepler solver
            // is singular at exactly e = 0.
            let orbit = ecliptic_orbit(400.0, 1e-6, 0.0, k as f64 * 5.0, &epoch);
            let helio = orbit.at(&epoch).position;
            let pos = apparent_equatorial(&helio, &Vector3::zeros());
            min_dec = min_dec.min(pos.dec_deg);
            max_dec = max_dec.max(pos.dec_deg);
        }
        assert!(min_dec < -20.0, "min dec = {min_dec:.1}");
        assert!(max_dec > 20.0, "max dec = {max_dec:.1}");
        assert!(max_dec < 23.5 && min_dec > -23.5);
    }

    #[test]
    fn parallax_amplitude_scales_inversely_with_distance() {
        // Over one year the apparent RA of a distant stationary object
        // oscillates with half-amplitude ~ asin(1 AU / r).
        let target = Vector3::new(0.0, 400.0, 0.0);
        let mut ra_min = f64::INFINITY;
        let mut ra_max = f64::NEG_INFINITY;
        for k in 0..=24 {
            let theta = std::f64::consts::TAU * k as f64 / 24.0;
            let earth = Vector3::new(theta.cos(), theta.sin(), 0.0);
            let pos = apparent_equatorial(&target, &earth);
            ra_min = ra_min.min(pos.ra_deg);
            ra_max = ra_max.max(pos.ra_deg);
        }
        let half_amplitude = (ra_max - ra_min) / 2.0;
        // ~0.143 deg of ecliptic longitude; the RA projection at lambda =
        // 90 deg stretches it by 1/cos(obliquity) (~9%).
        let expected = (1.0_f64 / 400.0).asin().to_degrees();
        assert!(
            (half_amplitude - expected).abs() < 0.03,
            "half-amplitude = {half_amplitude:.4} deg, expected ~{expected:.4}"
        );
    }

    #[test]
    fn light_time_displaces_by_sky_motion_over_delta_over_c() {
        // A target moving at constant velocity appears where it was a
        // light time ago: displaced *backward* along its motion by
        // v * delta/c (in ecliptic longitude; the RA projection at
        // lambda = 90 deg stretches it by 1/cos(obliquity)).
        let r = 400.0;
        let v = Vector3::new(0.001, 0.0, 0.0); // AU/day
        let target_at = |dt: f64| Vector3::new(v.x * dt, r, 0.0);
        let earth = Vector3::zeros();
        let instantaneous = apparent_equatorial(&target_at(0.0), &earth);
        let apparent = apparent_equatorial_light_time(target_at, &earth);
        // Light time at 400 AU is ~2.31 days.
        let lt_days = r / C_AUDAY;
        assert!((2.0..2.5).contains(&lt_days), "lt = {lt_days:.2} d");
        let lambda_shift = (v.x * lt_days / r).to_degrees();
        let expected_shift = -lambda_shift / 23.439_291_f64.to_radians().cos();
        let actual_shift = instantaneous.ra_deg - apparent.ra_deg;
        assert!(
            (actual_shift - expected_shift).abs() < 2e-6,
            "shift = {actual_shift:.2e} deg, expected {expected_shift:.2e}"
        );
    }

    #[test]
    fn ephemeris_earth_positions_are_one_au_heliocentric() {
        let Ok(kernel) = SpiceKernel::open("test_data/de421.bsp") else {
            eprintln!("skipping: test_data/de421.bsp not available");
            return;
        };
        let mut ephemeris = Ephemeris::from_kernel(kernel);
        let ts = Timescale::default();
        let times: Vec<Time> = (0..12)
            .map(|k| ts.tt_jd(2_456_000.0 + 30.0 * k as f64, None))
            .collect();
        let positions = earth_positions_from_ephemeris(&mut ephemeris, &times).unwrap();
        assert_eq!(positions.len(), times.len());
        for p in &positions {
            let r = p.norm();
            assert!((0.97..1.03).contains(&r), "|Earth| = {r:.4} AU");
            // Heliocentric ecliptic frame: Earth stays in the ecliptic
            // plane to well under a degree.
            assert!((p.z / r).asin().to_degrees().abs() < 0.01);
        }
        // The twelve samples actually sweep the orbit (not a constant).
        let spread = positions
            .iter()
            .map(|p| p.x)
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                (lo.min(x), hi.max(x))
            });
        assert!(spread.1 - spread.0 > 1.0);
    }

    #[test]
    fn closure_earth_positions_align_with_times() {
        let ts = Timescale::default();
        let times: Vec<Time> = (0..4)
            .map(|k| ts.tt_jd(2_451_545.0 + k as f64, None))
            .collect();
        let positions = earth_positions_with(
            |t| {
                let theta = std::f64::consts::TAU * (t.tt() - 2_451_545.0) / 365.25;
                Vector3::new(theta.cos(), theta.sin(), 0.0)
            },
            &times,
        );
        assert_eq!(positions.len(), 4);
        assert!((positions[0] - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
        assert!(positions[1].y > 0.0);
    }
}
