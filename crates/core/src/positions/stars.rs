//! Apparent places of catalog stars for an arbitrary observer
//!
//! Port of Skyfield's `Star._observe_from_bcrs()` in the shape the catalog
//! types of this crate use: a [`StarData`] row, an optional
//! [`ProperMotion`], and an optional parallax.  The observer is any
//! barycentric [`Position`] — a planet from an SPK kernel, a ground site
//! from `toposlib`, or a spacecraft — so a star direction can be obtained in
//! the same frame, and with the same aberration, as a planet observed from
//! that observer.
//!
//! The returned position is [`PositionKind::Astrometric`] with
//! `observer_barycentric` populated, so it flows through the ordinary
//! [`Position::apparent`] pipeline:
//!
//! ```ignore
//! let mars = kernel.at("mars", &t)?;
//! let star = StarData::new(32349, 101.2874, -16.7161, -1.46, None);
//! let pm = ProperMotion::new(-546.01, -1223.07);
//! let apparent = mars
//!     .observe_star(&star, Some(&pm), Some(379.21), &t)
//!     .apparent(&mut kernel, &t)?;
//! ```
//!
//! An observer moving with Mars picks up roughly 17″ of stellar aberration
//! (Mars's ~24 km/s orbital speed divided by the speed of light), exactly as
//! the planets observed from the same position do.

use crate::catalogs::StarData;
use crate::framelib::inertial::ProperMotion;
use crate::positions::{Position, PositionKind};
use crate::starlib::Star;
use crate::time::Time;

/// Target id given to catalog stars.
///
/// Stars have no NAIF body id.  This crate labels every star position with
/// the sentinel `-1`, which is outside the range of ephemeris body ids that
/// [`Position::observe`] can produce and is the value
/// [`Star::observe_from`](crate::starlib::Star::observe_from) has always
/// used.  Consumers that need to tell stars apart from solar-system targets
/// should compare `position.target` against this constant.
pub const STAR_TARGET_ID: i32 = -1;

impl Position {
    /// Astrometric direction from this observer to a catalog star.
    ///
    /// The catalog position in `star` is taken to refer to epoch J2000.0
    /// (the ICRS reference epoch).  It is carried forward to `epoch` with the
    /// supplied proper motion, placed at the distance implied by
    /// `parallax_mas`, and differenced against this observer's barycentric
    /// position — so both proper motion and annual parallax are applied.
    /// A star with no parallax is placed at one gigaparsec, following
    /// Skyfield, which makes the parallax term vanish without special-casing
    /// it.
    ///
    /// # Arguments
    /// * `star` — catalog row supplying ICRS right ascension and declination
    /// * `pm` — proper motion in mas/yr (`pmra` already includes cos δ);
    ///   `None` means the star is treated as fixed
    /// * `parallax_mas` — parallax in milliarcseconds; `None` or a
    ///   non-positive value places the star at one gigaparsec
    /// * `epoch` — the time of observation
    ///
    /// `StarData` carries no radial velocity, so the Doppler factor that
    /// Skyfield folds into the space motion is 1.  For a star with a known
    /// radial velocity, build a [`Star`](crate::starlib::Star) directly and
    /// call [`Star::observe_from`](crate::starlib::Star::observe_from), which
    /// takes `radial_km_per_s` and accepts a non-J2000 catalog epoch through
    /// [`Star::with_epoch`](crate::starlib::Star::with_epoch).
    ///
    /// # Panics
    /// Panics if this position is not [`PositionKind::Barycentric`], because
    /// proper motion and parallax are only meaningful against a barycentric
    /// observer.
    pub fn observe_star(
        &self,
        star: &StarData,
        pm: Option<&ProperMotion>,
        parallax_mas: Option<f64>,
        epoch: &Time,
    ) -> Position {
        assert_eq!(
            self.kind,
            PositionKind::Barycentric,
            "observe_star() requires a Barycentric position"
        );

        let pm = pm.copied().unwrap_or(ProperMotion::ZERO);
        let star = Star::new(
            star.ra_deg(),
            star.dec_deg(),
            pm.pmra,
            pm.pmdec,
            parallax_mas.unwrap_or(0.0),
            0.0,
        );
        star.observe_from(self, epoch)
    }
}

#[cfg(all(test, feature = "python-tests"))]
mod python_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::C_AUDAY;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::time::Timescale;
    use nalgebra::Vector3;

    /// Radians in one arcsecond
    const ARCSEC: f64 = std::f64::consts::PI / 180.0 / 3600.0;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
    }

    /// Sirius, from the Hipparcos entries embedded in `catalogs::hipparcos`.
    fn sirius() -> (StarData, ProperMotion, f64) {
        (
            StarData::new(32349, 101.2874, -16.7161, -1.46, None),
            ProperMotion::new(-546.01, -1223.07),
            379.21,
        )
    }

    #[test]
    fn test_observe_star_is_astrometric() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let earth = kernel.at("earth", &t).unwrap();

        let (star, pm, plx) = sirius();
        let astro = earth.observe_star(&star, Some(&pm), Some(plx), &t);

        assert_eq!(astro.kind, PositionKind::Astrometric);
        assert_eq!(astro.target, STAR_TARGET_ID);
        assert_eq!(astro.center, 399);
        assert!(astro.observer_barycentric.is_some());
        assert!(astro.light_time > 0.0);
        // Distance to Sirius is 2.64 pc = 5.4e5 AU
        assert!(
            astro.distance() > 4.0e5 && astro.distance() < 7.0e5,
            "Sirius distance {} AU",
            astro.distance()
        );
        assert!((astro.light_time - astro.distance() / C_AUDAY).abs() < 1e-9);
    }

    #[test]
    fn test_observe_star_apparent_pipeline() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let earth = kernel.at("earth", &t).unwrap();

        let (star, pm, plx) = sirius();
        let apparent = earth
            .observe_star(&star, Some(&pm), Some(plx), &t)
            .apparent(&mut kernel, &t)
            .unwrap();

        assert_eq!(apparent.kind, PositionKind::Apparent);
        let (ra_h, dec_d, _) = apparent.radec(None);
        // Sirius: 6h45m, -16.7 degrees, shifted at most ~20" by aberration
        assert!((ra_h - 101.2874 / 15.0).abs() < 0.01, "RA {ra_h} h");
        assert!((dec_d + 16.7161).abs() < 0.01, "Dec {dec_d} deg");
    }

    #[test]
    fn test_zero_motion_returns_catalog_direction() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let earth = kernel.at("earth", &t).unwrap();

        // No proper motion, no parallax: the star sits at one gigaparsec and
        // the astrometric direction is the catalog direction, to well under a
        // microarcsecond, whatever the observer does.
        let star = StarData::new(1, 123.456, -45.678, 5.0, None);
        let astro = earth.observe_star(&star, None, None, &t);
        let (ra_h, dec_d, _) = astro.radec(None);

        assert!(
            (ra_h * 15.0 - 123.456).abs() < 1e-9,
            "RA {} deg",
            ra_h * 15.0
        );
        assert!((dec_d + 45.678).abs() < 1e-9, "Dec {dec_d} deg");
    }

    /// A star seen from Mars is aberrated by |v|/c relative to the same star
    /// seen from a motionless observer at the solar system barycenter.
    #[test]
    fn test_aberration_from_mars_matches_v_over_c() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let mars = kernel.at("mars", &t).unwrap();
        let ssb = Position::barycentric(Vector3::zeros(), Vector3::zeros(), 0);

        // Zero parallax keeps the two observers' geometric parallax out of
        // the comparison, leaving aberration alone.
        let star = StarData::new(1, 30.0, 20.0, 5.0, None);

        let from_mars = mars
            .observe_star(&star, None, None, &t)
            .apparent(&mut kernel, &t)
            .unwrap();
        let from_ssb = ssb
            .observe_star(&star, None, None, &t)
            .apparent(&mut kernel, &t)
            .unwrap();

        let measured = from_mars.separation_from(&from_ssb);

        // Expected: (v/c) sin(theta) between Mars's velocity and the star.
        let speed_au_day = mars.velocity.norm();
        let cos_theta = mars
            .velocity
            .normalize()
            .dot(&from_ssb.position.normalize());
        let expected = speed_au_day / C_AUDAY * (1.0 - cos_theta * cos_theta).sqrt();

        // Mars's orbital speed runs from 21.97 km/s at aphelion to 26.50 km/s
        // at perihelion, so its full aberration constant v/c is 15.1"–18.2".
        let speed_km_s = speed_au_day * crate::constants::AU_KM / crate::constants::DAY_S;
        assert!(
            (21.9..26.6).contains(&speed_km_s),
            "Mars barycentric speed {speed_km_s} km/s"
        );
        let v_over_c_arcsec = speed_au_day / C_AUDAY / ARCSEC;
        assert!(
            (15.0..18.3).contains(&v_over_c_arcsec),
            "Mars v/c = {v_over_c_arcsec} arcsec"
        );
        println!(
            "Mars aberration: measured {:.4}\" expected {:.4}\" (v/c = {v_over_c_arcsec:.3}\")",
            measured / ARCSEC,
            expected / ARCSEC
        );
        assert!(
            (measured - expected).abs() < 0.02 * expected,
            "aberration: measured {} arcsec, expected {} arcsec",
            measured / ARCSEC,
            expected / ARCSEC
        );
    }

    #[test]
    fn test_proper_motion_moves_the_star() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tdb_jd(2451545.0);
        let t1 = ts.tdb_jd(2451545.0 + 50.0 * 365.25);

        let earth0 = kernel.at("earth", &t0).unwrap();
        let earth1 = kernel.at("earth", &t1).unwrap();

        let (star, pm, plx) = sirius();
        let a0 = earth0.observe_star(&star, Some(&pm), Some(plx), &t0);
        let a1 = earth1.observe_star(&star, Some(&pm), Some(plx), &t1);

        // Sirius moves 1.34"/yr, so 67" in 50 years; parallax adds < 0.8".
        let moved = a0.separation_from(&a1) / ARCSEC;
        assert!(
            (60.0..75.0).contains(&moved),
            "Sirius moved {moved} arcsec in 50 years"
        );
    }

    #[test]
    #[should_panic(expected = "observe_star() requires a Barycentric position")]
    fn test_observe_star_rejects_astrometric_observer() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        let (star, pm, plx) = sirius();
        let _ = mars.observe_star(&star, Some(&pm), Some(plx), &t);
    }
}
