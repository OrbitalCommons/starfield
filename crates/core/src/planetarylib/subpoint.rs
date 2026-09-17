//! Sub-observer and sub-solar points: which face of a body is turned toward
//! the observer, and which toward the Sun.
//!
//! The sub-observer point is where the line from the body's centre to the
//! observer pierces the body's surface — the centre of the apparent disc. The
//! sub-solar point is the same construction toward the Sun, and the great
//! circle 90° from it is the terminator. Together they say which hemisphere of
//! a texture to draw and where the day-night line falls across it.
//!
//! Both are methods on [`Position`] and both need a body-fixed
//! [`Frame`]: [`IauFrame`](crate::planetarylib::IauFrame) for most bodies,
//! [`ItrsFrame`](crate::framelib::ItrsFrame) for the Earth, or whichever frame
//! [`PlanetaryConstants::frame_for`](crate::planetarylib::PlanetaryConstants::frame_for)
//! picks. The frame is evaluated at the light-time corrected epoch
//! `t − light_time`, the moment the light now reaching the observer left the
//! body, which is what SPICE's `subpnt` does with aberration correction
//! `LT+S` and what JPL Horizons reports.
//!
//! # Planetocentric, planetographic, east and west
//!
//! Two latitudes and two longitude senses are in circulation, and Horizons
//! and SPICE report the ones that are least convenient to compute:
//!
//! * **Planetocentric** latitude is the angle at the centre of the body
//!   between its equator and the point. It is what the body-fixed vector
//!   gives directly.
//! * **Planetographic** (the SPICE term is *planetodetic*) latitude is the
//!   angle between the equator and the surface normal at the point, so it
//!   carries the body's oblateness. For the disc-centre construction used
//!   here the two are related by `tan φ_graphic = (a/c)² tan φ_centric`. The
//!   difference reaches 0.19° on Mars and 0.19° on the Earth, and 4° on
//!   Saturn.
//! * **Longitude** runs east in every body-fixed frame — the IAU prime
//!   meridian angle W increases eastward — but planetographic longitude is
//!   reported *west*-positive for bodies that rotate prograde, and
//!   east-positive for the retrograde rotators and for the three bodies the
//!   IAU exempts. [`LongitudeSense::for_body`] holds the table.
//!
//! The two methods here return planetographic points, since that is what the
//! references publish; [`SubPoint::to_planetocentric`] converts back, and
//! [`SubPoint::longitude_in`] applies a longitude sense.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::kernel::SpiceKernel;
//! use starfield_core::jplephem_ext::SpiceKernelExt;
//! use starfield_core::planetarylib::subpoint::LongitudeSense;
//! use starfield_core::planetarylib::IauFrame;
//! use starfield_core::planetlib::Body;
//! use starfield_core::time::Timescale;
//!
//! let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
//! let t = Timescale::default().utc((2007, 10, 3, 0, 0, 0.0));
//!
//! let frame = IauFrame::from_body(Body::Mars);
//! let radii = Body::Mars.radii_km();
//!
//! let earth = kernel.at("earth", &t).unwrap();
//! let mars = earth.observe("mars", &mut kernel, &t).unwrap();
//!
//! let sub_observer = mars.sub_observer_point(&frame, radii, &t);
//! println!(
//!     "sub-observer: {:.3}° W, {:.3}° N",
//!     sub_observer.longitude_in(LongitudeSense::West).to_degrees(),
//!     sub_observer.lat_rad.to_degrees(),
//! );
//! ```

use nalgebra::Vector3;

use crate::constants::{C_AUDAY, TAU};
use crate::framelib::Frame;
use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem_ext::SpiceKernelExt;
use crate::positions::Position;
use crate::time::Time;
use crate::Result;

/// How many times the Sun's position is re-evaluated to remove the light time
/// of the Sun → target leg. The leg is under 20 minutes for every body in
/// DE421 and the Sun moves by metres in that time, so one pass converges.
const SUN_LIGHT_TIME_ITERATIONS: usize = 2;

/// A point on a body's surface, given as a longitude and a latitude.
///
/// Produced by [`Position::sub_observer_point`] and
/// [`Position::sub_solar_point`]. The longitude is always **east**-positive
/// and in `[0, 2π)`; use [`longitude_in`](Self::longitude_in) to get the
/// west-positive value that the IAU and Horizons report for most bodies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubPoint {
    /// East-positive longitude in the body-fixed frame, radians in `[0, 2π)`.
    pub lon_rad: f64,
    /// Latitude in radians, positive north.
    pub lat_rad: f64,
    /// Whether `lat_rad` is planetocentric. When false it is planetographic,
    /// the oblate latitude of the surface normal, which is what
    /// [`Position::sub_observer_point`] returns.
    pub planetocentric: bool,
}

/// The direction in which a body's planetographic longitude is measured.
///
/// The IAU rule (Archinal et al. 2018 §2) is that planetographic longitude
/// increases in the direction *opposite* to the body's rotation, which makes
/// it west-positive for the prograde rotators and east-positive for the
/// retrograde ones — with the Earth, the Moon and the Sun exempted by long
/// convention and measured east-positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LongitudeSense {
    /// Longitude increases eastward, the sense of every body-fixed frame and
    /// of the reported longitudes of the Sun, Earth, Moon, Venus, Uranus and
    /// Pluto.
    East,
    /// Longitude increases westward, the sense in which the IAU and JPL
    /// Horizons report Mercury, Mars, Jupiter, Saturn and Neptune.
    West,
}

impl LongitudeSense {
    /// The sense in which planetographic longitude is reported for a body.
    ///
    /// | NAIF code | Body | Sense |
    /// |---|---|---|
    /// | 10 | Sun | east |
    /// | 199 | Mercury | west |
    /// | 299 | Venus | east (retrograde) |
    /// | 301 | Moon | east |
    /// | 399 | Earth | east |
    /// | 499 | Mars | west |
    /// | 599 | Jupiter | west |
    /// | 699 | Saturn | west |
    /// | 799 | Uranus | east (retrograde) |
    /// | 899 | Neptune | west |
    /// | 999 | Pluto | east (retrograde) |
    ///
    /// Every entry agrees with the `{West-longitude positive}` or
    /// `{East-longitude positive}` note that JPL Horizons prints in the
    /// `Target pole/equ` line of an observer table. A barycentre code (`4`
    /// for Mars, and so on) is read as the body of the same system, and an
    /// unknown code falls back to east, the sense of the underlying frame.
    pub fn for_body(naif_id: i32) -> Self {
        let body = if (1..=9).contains(&naif_id) {
            naif_id * 100 + 99
        } else {
            naif_id
        };
        match body {
            199 | 499 | 599 | 699 | 899 => LongitudeSense::West,
            _ => LongitudeSense::East,
        }
    }
}

impl SubPoint {
    /// The longitude of the point measured in the given sense, radians in
    /// `[0, 2π)`.
    ///
    /// East is the frame's own sense and returns [`lon_rad`](Self::lon_rad)
    /// unchanged; west returns `2π − lon_rad`, the value Horizons prints for
    /// a body whose [`LongitudeSense::for_body`] is
    /// [`West`](LongitudeSense::West).
    pub fn longitude_in(&self, sense: LongitudeSense) -> f64 {
        match sense {
            LongitudeSense::East => self.lon_rad,
            LongitudeSense::West => wrap(-self.lon_rad),
        }
    }

    /// The same point with a planetographic (planetodetic) latitude.
    ///
    /// `tan φ_graphic = (a/c)² tan φ_centric`, where `a` is the equatorial
    /// radius `radii_km[0]` and `c` the polar radius `radii_km[2]`. As
    /// Horizons and SPICE do, the body is treated as an oblate spheroid and
    /// the second equatorial radius `radii_km[1]` is ignored.
    ///
    /// Returns the point unchanged if it is planetographic already.
    pub fn to_planetographic(&self, radii_km: [f64; 3]) -> SubPoint {
        if !self.planetocentric {
            return *self;
        }
        SubPoint {
            lon_rad: self.lon_rad,
            lat_rad: scale_latitude(self.lat_rad, flattening_factor(radii_km)),
            planetocentric: false,
        }
    }

    /// The same point with a planetocentric latitude, the inverse of
    /// [`to_planetographic`](Self::to_planetographic).
    ///
    /// Returns the point unchanged if it is planetocentric already.
    pub fn to_planetocentric(&self, radii_km: [f64; 3]) -> SubPoint {
        if self.planetocentric {
            return *self;
        }
        SubPoint {
            lon_rad: self.lon_rad,
            lat_rad: scale_latitude(self.lat_rad, 1.0 / flattening_factor(radii_km)),
            planetocentric: true,
        }
    }

    /// The planetographic point that a body-fixed direction points at.
    ///
    /// The direction need not be a unit vector; only its direction is used,
    /// so this is the disc-centre ("intercept") construction rather than the
    /// nearest point of the ellipsoid. Horizons reports the disc centre, and
    /// says so: "This is NOT exactly the same as the *nearest* sub-point for
    /// a non-spherical target shape".
    fn from_body_fixed(direction: Vector3<f64>, radii_km: [f64; 3]) -> SubPoint {
        let r = direction.norm();
        let lat = if r > 0.0 {
            (direction.z / r).asin()
        } else {
            0.0
        };
        SubPoint {
            lon_rad: wrap(direction.y.atan2(direction.x)),
            lat_rad: lat,
            planetocentric: true,
        }
        .to_planetographic(radii_km)
    }
}

impl Position {
    /// The sub-observer point: where the centre of the apparent disc falls on
    /// the body's surface.
    ///
    /// `self` is the position of the body as seen by the observer, from
    /// [`observe`](Position::observe); `frame` is the body's body-fixed
    /// frame and `radii_km` its triaxial radii, both of which
    /// [`planetlib::Body`](crate::planetlib::Body) and
    /// [`PlanetaryConstants`](crate::planetarylib::PlanetaryConstants) can
    /// supply. The frame is evaluated at `t − light_time`.
    ///
    /// The returned latitude is planetographic and the longitude
    /// east-positive; see the [module documentation](self) for the
    /// conversions.
    ///
    /// Give this an *astrometric* position. An apparent one differs by the
    /// deflection and aberration of the incoming light, which turn the whole
    /// sky rather than the body and displace the sub-point by up to about
    /// 0.006°.
    pub fn sub_observer_point(&self, frame: &dyn Frame, radii_km: [f64; 3], t: &Time) -> SubPoint {
        let epoch = t.shift_days(-self.light_time);
        SubPoint::from_body_fixed(frame.rotation_at(&epoch) * -self.position, radii_km)
    }

    /// The sub-solar point: where the Sun stands overhead on the body, the
    /// centre of the lit hemisphere.
    ///
    /// Takes the same arguments as
    /// [`sub_observer_point`](Position::sub_observer_point) plus the kernel,
    /// which supplies the Sun. Both legs of the light path are corrected: the
    /// body is taken at `t − light_time` and the Sun at the further light
    /// time of the Sun → body leg before that, as Horizons does.
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::MissingObserver`](crate::StarfieldError::MissingObserver)
    /// if `self` does not carry the observer's barycentric position, and
    /// [`StarfieldError::EphemerisError`](crate::StarfieldError::EphemerisError)
    /// if the kernel cannot place the Sun.
    pub fn sub_solar_point(
        &self,
        frame: &dyn Frame,
        radii_km: [f64; 3],
        kernel: &mut SpiceKernel,
        t: &Time,
    ) -> Result<SubPoint> {
        let epoch = t.shift_days(-self.light_time);
        let body_to_sun = sun_seen_from_target(self, kernel, t)?;
        Ok(SubPoint::from_body_fixed(
            frame.rotation_at(&epoch) * body_to_sun,
            radii_km,
        ))
    }
}

/// The vector in AU from the target of `position` to the Sun, in the ICRF,
/// with the light time of both legs of the path removed: the body is taken at
/// `t − light_time` and the Sun at the light time of the Sun → body leg
/// before that, which is the geometry Horizons reports.
///
/// # Errors
///
/// Returns [`StarfieldError::MissingObserver`](crate::StarfieldError::MissingObserver)
/// if `position` does not carry the observer's barycentric position, and the
/// kernel's own errors if it cannot place the Sun.
pub(crate) fn sun_seen_from_target(
    position: &Position,
    kernel: &mut SpiceKernel,
    t: &Time,
) -> Result<Vector3<f64>> {
    let observer = position.require_observer()?;
    let epoch = t.shift_days(-position.light_time);

    // The body's own barycentric position at the moment the light left it.
    let body = observer.position + position.position;

    let mut sun = kernel.at("sun", &epoch)?.position;
    for _ in 0..SUN_LIGHT_TIME_ITERATIONS {
        let light_time = (body - sun).norm() / C_AUDAY;
        sun = kernel.at("sun", &epoch.shift_days(-light_time))?.position;
    }

    Ok(sun - body)
}

/// `(a / c)²`, the factor that turns a planetocentric latitude's tangent into
/// a planetographic one.
fn flattening_factor(radii_km: [f64; 3]) -> f64 {
    (radii_km[0] / radii_km[2]).powi(2)
}

/// Scale the tangent of a latitude, the operation both latitude conversions
/// perform.
fn scale_latitude(lat_rad: f64, factor: f64) -> f64 {
    (factor * lat_rad.tan()).atan()
}

/// Wrap an angle into `[0, 2π)`.
fn wrap(angle: f64) -> f64 {
    let wrapped = angle % TAU;
    if wrapped < 0.0 {
        wrapped + TAU
    } else {
        wrapped
    }
}

/// The JPL Horizons rows that the sub-point and position-angle tests compare
/// against, checked in so that neither test needs the network.
#[cfg(test)]
pub(crate) mod horizons_fixture {
    use crate::time::{Time, Timescale};

    /// The checked-in observer table.
    const ROWS: &str = include_str!("horizons_disk_orientation.csv");

    /// One row of the fixture: the disc orientation of `target` seen from the
    /// centre of `center` at a UT Julian date.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct DiskRow {
        pub target: i32,
        pub center: i32,
        pub jd_ut: f64,
        /// `ObsSub-LON`, degrees, in the sense Horizons reports for `target`.
        pub obs_sub_lon_deg: f64,
        /// `ObsSub-LAT`, planetodetic degrees.
        pub obs_sub_lat_deg: f64,
        /// `SunSub-LON`, degrees, in the sense Horizons reports for `target`.
        pub sun_sub_lon_deg: f64,
        /// `SunSub-LAT`, planetodetic degrees.
        pub sun_sub_lat_deg: f64,
        /// `SN.ang`, the sub-solar point's position angle in degrees.
        pub sn_ang_deg: f64,
        /// `NP.ang`, the north pole's position angle in degrees.
        pub np_ang_deg: f64,
    }

    impl DiskRow {
        /// The row's epoch, whose Julian date Horizons gives in UT.
        pub(crate) fn time(&self, ts: &Timescale) -> Time {
            ts.utc(ts.jd_to_calendar(self.jd_ut))
        }
    }

    /// Every row of the fixture, in file order.
    pub(crate) fn rows() -> Vec<DiskRow> {
        ROWS.lines()
            .filter(|line| {
                !line.starts_with('#') && !line.is_empty() && !line.starts_with("target")
            })
            .map(|line| {
                let f: Vec<f64> = line
                    .split(',')
                    .map(|v| v.trim().parse().expect("fixture holds numbers"))
                    .collect();
                assert_eq!(f.len(), 9, "fixture row {line:?} has the wrong width");
                DiskRow {
                    target: f[0] as i32,
                    center: f[1] as i32,
                    jd_ut: f[2],
                    obs_sub_lon_deg: f[3],
                    obs_sub_lat_deg: f[4],
                    sun_sub_lon_deg: f[5],
                    sun_sub_lat_deg: f[6],
                    sn_ang_deg: f[7],
                    np_ang_deg: f[8],
                }
            })
            .collect()
    }

    /// The rows for one geometry.
    pub(crate) fn rows_for(target: i32, center: i32) -> Vec<DiskRow> {
        rows()
            .into_iter()
            .filter(|row| row.target == target && row.center == center)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::horizons_fixture::{rows_for, DiskRow};
    use super::*;
    use crate::framelib::ItrsFrame;
    use crate::planetarylib::IauFrame;
    use crate::planetlib::Body;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
    }

    /// The body-fixed frame and radii Horizons used for a target: ITRF93 for
    /// the Earth, the IAU elements otherwise.
    fn frame_and_radii(naif_id: i32) -> (Box<dyn Frame>, [f64; 3]) {
        let radii = crate::planetarylib::body_constants(naif_id).unwrap().radii;
        if naif_id == 399 {
            (Box::new(ItrsFrame), radii)
        } else {
            (Box::new(IauFrame::from_naif_id(naif_id).unwrap()), radii)
        }
    }

    /// The signed difference of two angles in degrees, wrapped to ±180°.
    fn angle_error_deg(a: f64, b: f64) -> f64 {
        let mut diff = (a - b) % 360.0;
        if diff > 180.0 {
            diff -= 360.0;
        }
        if diff < -180.0 {
            diff += 360.0;
        }
        diff
    }

    /// Our sub-observer and sub-solar points for one fixture row.
    fn sub_points(row: &DiskRow) -> (SubPoint, SubPoint) {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = row.time(&ts);

        let (frame, radii) = frame_and_radii(row.target);
        let observer = kernel.at(&row.center.to_string(), &t).unwrap();
        let target = observer
            .observe(&row.target.to_string(), &mut kernel, &t)
            .unwrap();

        let sub_observer = target.sub_observer_point(frame.as_ref(), radii, &t);
        let sub_solar = target
            .sub_solar_point(frame.as_ref(), radii, &mut kernel, &t)
            .unwrap();
        (sub_observer, sub_solar)
    }

    #[test]
    fn test_longitude_sense_of_every_body() {
        for id in [199, 499, 599, 699, 899] {
            assert_eq!(LongitudeSense::for_body(id), LongitudeSense::West, "{id}");
        }
        for id in [10, 299, 301, 399, 799, 999] {
            assert_eq!(LongitudeSense::for_body(id), LongitudeSense::East, "{id}");
        }
        // Barycentre codes stand for the body of the same system.
        assert_eq!(LongitudeSense::for_body(4), LongitudeSense::West);
        assert_eq!(LongitudeSense::for_body(7), LongitudeSense::East);
    }

    #[test]
    fn test_longitude_in_west_is_the_complement() {
        let point = SubPoint {
            lon_rad: 1.0,
            lat_rad: 0.0,
            planetocentric: true,
        };
        assert_relative_eq!(
            point.longitude_in(LongitudeSense::West),
            TAU - 1.0,
            epsilon = 1e-15
        );
        assert_relative_eq!(point.longitude_in(LongitudeSense::East), 1.0);

        // Zero must not wrap to a full turn.
        let prime = SubPoint {
            lon_rad: 0.0,
            ..point
        };
        assert_relative_eq!(prime.longitude_in(LongitudeSense::West), 0.0);
    }

    #[test]
    fn test_latitude_conversions_round_trip() {
        let radii = Body::Mars.radii_km();
        for lat_deg in [-89.0_f64, -45.0, -1.0, 0.0, 23.5, 60.0, 89.9] {
            let centric = SubPoint {
                lon_rad: 0.5,
                lat_rad: lat_deg.to_radians(),
                planetocentric: true,
            };
            let graphic = centric.to_planetographic(radii);
            assert!(!graphic.planetocentric);
            assert_eq!(graphic.lon_rad, centric.lon_rad);
            assert!(
                graphic.lat_rad.abs() >= centric.lat_rad.abs(),
                "the oblate latitude is the larger one"
            );
            assert_relative_eq!(
                graphic.to_planetocentric(radii).lat_rad,
                centric.lat_rad,
                epsilon = 1e-14
            );
            // Converting a point that is already in the target form is a
            // no-op, not a second application of the scaling.
            assert_eq!(graphic.to_planetographic(radii), graphic);
        }
    }

    #[test]
    fn test_a_spherical_body_has_equal_latitudes() {
        let radii = [1000.0, 1000.0, 1000.0];
        let point = SubPoint {
            lon_rad: 0.0,
            lat_rad: 0.7,
            planetocentric: true,
        };
        assert_relative_eq!(point.to_planetographic(radii).lat_rad, 0.7, epsilon = 1e-15);
    }

    #[test]
    fn test_sub_observer_point_of_a_hand_built_geometry() {
        // An observer one AU along +x sees the point of the body-fixed frame
        // that faces +x, and the Sun in the same place if it is behind them.
        let radii = [1.0, 1.0, 1.0];
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let observer = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 0);
        let target = Position::astrometric(
            Vector3::new(-1.0, 0.0, 0.0),
            Vector3::zeros(),
            &observer,
            0,
            0.0,
        );

        let point = target.sub_observer_point(&crate::framelib::ICRS, radii, &t);
        assert_relative_eq!(point.lon_rad, 0.0, epsilon = 1e-12);
        assert_relative_eq!(point.lat_rad, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sub_solar_point_faces_the_sun() {
        // Seen from the Sun, a body's sub-solar and sub-observer points are
        // the same point.
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.utc((2007, 10, 3, 0, 0, 0.0));

        let radii = Body::Mars.radii_km();
        let frame = IauFrame::from_body(Body::Mars);

        let sun = kernel.at("sun", &t).unwrap();
        let mars = sun.observe("mars", &mut kernel, &t).unwrap();

        let sub_observer = mars.sub_observer_point(&frame, radii, &t);
        let sub_solar = mars
            .sub_solar_point(&frame, radii, &mut kernel, &t)
            .unwrap();

        // The light time of the Sun → Mars leg leaves a few arcseconds.
        assert!(
            angle_error_deg(
                sub_observer.lon_rad.to_degrees(),
                sub_solar.lon_rad.to_degrees()
            )
            .abs()
                < 0.01,
            "sub-observer {:?} and sub-solar {:?} should coincide",
            sub_observer,
            sub_solar
        );
        assert!((sub_observer.lat_rad - sub_solar.lat_rad).abs() < 1e-4);
    }

    #[test]
    fn test_a_position_without_an_observer_cannot_have_a_sub_solar_point() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let mars = kernel.at("mars", &t).unwrap();
        let frame = IauFrame::from_body(Body::Mars);
        assert!(mars
            .sub_solar_point(&frame, Body::Mars.radii_km(), &mut kernel, &t)
            .is_err());
    }

    #[test]
    fn test_sub_points_match_horizons() {
        let mut worst_lon = 0.0_f64;
        let mut worst_lat = 0.0_f64;

        for row in rows_for(499, 399).into_iter().chain(rows_for(399, 499)) {
            let (sub_observer, sub_solar) = sub_points(&row);
            let sense = LongitudeSense::for_body(row.target);

            for (label, point, lon_deg, lat_deg) in [
                (
                    "sub-observer",
                    sub_observer,
                    row.obs_sub_lon_deg,
                    row.obs_sub_lat_deg,
                ),
                (
                    "sub-solar",
                    sub_solar,
                    row.sun_sub_lon_deg,
                    row.sun_sub_lat_deg,
                ),
            ] {
                let lon_error =
                    angle_error_deg(point.longitude_in(sense).to_degrees(), lon_deg).abs();
                let lat_error = (point.lat_rad.to_degrees() - lat_deg).abs();
                assert!(
                    lon_error < 0.05,
                    "{label} longitude of {} from {} at JD {}: {} vs Horizons {} ({} deg)",
                    row.target,
                    row.center,
                    row.jd_ut,
                    point.longitude_in(sense).to_degrees(),
                    lon_deg,
                    lon_error
                );
                assert!(
                    lat_error < 0.05,
                    "{label} latitude of {} from {} at JD {}: {} vs Horizons {} ({} deg)",
                    row.target,
                    row.center,
                    row.jd_ut,
                    point.lat_rad.to_degrees(),
                    lat_deg,
                    lat_error
                );
                worst_lon = worst_lon.max(lon_error);
                worst_lat = worst_lat.max(lat_error);
            }
        }

        println!("worst longitude error {worst_lon:.4}°, worst latitude error {worst_lat:.4}°");
    }

    #[test]
    fn test_planetocentric_latitude_is_not_what_horizons_reports() {
        // A guard against reporting the planetocentric latitude by mistake:
        // for Mars at these epochs the two differ by more than the tolerance
        // of the comparison above.
        let radii = Body::Mars.radii_km();
        let row = rows_for(499, 399)[0];
        let (sub_observer, _) = sub_points(&row);
        let centric = sub_observer.to_planetocentric(radii);
        assert!(
            (centric.lat_rad.to_degrees() - row.obs_sub_lat_deg).abs() > 0.1,
            "the planetocentric and planetodetic latitudes should be distinguishable"
        );
    }
}
