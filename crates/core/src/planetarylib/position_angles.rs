//! Position angles of a body's north pole and of its bright limb.
//!
//! To draw a resolved body on a detector you need three things: how large the
//! disc is, which way its rotation axis leans, and which way the illuminated
//! crescent faces. The last two are position angles — angles measured on the
//! sky **east of celestial north**, counter-clockwise as the sky is drawn with
//! north up and east to the left, which is the convention of JPL Horizons'
//! `NP.ang` and `SN.ang` columns.
//!
//! * [`Position::north_pole_position_angle`] projects the body's rotation
//!   axis, taken from its body-fixed frame, onto the plane of the sky.
//! * [`Position::bright_limb_position_angle`] projects the direction from the
//!   body to the Sun. That is the position angle of the sub-solar point on
//!   the disc, and so of the midpoint of the illuminated limb; the terminator
//!   runs across the disc at right angles to it, and the unlit crescent lies
//!   180° away.
//!
//! Both angles are measured in the **ICRF** sky frame of the observer, not in
//! the equator and equinox of date. The two differ by the convergence of the
//! meridians between the two frames, which is the component of the
//! precession-nutation rotation along the line of sight: under 0.02° for a
//! body near the ecliptic within a few decades of J2000, since the precession
//! axis is nearly perpendicular to such a line of sight.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::kernel::SpiceKernel;
//! use starfield_core::jplephem_ext::SpiceKernelExt;
//! use starfield_core::planetarylib::IauFrame;
//! use starfield_core::planetlib::Body;
//! use starfield_core::time::Timescale;
//!
//! let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
//! let t = Timescale::default().utc((2007, 10, 3, 0, 0, 0.0));
//!
//! let earth = kernel.at("earth", &t).unwrap();
//! let mars = earth.observe("mars", &mut kernel, &t).unwrap();
//! let frame = IauFrame::from_body(Body::Mars);
//!
//! println!(
//!     "north pole at {:.2}°, bright limb at {:.2}° east of north",
//!     mars.north_pole_position_angle(&frame, &t).to_degrees(),
//!     mars.bright_limb_position_angle(&mut kernel, &t).unwrap().to_degrees(),
//! );
//! ```

use nalgebra::Vector3;

use crate::framelib::Frame;
use crate::jplephem::kernel::SpiceKernel;
use crate::planetarylib::subpoint::sun_seen_from_target;
use crate::positions::Position;
use crate::time::Time;
use crate::Result;

impl Position {
    /// The position angle of the body's north pole, radians east of celestial
    /// north in the observer's ICRF sky frame, in `[0, 2π)`.
    ///
    /// The pole direction comes from `frame`, whose third row is the
    /// body-fixed z axis expressed in the ICRF, evaluated at the light-time
    /// corrected epoch `t − light_time`. Horizons publishes the same angle as
    /// `NP.ang` (quantity 17) for an Earth-bound observer.
    ///
    /// The angle is undefined, and returns zero, if the body sits within a
    /// few microarcseconds of a celestial pole, where the sky's north
    /// direction is not defined.
    pub fn north_pole_position_angle(&self, frame: &dyn Frame, t: &Time) -> f64 {
        let epoch = t.shift_days(-self.light_time);
        let pole = frame.rotation_at(&epoch).row(2).transpose();
        position_angle(&self.position, &pole)
    }

    /// The position angle of the bright limb, radians east of celestial north
    /// in the observer's ICRF sky frame, in `[0, 2π)`.
    ///
    /// This is the direction from the centre of the disc toward the Sun as
    /// projected on the sky: the position angle of the sub-solar point, of
    /// the midpoint of the illuminated limb, and of the normal to the
    /// terminator. Horizons publishes the same angle as `SN.ang` (quantity
    /// 16) for an Earth-bound observer.
    ///
    /// Both legs of the light path are corrected, as in
    /// [`sub_solar_point`](Position::sub_solar_point).
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::MissingObserver`](crate::StarfieldError::MissingObserver)
    /// if `self` does not carry the observer's barycentric position, and
    /// [`StarfieldError::EphemerisError`](crate::StarfieldError::EphemerisError)
    /// if the kernel cannot place the Sun.
    pub fn bright_limb_position_angle(&self, kernel: &mut SpiceKernel, t: &Time) -> Result<f64> {
        let body_to_sun = sun_seen_from_target(self, kernel, t)?;
        Ok(position_angle(&self.position, &body_to_sun))
    }
}

/// The position angle of `direction` as seen along `line_of_sight`, radians
/// east of celestial north in `[0, 2π)`.
///
/// The sky frame at the end of the line of sight has north
/// `n̂ ∝ ẑ − (ẑ·û)û` and east `ê ∝ ẑ × û`, so the angle is
/// `atan2(v·ê, v·n̂)`. Only the component of `direction` in the plane of the
/// sky matters; its length does not.
fn position_angle(line_of_sight: &Vector3<f64>, direction: &Vector3<f64>) -> f64 {
    let u = line_of_sight.normalize();
    let z = Vector3::z();

    let east = z.cross(&u);
    let north = z - u * z.dot(&u);

    let angle = direction.dot(&east).atan2(direction.dot(&north));
    if angle < 0.0 {
        angle + std::f64::consts::TAU
    } else {
        angle
    }
}

#[cfg(test)]
mod tests {
    use super::super::subpoint::horizons_fixture::{rows_for, DiskRow};
    use super::*;
    use crate::framelib::ItrsFrame;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::planetarylib::IauFrame;
    use crate::planetlib::Body;
    use crate::time::Timescale;
    use approx::assert_relative_eq;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
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

    /// How far the sub-solar point stands from the centre of the disc, in
    /// arcseconds: the body's angular radius times the sine of the phase
    /// angle. It is what Horizons prints as `SN.dist`, and it says how well
    /// the bright limb's position angle is determined.
    fn sub_solar_offset_arcsec(target: &Position, kernel: &mut SpiceKernel, t: &Time) -> f64 {
        let radii = crate::planetarylib::body_constants(target.target)
            .unwrap()
            .radii;
        let angular_radius = radii[0] / (target.distance() * crate::constants::AU_KM);
        let phase = target.phase_angle(kernel, t).unwrap();
        (angular_radius * phase.sin()).to_degrees() * 3600.0
    }

    /// Our two position angles in degrees, and the sub-solar point's distance
    /// from the disc centre in arcseconds, for one fixture row.
    fn position_angles(row: &DiskRow) -> (f64, f64, f64) {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = row.time(&ts);

        let frame: Box<dyn Frame> = if row.target == 399 {
            Box::new(ItrsFrame)
        } else {
            Box::new(IauFrame::from_naif_id(row.target).unwrap())
        };

        let observer = kernel.at(&row.center.to_string(), &t).unwrap();
        let target = observer
            .observe(&row.target.to_string(), &mut kernel, &t)
            .unwrap();

        (
            target
                .north_pole_position_angle(frame.as_ref(), &t)
                .to_degrees(),
            target
                .bright_limb_position_angle(&mut kernel, &t)
                .unwrap()
                .to_degrees(),
            sub_solar_offset_arcsec(&target, &mut kernel, &t),
        )
    }

    #[test]
    fn test_position_angle_of_the_cardinal_directions() {
        // Looking along +x, the sky has north toward +z and east toward +y.
        let u = Vector3::new(1.0, 0.0, 0.0);
        assert_relative_eq!(position_angle(&u, &Vector3::z()), 0.0, epsilon = 1e-15);
        assert_relative_eq!(
            position_angle(&u, &Vector3::y()),
            FRAC_PI_2,
            epsilon = 1e-15
        );
        assert_relative_eq!(position_angle(&u, &(-Vector3::z())), PI, epsilon = 1e-15);
        assert_relative_eq!(
            position_angle(&u, &(-Vector3::y())),
            3.0 * FRAC_PI_2,
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_position_angle_ignores_the_radial_component() {
        let u = Vector3::new(0.3, -0.7, 0.2);
        let v = Vector3::new(0.1, 0.5, 0.9);
        let along = position_angle(&u, &v);
        assert_relative_eq!(position_angle(&u, &(v + u * 5.0)), along, epsilon = 1e-12);
        assert_relative_eq!(position_angle(&u, &(v * 3.0)), along, epsilon = 1e-12);
    }

    #[test]
    fn test_the_earths_pole_points_at_celestial_north() {
        // The Earth's rotation axis is the celestial pole itself, so from
        // anywhere its position angle is zero up to the precession and
        // nutation that separate the pole of date from the ICRF pole.
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.utc((2007, 10, 3, 0, 0, 0.0));

        let mars = kernel.at("mars", &t).unwrap();
        let earth = mars.observe("earth", &mut kernel, &t).unwrap();

        let pa = earth.north_pole_position_angle(&ItrsFrame, &t).to_degrees();
        assert!(
            angle_error_deg(pa, 0.0).abs() < 0.2,
            "the Earth's pole should sit at position angle 0, got {pa}"
        );
    }

    #[test]
    fn test_a_full_disc_has_its_bright_limb_opposite_the_sun() {
        // Seen from the Sun a body is fully lit, and the direction of the Sun
        // from the body is straight back along the line of sight, leaving the
        // bright limb angle to be decided by the sky-plane component alone.
        // Seen from the Earth at opposition the same holds to within a degree.
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.utc((2020, 10, 13, 23, 0, 0.0));

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();

        let limb = mars.bright_limb_position_angle(&mut kernel, &t).unwrap();
        assert!((0.0..TAU).contains(&limb));

        // At opposition the phase angle is a couple of degrees, so the Sun is
        // barely off the line of sight and the disc is all but full.
        assert!(mars.illuminated_fraction(&mut kernel, &t).unwrap() > 0.999);
    }

    #[test]
    fn test_a_position_without_an_observer_has_no_bright_limb() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let mars = kernel.at("mars", &t).unwrap();
        assert!(mars.bright_limb_position_angle(&mut kernel, &t).is_err());
    }

    #[test]
    fn test_position_angles_match_horizons_for_an_earth_observer() {
        // Horizons measures both angles from the north of the observer's own
        // reference frame of date, which for a geocentric observer is the
        // celestial pole of date. Against our ICRF angles that leaves only
        // the convergence of the meridians, well under the tolerance for a
        // target near the ecliptic.
        let mut worst_pole = 0.0_f64;
        let mut worst_limb = 0.0_f64;
        let mut worst_displacement = 0.0_f64;

        for row in rows_for(499, 399) {
            let (pole, limb, offset_arcsec) = position_angles(&row);
            let pole_error = angle_error_deg(pole, row.np_ang_deg).abs();
            let limb_error = angle_error_deg(limb, row.sn_ang_deg).abs();

            assert!(
                pole_error < 0.1,
                "north pole angle at JD {}: {pole} vs Horizons {} ({pole_error} deg)",
                row.jd_ut,
                row.np_ang_deg
            );

            // The bright limb's angle is ill-conditioned near full phase,
            // where the sub-solar point sits almost on top of the centre of
            // the disc: at the 2020 row Mars is a degree from opposition and
            // the point stands 0.4" from the centre of an 11" disc, so a
            // milliarcsecond of ephemeris disagreement — DE421 here against
            // Horizons' DE441 and mar099 — is a quarter of a degree of angle.
            // Hold the flat bound of the issue where the point stands clear
            // of the centre, and everywhere bound what the angle means for
            // where the point lands on the disc.
            let displacement_mas =
                2.0 * (limb_error.to_radians() / 2.0).sin() * offset_arcsec * 1000.0;
            assert!(
                displacement_mas < 5.0,
                "bright limb at JD {} puts the sub-solar point {displacement_mas} mas from \
                 where Horizons does",
                row.jd_ut
            );
            if offset_arcsec > 1.0 {
                assert!(
                    limb_error < 0.1,
                    "bright limb angle at JD {}: {limb} vs Horizons {} ({limb_error} deg)",
                    row.jd_ut,
                    row.sn_ang_deg
                );
            }

            worst_pole = worst_pole.max(pole_error);
            worst_limb = worst_limb.max(limb_error);
            worst_displacement = worst_displacement.max(displacement_mas);
        }

        println!(
            "worst north pole error {worst_pole:.4}°, worst bright limb error {worst_limb:.4}°, \
             worst sub-solar displacement {worst_displacement:.3} mas"
        );
    }

    #[test]
    fn test_position_angle_difference_matches_horizons_for_a_mars_observer() {
        // For an observer at Mars, Horizons measures both angles from Mars's
        // own pole rather than from celestial north — its Earth-from-Mars
        // NP.ang column is, to the digit, its Mars-from-Earth one, which is
        // what the mirror symmetry of the two views demands. Their
        // difference, however, is a property of the sky plane alone and is
        // free of any reference direction, so it is what we compare.
        let mut worst = 0.0_f64;

        for row in rows_for(399, 499) {
            let (pole, limb, _) = position_angles(&row);
            let error = angle_error_deg(limb - pole, row.sn_ang_deg - row.np_ang_deg).abs();
            assert!(
                error < 0.1,
                "bright limb minus north pole at JD {}: {} vs Horizons {} ({error} deg)",
                row.jd_ut,
                limb - pole,
                row.sn_ang_deg - row.np_ang_deg
            );
            worst = worst.max(error);
        }

        println!("worst bright limb minus north pole error {worst:.4}°");
    }

    #[test]
    fn test_the_two_views_share_one_pole_to_pole_angle() {
        // The claim the test above rests on: the position angle of Mars's
        // pole seen from the Earth, measured from the Earth's pole, equals
        // the position angle of the Earth's pole seen from Mars, measured
        // from Mars's pole. Reversing the line of sight mirrors the sky and
        // swapping the two directions negates the angle, and the two
        // reversals cancel.
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.utc((2007, 10, 3, 0, 0, 0.0));

        let mars_frame = IauFrame::from_body(Body::Mars);

        let earth = kernel.at("earth", &t).unwrap();
        let mars_seen = earth.observe("mars", &mut kernel, &t).unwrap();
        let from_earth = mars_seen.north_pole_position_angle(&mars_frame, &t);

        let mars = kernel.at("mars", &t).unwrap();
        let earth_seen = mars.observe("earth", &mut kernel, &t).unwrap();
        let epoch = t.shift_days(-earth_seen.light_time);
        let mars_pole = mars_frame.rotation_at(&epoch).row(2).transpose();
        let earth_pole = ItrsFrame.rotation_at(&epoch).row(2).transpose();
        // The angle from Mars's pole to the Earth's pole, in the sky of a
        // Mars-bound observer, is the same mirrored quantity.
        let from_mars = position_angle(&earth_seen.position, &earth_pole)
            - position_angle(&earth_seen.position, &mars_pole);

        assert!(
            angle_error_deg(from_earth.to_degrees(), from_mars.to_degrees()).abs() < 0.05,
            "{} from the Earth against {} from Mars",
            from_earth.to_degrees(),
            from_mars.to_degrees()
        );
    }
}
