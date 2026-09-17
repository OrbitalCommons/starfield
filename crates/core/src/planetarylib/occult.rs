//! Is one body hidden behind another?
//!
//! [`occultation`] answers that question for two bodies seen from one place —
//! from a Mars orbiter the Earth is regularly hidden behind Mars, and the Moon
//! behind the Earth — and reports whether the target is fully hidden, partly
//! hidden, or clear of the occulting body.
//!
//! [`eclipselib`](crate::eclipselib) covers the Sun, the Earth and the Moon
//! and works with shadow cones and penumbrae. This is the other half of the
//! same geometry: two discs on the sky and the question of which is in front.
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::SpiceKernel;
//! use starfield_core::jplephem_ext::SpiceKernelExt;
//! use starfield_core::planetarylib::occult::{occultation, Occultation};
//! use starfield_core::planetlib::Body;
//! use starfield_core::time::Timescale;
//!
//! let mut kernel = SpiceKernel::open("test_data/de421.bsp").unwrap();
//! let t = Timescale::default().utc((2007, 10, 3, 8, 30, 0.0));
//!
//! // Both targets are observed from the same place at the same moment.
//! let mars = kernel.at("mars", &t).unwrap();
//! let earth = mars.observe("earth", &mut kernel, &t).unwrap();
//! let moon = mars.observe("moon", &mut kernel, &t).unwrap();
//!
//! let state = occultation(
//!     &moon,
//!     &earth,
//!     Body::Moon.radii_km()[0],
//!     Body::Earth.radii_km()[0],
//! );
//! assert_eq!(state, Occultation::None);
//! ```

use crate::positions::Position;

/// How much of a target one body hides from an observer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occultation {
    /// The whole target is visible.
    None,
    /// Part of the target is hidden.
    Partial,
    /// The whole target is hidden.
    Full,
}

impl std::fmt::Display for Occultation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Occultation::None => write!(f, "None"),
            Occultation::Partial => write!(f, "Partial"),
            Occultation::Full => write!(f, "Full"),
        }
    }
}

/// Whether `occulter` hides `target` from the observer both are seen from.
///
/// Both positions must be of the same observer at the same moment — the
/// astrometric or apparent positions that
/// [`Position::observe`](crate::positions::Position::observe) and
/// [`Position::apparent`](crate::positions::Position::apparent) return, or any
/// other pair of vectors that run from one observer to the two bodies. Nothing
/// checks that; a pair from two different observers gives a meaningless
/// answer.
///
/// Both bodies are treated as spheres of the given radii, which is what the
/// question is worth: the flattening of an oblate planet moves a limb by a
/// fraction of a percent of its radius, while an occultation is decided by the
/// difference between two angles that are usually far further apart than that.
/// Use [`Body::radii_km`](crate::planetlib::Body::radii_km) and take the
/// equatorial radius for the conservative answer.
///
/// The test is the obvious one on three angles — the separation `s` of the two
/// bodies on the sky and their angular radii `rₜ` and `rₒ`:
///
/// ```text
/// s ≥ rₜ + rₒ   →  None
/// s ≤ rₒ − rₜ   →  Full
/// otherwise     →  Partial
/// ```
///
/// with one further condition: the occulter must be the nearer of the two, or
/// the result is [`Occultation::None`] however close the two discs are, since
/// then it is the target that passes in front.
///
/// An annular occultation — a nearer, smaller disc entirely inside a farther,
/// larger one — is reported as [`Occultation::Partial`], which is what the
/// middle case of the test above gives when `rₒ < rₜ`. Part of the target is
/// hidden and part is not, so `Partial` is the truth; a caller who cares about
/// the distinction can compare the two angular radii itself.
///
/// # Deviation from the proposed signature
///
/// Issue #170 proposes a leading `observer: &Position` argument. It carries no
/// information this function uses: the positions of the target and of the
/// occulter are already relative to the observer, and the epoch that would
/// come with it is already baked into them. It is left out rather than
/// accepted and ignored.
///
/// # Example
///
/// ```
/// use nalgebra::Vector3;
/// use starfield_core::planetarylib::occult::{occultation, Occultation};
/// use starfield_core::positions::Position;
///
/// // A body one AU away, right behind a body 0.5 AU away that is large
/// // enough to cover it.
/// let target = Position::barycentric(Vector3::new(1.0, 0.0, 0.0), Vector3::zeros(), 0);
/// let occulter = Position::barycentric(Vector3::new(0.5, 0.0, 0.0), Vector3::zeros(), 0);
/// assert_eq!(occultation(&target, &occulter, 6378.0, 6378.0), Occultation::Full);
///
/// // Swap them and the near body is the one in front of the far one.
/// assert_eq!(occultation(&occulter, &target, 6378.0, 6378.0), Occultation::None);
/// ```
pub fn occultation(
    target: &Position,
    occulter: &Position,
    target_radius_km: f64,
    occulter_radius_km: f64,
) -> Occultation {
    let target_distance = target.distance();
    let occulter_distance = occulter.distance();

    if occulter_distance >= target_distance {
        return Occultation::None;
    }

    let target_radius = target.angular_semi_diameter([target_radius_km; 3]);
    let occulter_radius = occulter.angular_semi_diameter([occulter_radius_km; 3]);
    let separation = target.separation_from(occulter);

    if separation >= target_radius + occulter_radius {
        Occultation::None
    } else if separation <= occulter_radius - target_radius {
        Occultation::Full
    } else {
        Occultation::Partial
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{AU_KM, DAY_S};
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::planetlib::Body;
    use crate::positions::PositionKind;
    use crate::searchlib::{find_discrete, DEFAULT_NUM, EPSILON_DISCRETE};
    use crate::time::{Time, Timescale};
    use nalgebra::Vector3;

    /// A position relative to the observer, which is all `occultation` reads.
    fn seen_from_here(position: Vector3<f64>) -> Position {
        Position {
            position,
            velocity: Vector3::zeros(),
            kind: PositionKind::Astrometric,
            center: 0,
            target: 0,
            light_time: 0.0,
            observer_barycentric: None,
        }
    }

    /// A body at `distance_au` whose direction is `angle` radians off the x
    /// axis, in the xy plane.
    fn at_angle(distance_au: f64, angle: f64) -> Position {
        seen_from_here(Vector3::new(angle.cos(), angle.sin(), 0.0) * distance_au)
    }

    #[test]
    fn test_a_target_clear_of_the_occulter_is_not_occulted() {
        // Two degrees apart, discs of a few arcminutes each.
        let target = at_angle(1.0, 0.0);
        let occulter = at_angle(0.5, 2f64.to_radians());
        assert_eq!(
            occultation(&target, &occulter, 6378.0, 3396.0),
            Occultation::None
        );
    }

    #[test]
    fn test_a_grazing_target_is_partly_occulted() {
        let target = at_angle(1.0, 0.0);
        let target_radius = target.angular_semi_diameter([6378.0; 3]);

        // Put the occulter's centre one target radius outside its own limb, so
        // the two discs overlap by half the target.
        let occulter_distance = 0.5;
        let occulter_radius = (3396.0 / (occulter_distance * AU_KM)).asin();
        let occulter = at_angle(occulter_distance, occulter_radius);
        assert!(target_radius < occulter_radius);
        assert_eq!(
            occultation(&target, &occulter, 6378.0, 3396.0),
            Occultation::Partial
        );
    }

    #[test]
    fn test_a_target_behind_the_occulter_is_fully_occulted() {
        let target = at_angle(1.0, 0.0);
        let target_radius = target.angular_semi_diameter([6378.0; 3]);
        let occulter_distance = 0.5;
        let occulter_radius = (3396.0 / (occulter_distance * AU_KM)).asin();
        // Inside the occulting disc by more than the target's own radius, but
        // not at its centre.
        let occulter = at_angle(occulter_distance, 0.5 * (occulter_radius - target_radius));
        assert_eq!(
            occultation(&target, &occulter, 6378.0, 3396.0),
            Occultation::Full
        );
    }

    #[test]
    fn test_an_occulter_behind_the_target_occults_nothing() {
        // Exactly the same line of sight, but the "occulter" is the far body.
        let target = at_angle(0.5, 0.0);
        let occulter = at_angle(1.0, 0.0);
        assert_eq!(
            occultation(&target, &occulter, 3396.0, 6378.0),
            Occultation::None
        );
        // Equal distances are not an occultation either.
        assert_eq!(
            occultation(&target, &at_angle(0.5, 0.0), 3396.0, 6378.0),
            Occultation::None
        );
    }

    #[test]
    fn test_an_annular_occultation_is_partial() {
        // A near, small disc entirely inside a far, large one.
        let target = at_angle(1.0, 0.0);
        let occulter = at_angle(0.5, 0.0);
        let target_radius = target.angular_semi_diameter([70000.0; 3]);
        let occulter_radius = occulter.angular_semi_diameter([1737.4; 3]);
        assert!(occulter_radius < target_radius);
        assert_eq!(
            occultation(&target, &occulter, 70000.0, 1737.4),
            Occultation::Partial
        );
    }

    /// Gravitational parameter of Mars, km³/s², from DE440.
    const GM_MARS: f64 = 42_828.375_214;

    /// Radius of the circular orbit the observer flies, in km.
    const ORBIT_RADIUS_KM: f64 = 17_000.0;

    /// One observer on a circular Mars orbit, and the state of the Earth
    /// behind Mars as seen from it.
    struct MarsOrbiter {
        kernel: SpiceKernel,
        /// Toward the Earth at the epoch: the orbit plane contains it, so the
        /// observer is bound to pass behind Mars once per revolution.
        toward_earth: Vector3<f64>,
        /// The second axis of the orbit plane.
        across: Vector3<f64>,
        /// TT Julian date at which the observer is between Mars and the Earth.
        epoch_tt: f64,
        /// Orbital period in days.
        period_days: f64,
        ts: Timescale,
    }

    impl MarsOrbiter {
        fn new() -> Self {
            let mut kernel = SpiceKernel::open("test_data/de421.bsp").expect("test_data/de421.bsp");
            let ts = Timescale::default();
            let t = ts.utc((2007, 10, 3, 8, 30, 0.0));

            let mars = kernel.at("mars", &t).unwrap();
            let earth = mars.observe("earth", &mut kernel, &t).unwrap();
            let toward_earth = earth.position.normalize();
            let across = toward_earth.cross(&Vector3::z()).normalize();

            let seconds = std::f64::consts::TAU * (ORBIT_RADIUS_KM.powi(3) / GM_MARS).sqrt();

            MarsOrbiter {
                kernel,
                toward_earth,
                across,
                epoch_tt: t.tt(),
                period_days: seconds / DAY_S,
                ts,
            }
        }

        /// The barycentric state of the observer at `t`.
        ///
        /// Mars from the ephemeris plus a circular offset that starts on the
        /// Earth-facing side and swings behind Mars half a revolution later.
        fn observer(&mut self, t: &Time) -> Position {
            let phase = std::f64::consts::TAU * (t.tt() - self.epoch_tt) / self.period_days;
            let radius_au = ORBIT_RADIUS_KM / AU_KM;
            let speed = std::f64::consts::TAU * radius_au / self.period_days;
            let (sin, cos) = phase.sin_cos();

            let mars = self.kernel.at("mars", t).unwrap();
            Position::barycentric(
                mars.position + radius_au * (cos * self.toward_earth + sin * self.across),
                mars.velocity + speed * (-sin * self.toward_earth + cos * self.across),
                -1,
            )
        }

        /// The state of the Earth's occultation by Mars at `t`.
        fn state(&mut self, t: &Time) -> Occultation {
            let observer = self.observer(t);
            let earth = observer.observe("earth", &mut self.kernel, t).unwrap();
            let mars = observer.observe("mars", &mut self.kernel, t).unwrap();
            occultation(
                &earth,
                &mars,
                Body::Earth.radii_km()[0],
                Body::Mars.radii_km()[0],
            )
        }
    }

    #[test]
    fn test_mars_occults_the_earth_once_per_orbit() {
        let mut orbiter = MarsOrbiter::new();
        let start = orbiter.epoch_tt;
        let end = start + orbiter.period_days;

        // At the epoch the observer is between Mars and the Earth.
        let t0 = orbiter.ts.tt_jd(start, None);
        assert_eq!(orbiter.state(&t0), Occultation::None);

        let mut sample = |jd: &[f64]| -> Vec<i64> {
            jd.iter()
                .map(|&jd| {
                    let t = orbiter.ts.tt_jd(jd, None);
                    match orbiter.state(&t) {
                        Occultation::None => 0,
                        Occultation::Partial => 1,
                        Occultation::Full => 2,
                    }
                })
                .collect()
        };

        let events = find_discrete(
            start,
            end,
            &mut sample,
            0.005,
            EPSILON_DISCRETE,
            DEFAULT_NUM,
        );

        let values: Vec<i64> = events.iter().map(|&(_, v)| v).collect();
        assert_eq!(
            values,
            vec![1, 2, 1, 0],
            "one ingress and one egress per orbit, got {:?}",
            events
        );

        // The total occultation runs from the second event to the third.
        let full_seconds = (events[2].0 - events[1].0) * DAY_S;
        // The observer moves at a constant angular rate about Mars, so the
        // Earth is wholly hidden while the angle from the anti-Earth direction
        // is less than the difference of the two angular radii.
        let mars_radius = (Body::Mars.radii_km()[0] / ORBIT_RADIUS_KM).asin();
        let earth_radius = 18.50 / 2.0 / 206_264.806_247_096_36;
        let expected = 2.0 * (mars_radius - earth_radius) / std::f64::consts::TAU
            * orbiter.period_days
            * DAY_S;
        assert!(
            (full_seconds - expected).abs() < 0.05 * expected,
            "total occultation lasted {full_seconds} s, expected about {expected} s"
        );

        // Halfway between ingress and egress the Earth is behind Mars.
        let middle = orbiter.ts.tt_jd(0.5 * (events[1].0 + events[2].0), None);
        assert_eq!(orbiter.state(&middle), Occultation::Full);
    }
}
