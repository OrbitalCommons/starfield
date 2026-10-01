//! Spherical caps and the smallest cap enclosing a set of sky directions.
//!
//! A spherical cap is the set of directions within an angular `radius` of a
//! `centre` direction: the footprint of a circular field of view, a cone
//! search, or the envelope of a slewing boresight.

use super::inertial::{Equatorial, InertialFrame};
use crate::coordinates::cartesian::Cartesian3;
use nalgebra::Vector3;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::f64::consts::FRAC_PI_2;

/// Below this norm a cross product or vector sum is treated as zero when
/// constructing caps through two or three boundary points.
const DEGENERATE_NORM: f64 = 1e-14;

/// A spherical cap: every direction within `radius` radians of `centre`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphericalCap {
    /// Direction of the cap's axis.
    pub centre: Equatorial,
    /// Angular radius (half-angle) in radians, in `[0, pi]`.
    pub radius: f64,
}

impl SphericalCap {
    /// Cap of angular `radius` radians about `centre`.
    pub fn new(centre: Equatorial, radius: f64) -> Self {
        Self { centre, radius }
    }

    /// Full opening angle (twice the radius) in radians.
    pub fn diameter(&self) -> f64 {
        2.0 * self.radius
    }

    /// Whether `point` lies inside or on the boundary of the cap.
    pub fn contains(&self, point: &Equatorial) -> bool {
        separation(&unit(&self.centre), &unit(point)) <= self.radius
    }

    /// The smallest cap containing every direction in `points`, or `None`
    /// for an empty slice.
    ///
    /// When the points fit in a cap of radius below 90 degrees (any set
    /// lying inside an open hemisphere) the result is the exact minimum,
    /// found with Welzl's randomised incremental algorithm adapted to the
    /// sphere: expected `O(n)` time, with a fixed internal seed so the result
    /// is deterministic. Its centre is generally *not* the mean direction;
    /// for points spread along a great-circle arc it is the arc midpoint.
    ///
    /// Sets that do not fit in a hemisphere have no unique minimum under
    /// this construction; for those the cap is centred on the normalised
    /// mean direction (or the first point if the mean vanishes) and still
    /// encloses every point, but is not guaranteed minimal.
    ///
    /// In every case `radius` is the largest angular separation between
    /// the returned centre and an input point, so [`contains`](Self::contains)
    /// holds for every input.
    pub fn enclosing(points: &[Equatorial]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        let vectors: Vec<Vector3<f64>> = points.iter().map(unit).collect();
        let centre = match welzl(&vectors) {
            Some(c) => c,
            None => mean_direction(&vectors),
        };
        let centre = Equatorial::from_cartesian(Cartesian3::from_vector3(centre));
        let axis = unit(&centre);
        let radius = vectors
            .iter()
            .map(|v| separation(&axis, v))
            .fold(0.0, f64::max);
        Some(Self::new(centre, radius))
    }
}

fn unit(eq: &Equatorial) -> Vector3<f64> {
    eq.to_cartesian().to_vector3()
}

/// Angle between unit vectors, accurate at all separations.
fn separation(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    a.cross(b).norm().atan2(a.dot(b))
}

/// Working cap: unit-vector centre and radius in radians.
#[derive(Clone, Copy)]
struct Cap {
    centre: Vector3<f64>,
    radius: f64,
}

impl Cap {
    /// Slack absorbing rounding in boundary points during the incremental
    /// construction; the final radius is recomputed exactly.
    const TOLERANCE: f64 = 1e-12;

    fn contains(&self, v: &Vector3<f64>) -> bool {
        separation(&self.centre, v) <= self.radius + Self::TOLERANCE
    }

    fn point(a: &Vector3<f64>) -> Self {
        Self {
            centre: *a,
            radius: 0.0,
        }
    }

    /// Smallest cap with `a` and `b` on its boundary.
    fn two(a: &Vector3<f64>, b: &Vector3<f64>) -> Self {
        let sum = a + b;
        let norm = sum.norm();
        if norm < DEGENERATE_NORM {
            // Antipodal pair: any hemisphere bounded by a great circle
            // through both works; the caller rejects radius >= pi/2.
            let centre = a.cross(&any_perpendicular(a)).normalize();
            return Self {
                centre,
                radius: FRAC_PI_2,
            };
        }
        Self {
            centre: sum / norm,
            radius: separation(a, b) / 2.0,
        }
    }

    /// Smallest cap with `a`, `b` and `c` on its boundary.
    fn three(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Self {
        let normal = (b - a).cross(&(c - a));
        let norm = normal.norm();
        if norm < DEGENERATE_NORM {
            // Coincident or co-great-circle points: the widest pair spans.
            return [Self::two(a, b), Self::two(a, c), Self::two(b, c)]
                .into_iter()
                .fold(Self::point(a), |best, cap| {
                    if cap.radius > best.radius {
                        cap
                    } else {
                        best
                    }
                });
        }
        let mut centre = normal / norm;
        if centre.dot(a) < 0.0 {
            centre = -centre;
        }
        Self {
            centre,
            radius: separation(&centre, a),
        }
    }
}

fn any_perpendicular(v: &Vector3<f64>) -> Vector3<f64> {
    if v.x.abs() < 0.9 {
        Vector3::x()
    } else {
        Vector3::y()
    }
}

/// Exact minimum enclosing cap centre, or `None` when the minimum cap is
/// not smaller than a hemisphere.
fn welzl(vectors: &[Vector3<f64>]) -> Option<Vector3<f64>> {
    let mut pts = vectors.to_vec();
    pts.shuffle(&mut StdRng::seed_from_u64(0x5ca1ab1e));

    let mut cap = Cap::point(&pts[0]);
    for i in 1..pts.len() {
        if cap.contains(&pts[i]) {
            continue;
        }
        cap = Cap::point(&pts[i]);
        for j in 0..i {
            if cap.contains(&pts[j]) {
                continue;
            }
            cap = Cap::two(&pts[i], &pts[j]);
            for k in 0..j {
                if !cap.contains(&pts[k]) {
                    cap = Cap::three(&pts[i], &pts[j], &pts[k]);
                }
            }
        }
        if cap.radius >= FRAC_PI_2 {
            return None;
        }
    }
    if pts.iter().all(|p| cap.contains(p)) {
        Some(cap.centre)
    } else {
        None
    }
}

fn mean_direction(vectors: &[Vector3<f64>]) -> Vector3<f64> {
    let sum: Vector3<f64> = vectors.iter().sum();
    let norm = sum.norm();
    if norm < DEGENERATE_NORM * vectors.len() as f64 {
        vectors[0]
    } else {
        sum / norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framelib::random::RandomEquatorial;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    fn deg(ra: f64, dec: f64) -> Equatorial {
        Equatorial::from_degrees(ra, dec)
    }

    #[test]
    fn empty_is_none() {
        assert!(SphericalCap::enclosing(&[]).is_none());
    }

    #[test]
    fn single_point_has_zero_radius() {
        let p = deg(12.0, -34.0);
        let cap = SphericalCap::enclosing(&[p]).unwrap();
        assert_abs_diff_eq!(cap.radius, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(cap.centre.angular_distance(&p), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn arc_along_equator_is_centred_on_midpoint() {
        let pts: Vec<Equatorial> = (0..=10).map(|i| deg(10.0 + i as f64, 0.0)).collect();
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert_abs_diff_eq!(cap.centre.ra_degrees(), 15.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cap.centre.dec_degrees(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cap.diameter().to_degrees(), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn clustered_points_do_not_bias_centre() {
        // Many points at one end of an arc pull the mean direction off the
        // midpoint; the minimum cap ignores the clustering.
        let mut pts = vec![deg(0.0, 0.0); 50];
        pts.push(deg(20.0, 0.0));
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert_abs_diff_eq!(cap.centre.ra_degrees(), 10.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cap.radius.to_degrees(), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn equilateral_triangle_uses_circumcentre() {
        // Three points 120 degrees apart in RA on a small circle about the
        // north pole: the minimum cap is that small circle.
        let pts = [deg(0.0, 80.0), deg(120.0, 80.0), deg(240.0, 80.0)];
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert_abs_diff_eq!(cap.centre.dec_degrees(), 90.0, epsilon = 1e-9);
        assert_abs_diff_eq!(cap.radius.to_degrees(), 10.0, epsilon = 1e-9);
    }

    #[test]
    fn straddles_ra_wrap() {
        let pts = [deg(359.0, 1.0), deg(1.0, -1.0), deg(0.0, 0.0)];
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert_abs_diff_eq!(
            cap.centre.angular_distance(&deg(0.0, 0.0)),
            0.0,
            epsilon = 1e-9
        );
        assert!(pts.iter().all(|p| cap.contains(p)));
    }

    #[test]
    fn random_sets_are_enclosed_and_minimal() {
        let centres: Vec<Equatorial> = RandomEquatorial::with_seed(7).take(20).collect();
        let mut offsets = RandomEquatorial::with_seed(11);
        for c in centres {
            // Scatter points within ~30 degrees of c.
            let pts: Vec<Equatorial> = (0..200)
                .map(|_| {
                    let o = offsets.next().unwrap();
                    let v = unit(&c) + 0.5 * unit(&o);
                    Equatorial::from_cartesian(Cartesian3::from_vector3(v.normalize()))
                })
                .collect();
            let cap = SphericalCap::enclosing(&pts).unwrap();
            assert!(pts.iter().all(|p| cap.contains(p)));
            // At least two points lie on the boundary of a minimum cap.
            let on_boundary = pts
                .iter()
                .filter(|p| (cap.centre.angular_distance(p) - cap.radius).abs() < 1e-9)
                .count();
            assert!(on_boundary >= 2, "only {on_boundary} boundary points");
            // Nudging the centre in any direction cannot shrink the cap.
            let (e, n, r) = crate::framelib::attitude::local_triad(&cap.centre);
            for k in 0..8 {
                let a = k as f64 * PI / 4.0;
                let moved = (r + 1e-6 * (a.cos() * e + a.sin() * n)).normalize();
                let worst = pts
                    .iter()
                    .map(|p| separation(&moved, &unit(p)))
                    .fold(0.0, f64::max);
                assert!(worst >= cap.radius - 1e-12);
            }
        }
    }

    #[test]
    fn beyond_hemisphere_still_encloses() {
        let pts = [
            deg(0.0, 0.0),
            deg(120.0, 0.0),
            deg(240.0, 0.0),
            deg(0.0, -60.0),
        ];
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert!(pts.iter().all(|p| cap.contains(p)));
        assert!(cap.radius >= FRAC_PI_2);
    }

    #[test]
    fn antipodal_pair_is_enclosed() {
        let pts = [deg(10.0, 20.0), deg(190.0, -20.0)];
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert!(pts.iter().all(|p| cap.contains(p)));
    }

    #[test]
    fn contains_excludes_outside_points() {
        let cap = SphericalCap::new(deg(0.0, 0.0), 1.0_f64.to_radians());
        assert!(cap.contains(&deg(0.5, 0.0)));
        assert!(!cap.contains(&deg(2.0, 0.0)));
        assert!(!cap.contains(&deg(180.0, 0.0)));
    }
}
