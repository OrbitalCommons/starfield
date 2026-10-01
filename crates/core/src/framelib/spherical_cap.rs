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

/// Below this norm a vector sum is treated as zero (an antipodal pair, or a
/// set whose mean direction vanishes).
const DEGENERATE_NORM: f64 = 1e-14;

/// Three boundary points are treated as degenerate (coincident, or on one
/// great circle within rounding) when the plane normal through them is
/// shorter than this fraction of the product of the two chord lengths, i.e.
/// when the triangle's angle at the first point is below about `1e-12` rad.
/// Relative, so a valid triangle of any size is never misclassified.
const DEGENERATE_SINE: f64 = 1e-12;

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
    /// lying inside an open hemisphere) the result is the minimum cap, found
    /// with Welzl's randomised incremental algorithm adapted to the sphere:
    /// expected `O(n)` time, with a fixed internal seed so the result is
    /// deterministic. "Minimum" holds up to floating-point rounding: the
    /// incremental steps accept points up to `1e-12` rad outside a
    /// candidate cap, and tests check the radius against brute force to a
    /// relative `1e-9` (absolute `1e-15` rad) for caps from `1e-8` rad to
    /// tens of degrees. Its centre is generally *not* the mean direction;
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

    /// Smallest cap with `a`, `b` and `c` on its boundary: their circumcap,
    /// or the widest pair's cap when that already contains the third point
    /// or the triangle is degenerate.
    ///
    /// The circumcentre is the normal of the plane through the three points.
    /// Computing it from raw unit-vector chords is ill-conditioned for small
    /// triangles: the plane's tilt depends on the chords' radial parts (of
    /// order the squared separation), which unit vectors carry only to about
    /// `1e-16` absolute. So the chords are built in a tangent frame at `a`,
    /// with each radial part rebuilt analytically from the tangential one;
    /// the centre is then accurate to the inputs' own `~1e-16` rad
    /// positional precision at every scale.
    fn three(a: &Vector3<f64>, b: &Vector3<f64>, c: &Vector3<f64>) -> Self {
        let widest = [Self::two(a, b), Self::two(a, c), Self::two(b, c)]
            .into_iter()
            .fold(Self::point(a), |best, cap| {
                if cap.radius > best.radius {
                    cap
                } else {
                    best
                }
            });
        if [a, b, c].iter().all(|p| widest.contains(p)) {
            return widest;
        }

        let e1 = a.cross(&any_perpendicular(a)).normalize();
        let e2 = a.cross(&e1);
        let chord = |p: &Vector3<f64>| {
            let (x, y) = (e1.dot(p), e2.dot(p));
            let t2 = x * x + y * y;
            // 1 - cos(theta) from sin^2(theta) without cancellation;
            // sign follows a.p for chords past 90 degrees.
            let cos = a.dot(p);
            let drop = if cos >= 0.0 {
                t2 / (1.0 + (1.0 - t2).max(0.0).sqrt())
            } else {
                1.0 - cos
            };
            Vector3::new(x, y, -drop)
        };
        let (db, dc) = (chord(b), chord(c));
        let normal = db.cross(&dc);
        let norm = normal.norm();
        if norm.is_nan() || norm <= DEGENERATE_SINE * db.norm() * dc.norm() {
            return widest;
        }
        let mut local = normal / norm;
        if local.z < 0.0 {
            local = -local;
        }
        let centre = (e1 * local.x + e2 * local.y + a * local.z).normalize();
        Self {
            centre,
            radius: separation(&centre, a)
                .max(separation(&centre, b))
                .max(separation(&centre, c)),
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

    /// Radius of the minimum cap by brute force over every pair and triple.
    fn brute_force_radius(pts: &[Equatorial]) -> f64 {
        let v: Vec<Vector3<f64>> = pts.iter().map(unit).collect();
        let encloses = |cap: &Cap| {
            v.iter()
                .all(|p| separation(&cap.centre, p) <= cap.radius * (1.0 + 1e-9) + 1e-15)
        };
        let mut best = f64::INFINITY;
        for i in 0..v.len() {
            for j in i + 1..v.len() {
                let cap = Cap::two(&v[i], &v[j]);
                if encloses(&cap) {
                    best = best.min(cap.radius);
                }
                for k in j + 1..v.len() {
                    let cap = Cap::three(&v[i], &v[j], &v[k]);
                    if encloses(&cap) {
                        best = best.min(cap.radius);
                    }
                }
            }
        }
        best
    }

    /// Tangent-plane offset `(x, y)` (east, north) about `origin`, mapped
    /// back to the sphere through its gnomonic inverse.
    fn offset(origin: &Equatorial, x: f64, y: f64) -> Equatorial {
        let (e, n, r) = crate::framelib::attitude::local_triad(origin);
        Equatorial::from_cartesian(Cartesian3::from_vector3((r + x * e + y * n).normalize()))
    }

    /// The scalene acute triangle (0,0), (4,0), (1,3) in units of `scale`
    /// radians, placed about `origin`.
    fn small_triangle(origin: Equatorial, scale: f64) -> [Equatorial; 3] {
        [(0.0, 0.0), (4.0, 0.0), (1.0, 3.0)].map(|(x, y)| offset(&origin, x * scale, y * scale))
    }

    #[test]
    fn small_acute_triangle_uses_its_circumcap() {
        // Planar circumradius of (0,0), (4,0), (1,3): R = abc / (4K) with
        // sides 4, sqrt(10), sqrt(18) and area 6.
        let planar_r = 4.0 * 10.0_f64.sqrt() * 18.0_f64.sqrt() / 24.0;
        for origin in [
            Equatorial::new(0.0, 0.0),
            deg(123.4, -56.7),
            deg(10.0, 89.0),
        ] {
            for scale in [1e-8, 1e-6, 1e-3] {
                let pts = small_triangle(origin, scale);
                let cap = SphericalCap::enclosing(&pts).unwrap();
                assert!(pts.iter().all(|p| cap.contains(p)));
                // Every vertex lies on the boundary of the circumcap.
                // (`angular_distance` uses acos, too coarse at this scale.)
                for p in &pts {
                    let d = separation(&unit(&cap.centre), &unit(p));
                    assert!(
                        (d - cap.radius).abs() <= 1e-9 * cap.radius + 1e-15,
                        "scale {scale}: vertex at {d}, radius {}",
                        cap.radius
                    );
                }
                // Gnomonic distortion is second order in the vertices' offsets
                // (up to 4 * scale).
                let expected = planar_r * scale;
                assert!(
                    (cap.radius - expected).abs()
                        <= expected * (1e-9 + 20.0 * scale * scale) + 1e-15,
                    "scale {scale}: radius {} vs planar {expected}",
                    cap.radius
                );
            }
        }
    }

    #[test]
    fn reviewer_triangle_is_not_degenerate() {
        let pts = [
            Equatorial::new(0.0, 0.0),
            Equatorial::new(4e-8, 0.0),
            Equatorial::new(1e-8, 3e-8),
        ];
        let cap = SphericalCap::enclosing(&pts).unwrap();
        assert!(pts.iter().all(|p| cap.contains(p)));
        let planar_r = 4.0 * 10.0_f64.sqrt() * 18.0_f64.sqrt() / 24.0 * 1e-8;
        assert!(
            (cap.radius - planar_r).abs() <= 1e-9 * planar_r,
            "{}",
            cap.radius
        );
    }

    #[test]
    fn small_random_sets_match_brute_force_at_every_scale() {
        let mut rng = StdRng::seed_from_u64(42);
        let origins: Vec<Equatorial> = RandomEquatorial::with_seed(3).take(6).collect();
        for origin in origins {
            for scale in [1e-8, 1e-5, 1e-2, 0.3] {
                for n in 3..=7 {
                    let pts: Vec<Equatorial> = (0..n)
                        .map(|_| {
                            use rand::Rng;
                            let (x, y): (f64, f64) =
                                (rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0));
                            offset(&origin, x * scale, y * scale)
                        })
                        .collect();
                    let cap = SphericalCap::enclosing(&pts).unwrap();
                    assert!(pts.iter().all(|p| cap.contains(p)));
                    let brute = brute_force_radius(&pts);
                    assert!(
                        cap.radius <= brute * (1.0 + 1e-9) + 1e-15,
                        "scale {scale}, n {n}: welzl {} vs brute {brute}",
                        cap.radius
                    );
                }
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
