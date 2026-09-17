//! Spherical cap ("cone") helper for cone-search filtering.
//!
//! A `Cone` is a circular patch of sky defined by a centre direction (RA/Dec)
//! and an angular radius. Membership tests use the unit-vector dot product
//! against `cos(radius)` — exact, branchless, and matches the convention used
//! by the rest of the crate (`GaiaCore::unit_vector`).

use nalgebra::Vector3;

/// A circular spherical cap centred at `(ra, dec)` with angular `radius`.
///
/// All angles are stored internally in radians; precomputed `centre` (unit
/// vector) and `cos_radius` make membership tests cheap.
#[derive(Debug, Clone, Copy)]
pub struct Cone {
    pub ra_rad: f64,
    pub dec_rad: f64,
    pub radius_rad: f64,
    pub centre: Vector3<f64>,
    pub cos_radius: f64,
}

impl Cone {
    /// Construct from degrees. RA/Dec are interpreted as ICRS spherical angles
    /// (the same convention used by `GaiaCore::ra` / `dec`).
    pub fn from_degrees(ra_deg: f64, dec_deg: f64, radius_deg: f64) -> Self {
        let ra_rad = ra_deg.to_radians();
        let dec_rad = dec_deg.to_radians();
        let radius_rad = radius_deg.to_radians();
        let centre = Vector3::new(
            dec_rad.cos() * ra_rad.cos(),
            dec_rad.cos() * ra_rad.sin(),
            dec_rad.sin(),
        );
        Self {
            ra_rad,
            dec_rad,
            radius_rad,
            centre,
            cos_radius: radius_rad.cos(),
        }
    }

    /// True when `v` (assumed unit-length) lies within the cone.
    pub fn contains_unit_vec(&self, v: &Vector3<f64>) -> bool {
        v.dot(&self.centre) >= self.cos_radius
    }

    /// True when the given RA/Dec (degrees) lies within the cone.
    pub fn contains_radec_deg(&self, ra_deg: f64, dec_deg: f64) -> bool {
        let ra = ra_deg.to_radians();
        let dec = dec_deg.to_radians();
        let v = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        self.contains_unit_vec(&v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centre_is_inside() {
        let c = Cone::from_degrees(123.4, -56.7, 1.0);
        assert!(c.contains_radec_deg(123.4, -56.7));
    }

    #[test]
    fn beyond_radius_is_outside() {
        let c = Cone::from_degrees(0.0, 0.0, 1.0);
        // Point 2° east of (0, 0): outside a 1° cone.
        assert!(!c.contains_radec_deg(2.0, 0.0));
        // Point 0.5° east of (0, 0): inside.
        assert!(c.contains_radec_deg(0.5, 0.0));
    }

    #[test]
    fn antipode_is_outside() {
        let c = Cone::from_degrees(45.0, 30.0, 1.0);
        assert!(!c.contains_radec_deg(225.0, -30.0));
    }
}
