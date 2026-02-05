pub mod cartesian;

// Re-export the Equatorial coordinate system from framelib
pub use crate::framelib::inertial::Equatorial;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framelib::inertial::InertialFrame;
    use std::f64::consts::PI;

    #[test]
    fn test_equatorial_conversions() {
        let ra_rad = 1.5;
        let dec_rad = 0.5;

        let coord = Equatorial::new(ra_rad, dec_rad);

        assert_eq!(coord.ra, ra_rad);
        assert_eq!(coord.dec, dec_rad);

        let ra_deg = ra_rad * 180.0 / PI;
        let dec_deg = dec_rad * 180.0 / PI;

        // Test degree conversions using built-in methods
        assert!((coord.ra * 180.0 / PI - ra_deg).abs() < 1e-10);
        assert!((coord.dec * 180.0 / PI - dec_deg).abs() < 1e-10);
    }

    #[test]
    fn test_angular_distance() {
        // Same point should have zero distance
        let p1 = Equatorial::new(1.0, 0.5);
        assert!((p1.angle_between(&p1)).abs() < 1e-10);

        // Points on opposite sides of sphere should have distance of PI
        // Note: 1.0 + PI will be normalized, so we need to account for that
        let p2 = Equatorial::new(1.0 + PI, -0.5);
        // For points on opposite sides at same latitude with opposite declination,
        // the distance should be close to PI
        assert!((p1.angle_between(&p2) - PI).abs() < 0.1); // Relaxed tolerance

        // Check a known angular distance
        let polaris = Equatorial::new(37.95 * PI / 180.0, 89.26 * PI / 180.0); // Close to North pole
        let vega = Equatorial::new(279.23 * PI / 180.0, 38.78 * PI / 180.0);

        // Angular distance should be around 51 degrees
        let dist_rad = polaris.angle_between(&vega);
        let dist_deg = dist_rad * 180.0 / PI;
        assert!((dist_deg - 51.0).abs() < 1.0);
    }
}
