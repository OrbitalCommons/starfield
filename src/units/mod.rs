//! Type-safe astronomical units built on the `uom` crate
//!
//! Provides custom unit definitions for astronomical calculations:
//! - `astronomical_unit_iau` for length with IAU 2012 exact value
//! - `au_per_day` for velocity (AU/day)
//!
//! Re-exports commonly used `uom::si::f64` quantity types.

// Re-export uom SI quantity types for convenience
pub use uom::si::angle;
pub use uom::si::f64::Angle;
pub use uom::si::f64::Length;
pub use uom::si::f64::Time;
pub use uom::si::f64::Velocity;

// Re-export built-in SI units commonly used in astronomy
pub use uom::si::angle::degree;
pub use uom::si::angle::radian;
pub use uom::si::angle::second as arcsecond;
pub use uom::si::length::astronomical_unit;
pub use uom::si::length::kilometer;
pub use uom::si::length::meter;
pub use uom::si::time::day;
pub use uom::si::time::second;
pub use uom::si::velocity::kilometer_per_second;
pub use uom::si::velocity::meter_per_second;

/// IAU 2012 exact Astronomical Unit in meters
pub const AU_M_EXACT: f64 = 149_597_870_700.0;

/// Speed of light in AU/day (derived from C and AU_M_EXACT)
pub const C_AUDAY: f64 = 299_792_458.0 * 86_400.0 / AU_M_EXACT;

/// Convert AU to kilometers using IAU 2012 exact value
pub fn au_to_km(au: f64) -> f64 {
    au * AU_M_EXACT / 1000.0
}

/// Convert kilometers to AU using IAU 2012 exact value
pub fn km_to_au(km: f64) -> f64 {
    km * 1000.0 / AU_M_EXACT
}

/// Convert AU/day to km/s
pub fn au_per_day_to_km_per_s(au_day: f64) -> f64 {
    au_day * AU_M_EXACT / (86_400.0 * 1000.0)
}

/// Convert km/s to AU/day
pub fn km_per_s_to_au_per_day(km_s: f64) -> f64 {
    km_s * 86_400.0 * 1000.0 / AU_M_EXACT
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_au_to_km_value() {
        let one_au = Length::new::<astronomical_unit>(1.0);
        let km = one_au.get::<kilometer>();
        // Built-in uom AU is slightly imprecise (1.495979e11 vs exact 1.495978707e11)
        // So we test our exact conversion function separately
        assert_relative_eq!(km, 149_597_900.0, epsilon = 200.0);
    }

    #[test]
    fn test_exact_au_conversion() {
        assert_relative_eq!(au_to_km(1.0), 149_597_870.700, epsilon = 1e-6);
        assert_relative_eq!(km_to_au(149_597_870.700), 1.0, epsilon = 1e-15);
    }

    #[test]
    fn test_c_auday() {
        // Speed of light should be ~173.14 AU/day
        assert_relative_eq!(C_AUDAY, 173.144_632_720, epsilon = 1e-3);
    }

    #[test]
    fn test_velocity_conversions() {
        // Earth orbital velocity ~30 km/s
        let au_day = km_per_s_to_au_per_day(29.78);
        assert_relative_eq!(au_day, 0.017_202, epsilon = 1e-3);

        let roundtrip = au_per_day_to_km_per_s(au_day);
        assert_relative_eq!(roundtrip, 29.78, epsilon = 1e-10);
    }

    #[test]
    fn test_angle_arcseconds() {
        let one_degree = Angle::new::<degree>(1.0);
        let asec = one_degree.get::<arcsecond>();
        assert_relative_eq!(asec, 3600.0, epsilon = 1e-10);
    }

    #[test]
    fn test_uom_length_velocity_time() {
        // uom dimensional analysis: Length / Time = Velocity
        let dist = Length::new::<meter>(1000.0);
        let time = Time::new::<second>(10.0);
        let vel: Velocity = dist / time;
        assert_relative_eq!(vel.get::<meter_per_second>(), 100.0, epsilon = 1e-10);
    }
}
