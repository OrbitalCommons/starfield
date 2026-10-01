//! Constants module for astronomical calculations

use std::f64::consts::PI;

// Astronomical distances
/// Astronomical Unit in meters (per IAU 2012 Resolution B2)
pub const AU_M: f64 = 149_597_870_700.0;
/// Astronomical Unit in kilometers
pub const AU_KM: f64 = 149_597_870.700;

// Time constants
/// Seconds in a day
pub const DAY_S: f64 = 86_400.0;
/// J2000.0 epoch as Julian date
pub const J2000: f64 = 2_451_545.0;
/// B1950 epoch as Julian date
pub const B1950: f64 = 2_433_282.423_5;
/// TT minus TAI in seconds
pub const TT_MINUS_TAI_S: f64 = 32.184;
/// TT minus TAI in days
pub const TT_MINUS_TAI: f64 = TT_MINUS_TAI_S / DAY_S;
/// Microseconds in a day
pub const DAY_US: f64 = 86_400_000_000.0;

// Angles
/// Arcseconds in a complete circle
pub const ASEC360: f64 = 1_296_000.0;
/// Arcseconds to radians conversion factor
pub const ASEC2RAD: f64 = 4.848_136_811_095_36e-6;
/// Degrees to radians conversion factor
pub const DEG2RAD: f64 = PI / 180.0;
/// Radians to degrees conversion factor
pub const RAD2DEG: f64 = 180.0 / PI;
/// Tau (2*PI) for full circle
pub const TAU: f64 = 2.0 * PI;

// Physics
//
// `C`, `PLANCK_CONSTANT` and `BOLTZMANN_CONSTANT` are defining constants of
// the SI as revised in 2019 (26th CGPM, Resolution 1; SI Brochure, 9th
// edition), so their values are exact by definition and identical in
// CODATA 2018 and later adjustments.
/// Speed of light in vacuum in m/s (exact, SI defining constant)
pub const C: f64 = 299_792_458.0;
/// Planck constant h in J s (exact, SI defining constant)
pub const PLANCK_CONSTANT: f64 = 6.626_070_15e-34;
/// Boltzmann constant k_B in J/K (exact, SI defining constant)
pub const BOLTZMANN_CONSTANT: f64 = 1.380_649e-23;
/// Heliocentric gravitational constant in m^3/s^2
pub const GS: f64 = 1.327_124_400_179_87e+20;
/// Solar GM in km^3/s^2 (Pitjeva 2005)
pub const GM_SUN: f64 = 132_712_440_042.0;

// Planetary gravitational parameters
//
// GM of each planetary system (barycenter), plus the Earth, Moon and Mars
// bodies themselves, in km^3/s^2. Values are the JPL DE440 constants
// (Park, Folkner, Williams & Boggs 2021, AJ 161, 105, Table 4), which are the
// same set the DE440 header carries in AU^3/day^2; multiply by
// [`GM_KM3_S2_TO_AU3_D2`] for that form. Use the body GM, not the system GM,
// for an orbit close enough to the planet that its moons do not matter.
/// GM of the Mercury system in km^3/s^2 (DE440)
pub const GM_MERCURY: f64 = 22_031.868_551;
/// GM of the Venus system in km^3/s^2 (DE440)
pub const GM_VENUS: f64 = 324_858.592;
/// GM of the Earth in km^3/s^2 (DE440)
pub const GM_EARTH: f64 = 398_600.435_507;
/// GM of the Moon in km^3/s^2 (DE440)
pub const GM_MOON: f64 = 4_902.800_118;
/// GM of the Mars system in km^3/s^2 (DE440)
pub const GM_MARS_SYSTEM: f64 = 42_828.375_816;
/// GM of Mars itself in km^3/s^2 (DE440); Phobos and Deimos add 1e-4 km^3/s^2
pub const GM_MARS: f64 = 42_828.375_214;
/// GM of the Jupiter system in km^3/s^2 (DE440)
pub const GM_JUPITER: f64 = 126_712_764.1;
/// GM of the Saturn system in km^3/s^2 (DE440)
pub const GM_SATURN: f64 = 37_940_584.841_8;
/// GM of the Uranus system in km^3/s^2 (DE440)
pub const GM_URANUS: f64 = 5_794_556.4;
/// GM of the Neptune system in km^3/s^2 (DE440)
pub const GM_NEPTUNE: f64 = 6_836_527.100_58;
/// GM of the Pluto system in km^3/s^2 (DE440)
pub const GM_PLUTO: f64 = 975.5;

// Earth constants
/// Earth's angular velocity in radians/s
pub const EARTH_ANGVEL: f64 = 7.292_115_0e-5;
/// Earth's equatorial radius in meters
pub const EARTH_RADIUS: f64 = 6_378_136.6;
/// IERS 2010 inverse Earth flattening
pub const IERS_2010_INVERSE_EARTH_FLATTENING: f64 = 298.25642;

// Derived constants
/// Speed of light in AU/day
pub const C_AUDAY: f64 = C * DAY_S / AU_M;
/// Factor converting a gravitational parameter from km^3/s^2 to AU^3/day^2
pub const GM_KM3_S2_TO_AU3_D2: f64 = DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);

// Calendar constants
/// First day of Gregorian calendar in Julian day number (1582-10-15)
pub const GREGORIAN_START: i32 = 2_299_161;
/// First day of Gregorian calendar in England (1752-09-14)
pub const GREGORIAN_START_ENGLAND: i32 = 2_361_222;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn radiation_constants_match_codata_2018() {
        // First radiation constant c1 = 2 pi h c^2 and second radiation
        // constant c2 = h c / k_B, as tabulated (to 10 significant figures)
        // in CODATA 2018.
        assert_relative_eq!(
            2.0 * PI * PLANCK_CONSTANT * C * C,
            3.741_771_852e-16,
            max_relative = 1e-10
        );
        assert_relative_eq!(
            PLANCK_CONSTANT * C / BOLTZMANN_CONSTANT,
            1.438_776_877e-2,
            max_relative = 1e-10
        );
    }

    #[test]
    fn stefan_boltzmann_constant_follows_from_h_c_and_k() {
        // sigma = 2 pi^5 k^4 / (15 h^3 c^2); CODATA 2018 gives
        // 5.670 374 419e-8 W m^-2 K^-4.
        let sigma = 2.0 * PI.powi(5) * BOLTZMANN_CONSTANT.powi(4)
            / (15.0 * PLANCK_CONSTANT.powi(3) * C * C);
        assert_relative_eq!(sigma, 5.670_374_419e-8, max_relative = 1e-10);
    }
}
