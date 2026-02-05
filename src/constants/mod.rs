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
/// Speed of light in m/s
pub const C: f64 = 299_792_458.0;
/// Heliocentric gravitational constant in m^3/s^2
pub const GS: f64 = 1.327_124_400_179_87e+20;
/// Solar GM in km^3/s^2 (Pitjeva 2005)
pub const GM_SUN: f64 = 132_712_440_042.0;

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

// Calendar constants
/// First day of Gregorian calendar in Julian day number (1582-10-15)
pub const GREGORIAN_START: i32 = 2_299_161;
/// First day of Gregorian calendar in England (1752-09-14)
pub const GREGORIAN_START_ENGLAND: i32 = 2_361_222;
