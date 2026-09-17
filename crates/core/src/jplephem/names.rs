//! Standard SPICE target names and ID numbers

use lazy_static::lazy_static;
use std::collections::HashMap;

lazy_static! {
    static ref TARGET_NAMES: HashMap<i32, &'static str> = {
        let mut m = HashMap::new();
        for &(id, name) in TARGET_NAME_PAIRS.iter() {
            m.entry(id).or_insert(name);
        }
        m
    };
    static ref TARGET_IDS: HashMap<String, i32> = {
        let mut m = HashMap::new();
        for &(id, name) in TARGET_NAME_PAIRS.iter() {
            m.insert(name.to_lowercase(), id);
        }
        m
    };
}

/// Get the canonical name of a target given its ID number
pub fn target_name(id: i32) -> Option<&'static str> {
    TARGET_NAMES.get(&id).copied()
}

/// Alias for target_name
pub fn get_target_name(id: i32) -> Option<&'static str> {
    target_name(id)
}

/// Get the ID number of a target given its name (case-insensitive)
pub fn target_id(name: &str) -> Option<i32> {
    TARGET_IDS.get(&name.to_lowercase()).copied()
}

/// Pairs of (id, name) for celestial bodies
const TARGET_NAME_PAIRS: &[(i32, &str)] = &[
    (0, "SOLAR SYSTEM BARYCENTER"),
    (0, "SSB"),
    (1, "MERCURY BARYCENTER"),
    (2, "VENUS BARYCENTER"),
    (3, "EARTH BARYCENTER"),
    (3, "EMB"),
    (3, "EARTH MOON BARYCENTER"),
    (3, "EARTH-MOON BARYCENTER"),
    (4, "MARS BARYCENTER"),
    (5, "JUPITER BARYCENTER"),
    (6, "SATURN BARYCENTER"),
    (7, "URANUS BARYCENTER"),
    (8, "NEPTUNE BARYCENTER"),
    (9, "PLUTO BARYCENTER"),
    (10, "SUN"),
    (199, "MERCURY"),
    (299, "VENUS"),
    (399, "EARTH"),
    (301, "MOON"),
    (499, "MARS"),
    (401, "PHOBOS"),
    (402, "DEIMOS"),
    (599, "JUPITER"),
    (501, "IO"),
    (502, "EUROPA"),
    (503, "GANYMEDE"),
    (504, "CALLISTO"),
    (699, "SATURN"),
    (799, "URANUS"),
    (899, "NEPTUNE"),
    (999, "PLUTO"),
];

/// Common NAIF target ID constants
pub mod targets {
    pub const SOLAR_SYSTEM_BARYCENTER: i32 = 0;
    pub const MERCURY_BARYCENTER: i32 = 1;
    pub const VENUS_BARYCENTER: i32 = 2;
    pub const EARTH_MOON_BARYCENTER: i32 = 3;
    pub const MARS_BARYCENTER: i32 = 4;
    pub const JUPITER_BARYCENTER: i32 = 5;
    pub const SATURN_BARYCENTER: i32 = 6;
    pub const URANUS_BARYCENTER: i32 = 7;
    pub const NEPTUNE_BARYCENTER: i32 = 8;
    pub const PLUTO_BARYCENTER: i32 = 9;
    pub const SUN: i32 = 10;
    pub const MERCURY: i32 = 199;
    pub const VENUS: i32 = 299;
    pub const EARTH: i32 = 399;
    pub const MOON: i32 = 301;
    pub const MARS: i32 = 499;
}
