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

/// Pairs of (id, name) for celestial bodies.
///
/// Includes the bodies with RADII or rotational elements in NAIF pck00011.tpc.
/// Names follow NAIF's canonical ID list (retrieved 2026-09-21):
/// <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html>.
/// PCK: <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc>.
/// Existing aliases are retained; the first name for each ID is canonical.
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
    (505, "AMALTHEA"),
    (506, "HIMALIA"),
    (507, "ELARA"),
    (508, "PASIPHAE"),
    (509, "SINOPE"),
    (510, "LYSITHEA"),
    (511, "CARME"),
    (512, "ANANKE"),
    (513, "LEDA"),
    (514, "THEBE"),
    (515, "ADRASTEA"),
    (516, "METIS"),
    (601, "MIMAS"),
    (602, "ENCELADUS"),
    (603, "TETHYS"),
    (604, "DIONE"),
    (605, "RHEA"),
    (606, "TITAN"),
    (607, "HYPERION"),
    (608, "IAPETUS"),
    (609, "PHOEBE"),
    (610, "JANUS"),
    (611, "EPIMETHEUS"),
    (612, "HELENE"),
    (613, "TELESTO"),
    (614, "CALYPSO"),
    (615, "ATLAS"),
    (616, "PROMETHEUS"),
    (617, "PANDORA"),
    (618, "PAN"),
    (632, "METHONE"),
    (633, "PALLENE"),
    (634, "POLYDEUCES"),
    (635, "DAPHNIS"),
    (649, "ANTHE"),
    (653, "AEGAEON"),
    (701, "ARIEL"),
    (702, "UMBRIEL"),
    (703, "TITANIA"),
    (704, "OBERON"),
    (705, "MIRANDA"),
    (706, "CORDELIA"),
    (707, "OPHELIA"),
    (708, "BIANCA"),
    (709, "CRESSIDA"),
    (710, "DESDEMONA"),
    (711, "JULIET"),
    (712, "PORTIA"),
    (713, "ROSALIND"),
    (714, "BELINDA"),
    (715, "PUCK"),
    (801, "TRITON"),
    (802, "NEREID"),
    (803, "NAIAD"),
    (804, "THALASSA"),
    (805, "DESPINA"),
    (806, "GALATEA"),
    (807, "LARISSA"),
    (808, "PROTEUS"),
    (901, "CHARON"),
    (1000005, "BORRELLY"),
    (1000012, "67P/CHURYUMOV-GERASIMENKO (1969 R1)"),
    (1000036, "HALLEY"),
    (1000041, "HARTLEY 2"),
    (1000093, "TEMPEL_1"),
    (1000107, "WILD 2"),
    (2000001, "CERES"),
    (2000002, "PALLAS"),
    (2000004, "VESTA"),
    (2000016, "PSYCHE"),
    (2000021, "LUTETIA"),
    (2000052, "52_EUROPA"),
    (2000216, "KLEOPATRA"),
    (2000253, "MATHILDE"),
    (2000433, "EROS"),
    (2000511, "DAVIDA"),
    (2002867, "STEINS"),
    (2004179, "TOUTATIS"),
    (2025143, "ITOKAWA"),
    (2431010, "IDA"),
    (9511010, "GASPRA"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn satellite_names_and_existing_aliases_resolve() {
        for (name, id) in [
            ("Titan", 606),
            ("Enceladus", 602),
            ("Rhea", 605),
            ("Iapetus", 608),
            ("Triton", 801),
            ("Titania", 703),
            ("Charon", 901),
        ] {
            assert_eq!(target_id(name), Some(id));
            assert_eq!(target_name(id), Some(name.to_uppercase().as_str()));
        }
        assert_eq!(target_id("EMB"), Some(3));
        assert_eq!(target_id("earth-moon barycenter"), Some(3));
        assert_eq!(target_id("Europa"), Some(502));
        assert_eq!(target_id("52_Europa"), Some(2000052));
    }

    #[test]
    fn names_cover_bodies_with_constants_in_pck00011() {
        // IDs extracted from BODY*_RADII/POLE_RA/POLE_DEC/PM assignments in
        // the upstream pck00011.tpc, independently of the name table above.
        // This list deliberately excludes barycenter phase-angle variables.
        let bodies = [
            10, 199, 299, 301, 399, 401, 402, 499, 501, 502, 503, 504, 505, 506, 507, 508, 509,
            510, 511, 512, 513, 514, 515, 516, 599, 601, 602, 603, 604, 605, 606, 607, 608, 609,
            610, 611, 612, 613, 614, 615, 616, 617, 618, 632, 633, 634, 635, 649, 653, 699, 701,
            702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 799, 801, 802,
            803, 804, 805, 806, 807, 808, 899, 901, 999, 1000005, 1000012, 1000036, 1000041,
            1000093, 1000107, 2000001, 2000002, 2000004, 2000016, 2000021, 2000052, 2000216,
            2000253, 2000433, 2000511, 2002867, 2004179, 2025143, 2431010, 9511010,
        ];
        for id in bodies {
            assert!(target_name(id).is_some(), "missing PCK body {id}");
        }
    }
}
