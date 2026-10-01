//! Default NAIF satellite SPK kernels for each planetary system.
//!
//! JPL planetary ephemerides such as DE440 carry each outer planet only as
//! its system barycentre. Positions of the planet centre and of its moons
//! come from a separate satellite SPK in NAIF's
//! `generic_kernels/spk/satellites/` directory. Several kernels exist per
//! system; [`default_satellite_kernel`] names the one holding the system's
//! major moons, chosen from NAIF's `aa_summaries.txt`:
//!
//! | System | Kernel | Bodies (NAIF IDs) | Coverage |
//! |--------|--------|-------------------|----------|
//! | Mars (4) | `mar099.bsp` | Phobos, Deimos (401-402), Mars (499) | 1600-2600 |
//! | Jupiter (5) | `jup365.bsp` | Io through Callisto (501-504), Amalthea (505), Thebe, Adrastea, Metis (514-516), Jupiter (599) | 1600-2200 |
//! | Saturn (6) | `sat441.bsp` | Mimas through Phoebe (601-609), Helene, Telesto, Calypso (612-614), Methone, Polydeuces (632, 634), Saturn (699) | 1750-2250 |
//! | Uranus (7) | `ura184_part-3.bsp` | Ariel, Umbriel, Titania, Oberon, Miranda (701-705), irregular moons (716-724), Uranus (799) | 1600-2399 |
//! | Neptune (8) | `nep097.bsp` | Triton (801), Neptune (899) | 1600-2399 |
//! | Pluto (9) | `plu060.bsp` | Charon, Nix, Hydra, Kerberos, Styx (901-905), Pluto (999) | 1800-2199 |
//!
//! Every one of these files also carries its system barycentre and the
//! Earth relative to the solar-system barycentre, so merged with a
//! planetary kernel each moon chains back to the solar-system barycentre.
//!
//! Bodies outside the default set need another kernel: `ura184_part-1.bsp`
//! and `ura184_part-2.bsp` hold Uranus' small inner moons, `nep105.bsp`
//! holds Nereid (802), and `sat415.bsp` and `sat455.bsp` through
//! `sat459.bsp` hold Saturn's remaining small and irregular moons. The
//! Earth-Moon system (3) needs no satellite kernel: the Moon is part of
//! every JPL planetary ephemeris.

use super::names::targets;

/// Default satellite SPK file for each planetary system barycentre.
///
/// Pairs of `(system barycentre NAIF ID, kernel file name)`, in system
/// order. See the [module documentation](self) for the bodies and time
/// span each file covers.
pub const DEFAULT_SATELLITE_KERNELS: [(i32, &str); 6] = [
    (targets::MARS_BARYCENTER, "mar099.bsp"),
    (targets::JUPITER_BARYCENTER, "jup365.bsp"),
    (targets::SATURN_BARYCENTER, "sat441.bsp"),
    (targets::URANUS_BARYCENTER, "ura184_part-3.bsp"),
    (targets::NEPTUNE_BARYCENTER, "nep097.bsp"),
    (targets::PLUTO_BARYCENTER, "plu060.bsp"),
];

/// NAIF satellite SPK holding the major moons of a planetary system.
///
/// `system_barycenter` is the NAIF ID of the system barycentre, `4`
/// (Mars) through `9` (Pluto). Returns `None` for any other ID, including
/// the Earth-Moon barycentre (3), whose Moon is in the planetary kernel,
/// and planet or moon IDs such as 499 or 401. The returned name resolves
/// through [`crate::Loader::open`] like any other SPK.
pub fn default_satellite_kernel(system_barycenter: i32) -> Option<&'static str> {
    DEFAULT_SATELLITE_KERNELS
        .iter()
        .find(|(id, _)| *id == system_barycenter)
        .map(|&(_, name)| name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{resolve_url, NAIF_SATELLITES_URL};

    #[test]
    fn each_outer_system_has_its_major_moon_kernel() {
        for (system, name) in [
            (4, "mar099.bsp"),
            (5, "jup365.bsp"),
            (6, "sat441.bsp"),
            (7, "ura184_part-3.bsp"),
            (8, "nep097.bsp"),
            (9, "plu060.bsp"),
        ] {
            assert_eq!(default_satellite_kernel(system), Some(name));
        }
    }

    #[test]
    fn non_system_ids_have_no_default_kernel() {
        for id in [0, 1, 2, 3, 10, 301, 399, 401, 499, 599, 801, -82] {
            assert_eq!(default_satellite_kernel(id), None, "id {id}");
        }
    }

    #[test]
    fn default_kernels_resolve_to_the_naif_satellite_directory() {
        for (_, name) in DEFAULT_SATELLITE_KERNELS {
            assert_eq!(
                resolve_url(name),
                Some(format!("{NAIF_SATELLITES_URL}{name}"))
            );
        }
    }
}
