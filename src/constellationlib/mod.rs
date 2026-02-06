//! Constellation identification from sky positions
//!
//! Identifies which of the 88 IAU constellations contains a given sky position,
//! using precomputed boundary grids in B1875 epoch coordinates (Delporte 1930).
//!
//! # Example
//!
//! ```no_run
//! use starfield::constellationlib::ConstellationMap;
//! use starfield::positions::Position;
//!
//! let map = ConstellationMap::new();
//! // Look up constellation for a position precessed to B1875
//! let name = map.from_ra_dec_b1875(12.0, 65.0);
//! assert_eq!(name, "UMa");
//! ```

mod data;

#[cfg(feature = "python-tests")]
mod python_tests;

use data::{CONSTELLATION_ABBREVIATIONS, DEC_BINS, RADEC_TO_INDEX, SORTED_DEC, SORTED_RA};

/// Julian date of the Besselian epoch B1875.0
const B1875_JD: f64 = 2_405_889.258_550_475;

/// Map of IAU constellation boundaries for position lookup
pub struct ConstellationMap {
    _private: (),
}

/// Full names of all 88 IAU constellations, indexed by abbreviation
static CONSTELLATION_NAMES: [(&str, &str); 88] = [
    ("And", "Andromeda"),
    ("Ant", "Antlia"),
    ("Aps", "Apus"),
    ("Aql", "Aquila"),
    ("Aqr", "Aquarius"),
    ("Ara", "Ara"),
    ("Ari", "Aries"),
    ("Aur", "Auriga"),
    ("Boo", "Bootes"),
    ("CMa", "Canis Major"),
    ("CMi", "Canis Minor"),
    ("CVn", "Canes Venatici"),
    ("Cae", "Caelum"),
    ("Cam", "Camelopardalis"),
    ("Cap", "Capricornus"),
    ("Car", "Carina"),
    ("Cas", "Cassiopeia"),
    ("Cen", "Centaurus"),
    ("Cep", "Cepheus"),
    ("Cet", "Cetus"),
    ("Cha", "Chamaeleon"),
    ("Cir", "Circinus"),
    ("Cnc", "Cancer"),
    ("Col", "Columba"),
    ("Com", "Coma Berenices"),
    ("CrA", "Corona Australis"),
    ("CrB", "Corona Borealis"),
    ("Crt", "Crater"),
    ("Cru", "Crux"),
    ("Crv", "Corvus"),
    ("Cyg", "Cygnus"),
    ("Del", "Delphinus"),
    ("Dor", "Dorado"),
    ("Dra", "Draco"),
    ("Equ", "Equuleus"),
    ("Eri", "Eridanus"),
    ("For", "Fornax"),
    ("Gem", "Gemini"),
    ("Gru", "Grus"),
    ("Her", "Hercules"),
    ("Hor", "Horologium"),
    ("Hya", "Hydra"),
    ("Hyi", "Hydrus"),
    ("Ind", "Indus"),
    ("LMi", "Leo Minor"),
    ("Lac", "Lacerta"),
    ("Leo", "Leo"),
    ("Lep", "Lepus"),
    ("Lib", "Libra"),
    ("Lup", "Lupus"),
    ("Lyn", "Lynx"),
    ("Lyr", "Lyra"),
    ("Men", "Mensa"),
    ("Mic", "Microscopium"),
    ("Mon", "Monoceros"),
    ("Mus", "Musca"),
    ("Nor", "Norma"),
    ("Oct", "Octans"),
    ("Oph", "Ophiuchus"),
    ("Ori", "Orion"),
    ("Pav", "Pavo"),
    ("Peg", "Pegasus"),
    ("Per", "Perseus"),
    ("Phe", "Phoenix"),
    ("Pic", "Pictor"),
    ("PsA", "Piscis Austrinus"),
    ("Psc", "Pisces"),
    ("Pup", "Puppis"),
    ("Pyx", "Pyxis"),
    ("Ret", "Reticulum"),
    ("Scl", "Sculptor"),
    ("Sco", "Scorpius"),
    ("Sct", "Scutum"),
    ("Ser", "Serpens"),
    ("Sex", "Sextans"),
    ("Sge", "Sagitta"),
    ("Sgr", "Sagittarius"),
    ("Tau", "Taurus"),
    ("Tel", "Telescopium"),
    ("TrA", "Triangulum Australe"),
    ("Tri", "Triangulum"),
    ("Tuc", "Tucana"),
    ("UMa", "Ursa Major"),
    ("UMi", "Ursa Minor"),
    ("Vel", "Vela"),
    ("Vir", "Virgo"),
    ("Vol", "Volans"),
    ("Vul", "Vulpecula"),
];

impl ConstellationMap {
    /// Create a new constellation map
    pub fn new() -> Self {
        ConstellationMap { _private: () }
    }

    /// Look up the constellation abbreviation for RA/Dec already in B1875 epoch
    ///
    /// # Arguments
    /// * `ra_hours` - Right ascension in hours [0, 24)
    /// * `dec_degrees` - Declination in degrees [-90, 90]
    ///
    /// # Returns
    /// Three-letter IAU constellation abbreviation (e.g. "UMa", "Ori")
    pub fn from_ra_dec_b1875(&self, ra_hours: f64, dec_degrees: f64) -> &'static str {
        let i = SORTED_RA.partition_point(|&v| v < ra_hours);
        let j = SORTED_DEC.partition_point(|&v| v <= dec_degrees);
        let k = RADEC_TO_INDEX[i * DEC_BINS + j] as usize;
        CONSTELLATION_ABBREVIATIONS[k]
    }

    /// Look up the constellation for an ICRS position by precessing to B1875
    ///
    /// This applies the IAU precession matrix to transform the position
    /// from ICRS (J2000) to B1875 epoch coordinates before lookup.
    ///
    /// # Arguments
    /// * `position` - An astrometric or apparent `Position`
    /// * `ts` - Timescale for computing precession
    ///
    /// # Returns
    /// Three-letter IAU constellation abbreviation
    pub fn constellation_of(
        &self,
        position: &crate::positions::Position,
        ts: &crate::time::Timescale,
    ) -> &'static str {
        let t_b1875 = ts.tdb_jd(B1875_JD);
        let (ra_hours, dec_degrees, _) = position.radec(Some(&t_b1875));
        // Ensure RA is in [0, 24)
        let ra = ra_hours.rem_euclid(24.0);
        self.from_ra_dec_b1875(ra, dec_degrees)
    }

    /// Get the full name for a constellation abbreviation
    ///
    /// Returns `None` if the abbreviation is not recognized.
    pub fn full_name(abbreviation: &str) -> Option<&'static str> {
        CONSTELLATION_NAMES
            .iter()
            .find(|(abbr, _)| *abbr == abbreviation)
            .map(|(_, name)| *name)
    }

    /// Return all 88 constellation abbreviation-name pairs
    pub fn all_names() -> &'static [(&'static str, &'static str); 88] {
        &CONSTELLATION_NAMES
    }
}

impl Default for ConstellationMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_north_pole_is_ursa_minor() {
        let map = ConstellationMap::new();
        assert_eq!(map.from_ra_dec_b1875(0.0, 90.0), "UMi");
    }

    #[test]
    fn test_south_pole_is_octans() {
        let map = ConstellationMap::new();
        assert_eq!(map.from_ra_dec_b1875(0.0, -90.0), "Oct");
    }

    #[test]
    fn test_orion_belt_region() {
        let map = ConstellationMap::new();
        // Orion's belt is roughly at RA ~5.5h, Dec ~ -1 deg
        assert_eq!(map.from_ra_dec_b1875(5.5, -1.0), "Ori");
    }

    #[test]
    fn test_sirius_region() {
        let map = ConstellationMap::new();
        // Sirius is roughly RA ~6.75h, Dec ~ -16.7 deg (B1875 coords are close)
        assert_eq!(map.from_ra_dec_b1875(6.75, -16.7), "CMa");
    }

    #[test]
    fn test_all_88_constellations_reachable() {
        let _map = ConstellationMap::new();
        let mut found = std::collections::HashSet::new();
        // Scan the grid to verify all 88 constellations appear
        for i in 0..=SORTED_RA.len() {
            for j in 0..=SORTED_DEC.len() {
                let k = RADEC_TO_INDEX[i * DEC_BINS + j] as usize;
                found.insert(CONSTELLATION_ABBREVIATIONS[k]);
            }
        }
        assert_eq!(found.len(), 88, "Not all constellations found in grid");
    }

    #[test]
    fn test_full_name_lookup() {
        assert_eq!(ConstellationMap::full_name("UMa"), Some("Ursa Major"));
        assert_eq!(ConstellationMap::full_name("Ori"), Some("Orion"));
        assert_eq!(ConstellationMap::full_name("XXX"), None);
    }

    #[test]
    fn test_constellation_count() {
        assert_eq!(CONSTELLATION_ABBREVIATIONS.len(), 88);
        assert_eq!(CONSTELLATION_NAMES.len(), 88);
    }

    #[test]
    fn test_constellation_of_with_precession() {
        let ts = crate::time::Timescale::default();
        let map = ConstellationMap::new();

        // Create a position pointing toward the north celestial pole (ICRS)
        // The pole should be in Ursa Minor
        let pos = crate::positions::Position::barycentric(
            nalgebra::Vector3::new(0.0, 0.0, 1.0),
            nalgebra::Vector3::zeros(),
            0,
        );
        let result = map.constellation_of(&pos, &ts);
        assert_eq!(result, "UMi");
    }
}
