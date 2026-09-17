//! Astronomical feature catalog for interesting objects and regions
//!
//! This module provides a collection of notable astronomical features
//! that can be used for targeting simulations, including:
//! - Major constellations
//! - Notable star clusters
//! - Galaxies and nebulae
//! - Other interesting celestial objects

use std::collections::HashMap;

/// Represents a region of interest in the sky
#[derive(Debug, Clone)]
pub struct SkyFeature {
    /// Name of the feature
    pub name: String,
    /// Type of feature (constellation, cluster, nebula, etc.)
    pub feature_type: FeatureType,
    /// Right ascension in degrees
    pub ra_deg: f64,
    /// Declination in degrees
    pub dec_deg: f64,
    /// Approximate diameter in degrees
    pub diameter_deg: f64,
    /// Brief description of the feature
    pub description: String,
}

/// Types of astronomical features
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FeatureType {
    /// A constellation (formal IAU division of the sky)
    Constellation,
    /// An open star cluster
    OpenCluster,
    /// A globular star cluster
    GlobularCluster,
    /// A nebula (emission, reflection, or dark)
    Nebula,
    /// A galaxy
    Galaxy,
    /// A star or multiple star system of interest
    Star,
    /// Other feature types
    Other,
}

impl SkyFeature {
    /// Create a new sky feature
    pub fn new(
        name: &str,
        feature_type: FeatureType,
        ra_deg: f64,
        dec_deg: f64,
        diameter_deg: f64,
        description: &str,
    ) -> Self {
        Self {
            name: name.to_string(),
            feature_type,
            ra_deg,
            dec_deg,
            diameter_deg,
            description: description.to_string(),
        }
    }
}

/// Catalog of interesting astronomical features
#[derive(Debug, Clone)]
pub struct FeatureCatalog {
    features: HashMap<String, SkyFeature>,
}

impl Default for FeatureCatalog {
    /// Get the default catalog with predefined features
    fn default() -> Self {
        let mut catalog = Self {
            features: HashMap::new(),
        };

        // Add all predefined features
        for feature in create_constellation_features() {
            catalog.add_feature(feature);
        }

        for feature in create_open_cluster_features() {
            catalog.add_feature(feature);
        }

        for feature in create_globular_cluster_features() {
            catalog.add_feature(feature);
        }

        for feature in create_nebula_features() {
            catalog.add_feature(feature);
        }

        for feature in create_galaxy_features() {
            catalog.add_feature(feature);
        }

        for feature in create_star_features() {
            catalog.add_feature(feature);
        }

        catalog
    }
}

impl FeatureCatalog {
    /// Create a new empty catalog
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    /// Add a feature to the catalog
    pub fn add_feature(&mut self, feature: SkyFeature) {
        self.features.insert(feature.name.clone(), feature);
    }

    /// Get a feature by name
    pub fn get_feature(&self, name: &str) -> Option<&SkyFeature> {
        self.features.get(name)
    }

    /// Get all features of a specific type
    pub fn get_features_by_type(&self, feature_type: &FeatureType) -> Vec<&SkyFeature> {
        self.features
            .values()
            .filter(|f| f.feature_type == *feature_type)
            .collect()
    }

    /// Get all features
    pub fn all_features(&self) -> Vec<&SkyFeature> {
        self.features.values().collect()
    }

    /// Get count of features
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Check if catalog is empty
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Create constellation features
fn create_constellation_features() -> Vec<SkyFeature> {
    vec![
        // Coordinates are for approximate center of constellation
        SkyFeature::new(
            "Andromeda",
            FeatureType::Constellation,
            0.8,
            38.5,
            30.0,
            "Named after the mythological princess, contains the Andromeda Galaxy.",
        ),
        SkyFeature::new(
            "Aquarius",
            FeatureType::Constellation,
            334.0,
            -10.0,
            25.0,
            "The water bearer, a zodiac constellation.",
        ),
        SkyFeature::new(
            "Aquila",
            FeatureType::Constellation,
            296.0,
            9.0,
            20.0,
            "The eagle, contains the bright star Altair.",
        ),
        SkyFeature::new(
            "Aries",
            FeatureType::Constellation,
            29.0,
            20.0,
            15.0,
            "The ram, a zodiac constellation.",
        ),
        SkyFeature::new(
            "Auriga",
            FeatureType::Constellation,
            90.0,
            40.0,
            20.0,
            "The charioteer, contains bright star Capella.",
        ),
        SkyFeature::new(
            "Boötes",
            FeatureType::Constellation,
            213.0,
            30.0,
            22.0,
            "The herdsman, contains bright star Arcturus.",
        ),
        SkyFeature::new(
            "Cancer",
            FeatureType::Constellation,
            130.0,
            20.0,
            15.0,
            "The crab, a zodiac constellation containing Beehive Cluster.",
        ),
        SkyFeature::new(
            "Canis Major",
            FeatureType::Constellation,
            105.0,
            -20.0,
            20.0,
            "The greater dog, contains brightest star Sirius.",
        ),
        SkyFeature::new(
            "Canis Minor",
            FeatureType::Constellation,
            114.0,
            6.0,
            10.0,
            "The lesser dog, contains bright star Procyon.",
        ),
        SkyFeature::new(
            "Capricornus",
            FeatureType::Constellation,
            320.0,
            -20.0,
            15.0,
            "The sea goat, a zodiac constellation.",
        ),
        SkyFeature::new(
            "Cassiopeia",
            FeatureType::Constellation,
            10.0,
            60.0,
            22.0,
            "Named after the mythological queen, has distinctive W shape.",
        ),
        SkyFeature::new(
            "Cepheus",
            FeatureType::Constellation,
            5.0,
            70.0,
            20.0,
            "Named after the mythological king, contains Delta Cephei.",
        ),
        SkyFeature::new(
            "Cetus",
            FeatureType::Constellation,
            24.0,
            -10.0,
            35.0,
            "The whale or sea monster, one of the largest constellations.",
        ),
        SkyFeature::new(
            "Corona Borealis",
            FeatureType::Constellation,
            235.0,
            30.0,
            10.0,
            "The northern crown, a small semicircular pattern of stars.",
        ),
        SkyFeature::new(
            "Cygnus",
            FeatureType::Constellation,
            305.0,
            40.0,
            25.0,
            "The swan, contains bright star Deneb and Northern Cross asterism.",
        ),
        SkyFeature::new(
            "Draco",
            FeatureType::Constellation,
            240.0,
            65.0,
            35.0,
            "The dragon, a large circumpolar constellation.",
        ),
        SkyFeature::new(
            "Gemini",
            FeatureType::Constellation,
            111.0,
            22.0,
            20.0,
            "The twins, a zodiac constellation with bright stars Castor and Pollux.",
        ),
        SkyFeature::new(
            "Hercules",
            FeatureType::Constellation,
            255.0,
            30.0,
            25.0,
            "Named after the mythological hero, contains M13 globular cluster.",
        ),
        SkyFeature::new(
            "Leo",
            FeatureType::Constellation,
            175.0,
            15.0,
            25.0,
            "The lion, a zodiac constellation with bright star Regulus.",
        ),
        SkyFeature::new(
            "Libra",
            FeatureType::Constellation,
            230.0,
            -15.0,
            20.0,
            "The scales, a zodiac constellation.",
        ),
        SkyFeature::new(
            "Lyra",
            FeatureType::Constellation,
            280.0,
            32.5,
            15.0,
            "The lyre, contains bright star Vega and Ring Nebula.",
        ),
        SkyFeature::new(
            "Ophiuchus",
            FeatureType::Constellation,
            255.0,
            0.0,
            30.0,
            "The serpent bearer, contains many globular clusters.",
        ),
        SkyFeature::new(
            "Orion",
            FeatureType::Constellation,
            85.0,
            0.0,
            25.0,
            "The hunter, contains Betelgeuse, Rigel, and Orion Nebula.",
        ),
        SkyFeature::new(
            "Pegasus",
            FeatureType::Constellation,
            340.0,
            20.0,
            25.0,
            "The winged horse, contains the Great Square of Pegasus.",
        ),
        SkyFeature::new(
            "Perseus",
            FeatureType::Constellation,
            55.0,
            45.0,
            25.0,
            "Named after Greek hero, contains Double Cluster and Algol.",
        ),
        SkyFeature::new(
            "Pisces",
            FeatureType::Constellation,
            5.0,
            10.0,
            30.0,
            "The fishes, a zodiac constellation.",
        ),
        SkyFeature::new(
            "Sagittarius",
            FeatureType::Constellation,
            280.0,
            -30.0,
            25.0,
            "The archer, zodiac constellation containing galactic center.",
        ),
        SkyFeature::new(
            "Scorpius",
            FeatureType::Constellation,
            255.0,
            -30.0,
            20.0,
            "The scorpion, zodiac constellation with bright star Antares.",
        ),
        SkyFeature::new(
            "Taurus",
            FeatureType::Constellation,
            65.0,
            15.0,
            25.0,
            "The bull, zodiac constellation with Aldebaran, Hyades and Pleiades.",
        ),
        SkyFeature::new(
            "Ursa Major",
            FeatureType::Constellation,
            165.0,
            56.0,
            35.0,
            "The great bear, contains the Big Dipper asterism.",
        ),
        SkyFeature::new(
            "Ursa Minor",
            FeatureType::Constellation,
            240.0,
            75.0,
            20.0,
            "The little bear, contains Polaris (North Star).",
        ),
        SkyFeature::new(
            "Virgo",
            FeatureType::Constellation,
            195.0,
            0.0,
            35.0,
            "The maiden, zodiac constellation with bright star Spica.",
        ),
    ]
}

/// Create open cluster features
fn create_open_cluster_features() -> Vec<SkyFeature> {
    vec![
        SkyFeature::new(
            "Pleiades",
            FeatureType::OpenCluster,
            56.75,
            24.12,
            2.0,
            "M45, The Seven Sisters, striking blue open cluster in Taurus.",
        ),
        SkyFeature::new(
            "Hyades",
            FeatureType::OpenCluster,
            66.75,
            15.87,
            5.5,
            "Closest open cluster to Earth, forms V-shape near Aldebaran.",
        ),
        SkyFeature::new(
            "Double Cluster",
            FeatureType::OpenCluster,
            34.75,
            57.13,
            1.0,
            "NGC 869 and NGC 884, pair of open clusters in Perseus.",
        ),
        SkyFeature::new(
            "Beehive Cluster",
            FeatureType::OpenCluster,
            130.0,
            19.98,
            1.5,
            "M44, Praesepe, large open cluster in Cancer.",
        ),
        SkyFeature::new(
            "Wild Duck Cluster",
            FeatureType::OpenCluster,
            279.25,
            -6.25,
            0.5,
            "M11, rich, compact open cluster in Scutum.",
        ),
        SkyFeature::new(
            "Jewel Box Cluster",
            FeatureType::OpenCluster,
            186.0,
            -60.30,
            0.2,
            "NGC 4755, colorful open cluster in Crux.",
        ),
        SkyFeature::new(
            "Butterfly Cluster",
            FeatureType::OpenCluster,
            265.0,
            -13.78,
            0.5,
            "M6, open cluster in Scorpius resembling a butterfly.",
        ),
        SkyFeature::new(
            "Ptolemy's Cluster",
            FeatureType::OpenCluster,
            267.92,
            -34.80,
            0.5,
            "M7, large, bright open cluster in Scorpius.",
        ),
    ]
}

/// Create globular cluster features
fn create_globular_cluster_features() -> Vec<SkyFeature> {
    vec![
        SkyFeature::new(
            "Omega Centauri",
            FeatureType::GlobularCluster,
            201.75,
            -47.48,
            0.5,
            "NGC 5139, largest globular cluster in Milky Way.",
        ),
        SkyFeature::new(
            "47 Tucanae",
            FeatureType::GlobularCluster,
            6.0,
            -72.08,
            0.5,
            "NGC 104, second brightest globular cluster.",
        ),
        SkyFeature::new(
            "Hercules Cluster",
            FeatureType::GlobularCluster,
            250.42,
            36.46,
            0.3,
            "M13, brightest globular cluster in northern hemisphere.",
        ),
        SkyFeature::new(
            "M22",
            FeatureType::GlobularCluster,
            279.10,
            -23.90,
            0.5,
            "Bright globular cluster in Sagittarius.",
        ),
        SkyFeature::new(
            "M15",
            FeatureType::GlobularCluster,
            322.50,
            12.17,
            0.3,
            "Dense, bright globular cluster in Pegasus.",
        ),
        SkyFeature::new(
            "M3",
            FeatureType::GlobularCluster,
            205.55,
            28.38,
            0.3,
            "Bright northern globular cluster in Canes Venatici.",
        ),
        SkyFeature::new(
            "M4",
            FeatureType::GlobularCluster,
            245.90,
            -26.53,
            0.4,
            "Nearest globular cluster to Earth in Scorpius.",
        ),
        SkyFeature::new(
            "M5",
            FeatureType::GlobularCluster,
            229.64,
            2.08,
            0.3,
            "Bright globular cluster in Serpens.",
        ),
    ]
}

/// Create nebula features
fn create_nebula_features() -> Vec<SkyFeature> {
    vec![
        SkyFeature::new(
            "Orion Nebula",
            FeatureType::Nebula,
            83.82,
            -5.39,
            1.0,
            "M42, brightest nebula visible to naked eye.",
        ),
        SkyFeature::new(
            "Lagoon Nebula",
            FeatureType::Nebula,
            271.0,
            -24.38,
            1.0,
            "M8, large emission nebula in Sagittarius.",
        ),
        SkyFeature::new(
            "Trifid Nebula",
            FeatureType::Nebula,
            270.6,
            -23.03,
            0.5,
            "M20, distinctive three-lobed emission/reflection nebula.",
        ),
        SkyFeature::new(
            "Eagle Nebula",
            FeatureType::Nebula,
            274.7,
            -13.84,
            0.7,
            "M16, site of 'Pillars of Creation' star-forming region.",
        ),
        SkyFeature::new(
            "Ring Nebula",
            FeatureType::Nebula,
            283.4,
            33.03,
            0.1,
            "M57, classic planetary nebula in Lyra.",
        ),
        SkyFeature::new(
            "Dumbbell Nebula",
            FeatureType::Nebula,
            299.9,
            22.72,
            0.2,
            "M27, bright planetary nebula in Vulpecula.",
        ),
        SkyFeature::new(
            "Veil Nebula",
            FeatureType::Nebula,
            311.5,
            30.67,
            3.0,
            "Large supernova remnant in Cygnus.",
        ),
        SkyFeature::new(
            "Crab Nebula",
            FeatureType::Nebula,
            83.63,
            22.02,
            0.2,
            "M1, supernova remnant in Taurus.",
        ),
        SkyFeature::new(
            "Horsehead Nebula",
            FeatureType::Nebula,
            85.24,
            -2.46,
            0.5,
            "IC 434, famous dark nebula in Orion.",
        ),
    ]
}

/// Create galaxy features
fn create_galaxy_features() -> Vec<SkyFeature> {
    vec![
        SkyFeature::new(
            "Andromeda Galaxy",
            FeatureType::Galaxy,
            10.68,
            41.27,
            3.0,
            "M31, nearest major galaxy to Milky Way.",
        ),
        SkyFeature::new(
            "Triangulum Galaxy",
            FeatureType::Galaxy,
            23.47,
            30.66,
            1.0,
            "M33, third-largest galaxy in Local Group.",
        ),
        SkyFeature::new(
            "Whirlpool Galaxy",
            FeatureType::Galaxy,
            202.47,
            47.20,
            0.3,
            "M51, classic spiral galaxy in Canes Venatici.",
        ),
        SkyFeature::new(
            "Sombrero Galaxy",
            FeatureType::Galaxy,
            190.0,
            -11.62,
            0.2,
            "M104, edge-on spiral galaxy with distinctive dust lane.",
        ),
        SkyFeature::new(
            "Pinwheel Galaxy",
            FeatureType::Galaxy,
            210.80,
            54.34,
            0.7,
            "M101, face-on spiral galaxy in Ursa Major.",
        ),
        SkyFeature::new(
            "Centaurus A",
            FeatureType::Galaxy,
            201.36,
            -43.02,
            0.5,
            "NGC 5128, peculiar elliptical galaxy with dust lane.",
        ),
        SkyFeature::new(
            "Bode's Galaxy",
            FeatureType::Galaxy,
            148.97,
            69.07,
            0.3,
            "M81, bright spiral galaxy in Ursa Major.",
        ),
        SkyFeature::new(
            "Cigar Galaxy",
            FeatureType::Galaxy,
            148.97,
            69.68,
            0.2,
            "M82, starburst galaxy near M81.",
        ),
        SkyFeature::new(
            "Southern Pinwheel Galaxy",
            FeatureType::Galaxy,
            204.25,
            -29.87,
            0.5,
            "M83, barred spiral galaxy in Hydra.",
        ),
    ]
}

/// Create star features
fn create_star_features() -> Vec<SkyFeature> {
    vec![
        SkyFeature::new(
            "Sirius",
            FeatureType::Star,
            101.29,
            -16.72,
            0.1,
            "Alpha Canis Majoris, brightest star in night sky.",
        ),
        SkyFeature::new(
            "Canopus",
            FeatureType::Star,
            96.0,
            -52.70,
            0.1,
            "Alpha Carinae, second brightest star in night sky.",
        ),
        SkyFeature::new(
            "Alpha Centauri",
            FeatureType::Star,
            219.9,
            -60.84,
            0.1,
            "Triple star system, closest star system to Solar System.",
        ),
        SkyFeature::new(
            "Arcturus",
            FeatureType::Star,
            213.92,
            19.18,
            0.1,
            "Alpha Boötis, brightest star in northern celestial hemisphere.",
        ),
        SkyFeature::new(
            "Vega",
            FeatureType::Star,
            279.24,
            38.78,
            0.1,
            "Alpha Lyrae, fifth brightest star in night sky.",
        ),
        SkyFeature::new(
            "Capella",
            FeatureType::Star,
            79.17,
            45.99,
            0.1,
            "Alpha Aurigae, bright yellow giant binary star.",
        ),
        SkyFeature::new(
            "Rigel",
            FeatureType::Star,
            78.63,
            -8.20,
            0.1,
            "Beta Orionis, blue supergiant in Orion.",
        ),
        SkyFeature::new(
            "Betelgeuse",
            FeatureType::Star,
            88.79,
            7.41,
            0.1,
            "Alpha Orionis, red supergiant variable star in Orion.",
        ),
        SkyFeature::new(
            "Antares",
            FeatureType::Star,
            247.35,
            -26.43,
            0.1,
            "Alpha Scorpii, red supergiant in Scorpius.",
        ),
        SkyFeature::new(
            "Polaris",
            FeatureType::Star,
            37.95,
            89.26,
            0.1,
            "Alpha Ursae Minoris, current North Star, close to NCP.",
        ),
        SkyFeature::new(
            "Mizar and Alcor",
            FeatureType::Star,
            200.98,
            54.93,
            0.1,
            "Famous double star in the handle of the Big Dipper.",
        ),
        SkyFeature::new(
            "Albireo",
            FeatureType::Star,
            292.68,
            27.96,
            0.1,
            "Beta Cygni, beautiful gold and blue double star system.",
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalog_creation() {
        let catalog = FeatureCatalog::default();
        assert!(!catalog.is_empty());

        // Check a few specific features to make sure they exist
        let pleiades = catalog.get_feature("Pleiades");
        assert!(pleiades.is_some());
        assert_eq!(pleiades.unwrap().feature_type, FeatureType::OpenCluster);

        let orion = catalog.get_feature("Orion");
        assert!(orion.is_some());
        assert_eq!(orion.unwrap().feature_type, FeatureType::Constellation);
    }

    #[test]
    fn test_get_features_by_type() {
        let catalog = FeatureCatalog::default();

        let constellations = catalog.get_features_by_type(&FeatureType::Constellation);
        assert!(!constellations.is_empty());

        let stars = catalog.get_features_by_type(&FeatureType::Star);
        assert!(!stars.is_empty());
    }
}
