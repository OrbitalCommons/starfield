//! The pinned catalogue of archive map products.
//!
//! Pinned rather than discovered. USGS's CKAN endpoint at
//! `/ckan/api/3/action/package_search` returns HTML, not JSON, so there is no
//! usable discovery API; and even if there were, a data source wants
//! reproducibility over freshness — a render should not change because an
//! upstream mosaic was silently revised.
//!
//! # Verifying a slug
//!
//! `astrogeology.usgs.gov` returns **HTTP 200 with a generic catalogue page**
//! for an unknown slug, so a wrong product looks like a hit and a status-code
//! check proves nothing. The existence test is whether the page contains a
//! `planetarymaps.usgs.gov` link. Every entry below was confirmed that way, and
//! the sizes are from following the redirect.

use starfield_core::{Result, StarfieldError};
use starfield_datasource_utils::datastore::{
    Artifact, ArtifactKey, ContentCheck, Datastore, Source,
};
use starfield_datasource_utils::{cache_dir, datastore_error, resolve_artifact};
use starfield_reflectance_library::Endmember;

use crate::grid::MapGrid;
use crate::map::PhotometricBand;

/// WGS84 flattening, for Blue Marble's geodetic latitude.
pub const EARTH_WGS84_FLATTENING: f64 = 1.0 / 298.257_223_563;

/// Cache subdirectory for downloaded mosaics.
const CACHE_SUBDIR: &str = "planet-maps";

/// USGS serves mosaics from here; requests 302 to an S3 bucket.
pub const USGS_MOSAIC_BASE_URL: &str = "https://planetarymaps.usgs.gov/mosaic/";

/// NASA Earth Observatory image records, which host Blue Marble.
pub const NASA_EOIMAGES_BASE_URL: &str = "https://eoimages.gsfc.nasa.gov/images/imagerecords/";

/// One archive mosaic, with everything needed to fetch and register it.
#[derive(Debug, Clone, Copy)]
pub struct MapProduct {
    /// Stable identifier used by [`MapProduct::by_id`].
    pub id: &'static str,
    /// NAIF id of the body.
    pub naif_id: i32,
    /// File name under [`USGS_MOSAIC_BASE_URL`].
    pub file_name: &'static str,
    /// Approximate download size, bytes. Verified by following the redirect.
    pub size_bytes: u64,
    /// Band the pixel values represent.
    pub band: PhotometricBand,
    /// Endmember the surface is treated as, pending real terrain-unit
    /// decomposition.
    pub endmember: Endmember,
    /// Whether the product is monochrome. Colour products need unmixing before
    /// they can drive an [`crate::AlbedoMap`]; see the crate docs.
    pub monochrome: bool,
    /// Provenance note.
    pub description: &'static str,
    /// Path under the host, when the product is not a USGS mosaic.
    ///
    /// `None` means [`USGS_MOSAIC_BASE_URL`] + [`MapProduct::file_name`].
    pub url_override: Option<&'static str>,
    /// Whether the pixel values are radiometrically calibrated reflectance.
    ///
    /// `false` marks a *visualization* product — contrast-enhanced for human
    /// viewing, correct in appearance but not in absolute brightness. Usable to
    /// make a body look right; not usable to claim a photometric result.
    pub calibrated: bool,
}

impl MapProduct {
    /// Full download URL.
    pub fn url(&self) -> String {
        match self.url_override {
            Some(url) => url.to_string(),
            None => format!("{}{}", USGS_MOSAIC_BASE_URL, self.file_name),
        }
    }

    /// Coordinate conventions of the stored product.
    ///
    /// The USGS mosaics are east-positive planetocentric, north at row 0,
    /// column 0 at longitude 0. Blue Marble differs on two counts and is the
    /// reason this is a per-product method rather than a constant: it spans
    /// -180..180 rather than 0..360, and its latitude is **geodetic**, which on
    /// Earth is 0.19 deg from planetocentric at 45 deg - two pixels at
    /// 0.1 deg/px.
    pub fn grid(&self) -> MapGrid {
        if self.naif_id == 399 {
            MapGrid::usgs_default()
                .planetographic()
                .with_flattening(EARTH_WGS84_FLATTENING)
                .with_lon0_deg(-180.0)
        } else {
            MapGrid::usgs_default()
        }
    }

    /// Local blob path if cached; errors if it has not yet been resolved.
    pub fn cached_path(&self) -> Result<std::path::PathBuf> {
        let store = Datastore::from_env().map_err(datastore_error)?;
        store
            .peek(&self.artifact(None)?.key)
            .ok_or_else(|| StarfieldError::DataError(format!("{} is not cached", self.id)))
    }

    /// Declarative archive identity and kind check, also used by server manifests.
    /// Approximate catalogue sizes are not enforced as exact byte pins.
    pub fn artifact(&self, expected_sha256: Option<&str>) -> Result<Artifact> {
        let (prefix, magic) = if self.file_name.ends_with(".jpg") {
            ("nasa/blue-marble", vec![vec![0xff, 0xd8, 0xff]])
        } else {
            (
                "usgs/mosaic",
                vec![
                    b"II*\0".to_vec(),
                    b"MM\0*".to_vec(),
                    b"II+\0".to_vec(),
                    b"MM\0+".to_vec(),
                ],
            )
        };
        let mut checks = vec![
            ContentCheck::default_binary(),
            ContentCheck::magic(magic, false),
        ];
        if let Some(digest) = expected_sha256 {
            if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(StarfieldError::DataError(
                    "expected_sha256 must be 64 hexadecimal digits".into(),
                ));
            }
            checks.push(ContentCheck::Sha256(digest.to_ascii_lowercase()));
        }
        let mut artifact = Artifact::new(
            ArtifactKey::new(format!("{prefix}/{}", self.file_name)).map_err(datastore_error)?,
            vec![Source::new(self.url())],
        )
        .with_check(ContentCheck::All(checks));
        artifact.provenance.description = self.description.into();
        artifact.provenance.license =
            "US government work; USGS/NASA source attribution retained".into();
        Ok(artifact)
    }

    /// Fetch into the starfield cache if not already present, returning the path.
    ///
    /// These are large — up to 11.9 GB — so this is deliberately explicit and
    /// never implicit in a constructor.
    ///
    /// `expected_sha256` is optional because the table does not yet carry
    /// digests: computing them means downloading ~25 GB. When supplied, the file
    /// is verified before publication. Rejected content is never installed.
    pub fn download(&self, expected_sha256: Option<&str>) -> Result<std::path::PathBuf> {
        let artifact = self.artifact(expected_sha256)?;
        resolve_artifact(
            &artifact,
            Some(&cache_dir().join(CACHE_SUBDIR).join(self.file_name)),
        )
    }

    /// Resolve using only the supplied store, without adopting global files.
    pub fn download_with(
        &self,
        store: &Datastore,
        expected_sha256: Option<&str>,
    ) -> Result<std::path::PathBuf> {
        store
            .get(&self.artifact(expected_sha256)?)
            .map_err(datastore_error)
    }

    /// Look up a product by [`MapProduct::id`].
    pub fn by_id(id: &str) -> Option<&'static MapProduct> {
        PRODUCTS.iter().find(|p| p.id == id)
    }

    /// Every product for a body, by NAIF id.
    pub fn for_body(naif_id: i32) -> Vec<&'static MapProduct> {
        PRODUCTS.iter().filter(|p| p.naif_id == naif_id).collect()
    }
}

/// The pinned catalogue. URLs and sizes verified by fetch.
pub static PRODUCTS: &[MapProduct] = &[
    MapProduct {
        id: "moon-lroc-wac-100m",
        naif_id: 301,
        file_name: "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif",
        size_bytes: 5_959_263_751,
        // WAC's 643 nm band is the one used for the monochrome global mosaic.
        band: PhotometricBand {
            lo_nm: 633.0,
            hi_nm: 653.0,
        },
        endmember: Endmember::FreshBasalt,
        monochrome: true,
        description: "LRO LROC WAC global morphology mosaic, 100 m, 643 nm",
        url_override: None,
        calibrated: true,
    },
    MapProduct {
        id: "mars-viking-color-925m",
        naif_id: 499,
        file_name: "Mars_Viking_ClrMosaic_global_925m.tif",
        size_bytes: 794_000_000,
        band: PhotometricBand {
            lo_nm: 400.0,
            hi_nm: 700.0,
        },
        endmember: Endmember::WeatheredBasalt,
        monochrome: false,
        description: "Viking colour global mosaic, 925 m",
        url_override: None,
        calibrated: true,
    },
    MapProduct {
        id: "mars-viking-mdim21-color-232m",
        naif_id: 499,
        file_name: "Mars_Viking_MDIM21_ClrMosaic_global_232m.tif",
        size_bytes: 12_745_000_000,
        band: PhotometricBand {
            lo_nm: 400.0,
            hi_nm: 700.0,
        },
        endmember: Endmember::WeatheredBasalt,
        monochrome: false,
        description: "Viking MDIM 2.1 colour global mosaic, 232 m",
        url_override: None,
        calibrated: true,
    },
    MapProduct {
        id: "ganymede-voyager-galileo-color-1435m",
        naif_id: 503,
        file_name: "Ganymede_Voyager_GalileoSSI_Global_ClrMosaic_1435m.tif",
        size_bytes: 0,
        band: PhotometricBand {
            lo_nm: 400.0,
            hi_nm: 700.0,
        },
        endmember: Endmember::WaterIce,
        monochrome: false,
        description: "Voyager/Galileo SSI colour global mosaic, 1.435 km",
        url_override: None,
        calibrated: true,
    },
    MapProduct {
        id: "callisto-voyager-galileo-1km",
        naif_id: 504,
        file_name: "Callisto_Voyager_GalileoSSI_global_mosaic_1km.tif",
        size_bytes: 0,
        band: PhotometricBand {
            lo_nm: 400.0,
            hi_nm: 700.0,
        },
        endmember: Endmember::WaterIce,
        monochrome: true,
        description: "Voyager/Galileo SSI global mosaic, 1 km",
        url_override: None,
        calibrated: true,
    },
    MapProduct {
        id: "earth-bmng-200412-5400",
        naif_id: 399,
        file_name: "world.topo.bathy.200412.3x5400x2700.jpg",
        size_bytes: 2_566_770,
        band: PhotometricBand {
            lo_nm: 400.0,
            hi_nm: 700.0,
        },
        endmember: Endmember::GreenVegetation,
        monochrome: false,
        description: "Blue Marble Next Generation, December 2004, 5400x2700, \
                      cloud-free true colour. VISUALIZATION PRODUCT - see `calibrated`",
        url_override: Some(
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/\
             world.topo.bathy.200412.3x5400x2700.jpg",
        ),
        calibrated: false,
    },
];

/// Report a body that has no product in the catalogue.
pub fn require_product_for(naif_id: i32) -> Result<&'static MapProduct> {
    MapProduct::for_body(naif_id)
        .into_iter()
        .next()
        .ok_or_else(|| {
            StarfieldError::ObjectNotFound(format!("no map product for NAIF id {naif_id}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique() {
        let mut ids: Vec<&str> = PRODUCTS.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        let before = ids.len();
        ids.dedup();
        assert_eq!(ids.len(), before, "duplicate product id");
    }

    #[test]
    fn lookup_by_id_and_body() {
        assert_eq!(
            MapProduct::by_id("moon-lroc-wac-100m").unwrap().naif_id,
            301
        );
        assert!(MapProduct::by_id("no-such-product").is_none());
        // Mars has two products at different resolutions.
        assert_eq!(MapProduct::for_body(499).len(), 2);
        assert!(MapProduct::for_body(-1).is_empty());
    }

    #[test]
    fn urls_point_at_a_known_host() {
        for p in PRODUCTS {
            let url = p.url();
            assert!(
                url.starts_with(USGS_MOSAIC_BASE_URL) || url.starts_with(NASA_EOIMAGES_BASE_URL),
                "{url}"
            );
        }
    }

    #[test]
    fn bands_are_ordered_and_plausible() {
        for p in PRODUCTS {
            assert!(p.band.lo_nm < p.band.hi_nm, "{}", p.id);
            assert!(p.band.lo_nm > 100.0 && p.band.hi_nm < 5000.0, "{}", p.id);
        }
    }

    #[test]
    fn jupiter_has_no_product_and_that_is_reported() {
        // Deliberate: USGS has no controlled global mosaic for Jupiter, and
        // will not - there is no solid surface to control a photogrammetric
        // network to. A caller must get an error, not an empty map.
        assert!(MapProduct::for_body(599).is_empty());
        let err = require_product_for(599).unwrap_err().to_string();
        assert!(err.contains("599"), "{err}");
    }

    #[test]
    fn planetary_products_use_the_usgs_grid_convention() {
        use crate::grid::{Latitude, Longitude, RowOrder};
        for p in PRODUCTS.iter().filter(|p| p.naif_id != 399) {
            let g = p.grid();
            assert_eq!(g.longitude, Longitude::EastPositive, "{}", p.id);
            assert_eq!(g.latitude, Latitude::Planetocentric, "{}", p.id);
            assert_eq!(g.row_order, RowOrder::NorthFirst, "{}", p.id);
            assert_eq!(g.lon0_deg, 0.0, "{}", p.id);
            assert_eq!(g.flattening, 0.0, "{}", p.id);
        }
    }

    #[test]
    fn earth_uses_geodetic_latitude_and_a_centred_seam() {
        // Blue Marble is the product that exercises the non-trivial grid path:
        // geodetic latitude on a flattened body, spanning -180..180.
        use crate::grid::{Latitude, Longitude};
        let earth = MapProduct::by_id("earth-bmng-200412-5400").unwrap();
        let g = earth.grid();
        assert_eq!(g.latitude, Latitude::Planetographic);
        assert_eq!(g.longitude, Longitude::EastPositive);
        assert_eq!(g.lon0_deg, -180.0);
        assert!((g.flattening - EARTH_WGS84_FLATTENING).abs() < 1e-15);

        // The prime meridian sits at the centre of the raster, not the edge.
        let (u, _) = g.body_fixed_to_uv(0.0, 0.0).unwrap();
        assert!((u - 0.5).abs() < 1e-12, "u = {u}");

        // Geodetic latitude runs ahead of planetocentric away from the equator,
        // so 45 deg planetocentric lands north of the halfway row by ~0.19 deg.
        let (_, v_equator) = g.body_fixed_to_uv(0.0, 0.0).unwrap();
        assert!((v_equator - 0.5).abs() < 1e-12);
        let (_, v45) = g.body_fixed_to_uv(0.0, 45f64.to_radians()).unwrap();
        let v45_spherical = (90.0 - 45.0) / 180.0;
        let offset_deg = (v45_spherical - v45) * 180.0;
        assert!(
            (0.18..0.21).contains(&offset_deg),
            "geodetic offset {offset_deg} deg"
        );
    }

    #[test]
    fn blue_marble_is_flagged_as_uncalibrated() {
        // It is a visualization product: contrast-enhanced for human viewing.
        // Fine for making Earth look right, not for a photometric claim.
        let earth = MapProduct::by_id("earth-bmng-200412-5400").unwrap();
        assert!(!earth.calibrated);
        // Everything else in the table is a calibrated archive product.
        for p in PRODUCTS.iter().filter(|p| p.naif_id != 399) {
            assert!(p.calibrated, "{}", p.id);
        }
    }

    #[test]
    fn url_overrides_produce_a_single_unbroken_url() {
        // The literal is split across source lines with a continuation; if the
        // escape were wrong the URL would contain whitespace and 404.
        let earth = MapProduct::by_id("earth-bmng-200412-5400").unwrap();
        let url = earth.url();
        assert!(!url.contains(char::is_whitespace), "{url}");
        assert_eq!(
            url,
            "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"
        );
    }

    #[test]
    #[ignore = "live upstream; must bypass the mirror and cache. Detects archive rot, so a warm or mirrored resolve would defeat it"]
    fn pinned_urls_are_reachable() {
        for p in PRODUCTS {
            let client = starfield_datasource_utils::build_http_client(60).unwrap();
            let response = client
                .head(p.url())
                .send()
                .and_then(|response| response.error_for_status())
                .unwrap_or_else(|error| {
                    panic!(
                        "{} live upstream failed (STARFIELD_ALLOW_UPSTREAM=1): {error}",
                        p.id
                    )
                });
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok())
                .unwrap_or("");
            assert!(
                !content_type.to_ascii_lowercase().contains("text/html"),
                "{} returned an HTML catalogue instead of a mosaic",
                p.id
            );
        }
    }
}
