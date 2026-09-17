//! Embedded endmember-abundance tiers.
//!
//! A tier stores, per texel, the fractional abundance of each endmember rather
//! than a single albedo — so a renderer gets colour that varies across the disk
//! and not merely brightness.
//!
//! # The file is self-describing
//!
//! The grid geometry lives in the header, not only in the product catalogue. A
//! loader that has never heard of MCD12C1 still cannot misregister the grid by
//! half a cell or mirror it, which is the failure this whole crate is arranged
//! to prevent. The header also carries its provenance, because the fitted class
//! weights are only defensible if you can tell which fit produced a given file.

use std::io::Read;

use starfield::{Result, StarfieldError};
use starfield_reflectance_library::{Endmember, EndmemberMix};

use crate::grid::{Latitude, Longitude, MapGrid, RowOrder};
use crate::map::{SampleFootprint, SurfaceSampler, MU_FLOOR};

/// Format magic. Bump on any layout change.
const MAGIC: &[u8] = b"SFEMv3\n";

/// Where a grid's coordinate bounds sit relative to its cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Registration {
    /// Cell *edges* lie on the bounds. A 1440-wide global grid starting at
    /// −180° has its first cell spanning −180.00 to −179.75.
    CellEdge,
    /// Cell *centres* lie on the bounds.
    CellCentre,
}

/// A per-texel endmember abundance grid.
#[derive(Debug, Clone)]
pub struct AbundanceTier {
    width: usize,
    height: usize,
    endmembers: Vec<Endmember>,
    grid: MapGrid,
    registration: Registration,
    naif_id: i32,
    scale: f32,
    provenance: String,
    /// Mip pyramid. Level 0 is full resolution; each level is plane-major,
    /// `planes[e * w * h + row * w + col]`, 255 = abundance 1.0.
    levels: Vec<TierLevel>,
}

#[derive(Debug, Clone)]
struct TierLevel {
    width: usize,
    height: usize,
    planes: Vec<u8>,
}

impl AbundanceTier {
    /// Parse a gzipped tier.
    pub fn from_gz_bytes(bytes: &[u8]) -> Result<Self> {
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(bytes)
            .read_to_end(&mut raw)
            .map_err(|e| StarfieldError::DataError(format!("tier: gunzip failed: {e}")))?;
        Self::from_bytes(&raw)
    }

    /// Parse an uncompressed tier.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let err = |m: String| StarfieldError::DataError(format!("tier: {m}"));
        if b.len() < MAGIC.len() || &b[..MAGIC.len()] != MAGIC {
            return Err(err(format!(
                "bad magic; expected {:?}",
                String::from_utf8_lossy(MAGIC)
            )));
        }
        let mut p = MAGIC.len();
        let need = |p: usize, n: usize, b: &[u8]| -> Result<()> {
            if p + n > b.len() {
                return Err(err(format!("truncated: wanted {n} bytes at offset {p}")));
            }
            Ok(())
        };

        need(p, 4 + 4 + 8 + 8 + 8, b)?;
        let width = u16::from_le_bytes([b[p], b[p + 1]]) as usize;
        let height = u16::from_le_bytes([b[p + 2], b[p + 3]]) as usize;
        let n_endmembers = b[p + 4] as usize;
        let lon_sense = b[p + 5];
        let lat_kind = b[p + 6];
        let registration = b[p + 7];
        p += 8;
        let naif_id = i32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        p += 4;
        let lon0_deg = f64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        p += 8;
        let lat0_deg = f64::from_le_bytes(b[p..p + 8].try_into().unwrap());
        p += 8;
        let payload_len = u64::from_le_bytes(b[p..p + 8].try_into().unwrap()) as usize;
        p += 8;
        need(p, 4, b)?;
        let scale = f32::from_le_bytes(b[p..p + 4].try_into().unwrap());
        p += 4;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(err(format!("scale {scale} must be finite and positive")));
        }

        let take_str = |p: &mut usize| -> Result<String> {
            need(*p, 2, b)?;
            let n = u16::from_le_bytes([b[*p], b[*p + 1]]) as usize;
            *p += 2;
            need(*p, n, b)?;
            let s = String::from_utf8_lossy(&b[*p..*p + n]).into_owned();
            *p += n;
            Ok(s)
        };
        let names_blob = take_str(&mut p)?;
        let provenance = take_str(&mut p)?;

        let endmembers: Vec<Endmember> = names_blob
            .split('\n')
            .filter(|s| !s.is_empty())
            .map(|s| Endmember::from_id(s).ok_or_else(|| err(format!("unknown endmember {s:?}"))))
            .collect::<Result<_>>()?;
        if endmembers.len() != n_endmembers {
            return Err(err(format!(
                "header says {n_endmembers} endmembers, names list has {}",
                endmembers.len()
            )));
        }

        // Declared length is checked against both the arithmetic and what is
        // actually present, so a truncated download fails here rather than
        // rendering as a dark stripe across the southern hemisphere.
        let expected = n_endmembers * width * height;
        if payload_len != expected {
            return Err(err(format!(
                "header payload_len {payload_len} != {n_endmembers} x {width} x {height} = {expected}"
            )));
        }
        need(p, payload_len, b)?;
        if b.len() - p != payload_len {
            return Err(err(format!(
                "payload is {} bytes, header declares {payload_len}",
                b.len() - p
            )));
        }
        let planes = b[p..p + payload_len].to_vec();

        let grid = MapGrid {
            longitude: match lon_sense {
                0 => Longitude::EastPositive,
                1 => Longitude::WestPositive,
                other => return Err(err(format!("unknown longitude sense {other}"))),
            },
            latitude: match lat_kind {
                0 => Latitude::Planetocentric,
                1 => Latitude::Planetographic,
                other => return Err(err(format!("unknown latitude kind {other}"))),
            },
            lon0_deg,
            // lat0 at the top means row 0 is the northern edge.
            row_order: if lat0_deg > 0.0 {
                RowOrder::NorthFirst
            } else {
                RowOrder::SouthFirst
            },
            flattening: crate::products::EARTH_WGS84_FLATTENING,
        };

        Ok(Self {
            width,
            height,
            endmembers,
            grid,
            registration: match registration {
                0 => Registration::CellEdge,
                1 => Registration::CellCentre,
                other => return Err(err(format!("unknown registration {other}"))),
            },
            naif_id,
            scale,
            provenance,
            levels: build_pyramid(TierLevel {
                width,
                height,
                planes,
            }),
        })
    }

    /// Grid dimensions, `(width, height)`.
    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Endmembers, in plane order.
    pub fn endmembers(&self) -> &[Endmember] {
        &self.endmembers
    }

    /// Coordinate conventions, read from the file's own header.
    pub fn grid(&self) -> &MapGrid {
        &self.grid
    }

    /// Whether cell edges or centres lie on the grid bounds.
    pub fn registration(&self) -> Registration {
        self.registration
    }

    /// NAIF id of the body.
    pub fn naif_id(&self) -> i32 {
        self.naif_id
    }

    /// Source archive, class-table version and build-script version.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Number of pyramid levels.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Full-scale value one raw unit represents.
    ///
    /// A one-endmember tier whose albedo exceeds the endmember's own band mean
    /// has an abundance above 1, which `u8/255` cannot hold. Rather than fork
    /// the format for that case, the header carries a scale: abundance is
    /// `raw / 255 * scale`. Composition tiers use `1.0`.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Abundance of plane `e` at `(col, row)` of `level`.
    fn plane_at(&self, level: usize, e: usize, col: usize, row: usize) -> f64 {
        let l = &self.levels[level];
        l.planes[e * l.width * l.height + row * l.width + col] as f64 / 255.0 * self.scale as f64
    }

    /// Mix at a texel of a given level.
    fn mix_at(&self, level: usize, u: f64, v: f64) -> EndmemberMix {
        let l = &self.levels[level];
        let col = ((u * l.width as f64) as usize).min(l.width - 1);
        let row = ((v * l.height as f64) as usize).min(l.height - 1);
        let mut weights = Vec::new();
        for (e, &endmember) in self.endmembers.iter().enumerate() {
            let w = self.plane_at(level, e, col, row);
            if w > 0.0 {
                weights.push((endmember, w));
            }
        }
        EndmemberMix::new(weights)
    }

    /// Choose a pyramid level for a sky footprint at emission cosine `mu`.
    ///
    /// Identical policy to [`crate::AlbedoMap::select_level`]: select from the
    /// stretched axis, clamp `mu` at [`MU_FLOOR`], never alias. Sharing the
    /// policy matters — a body with a tier and a body with a scalar map must
    /// not blur differently at the same geometry.
    pub fn select_level(&self, sky_radius_rad: f64, mu: f64) -> Option<SampleFootprint> {
        if !sky_radius_rad.is_finite() || sky_radius_rad <= 0.0 || !mu.is_finite() {
            return None;
        }
        let mu_clamped = mu < MU_FLOOR;
        let mu_eff = mu.max(MU_FLOOR);
        if mu_eff > 1.0 {
            return None;
        }
        let texels_per_rad = self.width as f64 / (2.0 * std::f64::consts::PI);
        let texels = sky_radius_rad / mu_eff * texels_per_rad;
        let level = if texels <= 1.0 {
            0
        } else {
            (texels.log2().floor() as usize).min(self.levels.len() - 1)
        };
        Some(SampleFootprint {
            level,
            texels,
            anisotropy: 1.0 / mu_eff,
            mu_clamped,
        })
    }

    /// Composition at a body-fixed position, full resolution, nearest texel.
    ///
    /// Input is **east-positive planetocentric radians**, as everywhere else in
    /// this crate; the header's conventions are applied internally.
    ///
    /// Endmembers with zero abundance are omitted, so a mix over open ocean has
    /// one entry rather than nine. Use [`AbundanceTier::sample_area`] unless the
    /// footprint is known to be sub-texel.
    pub fn sample(&self, lon_rad: f64, lat_rad: f64) -> Option<EndmemberMix> {
        let (u, v) = self.grid.body_fixed_to_uv(lon_rad, lat_rad)?;
        Some(self.mix_at(0, u, v))
    }

    /// Composition over a footprint, with the resolution level chosen from it.
    ///
    /// The tier equivalent of [`crate::AlbedoMap::sample_area`], and the reason
    /// both implement [`SurfaceSampler`]: a consumer should not branch on
    /// whether a body happens to have real composition data.
    pub fn sample_area(
        &self,
        lon_rad: f64,
        lat_rad: f64,
        sky_radius_rad: f64,
        mu: f64,
    ) -> Option<(EndmemberMix, SampleFootprint)> {
        let (u, v) = self.grid.body_fixed_to_uv(lon_rad, lat_rad)?;
        let footprint = self.select_level(sky_radius_rad, mu)?;
        Some((self.mix_at(footprint.level, u, v), footprint))
    }

    /// Largest plane sum over the grid, in raw u8 units.
    ///
    /// For a composition tier (`scale == 1.0`) this should not exceed
    /// `255 + n_endmembers`: abundances sum to at most 1, and each plane can
    /// round up by at most one unit. A larger value means the planes do not
    /// describe a partition and the file is corrupt. A scaled one-endmember
    /// tier is not a partition and this bound does not apply to it.
    pub fn max_plane_sum(&self) -> u32 {
        let l = &self.levels[0];
        let cells = l.width * l.height;
        (0..cells)
            .map(|i| {
                (0..self.endmembers.len())
                    .map(|e| l.planes[e * cells + i] as u32)
                    .sum::<u32>()
            })
            .max()
            .unwrap_or(0)
    }
}

impl SurfaceSampler for AbundanceTier {
    fn sample_area(
        &self,
        lon_rad: f64,
        lat_rad: f64,
        sky_radius_rad: f64,
        mu: f64,
    ) -> Option<(EndmemberMix, SampleFootprint)> {
        AbundanceTier::sample_area(self, lon_rad, lat_rad, sky_radius_rad, mu)
    }

    fn endmembers(&self) -> Vec<Endmember> {
        self.endmembers.clone()
    }
}

/// Box-filter every plane down to a 1x1 level.
///
/// Averaging abundances preserves the partition: if the planes summed to at
/// most 1 before, they do after, so a coarse level is still a valid mix.
fn build_pyramid(base: TierLevel) -> Vec<TierLevel> {
    let mut levels = vec![base];
    while levels.last().unwrap().width > 1 || levels.last().unwrap().height > 1 {
        let prev = levels.last().unwrap();
        let width = (prev.width / 2).max(1);
        let height = (prev.height / 2).max(1);
        let n = prev.planes.len() / (prev.width * prev.height);
        let mut planes = vec![0u8; n * width * height];
        for e in 0..n {
            for row in 0..height {
                for col in 0..width {
                    let mut sum = 0u32;
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let sr = (row * 2 + dy).min(prev.height - 1);
                            let sc = (col * 2 + dx).min(prev.width - 1);
                            sum += prev.planes[e * prev.width * prev.height + sr * prev.width + sc]
                                as u32;
                        }
                    }
                    planes[e * width * height + row * width + col] = (sum / 4) as u8;
                }
            }
        }
        levels.push(TierLevel {
            width,
            height,
            planes,
        });
    }
    levels
}

/// The embedded Moon tier: LROC WAC 643 nm mosaic at 0.1°, one endmember.
///
/// One endmember pending real lunar soils (#65): fresh basalt is the right
/// spectral family for mare and the wrong one for highlands, and the
/// mare/highland dichotomy is the Moon's largest albedo feature.
pub const MOON_TIER_GZ: &[u8] = include_bytes!("../data/moon_albedo_0p1deg.bin.gz");

/// Parse the embedded Moon tier.
pub fn moon_tier() -> Result<AbundanceTier> {
    AbundanceTier::from_gz_bytes(MOON_TIER_GZ)
}

/// The embedded Mars tier: USGS Viking colour mosaic at 0.1°, one endmember.
///
/// Mars has no composition data yet, so this is a **one-endmember** tier:
/// brightness varies across the disk and colour does not. It is the offline
/// equivalent of the scalar-map path, and it is what `AbundanceTier::scale`
/// exists for — a single endmember reproducing an albedo above its own band
/// mean has an abundance above 1.
pub const MARS_TIER_GZ: &[u8] = include_bytes!("../data/mars_albedo_0p1deg.bin.gz");

/// Parse the embedded Mars tier.
pub fn mars_tier() -> Result<AbundanceTier> {
    AbundanceTier::from_gz_bytes(MARS_TIER_GZ)
}

/// The embedded Earth tier: MODIS MCD12G1-derived endmember abundances at 0.25°.
pub const EARTH_TIER_GZ: &[u8] = include_bytes!("../data/earth_endmembers_0p25deg.bin.gz");

/// Parse the embedded Earth tier.
pub fn earth_tier() -> Result<AbundanceTier> {
    AbundanceTier::from_gz_bytes(EARTH_TIER_GZ)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier() -> AbundanceTier {
        earth_tier().unwrap()
    }

    #[test]
    fn embedded_tier_has_the_expected_shape() {
        let t = tier();
        assert_eq!(t.size(), (1440, 720));
        assert_eq!(t.endmembers().len(), 9);
        assert_eq!(t.naif_id(), 399);
        assert_eq!(t.registration(), Registration::CellEdge);
    }

    #[test]
    fn grid_conventions_come_from_the_file_not_the_catalogue() {
        // The point of the self-describing header: geometry survives without
        // any knowledge of which product this is.
        let t = tier();
        let g = t.grid();
        assert_eq!(g.longitude, Longitude::EastPositive);
        assert_eq!(g.latitude, Latitude::Planetographic);
        assert_eq!(g.lon0_deg, -180.0);
        assert_eq!(g.row_order, RowOrder::NorthFirst);
        assert!(g.flattening > 0.003 && g.flattening < 0.004);
    }

    #[test]
    fn provenance_names_the_granule_and_the_fit() {
        let t = tier();
        let p = t.provenance();
        assert!(p.contains("MCD12C1"), "{p}");
        assert!(p.contains("igbp-fitted-1"), "{p}");
        assert!(p.contains("coastal radius"), "{p}");
    }

    #[test]
    fn plane_sums_never_exceed_a_partition_plus_rounding() {
        // A corrupted plane cannot pass as a plausible mix.
        let t = tier();
        let max = t.max_plane_sum();
        let limit = 255 + t.endmembers().len() as u32;
        assert!(max <= limit, "max plane sum {max} exceeds {limit}");
    }

    #[test]
    fn open_ocean_is_ocean_and_sahara_is_arid() {
        let t = tier();
        // Mid-Pacific, far from any coast.
        let pacific = t.sample((-140f64).to_radians(), 0f64.to_radians()).unwrap();
        let ocean = pacific
            .weights()
            .iter()
            .find(|(e, _)| *e == Endmember::OpenOcean)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(ocean > 0.9, "mid-Pacific ocean fraction {ocean}");

        // Central Sahara.
        let sahara = t.sample(15f64.to_radians(), 23f64.to_radians()).unwrap();
        let arid = sahara
            .weights()
            .iter()
            .find(|(e, _)| *e == Endmember::AridSoil)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(arid > 0.5, "Sahara arid-soil fraction {arid}");
    }

    #[test]
    fn antarctica_is_snow() {
        let t = tier();
        let mix = t.sample(0f64.to_radians(), (-80f64).to_radians()).unwrap();
        let snow = mix
            .weights()
            .iter()
            .find(|(e, _)| *e == Endmember::Snow)
            .map(|(_, w)| *w)
            .unwrap_or(0.0);
        assert!(snow > 0.8, "Antarctic snow fraction {snow}");
    }

    #[test]
    fn closed_canopy_reproduces_the_fitted_split() {
        // Congo basin: uninterrupted evergreen broadleaf, so the mix should be
        // almost exactly the fitted 55/45 forest row.
        //
        // Deliberately not Manaus (-60, -3), which the tier resolves as a third
        // river water and a tenth urban. That is correct - it is a major river
        // confluence with a city of two million - but it makes a poor test of
        // the canopy fit.
        let t = tier();
        let mix = t.sample(20f64.to_radians(), 0f64.to_radians()).unwrap();
        let get = |want: Endmember| {
            mix.weights()
                .iter()
                .find(|(e, _)| *e == want)
                .map(|(_, w)| *w)
                .unwrap_or(0.0)
        };
        let veg = get(Endmember::GreenVegetation);
        let shade = get(Endmember::Shade);
        assert!((veg - 0.55).abs() < 0.03, "vegetation fraction {veg}");
        assert!((shade - 0.45).abs() < 0.03, "shade fraction {shade}");
        assert!(shade < veg);
    }

    #[test]
    fn a_river_city_resolves_as_water_and_urban() {
        // Manaus: the Rio Negro/Amazon confluence with a city of two million.
        // A tier that returned pure forest here would have lost the sub-pixel
        // class fractions that are the whole point of MCD12C1.
        let t = tier();
        let mix = t
            .sample((-60f64).to_radians(), (-3f64).to_radians())
            .unwrap();
        let get = |want: Endmember| {
            mix.weights()
                .iter()
                .find(|(e, _)| *e == want)
                .map(|(_, w)| *w)
                .unwrap_or(0.0)
        };
        assert!(get(Endmember::CoastalWater) > 0.1, "river water fraction");
        assert!(get(Endmember::Asphalt) > 0.02, "urban fraction");
    }

    #[test]
    fn mixes_omit_absent_endmembers() {
        let t = tier();
        let pacific = t.sample((-140f64).to_radians(), 0.0).unwrap();
        assert!(
            pacific.len() < t.endmembers().len(),
            "open ocean should not carry every endmember"
        );
    }

    #[test]
    fn pyramid_halves_down_to_one_texel() {
        let t = tier();
        // 1440x720 -> 11 levels before both axes reach 1.
        assert_eq!(t.levels(), 11);
    }

    #[test]
    fn coarse_levels_stay_valid_mixes() {
        // Averaging abundances preserves the partition, so a blurred texel is
        // still a mix and not something summing past 1.
        let t = tier();
        for level in 0..t.levels() {
            let mix = t.mix_at(level, 0.4, 0.55);
            assert!(
                mix.total_weight() <= 1.02,
                "level {level} total weight {}",
                mix.total_weight()
            );
        }
    }

    #[test]
    fn a_coarser_footprint_selects_a_coarser_level() {
        let t = tier();
        let fine = t.select_level(1e-5, 1.0).unwrap();
        let coarse = t.select_level(1e-2, 1.0).unwrap();
        assert_eq!(fine.level, 0);
        assert!(coarse.level > fine.level, "{coarse:?} vs {fine:?}");
        // And the limb coarsens further, with the clamp reported.
        let limb = t.select_level(1e-2, 1e-9).unwrap();
        assert!(limb.mu_clamped);
        assert!(limb.level >= coarse.level);
        assert!(limb.level < t.levels());
    }

    #[test]
    fn blurring_the_ocean_keeps_it_ocean() {
        // A large footprint in the mid-Pacific must stay water at every level;
        // if the pyramid were built per-texel-max or otherwise wrongly it would
        // drift toward land.
        let t = tier();
        for radius in [1e-5, 1e-3, 1e-2] {
            let (mix, fp) = t
                .sample_area((-140f64).to_radians(), 0.0, radius, 1.0)
                .unwrap();
            let water: f64 = mix
                .weights()
                .iter()
                .filter(|(e, _)| matches!(e, Endmember::OpenOcean | Endmember::CoastalWater))
                .map(|(_, w)| *w)
                .sum();
            assert!(water > 0.85, "level {} water {water}", fp.level);
        }
    }

    #[test]
    fn the_trait_gives_one_interface_for_both_backings() {
        use crate::map::{PhotometricBand, SurfaceSampler};
        use crate::MapGrid;

        let t = tier();
        let m = crate::AlbedoMap::new(
            MapGrid::usgs_default(),
            PhotometricBand::new(400.0, 700.0),
            Endmember::FreshBasalt,
            0.1,
            64,
            32,
            vec![0.05; 64 * 32],
        )
        .unwrap();

        let samplers: Vec<&dyn SurfaceSampler> = vec![&t, &m];
        for s in samplers {
            let (mix, fp) = s.sample_area(0.3, 0.2, 1e-4, 0.8).unwrap();
            assert!(!s.endmembers().is_empty());
            assert!(mix.total_weight() > 0.0);
            assert!(fp.anisotropy > 1.0);
        }
    }

    #[test]
    fn mars_tier_has_the_expected_shape() {
        let t = mars_tier().unwrap();
        assert_eq!(t.size(), (3600, 1800));
        assert_eq!(t.naif_id(), 499);
        assert_eq!(t.endmembers(), &[Endmember::WeatheredBasalt]);
        // Mars is planetocentric and starts at longitude 0, unlike Earth.
        assert_eq!(t.grid().latitude, Latitude::Planetocentric);
        assert_eq!(t.grid().lon0_deg, 0.0);
        // A one-endmember tier reproducing an albedo above the endmember's own
        // band mean needs a scale above 1.
        assert!(t.scale() > 1.0, "scale {}", t.scale());
    }

    #[test]
    fn the_mars_albedo_dichotomy_is_the_right_way_round() {
        // Registration check via real albedo features: Solis Lacus is a classic
        // dark marking, Amazonis a bright dust-mantled plain. A mirrored map
        // swaps them, and the difference is 2x so it cannot be noise.
        //
        // Deliberately not Syrtis Major vs Arabia Terra, the more famous pair:
        // in this product they differ by only 1.10x, against roughly 2.7x on
        // the real planet. The Viking colour mosaic is contrast-normalised, so
        // that pair does not discriminate here. See the crate docs.
        let t = mars_tier().unwrap();
        let at = |lon: f64, lat: f64| {
            t.sample(lon.to_radians(), lat.to_radians())
                .unwrap()
                .weights()[0]
                .1
        };
        let solis = at(270.0, -26.0);
        let amazonis = at(200.0, 15.0);
        assert!(
            amazonis > 1.8 * solis,
            "Amazonis {amazonis} should be well above Solis Lacus {solis}"
        );
    }

    #[test]
    fn mars_polar_values_are_present_but_not_a_cap() {
        // Viking's polar coverage is sparse and the mosaic fills the gaps with
        // black. The generator excludes those from its bin means, so the poles
        // carry a plausible value rather than an encoded dark cap -- but they
        // are not a real seasonal cap either, and nothing should read them as
        // one. The cap belongs in the seasonal overlay of the plan, not here.
        let t = mars_tier().unwrap();
        let at = |lat: f64| t.sample(0.0, lat.to_radians()).unwrap().weights()[0].1 * 0.147;
        for lat in [-89.5, -87.0, 87.0] {
            let a = at(lat);
            assert!((0.05..0.45).contains(&a), "lat {lat} albedo {a}");
        }
    }

    #[test]
    fn mars_abundance_reproduces_a_plausible_albedo() {
        // Abundance x the endmember's band mean is the albedo the tier encodes.
        // Mars' disk-averaged geometric albedo is ~0.17, and no texel should be
        // outside a plausible range for a rocky surface.
        let t = mars_tier().unwrap();
        let band_mean = 0.147;
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for lon in (0..360).step_by(10) {
            for lat in (-80..=80).step_by(10) {
                let a = t
                    .sample((lon as f64).to_radians(), (lat as f64).to_radians())
                    .unwrap()
                    .weights()[0]
                    .1
                    * band_mean;
                lo = lo.min(a);
                hi = hi.max(a);
            }
        }
        assert!(lo > 0.02, "darkest albedo {lo}");
        assert!(hi < 0.55, "brightest albedo {hi}");
    }

    #[test]
    fn moon_tier_has_the_expected_shape() {
        let t = moon_tier().unwrap();
        assert_eq!(t.size(), (3600, 1800));
        assert_eq!(t.naif_id(), 301);
        assert_eq!(t.endmembers(), &[Endmember::FreshBasalt]);
        // The WAC mosaic is centred on the prime meridian, unlike the USGS Mars
        // mosaics which start at it. Same agency, same projection, different
        // origin -- which is why the grid is stored per product.
        assert_eq!(t.grid().lon0_deg, -180.0);
        assert_eq!(mars_tier().unwrap().grid().lon0_deg, 0.0);
    }

    #[test]
    fn lunar_maria_are_darker_than_the_highlands() {
        // The Moon's largest albedo feature, and the registration check: get
        // the longitude origin wrong and this inverts, with maria reading
        // brighter than highlands. That is exactly what happened on the first
        // build of this tier.
        let t = moon_tier().unwrap();
        let at = |lon: f64, lat: f64| {
            t.sample(lon.to_radians(), lat.to_radians())
                .unwrap()
                .weights()[0]
                .1
                * 0.102
        };
        let maria = [
            ("Serenitatis", at(17.5, 28.0)),
            ("Tranquillitatis", at(31.4, 8.5)),
            ("Procellarum", at(302.0, 18.4)),
        ];
        let highlands = [("southern", at(0.0, -60.0)), ("farside", at(180.0, 0.0))];
        for (mare_name, mare) in maria {
            for (hl_name, hl) in highlands {
                assert!(
                    hl > 2.0 * mare,
                    "{hl_name} highlands {hl} should be well above {mare_name} {mare}"
                );
            }
        }
    }

    #[test]
    fn tycho_is_brighter_than_the_surrounding_highlands() {
        // Fresh ray craters are the brightest lunar terrain; Tycho is the
        // canonical one, and it sits inside the southern highlands so this is a
        // local contrast rather than a mare/highland one.
        let t = moon_tier().unwrap();
        let at = |lon: f64, lat: f64| {
            t.sample(lon.to_radians(), lat.to_radians())
                .unwrap()
                .weights()[0]
                .1
                * 0.102
        };
        assert!(at(348.68, -43.31) > at(0.0, -60.0));
    }

    #[test]
    fn lunar_albedos_are_physical_where_there_is_data() {
        // Mare ~0.06-0.08, highlands ~0.11-0.16 in reality.
        //
        // WAC's swath seams are no-data and encode as an empty mix, so this
        // skips them rather than asserting a physical albedo for ground that
        // was never imaged. An empty mix is how the tier says "no data" — a
        // real surface always has some reflectance.
        let t = moon_tier().unwrap();
        let mut valid = 0;
        let mut empty = 0;
        // Restricted to +/-70. Poleward of that, near-black is genuinely
        // physical -- permanently shadowed crater floors receive no direct
        // sunlight at all -- and texels there mix real shadow with partial
        // seam coverage, so a brightness floor would be asserting something
        // false about the Moon rather than about the data.
        for lon in (0..360).step_by(5) {
            for lat in (-70..=70).step_by(5) {
                let mix = t
                    .sample((lon as f64).to_radians(), (lat as f64).to_radians())
                    .unwrap();
                match mix.weights().first() {
                    None => empty += 1,
                    Some((_, w)) => {
                        let a = w * 0.102;
                        assert!((0.01..0.50).contains(&a), "{lon},{lat} albedo {a}");
                        valid += 1;
                    }
                }
            }
        }
        assert!(valid > 0);
        // Seams are ~1% of area; a large jump would mean the no-data floor had
        // started eating real terrain.
        let frac = empty as f64 / (valid + empty) as f64;
        assert!(frac < 0.05, "no-data fraction {frac}");
    }

    #[test]
    fn rejects_bad_magic_and_truncation() {
        assert!(AbundanceTier::from_bytes(b"not a tier").is_err());
        let mut raw = Vec::new();
        flate2::read::GzDecoder::new(EARTH_TIER_GZ)
            .read_to_end(&mut raw)
            .unwrap();
        // Chop the payload: the declared length no longer matches.
        let short = &raw[..raw.len() - 1000];
        let err = AbundanceTier::from_bytes(short).unwrap_err().to_string();
        assert!(
            err.contains("payload") || err.contains("truncated"),
            "{err}"
        );
    }
}
