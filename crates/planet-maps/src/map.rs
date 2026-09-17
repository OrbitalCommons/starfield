//! Raster albedo storage, mip pyramid, and footprint-aware sampling.

use starfield_reflectance_library::{Endmember, EndmemberMix};

use crate::grid::MapGrid;

/// Smallest emission cosine used when projecting a sky footprint onto the
/// surface.
///
/// The projection divides by `mu`, which is singular at the limb. Clamping at
/// `cos 87°` bounds the surface footprint at ~19× the sky footprint. Beyond
/// that the geometry is grazing enough that no map resolution is meaningful,
/// and an unclamped value would select a level past the top of the pyramid.
pub const MU_FLOOR: f64 = 0.05;

/// The photometric band a map's albedo values were measured in.
///
/// Required, not optional. A scalar mosaic is a reflectance *in some band* —
/// LROC WAC is monochrome at 643 nm — and converting it into an
/// [`EndmemberMix`] means matching that number against the endmember's
/// reflectance over the same band. Without the band recorded, the conversion
/// has no defined meaning and any colour derived from it is arbitrary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhotometricBand {
    /// Lower bound, nm.
    pub lo_nm: f64,
    /// Upper bound, nm.
    pub hi_nm: f64,
}

impl PhotometricBand {
    /// A band from explicit bounds.
    pub fn new(lo_nm: f64, hi_nm: f64) -> Self {
        Self { lo_nm, hi_nm }
    }

    /// A narrow band centred on a filter's effective wavelength.
    ///
    /// Monochrome mosaics quote one wavelength; this gives it the small finite
    /// width the band-mean machinery needs.
    pub fn monochrome(centre_nm: f64, width_nm: f64) -> Self {
        Self {
            lo_nm: centre_nm - 0.5 * width_nm,
            hi_nm: centre_nm + 0.5 * width_nm,
        }
    }
}

/// One resolution level of the pyramid.
#[derive(Debug, Clone)]
struct Level {
    width: usize,
    height: usize,
    /// Row-major, `height * width` normal-albedo values.
    data: Vec<f32>,
}

impl Level {
    /// Bilinear sample at fractional coordinates, wrapping in `u` and clamping in `v`.
    fn sample(&self, u: f64, v: f64) -> f64 {
        let x = u * self.width as f64 - 0.5;
        let y = (v * self.height as f64 - 0.5).clamp(0.0, self.height as f64 - 1.0);
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;

        let wrap = |i: i64| -> usize { i.rem_euclid(self.width as i64) as usize };
        let clamp_row = |j: i64| -> usize { j.clamp(0, self.height as i64 - 1) as usize };

        let (x0i, x1i) = (wrap(x0 as i64), wrap(x0 as i64 + 1));
        let (y0i, y1i) = (clamp_row(y0 as i64), clamp_row(y0 as i64 + 1));

        let at = |r: usize, c: usize| self.data[r * self.width + c] as f64;
        let top = at(y0i, x0i) * (1.0 - fx) + at(y0i, x1i) * fx;
        let bottom = at(y1i, x0i) * (1.0 - fx) + at(y1i, x1i) * fx;
        top * (1.0 - fy) + bottom * fy
    }
}

/// How a sample's resolution level was chosen. Exposed so consumers can assert
/// on it — notably at the terminator, where foreshortening is strongest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleFootprint {
    /// Pyramid level used. 0 is full resolution.
    pub level: usize,
    /// Surface footprint radius in texels of level 0, along the stretched axis.
    pub texels: f64,
    /// Ratio of the stretched axis to the unstretched one — 1.0 face-on,
    /// growing towards the limb.
    pub anisotropy: f64,
    /// Whether [`MU_FLOOR`] clamped the projection.
    pub mu_clamped: bool,
}

/// An equirectangular normal-albedo map in body-fixed coordinates.
#[derive(Debug, Clone)]
pub struct AlbedoMap {
    grid: MapGrid,
    band: PhotometricBand,
    endmember: Endmember,
    endmember_band_mean: f64,
    levels: Vec<Level>,
}

impl AlbedoMap {
    /// Build from a row-major albedo raster, generating the mip pyramid.
    ///
    /// `endmember` is the material this body's surface is treated as; the
    /// scalar albedo becomes that endmember's abundance (see
    /// [`AlbedoMap::sample_area`]).
    ///
    /// Returns `None` if the dimensions are zero or do not match `data`.
    pub fn new(
        grid: MapGrid,
        band: PhotometricBand,
        endmember: Endmember,
        endmember_band_mean: f64,
        width: usize,
        height: usize,
        data: Vec<f32>,
    ) -> Option<Self> {
        if width == 0 || height == 0 || data.len() != width * height {
            return None;
        }
        // Taken once here rather than per sample: it is a property of
        // (endmember, band), both of which this map already knows, and the
        // consumer wanted one reflectance-library call per render rather than
        // one per sub-sample. Rejecting it here also means `sample_area` cannot
        // fail for a reason unrelated to the position it was asked about.
        if !endmember_band_mean.is_finite() || endmember_band_mean <= 0.0 {
            return None;
        }
        let mut levels = vec![Level {
            width,
            height,
            data,
        }];
        while levels.last().unwrap().width > 1 || levels.last().unwrap().height > 1 {
            levels.push(downsample(levels.last().unwrap()));
        }
        Some(Self {
            grid,
            band,
            endmember,
            endmember_band_mean,
            levels,
        })
    }

    /// The stored product's coordinate conventions.
    pub fn grid(&self) -> &MapGrid {
        &self.grid
    }

    /// The band the albedo values were measured in.
    pub fn band(&self) -> PhotometricBand {
        self.band
    }

    /// The endmember this map's albedo is expressed against.
    pub fn endmember(&self) -> Endmember {
        self.endmember
    }

    /// The endmember's mean reflectance over [`AlbedoMap::band`], as supplied
    /// at construction.
    pub fn endmember_band_mean(&self) -> f64 {
        self.endmember_band_mean
    }

    /// Number of pyramid levels; level 0 is full resolution.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Dimensions of a level, or `None` if it does not exist.
    pub fn level_size(&self, level: usize) -> Option<(usize, usize)> {
        self.levels.get(level).map(|l| (l.width, l.height))
    }

    /// Point-sample the full-resolution albedo at a body-fixed position.
    ///
    /// Input is **east-positive planetocentric radians**. Use this only when the
    /// sample footprint is known to be sub-texel; otherwise
    /// [`AlbedoMap::sample_area`] avoids aliasing.
    pub fn sample_body_fixed(&self, lon_rad: f64, lat_rad: f64) -> Option<f64> {
        let (u, v) = self.grid.body_fixed_to_uv(lon_rad, lat_rad)?;
        Some(self.levels[0].sample(u, v))
    }

    /// Choose a pyramid level for a sky footprint at emission cosine `mu`.
    ///
    /// The sky footprint projects onto the surface stretched by `1/mu` along the
    /// emission direction and essentially unstretched across it. This selects
    /// from the **stretched** axis, which never aliases but over-blurs across
    /// the minor axis near the limb; true anisotropic filtering would sharpen
    /// that and is a deliberate follow-up rather than something to fake with a
    /// scalar level. `mu` is clamped at [`MU_FLOOR`].
    pub fn select_level(&self, sky_radius_rad: f64, mu: f64) -> Option<SampleFootprint> {
        if !sky_radius_rad.is_finite() || sky_radius_rad <= 0.0 || !mu.is_finite() {
            return None;
        }
        let mu_clamped = mu < MU_FLOOR;
        let mu_eff = mu.max(MU_FLOOR);
        if mu_eff > 1.0 {
            return None;
        }

        // Texels per radian of body-fixed longitude at level 0.
        let texels_per_rad = self.levels[0].width as f64 / (2.0 * std::f64::consts::PI);
        let texels = sky_radius_rad / mu_eff * texels_per_rad;
        let anisotropy = 1.0 / mu_eff;

        // A footprint spanning n texels is resolved by level log2(n).
        let level = if texels <= 1.0 {
            0
        } else {
            (texels.log2().floor() as usize).min(self.levels.len() - 1)
        };

        Some(SampleFootprint {
            level,
            texels,
            anisotropy,
            mu_clamped,
        })
    }

    /// Sample the surface composition over a footprint.
    ///
    /// `lon_rad`/`lat_rad` are east-positive planetocentric radians;
    /// `sky_radius_rad` is the sub-sample's angular radius **on the sky**, not
    /// projected onto the surface; `mu` is the emission cosine. The projection
    /// happens here so that its clamp and the level choice stay together — see
    /// [`AlbedoMap::select_level`].
    ///
    /// The returned mix expresses the sampled albedo as an abundance of this
    /// map's endmember, defined so that the mix's reflectance averaged over
    /// [`AlbedoMap::band`] reproduces the sampled albedo.
    ///
    /// A scalar map is a one-endmember approximation: brightness varies across
    /// the disk but colour does not. Where an [`crate::AbundanceTier`] exists
    /// for a body, prefer it — it carries real per-texel composition.
    pub fn sample_area(
        &self,
        lon_rad: f64,
        lat_rad: f64,
        sky_radius_rad: f64,
        mu: f64,
    ) -> Option<(EndmemberMix, SampleFootprint)> {
        let (u, v) = self.grid.body_fixed_to_uv(lon_rad, lat_rad)?;
        let footprint = self.select_level(sky_radius_rad, mu)?;
        let albedo = self.levels[footprint.level].sample(u, v);
        let abundance = albedo / self.endmember_band_mean;
        Some((
            EndmemberMix::new(vec![(self.endmember, abundance)]),
            footprint,
        ))
    }
}

/// One interface for sampling a body's surface composition, whether it is
/// backed by a scalar albedo map or by real per-texel endmember abundances.
///
/// A consumer should not have to branch on which a body happens to have: the
/// difference is data coverage, not physics. `AlbedoMap` returns a
/// one-endmember mix; `AbundanceTier` returns however many endmembers the
/// texel actually contains.
pub trait SurfaceSampler {
    /// Composition over a footprint, and how the resolution level was chosen.
    ///
    /// `lon_rad`/`lat_rad` are east-positive planetocentric radians,
    /// `sky_radius_rad` is the **sky** footprint radius, `mu` the emission
    /// cosine.
    fn sample_area(
        &self,
        lon_rad: f64,
        lat_rad: f64,
        sky_radius_rad: f64,
        mu: f64,
    ) -> Option<(EndmemberMix, SampleFootprint)>;

    /// Every endmember this surface can return, so a consumer can precompute
    /// band means once per render.
    fn endmembers(&self) -> Vec<Endmember>;
}

impl SurfaceSampler for AlbedoMap {
    fn sample_area(
        &self,
        lon_rad: f64,
        lat_rad: f64,
        sky_radius_rad: f64,
        mu: f64,
    ) -> Option<(EndmemberMix, SampleFootprint)> {
        AlbedoMap::sample_area(self, lon_rad, lat_rad, sky_radius_rad, mu)
    }

    fn endmembers(&self) -> Vec<Endmember> {
        vec![self.endmember]
    }
}

/// Box-filter one level to half size, rounding dimensions up so a 1-pixel axis
/// stays 1.
fn downsample(level: &Level) -> Level {
    let width = (level.width / 2).max(1);
    let height = (level.height / 2).max(1);
    let mut data = vec![0f32; width * height];
    for row in 0..height {
        for col in 0..width {
            let mut sum = 0.0f64;
            let mut count = 0.0f64;
            for dy in 0..2 {
                for dx in 0..2 {
                    let sr = (row * 2 + dy).min(level.height - 1);
                    let sc = (col * 2 + dx).min(level.width - 1);
                    sum += level.data[sr * level.width + sc] as f64;
                    count += 1.0;
                }
            }
            data[row * width + col] = (sum / count) as f32;
        }
    }
    Level {
        width,
        height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::MapGrid;

    fn uniform_map(width: usize, height: usize, value: f32) -> AlbedoMap {
        AlbedoMap::new(
            MapGrid::usgs_default(),
            PhotometricBand::monochrome(643.0, 20.0),
            Endmember::FreshBasalt,
            0.12,
            width,
            height,
            vec![value; width * height],
        )
        .unwrap()
    }

    #[test]
    fn rejects_mismatched_dimensions() {
        let g = MapGrid::usgs_default();
        let b = PhotometricBand::new(400.0, 700.0);
        assert!(AlbedoMap::new(g, b, Endmember::Sand, 0.1, 4, 4, vec![0.0; 15]).is_none());
        assert!(AlbedoMap::new(g, b, Endmember::Sand, 0.1, 0, 4, vec![]).is_none());
        // A non-positive band mean is rejected at construction now.
        assert!(AlbedoMap::new(g, b, Endmember::Sand, 0.0, 4, 4, vec![0.0; 16]).is_none());
        assert!(AlbedoMap::new(g, b, Endmember::Sand, f64::NAN, 4, 4, vec![0.0; 16]).is_none());
    }

    #[test]
    fn pyramid_halves_down_to_one_texel() {
        let m = uniform_map(64, 32, 0.2);
        assert_eq!(m.level_size(0), Some((64, 32)));
        assert_eq!(m.level_size(1), Some((32, 16)));
        assert_eq!(m.level_size(5), Some((2, 1)));
        assert_eq!(m.level_size(6), Some((1, 1)));
        assert_eq!(m.levels(), 7);
        assert_eq!(m.level_size(7), None);
    }

    #[test]
    fn a_uniform_map_samples_to_its_value_at_every_level() {
        // Box-filtering a constant must preserve it, or the pyramid is biased.
        let m = uniform_map(64, 32, 0.37);
        for level in 0..m.levels() {
            let s = m.levels[level].sample(0.3, 0.6);
            assert!((s - 0.37).abs() < 1e-6, "level {level} gave {s}");
        }
    }

    #[test]
    fn downsampling_preserves_the_mean() {
        // A gradient, so the filter has something to average.
        let (w, h) = (32usize, 16usize);
        let data: Vec<f32> = (0..w * h).map(|i| (i % w) as f32 / w as f32).collect();
        let m = AlbedoMap::new(
            MapGrid::usgs_default(),
            PhotometricBand::new(400.0, 700.0),
            Endmember::Sand,
            0.29,
            w,
            h,
            data.clone(),
        )
        .unwrap();
        let mean0: f64 = data.iter().map(|&x| x as f64).sum::<f64>() / data.len() as f64;
        for level in 0..m.levels() {
            let l = &m.levels[level];
            let mean: f64 = l.data.iter().map(|&x| x as f64).sum::<f64>() / l.data.len() as f64;
            assert!((mean - mean0).abs() < 1e-6, "level {level} mean {mean}");
        }
    }

    #[test]
    fn longitude_wraps_when_sampling() {
        // A map with a sharp seam: sampling either side of longitude 0 must
        // interpolate across the wrap rather than clamp.
        let (w, h) = (8usize, 4usize);
        let mut data = vec![0.0f32; w * h];
        for row in 0..h {
            data[row * w] = 1.0; // column 0 bright
        }
        let m = AlbedoMap::new(
            MapGrid::usgs_default(),
            PhotometricBand::new(400.0, 700.0),
            Endmember::Sand,
            0.29,
            w,
            h,
            data,
        )
        .unwrap();
        // Just west of the prime meridian is in the last column, which is dark,
        // but bilinear interpolation must pull in column 0's brightness.
        let just_west = m.sample_body_fixed((-1f64).to_radians(), 0.0).unwrap();
        assert!(just_west > 0.0, "wrap did not blend: {just_west}");
    }

    #[test]
    fn a_finer_footprint_selects_a_finer_level() {
        let m = uniform_map(2048, 1024, 0.1);
        // texels_per_rad = 2048 / 2pi = 326
        let tiny = m.select_level(1e-5, 1.0).unwrap();
        let coarse = m.select_level(1e-2, 1.0).unwrap();
        assert_eq!(tiny.level, 0);
        assert!(coarse.level > tiny.level, "{coarse:?} vs {tiny:?}");
        assert!(coarse.texels > tiny.texels);
    }

    #[test]
    fn foreshortening_coarsens_the_level_toward_the_limb() {
        let m = uniform_map(2048, 1024, 0.1);
        // 2048 texels over 2pi rad is 326 texels/rad, so 1e-2 rad spans ~3
        // texels face-on and ~16 at mu = 0.2 - far enough apart to separate
        // levels. A smaller footprint would sit under one texel at both.
        let face_on = m.select_level(1e-2, 1.0).unwrap();
        let oblique = m.select_level(1e-2, 0.2).unwrap();
        assert!(
            oblique.level > face_on.level,
            "limb {oblique:?} should be coarser than face-on {face_on:?}"
        );
        assert!((face_on.anisotropy - 1.0).abs() < 1e-12);
        assert!((oblique.anisotropy - 5.0).abs() < 1e-12);
        assert!(!face_on.mu_clamped && !oblique.mu_clamped);
    }

    #[test]
    fn grazing_emission_is_clamped_and_says_so() {
        // The singularity this clamp exists for: mu -> 0 would send the
        // footprint to infinity and the level past the top of the pyramid.
        let m = uniform_map(2048, 1024, 0.1);
        let grazing = m.select_level(1e-3, 1e-9).unwrap();
        assert!(grazing.mu_clamped);
        assert!((grazing.anisotropy - 1.0 / MU_FLOOR).abs() < 1e-9);
        assert!(grazing.level < m.levels());
        // And exactly at the floor it is not reported as clamped.
        assert!(!m.select_level(1e-3, MU_FLOOR).unwrap().mu_clamped);
    }

    #[test]
    fn level_never_runs_past_the_pyramid() {
        let m = uniform_map(64, 32, 0.1);
        let huge = m.select_level(3.0, MU_FLOOR).unwrap();
        assert!(huge.level < m.levels());
    }

    #[test]
    fn select_level_rejects_nonsense() {
        let m = uniform_map(64, 32, 0.1);
        assert!(m.select_level(0.0, 1.0).is_none());
        assert!(m.select_level(-1e-3, 1.0).is_none());
        assert!(m.select_level(f64::NAN, 1.0).is_none());
        assert!(m.select_level(1e-3, f64::NAN).is_none());
        assert!(m.select_level(1e-3, 1.5).is_none());
    }

    #[test]
    fn sampled_mix_reproduces_the_albedo_over_the_maps_band() {
        // The contract that makes the scalar-to-mix conversion meaningful:
        // mix reflectance averaged over the map's band equals the map albedo.
        let albedo = 0.24f32;
        let m = uniform_map(64, 32, albedo);
        let endmember_band_mean = m.endmember_band_mean();
        let (mix, _) = m.sample_area(0.5, 0.1, 1e-4, 1.0).unwrap();
        assert_eq!(mix.len(), 1);
        let (e, w) = mix.weights()[0];
        assert_eq!(e, Endmember::FreshBasalt);
        // reflectance = w * endmember_band_mean, and that must be the albedo.
        assert!((w * endmember_band_mean - albedo as f64).abs() < 1e-6);
    }

    #[test]
    fn sample_area_rejects_non_finite_positions() {
        let m = uniform_map(64, 32, 0.2);
        assert!(m.sample_area(f64::NAN, 0.0, 1e-4, 1.0).is_none());
    }

    #[test]
    fn band_helpers_are_consistent() {
        let b = PhotometricBand::monochrome(643.0, 20.0);
        assert_eq!(b.lo_nm, 633.0);
        assert_eq!(b.hi_nm, 653.0);
    }
}
