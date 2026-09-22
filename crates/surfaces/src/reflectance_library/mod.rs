//! Endmember surface reflectance spectra for resolved-body rendering.
//!
//! A global albedo map is one number per texel, but a real surface is a mixture
//! of a few materials whose *colours* differ. Multiplying a monochrome map by a
//! single disk-integrated spectrum gives a body whose colour is uniform and only
//! its brightness varies — visibly wrong for Mars and the Moon, the two bodies
//! most likely to be rendered large. Decomposing each texel into a few
//! endmembers with distinct spectra is what makes colour vary across the disk.
//!
//! This crate supplies the endmember spectra. The mixing fractions come from the
//! map crate; the radiometry stays with the consumer.
//!
//! # Source
//!
//! USGS Spectral Library Version 7 (Kokaly et al. 2017, USGS Data Series 1035,
//! doi:10.5066/F7RR1WDJ), a public-domain laboratory-measured library.
//! Reflectance is AREF (absolute reflectance), dimensionless.
//! `scripts/build_table.py` regenerates the embedded table from the archive and
//! records the curation.
//!
//! # Coverage is per-endmember and is not hidden
//!
//! The endmembers do not share a wavelength range: splib07 measured them on
//! different spectrometers, and deleted channels it could not calibrate. Most
//! span the silicon detector range, but [`Endmember::WaterIce`] **starts at
//! 859 nm** — splib07's only water-ice spectrum has no visible coverage at all.
//!
//! Rather than pad that gap, every accessor reports honestly: asking for a
//! visible-band mean of water ice returns `None`. A consumer that needs visible
//! ice must source it elsewhere, and this crate makes that obvious at the call
//! site instead of returning a fabricated number.
//!
//! # Example
//!
//! ```
//! use starfield_surfaces::reflectance_library::{Endmember, EndmemberMix, ReflectanceLibrary};
//!
//! let library = ReflectanceLibrary::load_embedded()?;
//!
//! // Vegetation's red edge: a steep rise between 680 and 750 nm.
//! let leaf = library.get(Endmember::GreenVegetation).unwrap();
//! let ratio = leaf.at_nm(750.0).unwrap() / leaf.at_nm(680.0).unwrap();
//! assert!(ratio > 4.0, "red edge ratio {ratio}");
//!
//! // A half-snow, half-vegetation texel, averaged over a red band.
//! let mix = EndmemberMix::new(vec![
//!     (Endmember::Snow, 0.5),
//!     (Endmember::GreenVegetation, 0.5),
//! ]);
//! let mean = mix.band_mean(&library, 600.0, 700.0).unwrap();
//! println!("mixed reflectance over 600-700 nm: {mean:.3}");
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

use std::collections::HashMap;

use starfield_core::data::source_utils::SampledCurve;
use starfield_core::{Result, StarfieldError};

const EMBEDDED_CSV: &str = include_str!("../../data/reflectance-library/splib07_endmembers.csv");

/// A material whose reflectance spectrum this crate ships.
///
/// This enum is the stable identifier shared between the reflectance library and
/// the map crate: a map stores mixing fractions keyed on these variants, and
/// resolves them here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Endmember {
    /// Open-ocean seawater, low chlorophyll.
    OpenOcean,
    /// Coastal seawater with chlorophyll.
    CoastalWater,
    /// Green aspen leaf — the canonical red-edge spectrum.
    GreenVegetation,
    /// Dry golden grass; non-photosynthetic vegetation.
    DryVegetation,
    /// Uncontaminated beach sand.
    Sand,
    /// Melting snow. splib07 has no dry or fresh snow spectrum, so this is
    /// darker in the near-infrared than fresh powder would be.
    Snow,
    /// Water ice at 77 K. **Covers 859 nm and longward only** — see the module
    /// docs.
    WaterIce,
    /// Fresh basalt; the closest laboratory analogue in this library to lunar
    /// mare and Mars dark terrain.
    FreshBasalt,
    /// Weathered basalt.
    WeatheredBasalt,
    /// Dried playa mud — the reddened arid surface for barren terrain.
    ///
    /// Iron-oxide reddening makes arid soil markedly redder than beach sand:
    /// its 700/450 nm ratio is 2.1 against Sand's 1.5. Barren terrain is a
    /// large fraction of Earth's illuminated disk, so using quartz beach sand
    /// there would bias the integrated colour blue.
    ///
    /// splib07 has **no bulk desert-soil spectrum** — its soils chapter is
    /// mineral mixtures, and the entries that redden more strongly are pure
    /// minerals rather than field-collected surfaces. This is a real arid field
    /// sample and is the closest honest choice in this archive.
    AridSoil,
    /// Old black asphalt road — the dark half of an urban surface.
    Asphalt,
    /// Photometric shade: identically zero reflectance at every wavelength.
    ///
    /// The standard shade endmember of spectral mixture analysis, and the only
    /// synthetic member of this enum — it has no laboratory spectrum because it
    /// is not a material. It accounts for sub-texel structure that removes light
    /// without changing its colour: canopy shadowing, gaps, and multiple
    /// scattering into the surface.
    ///
    /// It is not a nicety. splib07's vegetation spectra are **leaf-level**, and
    /// a leaf is far brighter than a canopy: pure [`Endmember::GreenVegetation`]
    /// integrates to a shortwave albedo of 0.218, while published forest
    /// class-means are 0.10–0.15. Without a shade component the vegetated
    /// surfaces of Earth cannot be made to match measured albedos at all.
    ///
    /// Because its reflectance is zero everywhere, it has no wavelength range
    /// of its own and never restricts a band a mix can be evaluated over.
    Shade,
    /// Light grey concrete road — the bright half of an urban surface.
    ///
    /// Urban land cover is strongly bimodal in brightness, so asphalt and
    /// concrete are separate endmembers rather than one "urban" average that
    /// would represent neither.
    Concrete,
}

impl Endmember {
    /// Every endmember in the library.
    pub const ALL: [Endmember; 13] = [
        Endmember::OpenOcean,
        Endmember::CoastalWater,
        Endmember::GreenVegetation,
        Endmember::DryVegetation,
        Endmember::Sand,
        Endmember::Snow,
        Endmember::WaterIce,
        Endmember::FreshBasalt,
        Endmember::WeatheredBasalt,
        Endmember::AridSoil,
        Endmember::Asphalt,
        Endmember::Concrete,
        Endmember::Shade,
    ];

    /// The identifier used in the embedded table and on the wire.
    pub fn id(&self) -> &'static str {
        match self {
            Endmember::OpenOcean => "OpenOcean",
            Endmember::CoastalWater => "CoastalWater",
            Endmember::GreenVegetation => "GreenVegetation",
            Endmember::DryVegetation => "DryVegetation",
            Endmember::Sand => "Sand",
            Endmember::Snow => "Snow",
            Endmember::WaterIce => "WaterIce",
            Endmember::FreshBasalt => "FreshBasalt",
            Endmember::WeatheredBasalt => "WeatheredBasalt",
            Endmember::AridSoil => "AridSoil",
            Endmember::Shade => "Shade",
            Endmember::Asphalt => "Asphalt",
            Endmember::Concrete => "Concrete",
        }
    }

    /// Parse an [`Endmember::id`].
    pub fn from_id(id: &str) -> Option<Endmember> {
        Endmember::ALL.into_iter().find(|e| e.id() == id)
    }
}

/// One endmember's measured reflectance against wavelength.
#[derive(Debug, Clone, PartialEq)]
pub struct Reflectance {
    endmember: Endmember,
    curve: SampledCurve,
}

impl Reflectance {
    /// Which material this describes.
    pub fn endmember(&self) -> Endmember {
        self.endmember
    }

    /// Inclusive wavelength coverage, nm. Differs per endmember.
    pub fn range_nm(&self) -> (f64, f64) {
        self.curve.range_nm()
    }

    /// Number of measured channels.
    pub fn len(&self) -> usize {
        self.curve.len()
    }

    /// Always false — a parsed reflectance has at least one channel.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Reflectance at `nm`, linearly interpolated. `None` outside coverage.
    pub fn at_nm(&self, nm: f64) -> Option<f64> {
        self.curve.at(nm)
    }

    /// Mean reflectance over `[lo_nm, hi_nm]`. `None` unless wholly covered.
    ///
    /// Call this once per band per endmember and reuse the result: the
    /// per-sub-sample cost then reduces to a weighted sum with no spectral work.
    pub fn band_mean(&self, lo_nm: f64, hi_nm: f64) -> Option<f64> {
        self.curve.mean_over(lo_nm, hi_nm)
    }

    /// The measured samples, as `(wavelength_nm, reflectance)`.
    pub fn samples(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.curve.samples()
    }

    /// The underlying curve, for consumers sharing one integration path.
    pub fn curve(&self) -> &SampledCurve {
        &self.curve
    }
}

/// The embedded endmember spectra.
#[derive(Debug, Clone)]
pub struct ReflectanceLibrary {
    spectra: HashMap<Endmember, Reflectance>,
}

impl ReflectanceLibrary {
    /// Parse the embedded table. No network, no I/O.
    pub fn load_embedded() -> Result<Self> {
        Self::parse(EMBEDDED_CSV)
    }

    /// Parse a table in the embedded CSV's format.
    pub fn parse(text: &str) -> Result<Self> {
        let mut columns: HashMap<Endmember, (Vec<f64>, Vec<f64>)> = HashMap::new();

        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("endmember") {
                continue;
            }
            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() != 3 {
                return Err(StarfieldError::DataError(format!(
                    "reflectance line {}: expected 3 columns, got {}",
                    line_no + 1,
                    fields.len()
                )));
            }
            let endmember = Endmember::from_id(fields[0]).ok_or_else(|| {
                StarfieldError::DataError(format!(
                    "reflectance line {}: unknown endmember {:?}",
                    line_no + 1,
                    fields[0]
                ))
            })?;
            let parse = |s: &str, what: &str| -> Result<f64> {
                s.trim().parse::<f64>().map_err(|e| {
                    StarfieldError::DataError(format!(
                        "reflectance line {}: bad {what}: {e}",
                        line_no + 1
                    ))
                })
            };
            let entry = columns.entry(endmember).or_default();
            entry.0.push(parse(fields[1], "wavelength")?);
            entry.1.push(parse(fields[2], "reflectance")?);
        }

        let mut spectra = HashMap::new();
        for (endmember, (wavelengths, values)) in columns {
            let curve = SampledCurve::new(wavelengths, values)
                .map_err(|e| StarfieldError::DataError(format!("{}: {e}", endmember.id())))?;
            spectra.insert(endmember, Reflectance { endmember, curve });
        }

        // Shade is synthetic: identically zero, spanning every wavelength any
        // real endmember covers, so it never narrows a mix's usable range.
        let (lo, hi) = spectra.values().fold((f64::MAX, f64::MIN), |(lo, hi), r| {
            let (a, b) = r.range_nm();
            (lo.min(a), hi.max(b))
        });
        if lo < hi {
            let curve = SampledCurve::new(vec![lo, hi], vec![0.0, 0.0])
                .map_err(|e| StarfieldError::DataError(format!("Shade: {e}")))?;
            spectra.insert(
                Endmember::Shade,
                Reflectance {
                    endmember: Endmember::Shade,
                    curve,
                },
            );
        }

        if spectra.is_empty() {
            return Err(StarfieldError::DataError(
                "reflectance library is empty".into(),
            ));
        }
        Ok(Self { spectra })
    }

    /// The spectrum for `endmember`, if the library has one.
    pub fn get(&self, endmember: Endmember) -> Option<&Reflectance> {
        self.spectra.get(&endmember)
    }

    /// How many endmembers are loaded.
    pub fn len(&self) -> usize {
        self.spectra.len()
    }

    /// Always false — a parsed library has at least one endmember.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Every loaded endmember, in a stable order.
    pub fn endmembers(&self) -> Vec<Endmember> {
        let mut ids: Vec<Endmember> = self.spectra.keys().copied().collect();
        ids.sort();
        ids
    }
}

/// A texel's material composition: endmembers and their fractional weights.
///
/// The map crate returns one of these per sample rather than a bare albedo, so
/// there is a single code path whether a body has one material or several. A
/// uniform surface is [`EndmemberMix::uniform`], a one-endmember mix.
#[derive(Debug, Clone, PartialEq)]
pub struct EndmemberMix {
    weights: Vec<(Endmember, f64)>,
}

impl EndmemberMix {
    /// Build from `(endmember, weight)` pairs. Weights are used as given and are
    /// not renormalised — see [`EndmemberMix::total_weight`].
    pub fn new(weights: Vec<(Endmember, f64)>) -> Self {
        Self { weights }
    }

    /// A pure surface of one material.
    pub fn uniform(endmember: Endmember) -> Self {
        Self {
            weights: vec![(endmember, 1.0)],
        }
    }

    /// The `(endmember, weight)` pairs.
    pub fn weights(&self) -> &[(Endmember, f64)] {
        &self.weights
    }

    /// Number of contributing endmembers.
    pub fn len(&self) -> usize {
        self.weights.len()
    }

    /// Whether the mix has no contributions.
    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    /// Sum of the weights.
    ///
    /// Not forced to 1. A mix covering only part of a texel — a partially
    /// cloud-covered pixel, say — is legitimately short, and silently
    /// renormalising it would hide the missing fraction.
    pub fn total_weight(&self) -> f64 {
        self.weights.iter().map(|(_, w)| w).sum()
    }

    /// Weighted reflectance at `nm`.
    ///
    /// `None` if any contributing endmember lacks coverage at `nm` — a mix
    /// evaluated over a partial subset of its materials is not the reflectance
    /// of the mix, and returning it would understate the result by however much
    /// was dropped.
    pub fn reflectance_at(&self, library: &ReflectanceLibrary, nm: f64) -> Option<f64> {
        let mut total = 0.0;
        for &(endmember, weight) in &self.weights {
            total += weight * library.get(endmember)?.at_nm(nm)?;
        }
        Some(total)
    }

    /// Weighted mean reflectance over `[lo_nm, hi_nm]`.
    ///
    /// `None` if any contributing endmember does not fully cover the band. This
    /// is the call to make once per band per render; the per-sub-sample loop then
    /// reduces to a weighted sum over precomputed values.
    pub fn band_mean(&self, library: &ReflectanceLibrary, lo_nm: f64, hi_nm: f64) -> Option<f64> {
        let mut total = 0.0;
        for &(endmember, weight) in &self.weights {
            total += weight * library.get(endmember)?.band_mean(lo_nm, hi_nm)?;
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> ReflectanceLibrary {
        ReflectanceLibrary::load_embedded().unwrap()
    }

    #[test]
    fn every_endmember_is_present() {
        let l = library();
        assert_eq!(l.len(), Endmember::ALL.len());
        for e in Endmember::ALL {
            assert!(l.get(e).is_some(), "{} missing", e.id());
        }
    }

    #[test]
    fn ids_round_trip() {
        for e in Endmember::ALL {
            assert_eq!(Endmember::from_id(e.id()), Some(e));
        }
        assert_eq!(Endmember::from_id("Unobtainium"), None);
    }

    #[test]
    fn reflectances_are_physical() {
        let l = library();
        for e in Endmember::ALL {
            for (nm, r) in l.get(e).unwrap().samples() {
                assert!(
                    (0.0..=1.0).contains(&r),
                    "{} reflectance {r} at {nm} nm is outside [0,1]",
                    e.id()
                );
            }
        }
    }

    #[test]
    fn vegetation_has_a_red_edge() {
        // The single most recognisable feature in terrestrial remote sensing:
        // chlorophyll absorbs hard in the red and the leaf scatters strongly in
        // the near-infrared, so reflectance jumps by a factor of several between
        // 680 and 750 nm. A wavelength-registration error would destroy this.
        let l = library();
        let leaf = l.get(Endmember::GreenVegetation).unwrap();
        let ratio = leaf.at_nm(750.0).unwrap() / leaf.at_nm(680.0).unwrap();
        assert!(ratio > 4.0, "red edge ratio {ratio}");
        // Dry vegetation has lost its chlorophyll and so has no red edge.
        let dry = l.get(Endmember::DryVegetation).unwrap();
        let dry_ratio = dry.at_nm(750.0).unwrap() / dry.at_nm(680.0).unwrap();
        assert!(dry_ratio < 2.0, "dry vegetation ratio {dry_ratio}");
    }

    #[test]
    fn snow_is_bright_in_the_visible_and_dark_in_the_swir() {
        let l = library();
        let snow = l.get(Endmember::Snow).unwrap();
        assert!(snow.at_nm(500.0).unwrap() > 0.7);
        assert!(snow.at_nm(1600.0).unwrap() < 0.1);
        assert!(snow.at_nm(500.0).unwrap() > 10.0 * snow.at_nm(1600.0).unwrap());
    }

    #[test]
    fn water_is_dark_and_blue() {
        let l = library();
        let ocean = l.get(Endmember::OpenOcean).unwrap();
        for nm in [450.0, 650.0, 850.0] {
            assert!(
                ocean.at_nm(nm).unwrap() < 0.1,
                "ocean too bright at {nm} nm"
            );
        }
        assert!(ocean.at_nm(450.0).unwrap() > ocean.at_nm(650.0).unwrap());
    }

    #[test]
    fn shade_is_zero_everywhere_and_never_narrows_a_band() {
        let l = library();
        let shade = l.get(Endmember::Shade).unwrap();
        for nm in [400.0, 550.0, 900.0, 2000.0] {
            assert_eq!(shade.at_nm(nm), Some(0.0), "at {nm} nm");
        }
        assert_eq!(shade.band_mean(400.0, 900.0), Some(0.0));
        // It spans at least as wide as any real endmember, so mixing it in
        // cannot make an otherwise-evaluable band return None.
        let (slo, shi) = shade.range_nm();
        for e in Endmember::ALL {
            let (lo, hi) = l.get(e).unwrap().range_nm();
            assert!(slo <= lo && shi >= hi, "{} exceeds Shade's range", e.id());
        }
    }

    #[test]
    fn shade_darkens_a_canopy_to_a_published_albedo() {
        // The reason Shade exists: leaf-level spectra are far brighter than
        // canopies. Pure green vegetation is well above published forest
        // class-means; 55/45 with shade lands inside them.
        let l = library();
        let leaf_only = EndmemberMix::uniform(Endmember::GreenVegetation);
        let canopy = EndmemberMix::new(vec![
            (Endmember::GreenVegetation, 0.55),
            (Endmember::Shade, 0.45),
        ]);
        let bare = leaf_only.band_mean(&l, 700.0, 900.0).unwrap();
        let shaded = canopy.band_mean(&l, 700.0, 900.0).unwrap();
        assert!((shaded - 0.55 * bare).abs() < 1e-12);
        assert!(shaded < bare);
    }

    #[test]
    fn arid_soil_is_redder_than_beach_sand() {
        // The distinction that matters for Earth's integrated colour: barren
        // terrain is iron-oxide reddened, beach quartz sand is not, and barren
        // is a large fraction of the illuminated disk.
        let l = library();
        let arid = l.get(Endmember::AridSoil).unwrap();
        let sand = l.get(Endmember::Sand).unwrap();
        let arid_slope = arid.at_nm(700.0).unwrap() / arid.at_nm(450.0).unwrap();
        let sand_slope = sand.at_nm(700.0).unwrap() / sand.at_nm(450.0).unwrap();
        assert!(
            arid_slope > 1.3 * sand_slope,
            "arid {arid_slope} vs sand {sand_slope}"
        );
        assert!(arid_slope > 2.0, "arid soil slope {arid_slope}");
    }

    #[test]
    fn urban_endmembers_bracket_a_real_city_in_brightness() {
        // Urban cover is bimodal: asphalt is among the darkest natural or
        // artificial surfaces, concrete among the brighter. Averaging them into
        // one "urban" endmember would represent neither, which is why they are
        // separate.
        let l = library();
        let asphalt = l.get(Endmember::Asphalt).unwrap().at_nm(600.0).unwrap();
        let concrete = l.get(Endmember::Concrete).unwrap().at_nm(600.0).unwrap();
        assert!(asphalt < 0.15, "asphalt reflectance {asphalt}");
        assert!(concrete > 2.0 * asphalt, "{concrete} vs {asphalt}");
        // Both are spectrally flatter than vegetation: no red edge.
        for e in [Endmember::Asphalt, Endmember::Concrete] {
            let r = l.get(e).unwrap();
            let edge = r.at_nm(750.0).unwrap() / r.at_nm(680.0).unwrap();
            assert!(edge < 1.5, "{} has a red edge: {edge}", e.id());
        }
    }

    #[test]
    fn basalt_is_dark_and_spectrally_flat() {
        let l = library();
        let basalt = l.get(Endmember::FreshBasalt).unwrap();
        let v = basalt.at_nm(550.0).unwrap();
        let nir = basalt.at_nm(900.0).unwrap();
        assert!(v < 0.2, "fresh basalt visible reflectance {v}");
        assert!((nir / v) < 2.0, "basalt is not flat: {nir} vs {v}");
        // Weathering brightens it.
        let weathered = l.get(Endmember::WeatheredBasalt).unwrap();
        assert!(weathered.at_nm(550.0).unwrap() > v);
    }

    #[test]
    fn water_ice_has_no_visible_coverage_and_says_so() {
        // splib07's only water-ice spectrum starts at 859 nm. The library does
        // not pad the gap; it reports the absence.
        let l = library();
        let ice = l.get(Endmember::WaterIce).unwrap();
        let (lo, _) = ice.range_nm();
        assert!(lo > 800.0, "water ice unexpectedly starts at {lo} nm");
        assert_eq!(ice.at_nm(550.0), None);
        assert_eq!(ice.band_mean(400.0, 700.0), None);
        // It is usable where it does have coverage.
        assert!(ice.band_mean(1000.0, 1100.0).is_some());
    }

    #[test]
    fn most_endmembers_span_the_silicon_response() {
        let l = library();
        for e in Endmember::ALL {
            if e == Endmember::WaterIce {
                continue;
            }
            let r = l.get(e).unwrap();
            assert!(
                r.band_mean(400.0, 900.0).is_some(),
                "{} does not cover 400-900 nm; range is {:?}",
                e.id(),
                r.range_nm()
            );
        }
    }

    #[test]
    fn a_uniform_mix_equals_its_endmember() {
        let l = library();
        let mix = EndmemberMix::uniform(Endmember::Sand);
        let direct = l.get(Endmember::Sand).unwrap().at_nm(600.0).unwrap();
        assert!((mix.reflectance_at(&l, 600.0).unwrap() - direct).abs() < 1e-12);
        assert_eq!(mix.total_weight(), 1.0);
    }

    #[test]
    fn a_mix_lies_between_its_components() {
        let l = library();
        let snow = l.get(Endmember::Snow).unwrap().at_nm(600.0).unwrap();
        let basalt = l.get(Endmember::FreshBasalt).unwrap().at_nm(600.0).unwrap();
        let mix = EndmemberMix::new(vec![(Endmember::Snow, 0.5), (Endmember::FreshBasalt, 0.5)]);
        let value = mix.reflectance_at(&l, 600.0).unwrap();
        assert!(
            value > basalt && value < snow,
            "{basalt} < {value} < {snow}"
        );
        assert!((value - 0.5 * (snow + basalt)).abs() < 1e-12);
    }

    #[test]
    fn a_mix_is_none_if_any_component_lacks_coverage() {
        // Half sand, half water ice, evaluated in the visible: sand has data and
        // ice does not, so the mix is not evaluable. Returning sand's half alone
        // would understate the result by whatever the ice contributes.
        let l = library();
        let mix = EndmemberMix::new(vec![(Endmember::Sand, 0.5), (Endmember::WaterIce, 0.5)]);
        assert_eq!(mix.reflectance_at(&l, 550.0), None);
        assert_eq!(mix.band_mean(&l, 400.0, 700.0), None);
    }

    #[test]
    fn partial_weights_are_not_renormalised() {
        let l = library();
        let mix = EndmemberMix::new(vec![(Endmember::Sand, 0.25)]);
        assert_eq!(mix.total_weight(), 0.25);
        let direct = l.get(Endmember::Sand).unwrap().at_nm(600.0).unwrap();
        assert!((mix.reflectance_at(&l, 600.0).unwrap() - 0.25 * direct).abs() < 1e-12);
    }

    #[test]
    fn parse_rejects_unknown_endmembers_and_short_rows() {
        let bad = "endmember,wavelength_nm,reflectance\nUnobtainium,500,0.5\n";
        assert!(ReflectanceLibrary::parse(bad)
            .unwrap_err()
            .to_string()
            .contains("unknown endmember"));
        let short = "endmember,wavelength_nm,reflectance\nSand,500\n";
        assert!(ReflectanceLibrary::parse(short)
            .unwrap_err()
            .to_string()
            .contains("expected 3 columns"));
    }

    #[test]
    fn parse_rejects_an_empty_table() {
        assert!(ReflectanceLibrary::parse("# nothing here\n").is_err());
    }
}
