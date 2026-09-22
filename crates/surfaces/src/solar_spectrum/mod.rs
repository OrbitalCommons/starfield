//! The solar spectrum at 1 nm resolution, for synthetic photometry.
//!
//! Embeds the **TSIS-1 Hybrid Solar Reference Spectrum (HSRS) version 2**
//! (Coddington et al. 2023, *Earth and Space Science*, doi:10.1029/2022EA002637),
//! the recognised international reference standard, resampled to 1 nm box means
//! over its full 202–2730 nm coverage.
//!
//! This is the illumination half of rendering a resolved solar-system body: a
//! reflectance or geometric albedo is dimensionless, and becomes photons at a
//! detector only once multiplied by the solar spectrum and the inverse-square
//! distance factors.
//!
//! # Box means, not point samples
//!
//! Every value is the mean of the native spectrum over `[n, n+1)` nm, labelled
//! by the bin's lower edge. [`SolarSpectrum::at_nm`] therefore returns the value
//! of the *bin containing* the requested wavelength; it does not interpolate,
//! because interpolating between box means of a line-blanketed spectrum implies
//! a resolution the data does not have.
//!
//! The resampling is done from the archive's native 0.001–0.01 nm product, not
//! from LISIRD's pre-smoothed 0.1 nm sibling. Averaging the smoothed product
//! instead differs from the true box mean by up to 1.1% across strong
//! Fraunhofer lines — larger than the archive's own 0.3% radiometric
//! uncertainty. `scripts/build_table.py` regenerates the table and records the
//! method.
//!
//! # Units and distance
//!
//! Spectral irradiance is W m⁻² nm⁻¹ **at 1 AU**, for solar-minimum conditions
//! between cycles 24 and 25. Scale to another heliocentric distance with
//! [`SolarSpectrum::scale_factor_at_au`], which is just 1/r² kept in one place
//! so it is obvious when it has been applied.
//!
//! # Example
//!
//! ```
//! use starfield_surfaces::solar_spectrum::SolarSpectrum;
//!
//! let sun = SolarSpectrum::load_embedded()?;
//!
//! // Integrating the archive recovers ~97% of the solar constant; the rest
//! // lies outside its 202-2730 nm coverage.
//! let total = sun.total_irradiance();
//! assert!((total - 1325.8).abs() < 1.0, "{total}");
//!
//! // Irradiance reaching Mars over a silicon detector's range.
//! let at_1au = sun.irradiance_over(300.0, 1100.0).unwrap();
//! let at_mars = at_1au * SolarSpectrum::scale_factor_at_au(1.524);
//! println!("{at_mars:.1} W/m^2 over 300-1100 nm at Mars");
//! # Ok::<(), starfield_core::StarfieldError>(())
//! ```

use starfield_core::{Result, StarfieldError};

const EMBEDDED_CSV: &str = include_str!("../../data/solar-spectrum/tsis1_hsrs_v2_1nm.csv");

/// Width of every bin, in nanometres.
pub const BIN_WIDTH_NM: f64 = 1.0;

/// Converts *F_λ* in W m⁻² nm⁻¹ to *F_ν* in erg s⁻¹ cm⁻² Hz⁻¹, given `λ²` in nm².
///
/// From `F_ν = F_λ · λ²/c` with the CGS unit change folded in:
///
/// ```text
/// W m^-2 nm^-1  =  1e7 erg/s · 1e-4 cm^-2 · 1e7 cm^-1  =  1e10 erg s^-1 cm^-2 cm^-1
/// λ_cm = λ_nm · 1e-7,  so λ_cm² = λ_nm² · 1e-14
/// factor = 1e10 · 1e-14 / c_cgs = 1e-4 / 2.99792458e10
/// ```
pub const F_LAMBDA_W_M2_NM_TO_F_NU_CGS: f64 = 1e-4 / 2.997_924_58e10;

/// One 1 nm bin of the reference spectrum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bin {
    /// Lower edge of the bin, nm. The bin covers `[wavelength_nm, wavelength_nm + 1)`.
    pub wavelength_nm: f64,
    /// Mean spectral irradiance over the bin at 1 AU, W m⁻² nm⁻¹.
    pub irradiance: f64,
    /// Mean archive uncertainty over the bin, W m⁻² nm⁻¹.
    pub uncertainty: f64,
}

/// The TSIS-1 HSRS v2 reference spectrum, on a contiguous 1 nm grid.
#[derive(Debug, Clone)]
pub struct SolarSpectrum {
    first_nm: f64,
    bins: Vec<Bin>,
}

impl SolarSpectrum {
    /// Parse the embedded table. No network, no I/O.
    pub fn load_embedded() -> Result<Self> {
        Self::parse(EMBEDDED_CSV)
    }

    /// Parse a table in the embedded CSV's format.
    ///
    /// Lines beginning with `#` are provenance comments and are skipped, as is
    /// the single header row.
    pub fn parse(text: &str) -> Result<Self> {
        let mut bins = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("wavelength_nm") {
                continue;
            }
            let mut fields = line.split(',');
            let mut next_f64 = |what: &str| -> Result<f64> {
                fields
                    .next()
                    .ok_or_else(|| {
                        StarfieldError::DataError(format!(
                            "solar spectrum line {}: missing {what}",
                            line_no + 1
                        ))
                    })?
                    .trim()
                    .parse::<f64>()
                    .map_err(|e| {
                        StarfieldError::DataError(format!(
                            "solar spectrum line {}: bad {what}: {e}",
                            line_no + 1
                        ))
                    })
            };
            bins.push(Bin {
                wavelength_nm: next_f64("wavelength")?,
                irradiance: next_f64("irradiance")?,
                uncertainty: next_f64("uncertainty")?,
            });
        }

        if bins.is_empty() {
            return Err(StarfieldError::DataError(
                "solar spectrum table is empty".into(),
            ));
        }

        // The accessors index arithmetically, so a gap would silently return the
        // wrong bin. Verify contiguity once, here.
        let first_nm = bins[0].wavelength_nm;
        for (i, bin) in bins.iter().enumerate() {
            let expected = first_nm + i as f64 * BIN_WIDTH_NM;
            if (bin.wavelength_nm - expected).abs() > 1e-9 {
                return Err(StarfieldError::DataError(format!(
                    "solar spectrum grid is not contiguous: bin {i} is at {} nm, expected {expected} nm",
                    bin.wavelength_nm
                )));
            }
        }

        Ok(Self { first_nm, bins })
    }

    /// Number of 1 nm bins.
    pub fn len(&self) -> usize {
        self.bins.len()
    }

    /// Always false — a successful parse has at least one bin.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Half-open wavelength coverage `[lo, hi)`, in nm.
    pub fn range_nm(&self) -> (f64, f64) {
        (self.first_nm, self.first_nm + self.bins.len() as f64)
    }

    /// Every bin, ascending.
    pub fn bins(&self) -> &[Bin] {
        &self.bins
    }

    /// The bin containing `nm`, or `None` outside [`SolarSpectrum::range_nm`].
    pub fn bin_at(&self, nm: f64) -> Option<&Bin> {
        if !nm.is_finite() {
            return None;
        }
        let offset = (nm - self.first_nm).floor();
        if offset < 0.0 {
            return None;
        }
        self.bins.get(offset as usize)
    }

    /// Spectral irradiance at `nm`, W m⁻² nm⁻¹ at 1 AU.
    ///
    /// Returns the box mean of the containing bin — see the module docs on why
    /// this does not interpolate.
    pub fn at_nm(&self, nm: f64) -> Option<f64> {
        self.bin_at(nm).map(|b| b.irradiance)
    }

    /// Archive uncertainty at `nm`, W m⁻² nm⁻¹ at 1 AU.
    pub fn uncertainty_at_nm(&self, nm: f64) -> Option<f64> {
        self.bin_at(nm).map(|b| b.uncertainty)
    }

    /// Spectral irradiance at `nm` as *F_ν* in CGS, erg s⁻¹ cm⁻² Hz⁻¹ at 1 AU.
    ///
    /// The archive tabulates *F_λ*; astronomical photometry conventionally works
    /// in *F_ν*. The conversion is `F_ν = F_λ · λ²/c` plus the CGS unit change,
    /// folded into [`F_LAMBDA_W_M2_NM_TO_F_NU_CGS`].
    ///
    /// This is a unit change, not radiometry: no distance, geometry or solar
    /// model enters. It lives here so the factor is written down once, in the
    /// crate that owns the units, rather than being re-derived by each consumer.
    /// Like [`SolarSpectrum::at_nm`], it returns the containing bin's box mean
    /// and does not interpolate.
    pub fn f_nu_cgs_at_nm(&self, nm: f64) -> Option<f64> {
        self.at_nm(nm)
            .map(|f_lambda| f_lambda * nm * nm * F_LAMBDA_W_M2_NM_TO_F_NU_CGS)
    }

    /// Irradiance integrated over `[lo_nm, hi_nm)`, W m⁻² at 1 AU.
    ///
    /// Bins are weighted by how much of them the interval covers, so partial
    /// bins at the ends contribute proportionally. Returns `None` unless the
    /// interval is valid and wholly inside [`SolarSpectrum::range_nm`] — a
    /// partially covered band would report a real-looking number that silently
    /// omits the missing part.
    pub fn irradiance_over(&self, lo_nm: f64, hi_nm: f64) -> Option<f64> {
        self.integrate(lo_nm, hi_nm, |_| 1.0)
    }

    /// Mean spectral irradiance over `[lo_nm, hi_nm)`, W m⁻² nm⁻¹ at 1 AU.
    pub fn mean_over(&self, lo_nm: f64, hi_nm: f64) -> Option<f64> {
        self.irradiance_over(lo_nm, hi_nm)
            .map(|total| total / (hi_nm - lo_nm))
    }

    /// Irradiance over `[lo_nm, hi_nm)` weighted by a response function,
    /// W m⁻² at 1 AU.
    ///
    /// `response` is evaluated once per bin at the bin centre — appropriate for
    /// a filter transmission or detector quantum-efficiency curve, which vary
    /// slowly compared with 1 nm. Call this once per band rather than per pixel.
    pub fn weighted_irradiance<F>(&self, lo_nm: f64, hi_nm: f64, response: F) -> Option<f64>
    where
        F: FnMut(f64) -> f64,
    {
        self.integrate(lo_nm, hi_nm, response)
    }

    /// Total irradiance across the whole archive range, W m⁻² at 1 AU.
    ///
    /// This is ~97% of the solar constant; the remainder lies outside the
    /// archive's 202–2730 nm coverage.
    pub fn total_irradiance(&self) -> f64 {
        self.bins.iter().map(|b| b.irradiance).sum::<f64>() * BIN_WIDTH_NM
    }

    /// Multiplier converting an irradiance at 1 AU to one at `au`.
    ///
    /// Simply 1/r², kept as a named function so its application is visible at
    /// the call site rather than being an anonymous stray division.
    pub fn scale_factor_at_au(au: f64) -> f64 {
        1.0 / (au * au)
    }

    fn integrate<F>(&self, lo_nm: f64, hi_nm: f64, mut response: F) -> Option<f64>
    where
        F: FnMut(f64) -> f64,
    {
        if !lo_nm.is_finite() || !hi_nm.is_finite() || lo_nm >= hi_nm {
            return None;
        }
        let (min, max) = self.range_nm();
        if lo_nm < min || hi_nm > max {
            return None;
        }

        let first = ((lo_nm - self.first_nm).floor() as usize).min(self.bins.len() - 1);
        let mut total = 0.0;
        for bin in self.bins.iter().skip(first) {
            let bin_lo = bin.wavelength_nm;
            if bin_lo >= hi_nm {
                break;
            }
            let overlap = (bin_lo + BIN_WIDTH_NM).min(hi_nm) - bin_lo.max(lo_nm);
            if overlap <= 0.0 {
                continue;
            }
            total += bin.irradiance * response(bin_lo + 0.5 * BIN_WIDTH_NM) * overlap;
        }
        Some(total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sun() -> SolarSpectrum {
        SolarSpectrum::load_embedded().unwrap()
    }

    #[test]
    fn embedded_table_covers_the_archive_range() {
        let s = sun();
        assert_eq!(s.len(), 2528);
        let (lo, hi) = s.range_nm();
        assert_eq!(lo, 202.0);
        assert_eq!(hi, 2730.0);
    }

    #[test]
    fn integrated_irradiance_is_97_percent_of_the_solar_constant() {
        // The archive states it spans >97% of the energy in TSI. TSI at solar
        // minimum is ~1361 W/m^2, so this is the headline physical check: it
        // fails if the table is mis-scaled, truncated, or in the wrong units.
        let total = sun().total_irradiance();
        assert!(
            (1320.0..1335.0).contains(&total),
            "integrated irradiance {total} W/m^2 is not ~97% of 1361"
        );
    }

    #[test]
    fn spectrum_peaks_in_the_blue() {
        // Solar spectral irradiance per unit wavelength peaks near 450 nm.
        let s = sun();
        let peak = s
            .bins()
            .iter()
            .max_by(|a, b| a.irradiance.total_cmp(&b.irradiance))
            .unwrap();
        assert!(
            (440.0..=520.0).contains(&peak.wavelength_nm),
            "peak at {} nm",
            peak.wavelength_nm
        );
    }

    #[test]
    fn irradiance_falls_steeply_into_the_ultraviolet() {
        let s = sun();
        assert!(s.at_nm(250.0).unwrap() < 0.1 * s.at_nm(500.0).unwrap());
    }

    #[test]
    fn at_nm_returns_the_containing_bin_without_interpolating() {
        let s = sun();
        let bin = s.at_nm(500.0).unwrap();
        assert_eq!(s.at_nm(500.4).unwrap(), bin);
        assert_eq!(s.at_nm(500.9).unwrap(), bin);
        assert_ne!(s.at_nm(501.0).unwrap(), bin);
    }

    #[test]
    fn out_of_range_and_non_finite_are_rejected() {
        let s = sun();
        assert_eq!(s.at_nm(201.9), None);
        assert_eq!(s.at_nm(2730.0), None);
        assert_eq!(s.at_nm(f64::NAN), None);
        assert_eq!(s.irradiance_over(100.0, 500.0), None);
        assert_eq!(s.irradiance_over(2000.0, 3000.0), None);
        assert_eq!(s.irradiance_over(500.0, 500.0), None);
        assert_eq!(s.irradiance_over(600.0, 500.0), None);
    }

    #[test]
    fn whole_bin_integral_matches_the_bin_value() {
        let s = sun();
        let bin = s.at_nm(500.0).unwrap();
        let integral = s.irradiance_over(500.0, 501.0).unwrap();
        assert!((integral - bin * BIN_WIDTH_NM).abs() < 1e-12);
    }

    #[test]
    fn partial_bins_contribute_proportionally() {
        let s = sun();
        let bin = s.at_nm(500.0).unwrap();
        let half = s.irradiance_over(500.0, 500.5).unwrap();
        assert!((half - 0.5 * bin).abs() < 1e-12, "{half} vs {}", 0.5 * bin);
    }

    #[test]
    fn sub_band_integrals_sum_to_the_whole() {
        let s = sun();
        let whole = s.irradiance_over(400.0, 700.0).unwrap();
        let parts = s.irradiance_over(400.0, 512.5).unwrap()
            + s.irradiance_over(512.5, 631.25).unwrap()
            + s.irradiance_over(631.25, 700.0).unwrap();
        assert!((whole - parts).abs() < 1e-9, "{whole} vs {parts}");
    }

    #[test]
    fn mean_over_is_the_integral_divided_by_width() {
        let s = sun();
        let mean = s.mean_over(400.0, 700.0).unwrap();
        let integral = s.irradiance_over(400.0, 700.0).unwrap();
        assert!((mean - integral / 300.0).abs() < 1e-12);
    }

    #[test]
    fn unit_response_reproduces_the_plain_integral() {
        let s = sun();
        let plain = s.irradiance_over(400.0, 700.0).unwrap();
        let weighted = s.weighted_irradiance(400.0, 700.0, |_| 1.0).unwrap();
        assert!((plain - weighted).abs() < 1e-12);
    }

    #[test]
    fn zero_response_gives_zero() {
        let s = sun();
        assert_eq!(s.weighted_irradiance(400.0, 700.0, |_| 0.0).unwrap(), 0.0);
    }

    #[test]
    fn distance_scaling_follows_inverse_square() {
        assert!((SolarSpectrum::scale_factor_at_au(1.0) - 1.0).abs() < 1e-12);
        assert!((SolarSpectrum::scale_factor_at_au(2.0) - 0.25).abs() < 1e-12);
        // Solar constant at Mars is ~586 W/m^2; our range holds 97% of it.
        let at_mars = sun().total_irradiance() * SolarSpectrum::scale_factor_at_au(1.524);
        assert!(
            (560.0..=580.0).contains(&at_mars),
            "{at_mars} W/m^2 at Mars"
        );
    }

    #[test]
    fn uncertainty_is_a_small_fraction_of_irradiance_in_the_visible() {
        // The archive quotes 0.3% over 460-2365 nm.
        let s = sun();
        for nm in [500.0, 700.0, 1000.0, 2000.0] {
            let ratio = s.uncertainty_at_nm(nm).unwrap() / s.at_nm(nm).unwrap();
            assert!(ratio < 0.02, "{nm} nm: uncertainty ratio {ratio}");
        }
    }

    #[test]
    fn f_nu_conversion_reproduces_the_solar_ab_magnitude() {
        // The strongest available check that the F_lambda -> F_nu unit change is
        // right: turn it into an AB magnitude and compare with the known solar
        // value. AB = -2.5 log10(F_nu) - 48.60, F_nu in erg s^-1 cm^-2 Hz^-1.
        // The Sun is V = -26.75, and its AB magnitude near V is about -26.7.
        let s = sun();
        let f_nu = s.f_nu_cgs_at_nm(550.0).unwrap();
        let ab = -2.5 * f_nu.log10() - 48.60;
        assert!(
            (-26.9..=-26.5).contains(&ab),
            "solar AB magnitude at 550 nm came out {ab}, expected about -26.7"
        );
    }

    #[test]
    fn f_nu_conversion_matches_a_hand_computed_value() {
        let s = sun();
        let f_lambda = s.at_nm(500.0).unwrap();
        // F_nu = F_lambda * 1e10 (to CGS per cm) * (500e-7 cm)^2 / c_cgs
        let expected = f_lambda * 1e10 * (500e-7f64).powi(2) / 2.997_924_58e10;
        let got = s.f_nu_cgs_at_nm(500.0).unwrap();
        assert!(
            (got - expected).abs() < 1e-18,
            "{got} vs hand-computed {expected}"
        );
    }

    #[test]
    fn f_nu_respects_the_same_range_and_bin_discipline() {
        let s = sun();
        assert_eq!(s.f_nu_cgs_at_nm(201.9), None);
        assert_eq!(s.f_nu_cgs_at_nm(2730.0), None);
        assert_eq!(s.f_nu_cgs_at_nm(f64::NAN), None);
        // Within a bin the F_lambda value is constant, so F_nu varies only
        // through the explicit lambda^2 factor.
        let a = s.f_nu_cgs_at_nm(500.0).unwrap();
        let b = s.f_nu_cgs_at_nm(500.5).unwrap();
        assert!(b > a);
    }

    #[test]
    fn parse_rejects_a_non_contiguous_grid() {
        let text = "wavelength_nm,irradiance_w_m2_nm,uncertainty_w_m2_nm,n_samples\n\
                    500,1.0,0.01,1000\n\
                    502,1.0,0.01,1000\n";
        let err = SolarSpectrum::parse(text).unwrap_err();
        assert!(err.to_string().contains("not contiguous"), "{err}");
    }

    #[test]
    fn parse_rejects_an_empty_table() {
        let err = SolarSpectrum::parse("# only a comment\n").unwrap_err();
        assert!(err.to_string().contains("empty"), "{err}");
    }
}
