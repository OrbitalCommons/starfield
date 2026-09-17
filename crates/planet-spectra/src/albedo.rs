//! Sampled spectral albedo and its accessors.

use starfield_datasource_utils::SampledCurve;

use crate::body::SpectralBody;

/// What a tabulated albedo actually measures.
///
/// Archives are rarely all one thing, and the difference is large enough to
/// matter: a full-disk albedo at 6.8° phase is several percent below the
/// geometric albedo of the same body at the same wavelength. Mixing the two
/// silently is the most likely way to get a planet's brightness wrong, so the
/// distinction is carried in the type rather than in a doc comment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlbedoKind {
    /// Disk-integrated albedo at zero phase angle.
    Geometric,
    /// Disk-integrated albedo (I/F averaged over the disk) at a non-zero phase
    /// angle. Converting this to [`AlbedoKind::Geometric`] requires a phase
    /// function, which this crate does not yet ship.
    FullDisk {
        /// Phase angle of the observation, in degrees.
        phase_angle_deg: f64,
    },
}

/// A body's albedo sampled against wavelength.
///
/// Wavelengths are in nanometres and strictly ascending; albedo is
/// dimensionless. Between samples the albedo is treated as piecewise linear —
/// both [`SpectralAlbedo::at_nm`] and [`SpectralAlbedo::mean_over`] are exact
/// for that interpretation.
///
/// This type deliberately does not implement any spectrum or radiance trait.
/// An albedo is a dimensionless reflectance; turning it into a flux at a
/// detector needs the solar spectrum and the heliocentric and observer
/// distances, none of which belong in a data crate. Consumers multiply through
/// themselves.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralAlbedo {
    body: SpectralBody,
    kind: AlbedoKind,
    source: &'static str,
    curve: SampledCurve,
}

impl SpectralAlbedo {
    /// Build from parallel wavelength and albedo arrays.
    ///
    /// # Panics
    /// Panics if the arrays differ in length, are empty, or if the wavelengths
    /// are not strictly ascending. These are construction-time programming
    /// errors in a data crate whose inputs are vendored, not user input.
    pub fn new(
        body: SpectralBody,
        kind: AlbedoKind,
        source: &'static str,
        wavelengths_nm: Vec<f64>,
        albedo: Vec<f64>,
    ) -> Self {
        let curve = SampledCurve::new(wavelengths_nm, albedo)
            .unwrap_or_else(|e| panic!("invalid spectral albedo for {}: {e}", body.name()));
        Self {
            body,
            kind,
            source,
            curve,
        }
    }

    /// Build from an already-validated curve.
    pub fn from_curve(
        body: SpectralBody,
        kind: AlbedoKind,
        source: &'static str,
        curve: SampledCurve,
    ) -> Self {
        Self {
            body,
            kind,
            source,
            curve,
        }
    }

    /// The body this spectrum describes.
    pub fn body(&self) -> SpectralBody {
        self.body
    }

    /// Whether this is a geometric or a non-zero-phase full-disk albedo.
    pub fn kind(&self) -> AlbedoKind {
        self.kind
    }

    /// Provenance string naming the archive and product this came from.
    pub fn source(&self) -> &'static str {
        self.source
    }

    /// The underlying sampled curve, for consumers that want to share one
    /// integration path across albedos, reflectances and the solar spectrum.
    pub fn curve(&self) -> &SampledCurve {
        &self.curve
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.curve.len()
    }

    /// Always false — construction rejects empty spectra.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Inclusive wavelength bounds, in nanometres.
    pub fn range_nm(&self) -> (f64, f64) {
        self.curve.range_nm()
    }

    /// The raw samples, as `(wavelength_nm, albedo)` pairs.
    pub fn samples(&self) -> impl Iterator<Item = (f64, f64)> + '_ {
        self.curve.samples()
    }

    /// Albedo at `nm`, linearly interpolated between samples.
    ///
    /// Returns `None` outside [`SpectralAlbedo::range_nm`] rather than
    /// extrapolating or clamping — a detector band that runs off the end of the
    /// measured range is a real problem for the caller to handle, not something
    /// to paper over with the edge value.
    pub fn at_nm(&self, nm: f64) -> Option<f64> {
        self.curve.at(nm)
    }

    /// Mean albedo over `[lo_nm, hi_nm]`, weighted by wavelength.
    ///
    /// Integrates the piecewise-linear spectrum exactly over the interval and
    /// divides by its width. Returns `None` unless the interval is valid and
    /// wholly inside [`SpectralAlbedo::range_nm`]; a partially covered band
    /// would silently report the mean of the covered part, which reads as a
    /// real answer while being biased by however much was missing.
    ///
    /// This is an unweighted mean over wavelength, not a filter-weighted or
    /// photon-weighted one. Consumers applying a filter response or a quantum
    /// efficiency curve should iterate [`SpectralAlbedo::samples`] and do their
    /// own weighted integral.
    pub fn mean_over(&self, lo_nm: f64, hi_nm: f64) -> Option<f64> {
        self.curve.mean_over(lo_nm, hi_nm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> SpectralAlbedo {
        // albedo = wavelength/1000, so every interpolation has a closed form.
        SpectralAlbedo::new(
            SpectralBody::Jupiter,
            AlbedoKind::Geometric,
            "test",
            vec![400.0, 500.0, 600.0],
            vec![0.4, 0.5, 0.6],
        )
    }

    #[test]
    fn at_nm_hits_exact_samples() {
        let s = ramp();
        assert_eq!(s.at_nm(400.0), Some(0.4));
        assert_eq!(s.at_nm(500.0), Some(0.5));
        assert_eq!(s.at_nm(600.0), Some(0.6));
    }

    #[test]
    fn at_nm_interpolates_linearly() {
        let s = ramp();
        assert!((s.at_nm(450.0).unwrap() - 0.45).abs() < 1e-12);
        assert!((s.at_nm(575.0).unwrap() - 0.575).abs() < 1e-12);
    }

    #[test]
    fn at_nm_refuses_to_extrapolate() {
        let s = ramp();
        assert_eq!(s.at_nm(399.9), None);
        assert_eq!(s.at_nm(600.1), None);
    }

    #[test]
    fn mean_over_a_linear_ramp_is_the_midpoint() {
        let s = ramp();
        assert!((s.mean_over(400.0, 600.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((s.mean_over(400.0, 500.0).unwrap() - 0.45).abs() < 1e-12);
        // An interval inside a single segment.
        assert!((s.mean_over(410.0, 430.0).unwrap() - 0.42).abs() < 1e-12);
    }

    #[test]
    fn mean_over_rejects_partial_coverage() {
        let s = ramp();
        assert_eq!(s.mean_over(350.0, 500.0), None);
        assert_eq!(s.mean_over(500.0, 700.0), None);
    }

    #[test]
    fn mean_over_rejects_degenerate_intervals() {
        let s = ramp();
        assert_eq!(s.mean_over(500.0, 500.0), None);
        assert_eq!(s.mean_over(550.0, 450.0), None);
    }

    #[test]
    fn non_finite_bounds_are_rejected() {
        let s = ramp();
        assert_eq!(s.mean_over(f64::NAN, 500.0), None);
        assert_eq!(s.mean_over(400.0, f64::NAN), None);
        assert_eq!(s.mean_over(f64::NEG_INFINITY, f64::INFINITY), None);
        assert_eq!(s.at_nm(f64::NAN), None);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn construction_rejects_unsorted_wavelengths() {
        SpectralAlbedo::new(
            SpectralBody::Jupiter,
            AlbedoKind::Geometric,
            "test",
            vec![500.0, 400.0],
            vec![0.5, 0.4],
        );
    }

    #[test]
    #[should_panic(expected = "wavelengths but")]
    fn construction_rejects_mismatched_lengths() {
        SpectralAlbedo::new(
            SpectralBody::Jupiter,
            AlbedoKind::Geometric,
            "test",
            vec![400.0, 500.0],
            vec![0.4],
        );
    }
}
