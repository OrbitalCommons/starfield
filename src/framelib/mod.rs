mod frame_rotations;
pub mod inertial;
pub mod random;

use crate::constants::ASEC2RAD;
use crate::time::Time;
use nalgebra::Matrix3;
use once_cell::sync::Lazy;

/// A reference frame that can produce a rotation matrix at a given time.
///
/// Matches Skyfield's frame abstraction. Inertial frames return a constant
/// matrix; dynamical frames (e.g. ecliptic of date) compute time-dependent
/// rotations from precession and nutation.
pub trait Frame {
    /// Rotation matrix from ICRF to this frame at time `t`.
    fn rotation_at(&self, t: &Time) -> Matrix3<f64>;
}

/// ICRS (International Celestial Reference System) — identity frame.
pub struct IcrsFrame;

impl Frame for IcrsFrame {
    fn rotation_at(&self, _t: &Time) -> Matrix3<f64> {
        Matrix3::identity()
    }
}

/// True equator and equinox of date (TETE).
///
/// Applies the full precession-nutation matrix M to rotate from ICRF
/// to the true equator and equinox at the given epoch.
pub struct TrueEquatorFrame;

impl Frame for TrueEquatorFrame {
    fn rotation_at(&self, t: &Time) -> Matrix3<f64> {
        t.m_matrix()
    }
}

/// Ecliptic frame at J2000 (static rotation by mean obliquity).
pub struct EclipticJ2000Frame;

impl Frame for EclipticJ2000Frame {
    fn rotation_at(&self, _t: &Time) -> Matrix3<f64> {
        *ECLIPJ2000_MATRIX
    }
}

/// True ecliptic of date (dynamical ecliptic with nutation).
///
/// Matches Skyfield's `ecliptic_frame`: `R_x(-true_obliquity) × M`.
pub struct EclipticOfDateFrame;

impl Frame for EclipticOfDateFrame {
    fn rotation_at(&self, t: &Time) -> Matrix3<f64> {
        let mean_obliq = crate::nutationlib::mean_obliquity(t.tdb());
        let (_d_psi, d_eps) = crate::nutationlib::iau2000a_nutation(t.tt());
        let true_obliq = mean_obliq + d_eps;

        let (s, c) = (-true_obliq).sin_cos();
        #[rustfmt::skip]
        let rx = Matrix3::new(
            1.0, 0.0, 0.0,
            0.0,   c,  -s,
            0.0,   s,   c,
        );
        rx * t.m_matrix()
    }
}

/// Galactic coordinate frame (static rotation).
pub struct GalacticFrame;

impl Frame for GalacticFrame {
    fn rotation_at(&self, _t: &Time) -> Matrix3<f64> {
        *GALACTIC_MATRIX
    }
}

/// Pre-built frame instances for convenience.
pub static ICRS: IcrsFrame = IcrsFrame;
pub static TRUE_EQUATOR_OF_DATE: TrueEquatorFrame = TrueEquatorFrame;
pub static ECLIPTIC_J2000: EclipticJ2000Frame = EclipticJ2000Frame;
pub static ECLIPTIC_OF_DATE: EclipticOfDateFrame = EclipticOfDateFrame;
pub static GALACTIC: GalacticFrame = GalacticFrame;

/// ECLIPJ2000 rotation matrix from the SPICE frame table.
static ECLIPJ2000_MATRIX: Lazy<Matrix3<f64>> = Lazy::new(|| {
    frame_rotations::INERTIAL_FRAMES
        .get("ECLIPJ2000")
        .copied()
        .unwrap_or_else(Matrix3::identity)
});

/// Galactic rotation matrix from the SPICE frame table.
static GALACTIC_MATRIX: Lazy<Matrix3<f64>> = Lazy::new(|| {
    frame_rotations::INERTIAL_FRAMES
        .get("GALACTIC")
        .copied()
        .unwrap_or_else(Matrix3::identity)
});

/// ICRS to J2000 frame bias matrix
///
/// Frame bias parameters from IERS (2003) Conventions, Chapter 5.
/// This small rotation accounts for the offset between the ICRS
/// (International Celestial Reference System) and the dynamical
/// J2000 mean equator and equinox.
pub static ICRS_TO_J2000: Lazy<Matrix3<f64>> = Lazy::new(|| {
    let xi0 = -0.0166170 * ASEC2RAD;
    let eta0 = -0.0068192 * ASEC2RAD;
    let da0 = -0.01460 * ASEC2RAD;

    let yx = -da0;
    let zx = xi0;
    let xy = da0;
    let zy = eta0;
    let xz = -xi0;
    let yz = -eta0;

    let xx = 1.0 - 0.5 * (yx * yx + zx * zx);
    let yy = 1.0 - 0.5 * (yx * yx + zy * zy);
    let zz = 1.0 - 0.5 * (zy * zy + zx * zx);

    Matrix3::new(xx, xy, xz, yx, yy, yz, zx, zy, zz)
});
