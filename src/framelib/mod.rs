mod frame_rotations;
pub mod inertial;
pub mod random;

use crate::constants::ASEC2RAD;
use nalgebra::Matrix3;
use once_cell::sync::Lazy;

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
