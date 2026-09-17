mod frame_rotations;
pub mod inertial;
pub mod random;

#[cfg(feature = "python-tests")]
mod python_tests;

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

/// ITRS — the Earth-fixed frame of the IERS conventions.
///
/// `rotation_at` returns [`Time::c_matrix`], which already is the complete
/// ICRF → ITRS chain: frame bias, precession and nutation (the `M` matrix),
/// the daily rotation `R_z(-GAST)`, and polar motion `W` when a polar motion
/// table has been loaded into the [`crate::time::Timescale`]. Nothing is
/// recomputed here; this type only gives that rotation a [`Frame`] identity so
/// it can be used wherever body-fixed frames are accepted.
///
/// Use this frame for the Earth. The WGCCRE / IAU rotational elements that
/// serve other bodies are only accurate to roughly a tenth of a degree for the
/// Earth, because they ignore nutation and the observed variation of the
/// rotation rate; `c_matrix` is IERS-grade.
///
/// The transpose maps ITRS → ICRF (see [`Time::ct_matrix`]), which is how a
/// site on the ground becomes a geocentric offset.
///
/// Matches Skyfield's `skyfield.framelib.itrs`.
pub struct ItrsFrame;

impl Frame for ItrsFrame {
    fn rotation_at(&self, t: &Time) -> Matrix3<f64> {
        t.c_matrix()
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
pub static ITRS: ItrsFrame = ItrsFrame;

/// ECLIPJ2000 rotation matrix: equatorial J2000 to ecliptic J2000.
///
/// This is a rotation about the X-axis by the mean obliquity at J2000
/// (approximately 23.4393 degrees).
pub static ECLIPJ2000_MATRIX: Lazy<Matrix3<f64>> = Lazy::new(|| {
    frame_rotations::INERTIAL_FRAMES
        .get("ECLIPJ2000")
        .copied()
        .unwrap_or_else(Matrix3::identity)
});

/// Inverse ECLIPJ2000 rotation matrix: ecliptic J2000 to equatorial J2000.
pub static ECLIPJ2000_MATRIX_INV: Lazy<Matrix3<f64>> =
    Lazy::new(|| ECLIPJ2000_MATRIX.try_inverse().unwrap());

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::time::Timescale;
    use crate::toposlib::WGS84;

    /// `ItrsFrame` and `GeographicPosition::at` must agree.
    ///
    /// Both go through `Time::c_matrix`, so this is a guard against one of them
    /// drifting away from the other, not a discovery.
    #[test]
    fn test_itrs_frame_matches_geographic_position() {
        let mut kernel = SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421");
        let ts = Timescale::default();
        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);

        for jd in [2451545.0, 2455000.5, 2458849.5] {
            let t = ts.tdb_jd(jd);
            let earth = kernel.at("earth", &t).unwrap();
            let observer = boston.at(&t, &mut kernel).unwrap();
            let gcrs_offset = observer.position - earth.position;

            let from_frame = ItrsFrame.rotation_at(&t).transpose() * boston.itrs_xyz;

            for axis in 0..3 {
                let diff = (from_frame[axis] - gcrs_offset[axis]).abs();
                assert!(
                    diff < 1e-15,
                    "axis {axis} at JD {jd}: frame={} topos={} diff={diff}",
                    from_frame[axis],
                    gcrs_offset[axis]
                );
            }
        }
    }

    /// The ITRS rotation is orthonormal at every epoch.
    #[test]
    fn test_itrs_frame_is_orthonormal() {
        let ts = Timescale::default();
        for jd in [2451545.0, 2458849.5, 2460000.5] {
            let r = ItrsFrame.rotation_at(&ts.tt_jd(jd, None));
            let should_be_identity = r * r.transpose();
            for i in 0..3 {
                for j in 0..3 {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (should_be_identity[(i, j)] - expected).abs() < 1e-12,
                        "R·Rᵀ[{i},{j}] at JD {jd} = {}",
                        should_be_identity[(i, j)]
                    );
                }
            }
        }
    }
}
