//! Body-fixed frames read from a binary PCK kernel.
//!
//! A binary PCK (`.bpc`) stores the orientation of a frame as Chebyshev
//! coefficients for three Euler angles; see
//! [`jplephem::pck`](crate::jplephem::pck). [`PckFrame`] turns those angles
//! into the rotation matrix that carries a vector from the reference frame of
//! the segment — J2000 in every kernel NAIF publishes — into the body-fixed
//! frame, and implements [`Frame`] so it can be used wherever the inertial and
//! Earth-fixed frames of [`framelib`](crate::framelib) are.
//!
//! Build one with
//! [`PlanetaryConstants::build_frame_named`](crate::planetarylib::PlanetaryConstants::build_frame_named),
//! which is the port of Skyfield's `planetarylib.PlanetaryConstants`, or let
//! [`PlanetaryConstants::frame_for`](crate::planetarylib::PlanetaryConstants::frame_for)
//! reach for one on its own: given the Moon and a loaded `moon_pa_*.bpc` it
//! returns the principal-axes frame rather than the IAU elements.
//!
//! This is the most accurate body-fixed frame available for the Moon: the
//! lunar principal-axes frames `MOON_PA_DE421` (frame class id 31006) and
//! `MOON_PA_DE440` (31008) come from the same fits as the ephemerides
//! themselves, where the IAU rotational elements of
//! [`RotationalElements`](crate::planetarylib::RotationalElements) are a
//! truncated series. `ITRF93` (3000) is likewise published as a binary PCK,
//! though for the Earth [`ItrsFrame`](crate::framelib::ItrsFrame) is the
//! frame to prefer.

use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};

use crate::framelib::Frame;
use crate::jplephem::pck::PckSegment;
use crate::time::Time;
use crate::Result;

/// A body-fixed frame whose orientation comes from a binary PCK segment.
///
/// The rotation is Skyfield's `rot_z(-W) · rot_x(-δ) · rot_z(-α)` on the three
/// angles the segment stores. Those angles are the SPICE Euler angles of the
/// rotation itself, not the right ascension and declination of a pole, so no
/// quarter turn is folded in here; that difference is what separates this
/// frame from one built out of IAU rotational elements.
///
/// `matrix`, when present, is the fixed offset of a text-kernel *TK frame*
/// defined relative to the PCK frame — the way `MOON_ME` is defined relative
/// to `MOON_PA_DE421` in `moon_080317.tf`. It is applied on the left, after
/// the PCK rotation.
#[derive(Debug, Clone)]
pub struct PckFrame {
    /// NAIF id of the body at the centre of the frame, such as 301.
    center: i32,
    /// The segment supplying the Euler angles.
    segment: Arc<PckSegment>,
    /// Fixed rotation applied after the segment's, for TK frames.
    matrix: Option<Matrix3<f64>>,
}

impl PckFrame {
    /// Build a frame around one binary PCK segment.
    ///
    /// `center` is the NAIF id of the body the frame is fixed to, and
    /// `matrix`, if given, is the constant TK-frame offset applied on the left
    /// of the segment's rotation.
    pub fn new(center: i32, segment: Arc<PckSegment>, matrix: Option<Matrix3<f64>>) -> Self {
        Self {
            center,
            segment,
            matrix,
        }
    }

    /// NAIF id of the body at the centre of the frame.
    pub fn center(&self) -> i32 {
        self.center
    }

    /// The segment the angles are read from.
    pub fn segment(&self) -> &PckSegment {
        &self.segment
    }

    /// The constant TK-frame offset, if this frame has one.
    pub fn matrix(&self) -> Option<&Matrix3<f64>> {
        self.matrix.as_ref()
    }

    /// The rotation from the segment's reference frame to this frame at `t`.
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::EphemerisError`](crate::StarfieldError) if
    /// `t` lies outside the span the segment covers, or if the segment is not
    /// a supported data type.
    pub fn try_rotation_at(&self, t: &Time) -> Result<Matrix3<f64>> {
        let (angles, _rates) = self.segment.compute(t.tdb())?;
        Ok(self.assemble(&angles))
    }

    /// The rotation and its rate of change at `t`.
    ///
    /// The second matrix is the derivative of the first with respect to time,
    /// in units of one day, so `dR/dt · r` is the velocity a fixed body-frame
    /// vector `r` acquires from the body's rotation, per day. Ported from
    /// Skyfield's `planetarylib.Frame.rotation_and_rate_at`.
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::EphemerisError`](crate::StarfieldError) if
    /// `t` lies outside the span the segment covers, or if the segment is not
    /// a supported data type.
    pub fn rotation_and_rate_at(&self, t: &Time) -> Result<(Matrix3<f64>, Matrix3<f64>)> {
        let (angles, rates) = self.segment.compute(t.tdb())?;
        let (ra, dec, w) = (angles[0], angles[1], angles[2]);
        let (ra_dot, dec_dot, w_dot) = (rates[0], rates[1], rates[2]);

        let r = rot_z(-w) * rot_x(-dec) * rot_z(-ra);

        let (sa, ca) = w.sin_cos();
        let u = dec.cos();
        let v = -dec.sin();

        let domega0 = w_dot + u * ra_dot;
        let domega1 = ca * dec_dot - sa * v * ra_dot;
        let domega2 = sa * dec_dot + ca * v * ra_dot;

        #[rustfmt::skip]
        let drdt_rt = Matrix3::new(
                 0.0,  domega0,  domega2,
            -domega0,      0.0,  domega1,
            -domega2, -domega1,      0.0,
        );

        let dr_dt = drdt_rt * r;

        Ok(match self.matrix {
            Some(m) => (m * r, m * dr_dt),
            None => (r, dr_dt),
        })
    }

    /// Assemble the rotation matrix from three Euler angles in radians.
    fn assemble(&self, angles: &Vector3<f64>) -> Matrix3<f64> {
        let r = rot_z(-angles[2]) * rot_x(-angles[1]) * rot_z(-angles[0]);
        match self.matrix {
            Some(m) => m * r,
            None => r,
        }
    }
}

impl Frame for PckFrame {
    /// The rotation from the segment's reference frame — J2000 in the kernels
    /// NAIF publishes — to this body-fixed frame.
    ///
    /// The [`Frame`] trait cannot report an error, so an epoch the kernel does
    /// not cover produces a matrix of `NaN`. Call
    /// [`try_rotation_at`](Self::try_rotation_at) to see the error instead.
    fn rotation_at(&self, t: &Time) -> Matrix3<f64> {
        self.try_rotation_at(t)
            .unwrap_or_else(|_| Matrix3::from_element(f64::NAN))
    }
}

/// Rotation of `angle` radians about the x axis.
///
/// The same convention as Skyfield's `functions.rot_x`: the matrix rotates a
/// vector counter-clockwise about the axis, as seen looking down the axis
/// toward the origin.
pub(crate) fn rot_x(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    #[rustfmt::skip]
    let m = Matrix3::new(
        1.0, 0.0, 0.0,
        0.0,   c,  -s,
        0.0,   s,   c,
    );
    m
}

/// Rotation of `angle` radians about the y axis, in the convention of
/// Skyfield's `functions.rot_y`.
pub(crate) fn rot_y(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    #[rustfmt::skip]
    let m = Matrix3::new(
          c, 0.0,   s,
        0.0, 1.0, 0.0,
         -s, 0.0,   c,
    );
    m
}

/// Rotation of `angle` radians about the z axis, in the convention of
/// Skyfield's `functions.rot_z`.
pub(crate) fn rot_z(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    #[rustfmt::skip]
    let m = Matrix3::new(
          c,  -s, 0.0,
          s,   c, 0.0,
        0.0, 0.0, 1.0,
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem::pck::test_support;
    use crate::planetarylib::PlanetaryConstants;
    use nalgebra::{Rotation3, Vector3};

    /// The sign convention of the `rot_*` helpers, checked before anything is
    /// built on top of them.
    ///
    /// Skyfield's `functions.rot_z(theta)` is
    /// `[[c, -s, 0], [s, c, 0], [0, 0, 1]]`, which is the active,
    /// right-handed rotation: it carries the x axis a quarter turn toward the
    /// y axis when `theta` is a quarter turn. nalgebra's
    /// `Rotation3::from_axis_angle` is the independent witness.
    #[test]
    fn test_rotation_sign_convention_matches_skyfield() {
        let angle = 0.3;
        for (ours, axis) in [
            (rot_x(angle), Vector3::x_axis()),
            (rot_y(angle), Vector3::y_axis()),
            (rot_z(angle), Vector3::z_axis()),
        ] {
            let theirs = *Rotation3::from_axis_angle(&axis, angle).matrix();
            assert!(
                (ours - theirs).abs().max() < 1e-15,
                "rot about {axis:?} disagrees with nalgebra:\n{ours}\n{theirs}"
            );
        }

        // Skyfield's literal matrices, spelled out.
        let (s, c) = angle.sin_cos();
        assert_eq!(
            rot_z(angle),
            Matrix3::new(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)
        );
        assert_eq!(
            rot_x(angle),
            Matrix3::new(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)
        );

        // A quarter turn about z sends x̂ to ŷ.
        let quarter = rot_z(std::f64::consts::FRAC_PI_2) * Vector3::x();
        assert!((quarter - Vector3::y()).norm() < 1e-15);
    }

    /// The rotation matrices Skyfield produces for `MOON_PA_DE421` at three
    /// epochs, from `moon_pa_de421_1900-2050.bpc` and `moon_080317.tf`:
    ///
    /// ```python
    /// pc = PlanetaryConstants()
    /// pc.read_text(load('moon_080317.tf'))
    /// pc.read_binary(load('moon_pa_de421_1900-2050.bpc'))
    /// pc.build_frame_named('MOON_PA_DE421').rotation_at(ts.tdb(y, m, d))
    /// ```
    ///
    /// Each entry is a TDB Julian date and the nine elements of the matrix in
    /// row order.
    const MOON_PA_DE421_GOLDEN: [(f64, [f64; 9]); 3] = [
        (
            2451544.5,
            [
                0.850042543689892,
                0.4719025905658242,
                0.2339564466615154,
                -0.5262425521270344,
                0.7796926815587792,
                0.3393347884530935,
                -0.022281163525359693,
                -0.41156684431686724,
                0.9111071739433357,
            ],
        ),
        (
            2455362.5,
            [
                0.48606359354917644,
                -0.7988143368227282,
                -0.35445428240116666,
                0.8735378827586621,
                0.4561395395613729,
                0.1699067033234794,
                0.025956702632943758,
                -0.3922147061689602,
                0.9195074082644583,
            ],
        ),
        (
            2460735.5,
            [
                -0.9971502695940774,
                0.06991210219677377,
                0.028348506396042604,
                -0.07544066971979335,
                -0.925127101746052,
                -0.37208675193720486,
                0.00021260453350083338,
                -0.37316503531464174,
                0.9277649547261065,
            ],
        ),
    ];

    /// Skyfield's `MOON_ME_DE421` rotation at 2000-01-01 TDB: the same segment
    /// seen through the TK-frame offset that `moon_080317.tf` defines.
    const MOON_ME_DE421_GOLDEN: [f64; 9] = [
        0.8502072337338002,
        0.47148903601337244,
        0.23419169205140472,
        -0.5259625840805721,
        0.7798486288034029,
        0.33941048348679465,
        -0.02260574825141532,
        -0.41174531578711776,
        0.9110185371732895,
    ];

    /// The angles Skyfield reads from the `MOON_PA_DE421` segment at
    /// 2000-01-01 TDB, in radians.
    const MOON_PA_DE421_ANGLES: [f64; 3] = [
        -0.054084614394338155,
        0.42483397988172855,
        2564.1432197564877,
    ];

    /// Their rates at the same epoch, in radians per day.
    const MOON_PA_DE421_RATES: [f64; 3] = [
        -0.00013824229206887688,
        4.2546358627903764e-05,
        0.23011790298638823,
    ];

    /// The two kernels the ignored tests need, read from the data directory or
    /// downloaded into it.
    fn moon_constants() -> PlanetaryConstants {
        let loader = crate::Loader::new();
        let mut pc = loader.open_text_pck("moon_080317.tf").unwrap();
        pc.read_binary(
            loader
                .open_binary_pck("moon_pa_de421_1900-2050.bpc")
                .unwrap(),
        );
        pc
    }

    /// Compare a rotation with a golden matrix, element by element.
    fn assert_matches(actual: &Matrix3<f64>, expected: &[f64; 9], tolerance: f64, label: &str) {
        for row in 0..3 {
            for column in 0..3 {
                let want = expected[row * 3 + column];
                let got = actual[(row, column)];
                assert!(
                    (got - want).abs() < tolerance,
                    "{label}: element ({row}, {column}) is {got}, expected {want}"
                );
            }
        }
    }

    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc and moon_080317.tf"]
    fn test_moon_pa_de421_matches_skyfield() {
        let pc = moon_constants();
        let frame = pc.build_frame_named("MOON_PA_DE421").unwrap();
        assert_eq!(frame.center(), 301);
        assert!(frame.matrix().is_none());
        assert_eq!(frame.segment().body, 31006);
        assert_eq!(frame.segment().frame, 1);

        let ts = crate::time::Timescale::default();
        for (jd, expected) in MOON_PA_DE421_GOLDEN {
            let t = ts.tdb_jd(jd);
            assert_matches(&frame.rotation_at(&t), &expected, 1e-9, &format!("JD {jd}"));
        }
    }

    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc and moon_080317.tf"]
    fn test_moon_pa_de421_angles_and_rates_match_skyfield() {
        let pc = moon_constants();
        let frame = pc.build_frame_named("MOON_PA_DE421").unwrap();
        let ts = crate::time::Timescale::default();

        let (angles, rates) = frame.segment().compute(2451544.5).unwrap();
        for i in 0..3 {
            assert!((angles[i] - MOON_PA_DE421_ANGLES[i]).abs() < 1e-9);
            assert!((rates[i] - MOON_PA_DE421_RATES[i]).abs() < 1e-12);
        }

        let t = ts.tdb_jd(2451544.5);
        let (rotation, rate) = frame.rotation_and_rate_at(&t).unwrap();
        assert_matches(&rotation, &MOON_PA_DE421_GOLDEN[0].1, 1e-9, "rotation");

        // The rate matrix must be the derivative of the rotation, so a central
        // difference over a minute has to reproduce it.
        let h = 1.0 / 1440.0;
        let ahead = frame.try_rotation_at(&ts.tdb_jd(2451544.5 + h)).unwrap();
        let behind = frame.try_rotation_at(&ts.tdb_jd(2451544.5 - h)).unwrap();
        let numeric = (ahead - behind) / (2.0 * h);
        assert!(
            (rate - numeric).abs().max() < 1e-6,
            "rate matrix disagrees with a central difference:\n{rate}\n{numeric}"
        );
    }

    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc and moon_080317.tf"]
    fn test_moon_me_de421_applies_the_tk_offset() {
        let pc = moon_constants();
        let frame = pc.build_frame_named("MOON_ME_DE421").unwrap();

        // MOON_ME_DE421 is frame 31007, a TK frame defined by three small
        // rotations away from MOON_PA_DE421, which is the frame the kernel
        // actually carries.
        assert_eq!(frame.segment().body, 31006);
        let matrix = frame.matrix().expect("the TK offset should be present");
        assert!((matrix - Matrix3::identity()).abs().max() < 1e-3);

        let t = crate::time::Timescale::default().tdb_jd(2451544.5);
        assert_matches(
            &frame.rotation_at(&t),
            &MOON_ME_DE421_GOLDEN,
            1e-9,
            "MOON_ME_DE421",
        );
    }

    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc and moon_080317.tf"]
    fn test_an_epoch_outside_the_kernel_is_an_error() {
        let pc = moon_constants();
        let frame = pc.build_frame_named("MOON_PA_DE421").unwrap();
        let t = crate::time::Timescale::default().tdb_jd(2000000.5);

        assert!(frame.try_rotation_at(&t).is_err());
        assert!(frame.rotation_at(&t)[(0, 0)].is_nan());
    }

    #[test]
    fn test_a_frame_built_from_a_synthetic_kernel() {
        let pck = crate::jplephem::pck::PCK::from_bytes(&test_support::one_day_pck(2)).unwrap();
        let mut pc = PlanetaryConstants::new();
        pc.read_text(
            "KPL/FK\n\\begindata\nFRAME_MOON_PA_DE421 = 31006\n\
             FRAME_31006_CENTER = 301\n\\begintext\n",
        )
        .unwrap();
        pc.read_binary(pck);
        assert_eq!(pc.segments().len(), 1);

        let frame = pc.build_frame_named("MOON_PA_DE421").unwrap();
        assert_eq!(frame.center(), 301);

        let t = crate::time::Timescale::default().tdb_jd(2451545.25);
        let (angles, _) = frame.segment().compute(2451545.25).unwrap();
        let expected = rot_z(-angles[2]) * rot_x(-angles[1]) * rot_z(-angles[0]);
        assert!((frame.rotation_at(&t) - expected).abs().max() < 1e-15);
        assert!((frame.rotation_at(&t).determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_a_tk_frame_folds_in_its_offset() {
        let pck = crate::jplephem::pck::PCK::from_bytes(&test_support::one_day_pck(2)).unwrap();
        let mut pc = PlanetaryConstants::new();
        pc.read_text(
            "KPL/FK\n\\begindata\n\
             FRAME_MOON_PA_DE421 = 31006\n\
             FRAME_31006_CENTER = 301\n\
             FRAME_MOON_ME_DE421 = 31007\n\
             FRAME_31007_CENTER = 301\n\
             TKFRAME_31007_RELATIVE = 'MOON_PA_DE421'\n\
             TKFRAME_31007_SPEC = 'ANGLES'\n\
             TKFRAME_31007_ANGLES = ( 67.92, 78.56, 0.30 )\n\
             TKFRAME_31007_AXES = ( 3, 2, 1 )\n\
             TKFRAME_31007_UNITS = 'ARCSECONDS'\n\
             \\begintext\n",
        )
        .unwrap();
        pc.read_binary(pck);

        let frame = pc.build_frame_named("MOON_ME_DE421").unwrap();
        assert_eq!(frame.segment().body, 31006);

        // The three angles are the ones moon_080317.tf gives, applied in the
        // order z, y, x, each on the left of the last.
        let scale = crate::constants::ASEC2RAD;
        let expected = rot_x(0.30 * scale) * rot_y(78.56 * scale) * rot_z(67.92 * scale);
        let matrix = frame.matrix().expect("the TK offset should be present");
        assert!((matrix - expected).abs().max() < 1e-15);

        let plain = pc.build_frame_named("MOON_PA_DE421").unwrap();
        let t = crate::time::Timescale::default().tdb_jd(2451545.0);
        assert!(
            (frame.rotation_at(&t) - expected * plain.rotation_at(&t))
                .abs()
                .max()
                < 1e-15
        );
    }

    #[test]
    fn test_a_frame_with_no_segment_is_an_error() {
        let mut pc = PlanetaryConstants::new();
        pc.read_text(
            "KPL/FK\n\\begindata\nFRAME_MOON_PA_DE421 = 31006\n\
             FRAME_31006_CENTER = 301\n\\begintext\n",
        )
        .unwrap();
        assert!(pc.build_frame_named("MOON_PA_DE421").is_err());
        assert!(pc.build_frame_named("NO_SUCH_FRAME").is_err());
    }

    /// A verbatim excerpt of `pck00011.tpc`, the source of the IAU elements
    /// the Moon falls back on when no binary kernel has been read.
    const EXCERPT: &str = include_str!("pck00011_excerpt.tpc");

    /// A synthetic principal-axes kernel covering one day about J2000, whose
    /// three Euler angles are constant and of the test's choosing.
    fn synthetic_moon_pa(frame_id: i32, angles: [f64; 3]) -> crate::jplephem::pck::PCK {
        let coefficients = std::array::from_fn(|i| vec![angles[i], 0.0, 0.0]);
        let bytes = test_support::synthetic_pck(
            frame_id,
            2,
            &[(0.0, crate::jplephem::S_PER_DAY / 2.0, coefficients)],
        )
        .unwrap();
        crate::jplephem::pck::PCK::from_bytes(&bytes).unwrap()
    }

    /// `frame_for(301)` prefers a loaded principal-axes segment to the IAU
    /// elements, and needs no frame kernel to do it.
    #[test]
    fn test_frame_for_the_moon_prefers_a_principal_axes_kernel() {
        use crate::planetarylib::IauFrame;
        use crate::planetlib::Body;

        let angles = [0.375, 1.0, 0.5];
        let mut pc = PlanetaryConstants::new();
        pc.read_text(EXCERPT).unwrap();
        pc.read_binary(synthetic_moon_pa(test_support::MOON_PA_DE421, angles));

        let t = crate::time::Timescale::default().tdb_jd(2451545.0);
        let expected = rot_z(-angles[2]) * rot_x(-angles[1]) * rot_z(-angles[0]);
        let frame = pc.frame_for(301).unwrap();
        assert!((frame.rotation_at(&t) - expected).abs().max() < 1e-14);

        // Without the binary kernel the same call gives the IAU elements.
        let mut iau_only = PlanetaryConstants::new();
        iau_only.read_text(EXCERPT).unwrap();
        assert_eq!(
            iau_only.frame_for(301).unwrap().rotation_at(&t),
            IauFrame::from_body(Body::Moon).rotation_at(&t)
        );
        // And the two frames really are different frames.
        assert!(
            (frame.rotation_at(&t) - IauFrame::from_body(Body::Moon).rotation_at(&t))
                .abs()
                .max()
                > 1e-3
        );
    }

    /// `MOON_PA_DE440` wins when both principal-axes kernels are loaded.
    #[test]
    fn test_frame_for_the_moon_prefers_de440_to_de421() {
        let de421 = [0.375, 1.0, 0.5];
        let de440 = [0.25, 0.75, 0.125];

        let mut pc = PlanetaryConstants::new();
        pc.read_binary(synthetic_moon_pa(test_support::MOON_PA_DE421, de421));
        pc.read_binary(synthetic_moon_pa(31008, de440));

        let t = crate::time::Timescale::default().tdb_jd(2451545.0);
        let expected = rot_z(-de440[2]) * rot_x(-de440[1]) * rot_z(-de440[0]);
        let rotation = pc.frame_for(301).unwrap().rotation_at(&t);
        assert!((rotation - expected).abs().max() < 1e-14);
    }

    /// The real kernel, against the same golden matrices Skyfield produced —
    /// and without a frame kernel, which `frame_for` does not need.
    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc"]
    fn test_frame_for_the_moon_matches_the_published_kernel() {
        let loader = crate::Loader::new();
        let mut pc = PlanetaryConstants::new();
        pc.read_binary(
            loader
                .open_binary_pck("moon_pa_de421_1900-2050.bpc")
                .unwrap(),
        );
        assert!(pc.build_frame_named("MOON_PA_DE421").is_err());

        let frame = pc.frame_for(301).unwrap();
        let ts = crate::time::Timescale::default();
        for (jd, expected) in MOON_PA_DE421_GOLDEN {
            let t = ts.tdb_jd(jd);
            assert_matches(&frame.rotation_at(&t), &expected, 1e-9, &format!("JD {jd}"));
        }
    }

    /// A body other than the Moon ignores the lunar segments entirely: Mars
    /// comes from the embedded table, and a body the table lacks has nothing.
    #[test]
    fn test_frame_for_another_body_ignores_the_lunar_segments() {
        let mut pc = PlanetaryConstants::new();
        pc.read_binary(synthetic_moon_pa(
            test_support::MOON_PA_DE421,
            [0.375, 1.0, 0.5],
        ));
        let t = crate::time::Timescale::default().tdb_jd(2455362.5);
        assert_eq!(
            pc.frame_for(499).unwrap().rotation_at(&t),
            crate::planetarylib::IauFrame::from_body(crate::planetlib::Body::Mars).rotation_at(&t)
        );
        assert!(pc.frame_for(401).is_err());
        assert!(pc.frame_for(301).is_ok());
    }

    #[test]
    fn test_rotations_are_orthonormal() {
        for m in [rot_x(1.1), rot_y(-0.4), rot_z(2.7)] {
            assert!((m * m.transpose() - Matrix3::identity()).abs().max() < 1e-15);
            assert!((m.determinant() - 1.0).abs() < 1e-15);
        }
    }
}
