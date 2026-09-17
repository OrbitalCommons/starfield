//! Body-fixed frames built from the IAU WGCCRE rotational elements.
//!
//! [`IauFrame`] is what SPICE calls `IAU_MARS`, `IAU_JUPITER` and so on: the
//! frame whose z axis is the body's north pole and whose x axis is its prime
//! meridian, both given by the polynomial and periodic series of Archinal et
//! al. (2018). The series themselves are evaluated by
//! [`RotationalElements::evaluate`], so a caller who wants the pole direction
//! without a matrix need not build a frame at all.
//!
//! The elements come either from a text kernel read into
//! [`PlanetaryConstants`] or from the embedded IAU 2015 table; the two agree,
//! and [`IauFrame::new`] and [`IauFrame::from_body`] are the respective
//! constructors.
//!
//! # Example
//!
//! ```
//! use starfield_core::framelib::Frame;
//! use starfield_core::planetarylib::IauFrame;
//! use starfield_core::planetlib::Body;
//! use starfield_core::time::Timescale;
//!
//! let ts = Timescale::default();
//! let t = ts.tdb_jd(2451545.0);
//! let mars = IauFrame::from_body(Body::Mars);
//!
//! // The pole of Mars at J2000, periodic terms included.
//! let (ra, dec, w) = mars.pole_and_meridian(&t);
//! assert!((ra.to_degrees() - 317.680854).abs() < 1e-6);
//! assert!((dec.to_degrees() - 52.886439).abs() < 1e-6);
//! assert!((w.to_degrees() - 176.632060).abs() < 1e-6);
//!
//! // ICRF → body-fixed, so the third row is the body's north pole in ICRF.
//! let r = mars.rotation_at(&t);
//! assert!((r.determinant() - 1.0).abs() < 1e-12);
//! ```

use nalgebra::Matrix3;

use crate::constants::{DEG2RAD, J2000};
use crate::framelib::Frame;
use crate::planetarylib::{body_constants, PlanetaryConstants, RotationalElements};
use crate::planetlib::Body;
use crate::time::Time;
use crate::{Result, StarfieldError};

/// Days in a Julian century, the unit of the pole and nutation/precession
/// polynomials.
const DAYS_PER_CENTURY: f64 = 36525.0;

/// A quarter turn in radians, the offset between the pole angles and the
/// Euler angles of the rotation.
const QUARTER_TURN: f64 = std::f64::consts::FRAC_PI_2;

impl RotationalElements {
    /// The pole right ascension, pole declination and prime meridian angle W
    /// at TDB time `t`, in radians.
    ///
    /// This is Archinal et al. (2018) §2 verbatim,
    ///
    /// ```text
    /// α = α₀ + α₁ T + α₂ T² + Σ aᵢ sin θᵢ
    /// δ = δ₀ + δ₁ T + δ₂ T² + Σ dᵢ cos θᵢ
    /// W = W₀ + W₁ d + W₂ d² + Σ wᵢ sin θᵢ
    /// ```
    ///
    /// where `d` is TDB days since J2000, `T = d / 36525` is TDB Julian
    /// centuries since J2000, and the nutation/precession angles are
    /// `θᵢ = θᵢ₀ + θᵢ₁ T + θᵢ₂ T²`. The quadratic term of `θᵢ` is used only by
    /// the bodies whose kernel declares `BODYn_MAX_PHASE_DEGREE = 2` — as of
    /// `pck00011.tpc`, the Mars system alone — and is zero elsewhere.
    ///
    /// W increases eastward: it is the angle from the ascending node of the
    /// body's equator on the ICRF equator to the body's prime meridian. See
    /// [`IauFrame`] for what that means for reported longitudes.
    ///
    /// # Example
    ///
    /// ```
    /// use starfield_core::planetlib::Body;
    /// use starfield_core::time::Timescale;
    ///
    /// let t = Timescale::default().tdb_jd(2451545.0);
    /// let (ra, dec, w) = Body::Jupiter.rotational_elements().evaluate(&t);
    /// assert!((ra.to_degrees() - 268.057204).abs() < 1e-6);
    /// assert!((dec.to_degrees() - 64.495810).abs() < 1e-6);
    /// assert!((w.to_degrees() - 284.95).abs() < 1e-6);
    /// ```
    pub fn evaluate(&self, t: &Time) -> (f64, f64, f64) {
        self.evaluate_with_rates(t).0
    }

    /// The angles of [`evaluate`](Self::evaluate) together with their time
    /// derivatives.
    ///
    /// Returns `((α, δ, W), (dα/dt, dδ/dt, dW/dt))`, the angles in radians and
    /// the rates in radians per day, both at TDB time `t`. The rates are the
    /// analytic derivatives of the same series, not a finite difference.
    pub fn evaluate_with_rates(&self, t: &Time) -> ((f64, f64, f64), (f64, f64, f64)) {
        let days = t.tdb() - J2000;
        let centuries = days / DAYS_PER_CENTURY;

        // Degrees throughout, converted once at the end.
        let mut ra = poly(self.pole_ra, centuries);
        let mut dec = poly(self.pole_dec, centuries);
        let mut w = poly(self.pm, days);

        // Rates in degrees per day; the pole polynomials are per century.
        let mut ra_rate = poly_rate(self.pole_ra, centuries) / DAYS_PER_CENTURY;
        let mut dec_rate = poly_rate(self.pole_dec, centuries) / DAYS_PER_CENTURY;
        let mut w_rate = poly_rate(self.pm, days);

        for (i, &(angle, angle_rate)) in self.nut_prec_angles.iter().enumerate() {
            let accel = self.nut_prec_angle_accel.get(i).copied().unwrap_or(0.0);
            let theta = (angle + angle_rate * centuries + accel * centuries * centuries) * DEG2RAD;
            // Radians per day, so it can multiply a degree amplitude directly.
            let theta_rate = (angle_rate + 2.0 * accel * centuries) * DEG2RAD / DAYS_PER_CENTURY;
            let (sin_theta, cos_theta) = theta.sin_cos();

            if let Some(&amplitude) = self.nut_prec_ra.get(i) {
                ra += amplitude * sin_theta;
                ra_rate += amplitude * cos_theta * theta_rate;
            }
            if let Some(&amplitude) = self.nut_prec_dec.get(i) {
                dec += amplitude * cos_theta;
                dec_rate -= amplitude * sin_theta * theta_rate;
            }
            if let Some(&amplitude) = self.nut_prec_pm.get(i) {
                w += amplitude * sin_theta;
                w_rate += amplitude * cos_theta * theta_rate;
            }
        }

        (
            (ra * DEG2RAD, dec * DEG2RAD, w * DEG2RAD),
            (ra_rate * DEG2RAD, dec_rate * DEG2RAD, w_rate * DEG2RAD),
        )
    }
}

/// A quadratic polynomial `c₀ + c₁ x + c₂ x²`.
fn poly(c: [f64; 3], x: f64) -> f64 {
    c[0] + (c[1] + c[2] * x) * x
}

/// The derivative `c₁ + 2 c₂ x` of [`poly`].
fn poly_rate(c: [f64; 3], x: f64) -> f64 {
    c[1] + 2.0 * c[2] * x
}

/// The body-fixed frame of the IAU WGCCRE rotational elements.
///
/// `rotation_at` returns the ICRF → body-fixed rotation
/// `R_z(W) · R_x(90° − δ) · R_z(90° + α)` in the frame-rotation sense, which
/// is SPICE's `pxform('J2000', 'IAU_<BODY>', et)` and Skyfield's
/// `planetarylib.Frame.rotation_at`.
///
/// # Longitude convention — the classic trap
///
/// The prime meridian angle W of the IAU elements always increases
/// **eastward**, so the body-fixed x axis of this frame is at east longitude
/// zero and longitudes measured in it are **east**-positive. That is *not*
/// how planetographic longitudes are reported for most bodies: Mars, Venus and
/// every other body that rotates in the direction the IAU calls prograde have
/// planetographic longitude measured **west**-positive, so
///
/// ```text
/// planetographic longitude (west) = 360° − east longitude
/// ```
///
/// for Mars, Mercury, Jupiter, Saturn, Uranus, Neptune and the Sun, while
/// Venus and Pluto, which rotate the other way, are east-positive and need no
/// flip. The Earth and the Moon are east-positive by convention as well.
/// JPL Horizons reports planetographic longitude; SPICE `reclat` on a vector
/// in this frame returns east longitude. Convert deliberately.
///
/// # The Earth
///
/// `IauFrame::new(399, ..)` is permitted, but prefer
/// [`ItrsFrame`](crate::framelib::ItrsFrame). The IAU elements for the Earth
/// ignore nutation and the observed variation of the rotation rate and are
/// good to only about a tenth of a degree, where
/// [`Time::c_matrix`](crate::time::Time::c_matrix) behind `ItrsFrame` is
/// IERS-grade. [`PlanetaryConstants::frame_for`] makes that choice for you.
#[derive(Debug, Clone, PartialEq)]
pub struct IauFrame {
    /// NAIF integer code of the body whose equator and prime meridian define
    /// the frame, such as 499 for Mars.
    pub body: i32,
    /// The pole, prime meridian and nutation/precession series.
    pub elements: RotationalElements,
}

impl IauFrame {
    /// Build the frame of `body` from constants read out of text kernels.
    ///
    /// # Errors
    ///
    /// Returns [`StarfieldError::DataError`] if the kernels read so far do not
    /// assign all three of `BODYnnn_POLE_RA`, `BODYnnn_POLE_DEC` and
    /// `BODYnnn_PM`. The embedded table is not consulted; use
    /// [`from_body`](Self::from_body) or [`from_naif_id`](Self::from_naif_id)
    /// for that.
    pub fn new(body: i32, constants: &PlanetaryConstants) -> Result<Self> {
        let elements = constants.rotational_elements(body).ok_or_else(|| {
            StarfieldError::DataError(format!(
                "the text kernels read so far define no rotational elements for body {}",
                body
            ))
        })?;
        Ok(Self { body, elements })
    }

    /// Build the frame of `body` from the embedded IAU WGCCRE 2015 table,
    /// which needs no kernel and no network.
    pub fn from_body(body: Body) -> Self {
        Self {
            body: body.naif_id(),
            elements: body.rotational_elements().clone(),
        }
    }

    /// Build the frame of a NAIF code from the embedded IAU WGCCRE 2015 table.
    ///
    /// Returns `None` for a body the table does not cover; see
    /// [`body_constants`](crate::planetarylib::body_constants).
    pub fn from_naif_id(naif_id: i32) -> Option<Self> {
        body_constants(naif_id).map(|constants| Self {
            body: naif_id,
            elements: constants.elements.clone(),
        })
    }

    /// The pole right ascension, pole declination and prime meridian angle W
    /// at TDB time `t`, in radians.
    ///
    /// A thin pass-through to [`RotationalElements::evaluate`].
    pub fn pole_and_meridian(&self, t: &Time) -> (f64, f64, f64) {
        self.elements.evaluate(t)
    }

    /// The ICRF → body-fixed rotation and its time derivative at `t`.
    ///
    /// The second matrix is `dR/dt` in units of **per day**, the analytic
    /// derivative of the same series rather than a difference of two epochs.
    /// A point `r` fixed on the body's surface therefore has ICRF velocity
    /// `(dR/dt)ᵀ · r` per day, which is how a location that turns with the
    /// body acquires the velocity that aberration needs.
    ///
    /// Ports Skyfield's `planetarylib.Frame.rotation_and_rate_at`, and equals
    /// the derivative block of SPICE's `sxform` scaled from seconds to days.
    pub fn rotation_and_rate_at(&self, t: &Time) -> (Matrix3<f64>, Matrix3<f64>) {
        let ((ra, dec, w), (ra_rate, dec_rate, w_rate)) = self.elements.evaluate_with_rates(t);
        let (phi, theta, psi) = euler_angles(ra, dec, w);
        // φ = 90° + α and θ = 90° − δ, so the rates carry those signs.
        let (phi_rate, theta_rate, psi_rate) = (ra_rate, -dec_rate, w_rate);

        let r = euler_rotation(phi, theta, psi);

        let (sin_psi, cos_psi) = psi.sin_cos();
        let u = theta.cos();
        let v = -theta.sin();

        let d0 = psi_rate + u * phi_rate;
        let d1 = cos_psi * theta_rate - sin_psi * v * phi_rate;
        let d2 = sin_psi * theta_rate + cos_psi * v * phi_rate;

        #[rustfmt::skip]
        let skew = Matrix3::new(
            0.0,  d0,  d2,
            -d0, 0.0,  d1,
            -d2, -d1, 0.0,
        );

        (r, skew * r)
    }
}

impl Frame for IauFrame {
    fn rotation_at(&self, t: &Time) -> Matrix3<f64> {
        let (ra, dec, w) = self.elements.evaluate(t);
        let (phi, theta, psi) = euler_angles(ra, dec, w);
        euler_rotation(phi, theta, psi)
    }
}

/// The 3-1-3 Euler angles `(φ, θ, ψ) = (90° + α, 90° − δ, W)` of the ICRF →
/// body-fixed rotation.
fn euler_angles(ra: f64, dec: f64, w: f64) -> (f64, f64, f64) {
    (ra + QUARTER_TURN, QUARTER_TURN - dec, w)
}

/// The 3-1-3 frame rotation `R_z(ψ) · R_x(θ) · R_z(φ)`.
fn euler_rotation(phi: f64, theta: f64, psi: f64) -> Matrix3<f64> {
    rot_z(-psi) * rot_x(-theta) * rot_z(-phi)
}

/// Rotate a vector about the x axis by `angle` radians, counterclockwise seen
/// from +x. The transpose rotates the frame instead, which is why the callers
/// above pass negated angles.
#[rustfmt::skip]
fn rot_x(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(
        1.0, 0.0, 0.0,
        0.0,   c,  -s,
        0.0,   s,   c,
    )
}

/// Rotate a vector about the z axis by `angle` radians, counterclockwise seen
/// from +z. The transpose rotates the frame instead.
#[rustfmt::skip]
fn rot_z(angle: f64) -> Matrix3<f64> {
    let (s, c) = angle.sin_cos();
    Matrix3::new(
          c,  -s, 0.0,
          s,   c, 0.0,
        0.0, 0.0, 1.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::TAU;
    use crate::planetarylib::PlanetaryConstants;
    use crate::time::Timescale;
    use nalgebra::Vector3;

    /// A verbatim excerpt of `pck00011.tpc`, the same one the parser tests read.
    const EXCERPT: &str = include_str!("pck00011_excerpt.tpc");

    /// The three TDB Julian dates of the checked-in golden matrices:
    /// 2000-01-01 12:00, 2010-06-15 00:00 and 2025-03-01 00:00 TDB.
    const GOLDEN_EPOCHS: [f64; 3] = [2451545.0, 2455362.5, 2460735.5];

    /// One arcsecond in radians, the acceptance of the golden comparison.
    const ONE_ARCSEC: f64 = 4.84813681109536e-6;

    // The golden matrices below are `spiceypy.pxform('J2000', 'IAU_<BODY>', et)`
    // from SpiceyPy 8.2.0 (CSPICE N0067) with `pck00011.tpc` and
    // `naif0012.tls` furnished, at `GOLDEN_EPOCHS`, row major. SPICE ET is TDB
    // seconds past J2000, so `et = (jd_tdb − 2451545.0) × 86400` exactly;
    // `spiceypy.str2et('JD 2451545.0 TDB')` returns 0.0, confirming it. The
    // live test that regenerates them is `test_pxform_matches_spiceypy` in
    // `planetarylib::python_tests`.

    /// `pxform('J2000', 'IAU_MARS', et)`.
    #[rustfmt::skip]
    const MARS_GOLDEN: [[f64; 9]; 3] = [
        [
            -0.7067364464274375, -0.7065882946541798, 0.035448231956131226,
            0.5490619907161186, -0.5793961327942199, -0.6023545896346731,
            0.44615527077685696, -0.40624266536246617, 0.7974411396443181,
        ],
        [
            -0.8863118400793911, -0.3242054226337199, 0.33066926992083,
            0.12413169701212334, -0.8542466254129581, -0.5048306892092274,
            0.4461419549111853, -0.406390879456545, 0.7973730677434258,
        ],
        [
            -0.7705388191649774, -0.6276401367684895, 0.11107559082478148,
            0.45524045028887483, -0.6638869402697514, -0.5933045280124281,
            0.44612336923840346, -0.4065980684367541, 0.7972778374963622,
        ],
    ];

    /// `pxform('J2000', 'IAU_EARTH', et)`.
    #[rustfmt::skip]
    const EARTH_GOLDEN: [[f64; 9]; 3] = [
        [
            0.17617425963267894, -0.9843589945964213, -0.0,
            0.9843589945964213, 0.17617425963267894, 0.0,
            0.0, 0.0, 1.0,
        ],
        [
            -0.12710409127331979, -0.9918893756895991, 0.00012796749876899007,
            0.9918888635304756, -0.12710415568613648, -0.0010079739757959242,
            0.0010160638784498816, -1.1880792978850964e-06, 0.9999994838062585,
        ],
        [
            -0.9295067370275555, 0.3687981081213379, 0.002276239554384584,
            -0.36879702042479284, -0.9295095241032796, 0.0008957271289283591,
            0.0024461288154828125, -6.885964552833245e-06, 0.9999970081987254,
        ],
    ];

    /// `pxform('J2000', 'IAU_MOON', et)`.
    #[rustfmt::skip]
    const MOON_GOLDEN: [[f64; 9]; 3] = [
        [
            0.7842270520919169, 0.5578471124601639, 0.2716514860755947,
            -0.6200619152508559, 0.7205566654668131, 0.31035675134719964,
            -0.022608671404182493, -0.4118309009426129, 0.9109797785934293,
        ],
        [
            0.485787326776981, -0.7991110745205956, -0.3541640350479052,
            0.873697712957627, 0.4558665196455951, 0.16981761580832655,
            0.025748388599811645, -0.39192755305636695, 0.9196356961535158,
        ],
        [
            -0.9971245661890603, 0.07008609646347157, 0.02881906634141836,
            -0.07577768953654991, -0.9251083536084237, -0.37206488124037446,
            0.0005841838583585007, -0.3731788755629459, 0.927759228228313,
        ],
    ];

    /// `pxform('J2000', 'IAU_JUPITER', et)`.
    #[rustfmt::skip]
    const JUPITER_GOLDEN: [[f64; 9]; 3] = [
        [
            0.2282653328760834, -0.8802481155891952, -0.4160026355789609,
            0.9734895258323475, 0.19994923179133894, 0.11107856589263648,
            -0.014597290902157951, -0.4303295942736501, 0.9025537986129101,
        ],
        [
            0.8252686343955254, 0.5044629469772228, 0.25386771399687713,
            -0.564551467378859, 0.7485631397080112, 0.3477569072654942,
            -0.01460553878916179, -0.43031425841765925, 0.9025609770199121,
        ],
        [
            0.5635371482080168, -0.7491787390204209, -0.3480762841524078,
            0.8259618325198612, 0.5035472121650848, 0.2534309695779181,
            -0.014592251711380493, -0.43031549137875935, 0.9025606040978349,
        ],
    ];

    // The rate matrices are the lower-left 3×3 block of
    // `spiceypy.sxform('J2000', 'IAU_<BODY>', et)`, which is dR/dt in radians
    // per second, scaled by 86400 to the per-day units of
    // `IauFrame::rotation_and_rate_at`.

    /// `d/dt pxform('J2000', 'IAU_MARS', et)`, per day.
    #[rustfmt::skip]
    const MARS_GOLDEN_RATE: [[f64; 9]; 3] = [
        [
            3.3625765858601433, -3.548349558853409, -3.688952291275563,
            4.328209688475353, 4.327302448491811, -0.21709285285823873,
            -1.4796778249337661e-09, -9.959557787210436e-08, -4.9909397664365254e-08,
        ],
        [
            0.7602098728933745, -5.23159461621461, -3.091694406777728,
            5.4279690273728, 1.9855054923684288, -2.0250915440560506,
            1.0296311372929939e-08, -5.658707971357899e-08, -3.460123083044737e-08,
        ],
        [
            2.787992814512526, -4.065789900017914, -3.6335277155619616,
            4.718949525227956, 3.8438065131101253, -0.6802514040714048,
            -3.501346145931982e-08, -3.949461999639012e-08, -5.495108337902155e-10,
        ],
    ];

    /// `d/dt pxform('J2000', 'IAU_EARTH', et)`, per day.
    #[rustfmt::skip]
    const EARTH_GOLDEN_RATE: [[f64; 9]; 3] = [
        [
            6.201842983146044, 1.1099660813907002, -4.68904923897179e-08,
            -1.1099660813907002, 6.201842983146044, -2.619967187097069e-07,
            2.661597243972188e-07, -8.724170523020214e-24, 0.0,
        ],
        [
            6.249284074356058, -0.8008054179400418, -0.006350593300052623,
            0.8008050118459835, 6.249287301123502, -0.0008065088932400845,
            2.661590411465362e-07, -6.224381520496189e-10, -2.7043546673482926e-10,
        ],
        [
            -2.3235640912365687, -5.856270071665599, 0.005643675842525499,
            5.856252512243883, -2.323570944753329, -0.014341094188636013,
            2.66155764358499e-07, -1.4984927024877563e-09, -6.510635510245715e-10,
        ],
    ];

    /// `d/dt pxform('J2000', 'IAU_MOON', et)`, per day.
    #[rustfmt::skip]
    const MOON_GOLDEN_RATE: [[f64; 9]; 3] = [
        [
            -0.1426099614652412, 0.1656968463687124, 0.07143374280266697,
            -0.1803671886168799, -0.12833783140240002, -0.062393179750668944,
            1.0382890828401112e-05, -9.997442620146057e-05, -4.4938225423633714e-05,
        ],
        [
            0.20090262568018724, 0.1048667522849226, 0.038953041547256234,
            -0.11170342367891953, 0.18371535232994096, 0.08153069079312492,
            -3.5873073851044095e-05, -0.0001288626619417904, -5.391390757967345e-05,
        ],
        [
            -0.01742578557159137, -0.21278289446826737, -0.08544886159472152,
            0.22929896515824316, -0.01609217763750278, -0.0066889888245087615,
            0.00011454311162761749, -6.997852500572557e-05, -2.8220060459098083e-05,
        ],
    ];

    /// `d/dt pxform('J2000', 'IAU_JUPITER', et)`, per day.
    #[rustfmt::skip]
    const JUPITER_GOLDEN_RATE: [[f64; 9]; 3] = [
        [
            14.790926739868896, 3.0379725307841534, 1.6876965681006701,
            -3.4681994272173347, 13.374242909809926, 6.320627338357702,
            -4.924638001776149e-09, 4.014758067119235e-10, 1.1177233719288335e-10,
        ],
        [
            -8.577636616605558, 11.37345834606923, 5.283720884503052,
            -12.538900110956867, -7.664668492887371, -3.8571948308613533,
            2.089141670765846e-09, 6.330105313103222e-09, 3.0518133215960594e-09,
        ],
        [
            12.549432360111282, 7.650755071451132, 3.850559052382348,
            -8.562225329853145, 11.382811577450902, 5.288573411922915,
            -3.2183428802661634e-09, -1.2967125856405486e-09, -6.70268988168391e-10,
        ],
    ];

    /// The angle, in radians, of the rotation that carries `b` onto `a`.
    ///
    /// Taken from the sine and cosine together rather than from `acos` of the
    /// trace alone, which loses half its digits for the small angles this
    /// comparison is looking for.
    fn angle_between(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let d = a * b.transpose();
        let axis = Vector3::new(
            d[(2, 1)] - d[(1, 2)],
            d[(0, 2)] - d[(2, 0)],
            d[(1, 0)] - d[(0, 1)],
        ) * 0.5;
        axis.norm().atan2(0.5 * (d.trace() - 1.0))
    }

    /// Wrap an angle difference into (−π, π].
    fn wrap(difference: f64) -> f64 {
        difference - TAU * (difference / TAU).round()
    }

    /// The Earth's elements carry no periodic terms, so its angles are the
    /// bare polynomials and can be checked by hand.
    #[test]
    fn test_evaluate_earth_is_the_bare_polynomial() {
        let ts = Timescale::default();
        let elements = Body::Earth.rotational_elements();
        assert!(elements.nut_prec_ra.is_empty());

        // BODY399_POLE_RA = ( 0.0 -0.641 0.0 ), POLE_DEC = ( 90.0 -0.557 0.0 ),
        // PM = ( 190.147 360.9856235 0.0 ).
        let (ra, dec, w) = elements.evaluate(&ts.tdb_jd(2451545.0));
        assert!(ra.abs() < 1e-15, "α(J2000) = {}", ra.to_degrees());
        assert!((dec.to_degrees() - 90.0).abs() < 1e-12);
        assert!((w.to_degrees() - 190.147).abs() < 1e-12);

        // One Julian century later: T = 1, d = 36525.
        let (ra, dec, w) = elements.evaluate(&ts.tdb_jd(2451545.0 + 36525.0));
        assert!(
            (ra.to_degrees() + 0.641).abs() < 1e-12,
            "α = {}",
            ra.to_degrees()
        );
        assert!((dec.to_degrees() - 89.443).abs() < 1e-12);
        let expected_w = 190.147 + 360.9856235 * 36525.0;
        assert!(
            (w.to_degrees() - expected_w).abs() < 1e-6,
            "W = {}",
            w.to_degrees()
        );
    }

    /// The angles themselves match SPICE `bodeul`, periodic terms included.
    ///
    /// Checking the angles apart from the matrix separates a mistake in the
    /// series from a mistake in assembling the rotation. Values are
    /// `spiceypy.bodeul(body, et)` in radians, with W wrapped into `[0, τ)`.
    #[test]
    fn test_evaluate_matches_spice_bodeul() {
        let ts = Timescale::default();
        // (label, NAIF code, TDB JD, α, δ, W)
        let cases = [
            (
                "Mars at J2000",
                499,
                2451545.0,
                5.544576879956258,
                0.9230424950070161,
                3.0828110069012546,
            ),
            (
                "Mars at T=1",
                499,
                2451545.0 + 36525.0,
                5.542684477363816,
                0.9219636387313781,
                2.5532431718311273,
            ),
            (
                "Moon at J2000",
                301,
                2451545.0,
                4.657546083023791,
                1.1456533675897982,
                0.7189929926922299,
            ),
            (
                "Jupiter in 2010",
                599,
                2455362.5,
                4.67846044163541,
                1.1256809093134712,
                0.6305874587415019,
            ),
        ];

        for (label, body, jd, ra_expected, dec_expected, w_expected) in cases {
            let elements = &body_constants(body).unwrap().elements;
            let (ra, dec, w) = elements.evaluate(&ts.tdb_jd(jd));
            assert!(
                wrap(ra - ra_expected).abs() < 1e-12,
                "{label}: α = {ra}, SPICE says {ra_expected}"
            );
            assert!(
                (dec - dec_expected).abs() < 1e-12,
                "{label}: δ = {dec}, SPICE says {dec_expected}"
            );
            assert!(
                wrap(w - w_expected).abs() < 1e-9,
                "{label}: W = {w}, SPICE says {w_expected}"
            );
        }
    }

    /// The quadratic phase-angle term is honoured.
    ///
    /// `BODY4_MAX_PHASE_DEGREE = 2` gives the fifth Mars-system angle a
    /// deg/century² term. Mars itself has zero amplitude on that angle — the
    /// term is there for Phobos and Deimos — so this uses a hand-sized kernel
    /// whose whole answer is one periodic term, and checks it in closed form.
    #[test]
    fn test_quadratic_phase_angle_is_honoured() {
        let mut pc = PlanetaryConstants::new();
        pc.read_text(concat!(
            "KPL/PCK\n\\begindata\n",
            "BODY4_MAX_PHASE_DEGREE = 2\n",
            "BODY4_NUT_PREC_ANGLES = ( 10.0 100.0 20.0 )\n",
            "BODY499_POLE_RA = ( 0.0 0.0 0.0 )\n",
            "BODY499_POLE_DEC = ( 90.0 0.0 0.0 )\n",
            "BODY499_PM = ( 0.0 0.0 0.0 )\n",
            "BODY499_NUT_PREC_RA = ( 2.0 )\n",
            "\\begintext\n",
        ))
        .unwrap();
        let elements = pc.rotational_elements(499).unwrap();
        assert_eq!(elements.nut_prec_angle_accel, vec![20.0]);

        // At T = 1 the angle is 10 + 100 + 20 = 130 degrees, so α = 2 sin 130°.
        let t = Timescale::default().tdb_jd(2451545.0 + 36525.0);
        let ra = elements.evaluate(&t).0.to_degrees();
        let expected = 2.0 * 130.0_f64.to_radians().sin();
        assert!(
            (ra - expected).abs() < 1e-12,
            "α = {ra}, expected {expected}"
        );

        // Without the quadratic term the angle would be 110 degrees instead.
        let mut linear = elements.clone();
        linear.nut_prec_angle_accel = vec![0.0];
        let ra_linear = linear.evaluate(&t).0.to_degrees();
        let expected_linear = 2.0 * 110.0_f64.to_radians().sin();
        assert!((ra_linear - expected_linear).abs() < 1e-12);
        assert!((ra - ra_linear).abs() > 0.1);
    }

    /// `rotation_at` matches SpiceyPy `pxform` to well under an arcsecond.
    #[test]
    fn test_rotation_matches_spiceypy_golden() {
        let ts = Timescale::default();
        let cases = [
            (Body::Mars, MARS_GOLDEN),
            (Body::Earth, EARTH_GOLDEN),
            (Body::Moon, MOON_GOLDEN),
            (Body::Jupiter, JUPITER_GOLDEN),
        ];

        for (body, golden) in cases {
            let frame = IauFrame::from_body(body);
            for (jd, expected) in GOLDEN_EPOCHS.iter().zip(golden.iter()) {
                let expected = Matrix3::from_row_slice(expected);
                let actual = frame.rotation_at(&ts.tdb_jd(*jd));

                let angle = angle_between(&actual, &expected);
                assert!(
                    angle < ONE_ARCSEC,
                    "{} at JD {jd}: {} arcsec from SPICE",
                    body.name(),
                    angle / ONE_ARCSEC
                );
                // The floor is floating point, not physics: the Mars M5 phase
                // angle passes 4×10⁶ degrees within these epochs, where a
                // double holds about 10⁻¹¹ radians.
                let elementwise = (actual - expected).abs().max();
                assert!(
                    elementwise < 1e-10,
                    "{} at JD {jd}: largest element differs by {elementwise}",
                    body.name()
                );
            }
        }
    }

    /// The rotation of an `IauFrame` is a proper rotation at every epoch.
    #[test]
    fn test_rotation_is_orthonormal() {
        let ts = Timescale::default();
        for body in [Body::Mars, Body::Moon, Body::Jupiter, Body::Earth] {
            let frame = IauFrame::from_body(body);
            for jd in GOLDEN_EPOCHS {
                let r = frame.rotation_at(&ts.tdb_jd(jd));
                let residual = (r * r.transpose() - Matrix3::identity()).abs().max();
                assert!(
                    residual < 1e-14,
                    "{} at JD {jd}: max |R·Rᵀ − I| = {residual}",
                    body.name()
                );
                assert!(
                    (r.determinant() - 1.0).abs() < 1e-14,
                    "{} at JD {jd}: det R = {}",
                    body.name(),
                    r.determinant()
                );
            }
        }
    }

    /// The third row of the rotation is the body's north pole in ICRF.
    ///
    /// This is the test that pins the sign convention of `rot_x` and `rot_z`:
    /// the rows of a frame rotation are the new frame's axes written in the
    /// old frame, and the z axis of a body-fixed frame is the body's pole. Get
    /// a sign wrong and this fails while orthonormality still passes.
    #[test]
    fn test_third_row_is_the_pole() {
        let ts = Timescale::default();
        let t = ts.tdb_jd(2455362.5);
        for body in [Body::Mars, Body::Jupiter, Body::Moon] {
            let frame = IauFrame::from_body(body);
            let (ra, dec, _w) = frame.pole_and_meridian(&t);
            let pole = Vector3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
            let row = frame.rotation_at(&t).row(2).transpose();
            assert!(
                (row - pole).abs().max() < 1e-14,
                "{}: third row {row} is not the pole {pole}",
                body.name()
            );
        }
    }

    /// The rate matrix matches the derivative block of SpiceyPy `sxform`.
    #[test]
    fn test_rotation_rate_matches_spiceypy_golden() {
        let ts = Timescale::default();
        let cases = [
            (Body::Mars, MARS_GOLDEN_RATE),
            (Body::Earth, EARTH_GOLDEN_RATE),
            (Body::Moon, MOON_GOLDEN_RATE),
            (Body::Jupiter, JUPITER_GOLDEN_RATE),
        ];

        for (body, golden) in cases {
            let frame = IauFrame::from_body(body);
            for (jd, expected) in GOLDEN_EPOCHS.iter().zip(golden.iter()) {
                let expected = Matrix3::from_row_slice(expected);
                let (rotation, rate) = frame.rotation_and_rate_at(&ts.tdb_jd(*jd));
                assert_eq!(rotation, frame.rotation_at(&ts.tdb_jd(*jd)));

                let error = (rate - expected).abs().max() / expected.abs().max();
                assert!(
                    error < 1e-9,
                    "{} at JD {jd}: the rate matrix is {error} (relative) from SPICE",
                    body.name()
                );
            }
        }
    }

    /// The rate matrix is also the central difference of the rotation.
    ///
    /// A guard against the two ever parting company, at a tolerance set by
    /// how well a difference quotient can do: the unwrapped W of a fast
    /// rotator is thousands of radians, so its sine is good to about 10⁻¹³.
    #[test]
    fn test_rotation_rate_matches_finite_difference() {
        let ts = Timescale::default();
        let jd = 2455362.5;
        // A power of two, so `jd ± step` is exact.
        let step = f64::exp2(-13.0); // days

        for body in [Body::Mars, Body::Moon, Body::Jupiter] {
            let frame = IauFrame::from_body(body);
            let rate = frame.rotation_and_rate_at(&ts.tdb_jd(jd)).1;

            let ahead = frame.rotation_at(&ts.tdb_jd(jd + step));
            let behind = frame.rotation_at(&ts.tdb_jd(jd - step));
            let numeric = (ahead - behind) / (2.0 * step);

            let error = (rate - numeric).abs().max() / rate.abs().max();
            assert!(
                error < 1e-6,
                "{}: the rate matrix is {error} (relative) from the finite difference",
                body.name()
            );
        }
    }

    /// The kernel and the embedded table build the same frame.
    #[test]
    fn test_new_from_kernel_matches_the_embedded_table() {
        let mut pc = PlanetaryConstants::new();
        pc.read_text(EXCERPT).unwrap();
        for body in [Body::Mars, Body::Moon, Body::Jupiter, Body::Earth] {
            let from_kernel = IauFrame::new(body.naif_id(), &pc).unwrap();
            assert_eq!(from_kernel, IauFrame::from_body(body));
        }
    }

    /// A body the kernels say nothing about is an error, not a silent default.
    #[test]
    fn test_new_rejects_an_unknown_body() {
        let pc = PlanetaryConstants::new();
        assert!(IauFrame::new(499, &pc).is_err());
        assert!(IauFrame::from_naif_id(499).is_some());
        assert!(IauFrame::from_naif_id(-1).is_none());
    }

    /// `frame_for` hands the Earth to `ItrsFrame` and everything else to
    /// `IauFrame`.
    #[test]
    fn test_frame_for_picks_itrs_for_the_earth() {
        use crate::framelib::ItrsFrame;

        let mut pc = PlanetaryConstants::new();
        pc.read_text(EXCERPT).unwrap();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2455362.5);

        let earth = pc.frame_for(399).unwrap().rotation_at(&t);
        assert_eq!(earth, ItrsFrame.rotation_at(&t));
        // The IAU elements are the coarse alternative, good to ~0.1 degree.
        let iau = IauFrame::from_body(Body::Earth).rotation_at(&t);
        let offset = angle_between(&iau, &earth).to_degrees();
        assert!(offset < 0.2, "IAU and ITRS Earth differ by {offset} deg");

        let mars = pc.frame_for(499).unwrap().rotation_at(&t);
        assert_eq!(mars, IauFrame::from_body(Body::Mars).rotation_at(&t));
        assert!(pc.frame_for(-1).is_err());
    }

    /// With no kernel read, `frame_for` falls back to the embedded table for
    /// the bodies it covers, and kernel values win once one is read.
    #[test]
    fn test_frame_for_falls_back_to_the_embedded_table() {
        let ts = Timescale::default();
        let t = ts.tdb_jd(2455362.5);

        let empty = PlanetaryConstants::new();
        let mars = empty.frame_for(499).unwrap().rotation_at(&t);
        assert_eq!(mars, IauFrame::from_body(Body::Mars).rotation_at(&t));
        let moon = empty.frame_for(301).unwrap().rotation_at(&t);
        assert_eq!(moon, IauFrame::from_body(Body::Moon).rotation_at(&t));
        // Phobos is in pck00011.tpc but not in the embedded table.
        assert!(empty.frame_for(401).is_err());

        let mut pc = PlanetaryConstants::new();
        pc.read_text(concat!(
            "KPL/PCK\n\\begindata\n",
            "BODY499_POLE_RA = ( 300.0 0.0 0.0 )\n",
            "BODY499_POLE_DEC = ( 50.0 0.0 0.0 )\n",
            "BODY499_PM = ( 0.0 0.0 0.0 )\n",
            "\\begintext\n",
        ))
        .unwrap();
        let from_kernel = pc.frame_for(499).unwrap().rotation_at(&t);
        assert_ne!(from_kernel, mars, "kernel values must override the table");
        let (ra, dec, _) = IauFrame::new(499, &pc).unwrap().pole_and_meridian(&t);
        assert!((ra.to_degrees() - 300.0).abs() < 1e-12);
        assert!((dec.to_degrees() - 50.0).abs() < 1e-12);
    }
}
