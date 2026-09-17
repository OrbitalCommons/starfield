//! Nutation computations based on the IAU 2000A model
//!
//! Implements nutation in longitude (delta-psi) and obliquity (delta-epsilon),
//! mean obliquity of the ecliptic, the nutation rotation matrix, and the
//! complementary terms of the equation of equinoxes.
//!
//! Full IAU 2000A model: 678 lunisolar terms + 687 planetary terms.

#[cfg(all(test, feature = "python-tests"))]
mod python_tests;

mod iau2000a_data;

use crate::constants::{ASEC2RAD, J2000, TAU};
use nalgebra::Matrix3;

/// Conversion factor from 0.1 microarcsecond to radians
const TENTH_USEC_2_RAD: f64 = ASEC2RAD / 1e7;

/// Planetary fundamental argument coefficients [constant, rate] in radians
/// From Souchay et al. (1999) and IERS Conventions (2003)
const PLANETARY_FA: [[f64; 2]; 14] = [
    // Moon: l
    [2.35555598, 8328.6914269554],
    // Sun: l'
    [6.24006013, 628.301955],
    // Moon: F
    [1.627905234, 8433.466158131],
    // Moon-Sun: D
    [5.198466741, 7771.3771468121],
    // Moon: Omega
    [2.18243920, -33.757045],
    // Mercury
    [4.402608842, 2608.7903141574],
    // Venus
    [3.176146697, 1021.3285546211],
    // Earth
    [1.753470314, 628.3075849991],
    // Mars
    [6.203480913, 334.0612426700],
    // Jupiter
    [0.599546497, 52.9690962641],
    // Saturn
    [0.874016757, 21.3299104960],
    // Uranus
    [5.481293871, 7.4781598567],
    // Neptune
    [5.321159000, 3.8127774000],
    // General precession in longitude
    [0.02438175, 0.00000538691],
];

/// Fundamental argument polynomial coefficients
///
/// Each row: [constant, t^1, t^2, t^3, t^4] in arcseconds
/// From IERS Conventions (2003), Chapter 5
#[rustfmt::skip]
const FA_COEFFS: [[f64; 5]; 5] = [
    // Mean Anomaly of the Moon (l)
    [485868.249036, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
    // Mean Anomaly of the Sun (l')
    [1287104.79305, 129596581.0481, -0.5532, 0.000136, -0.00001149],
    // Mean Longitude of Moon - Ascending Node (F)
    [335779.526232, 1739527262.8478, -12.7512, -0.001037, 0.00000417],
    // Mean Elongation of Moon from Sun (D)
    [1072260.70369, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
    // Mean Longitude of Ascending Node of Moon (Omega)
    [450160.398036, -6962890.5431, 7.4722, 0.007702, -0.00005939],
];

/// Compute the five fundamental arguments (Delaunay variables) in radians
///
/// # Arguments
/// * `t` - TDB/TT time in Julian centuries since J2000.0
pub fn fundamental_arguments(t: f64) -> [f64; 5] {
    let mut fa = [0.0f64; 5];
    for (i, coeffs) in FA_COEFFS.iter().enumerate() {
        let val = coeffs[0] + (coeffs[1] + (coeffs[2] + (coeffs[3] + coeffs[4] * t) * t) * t) * t;
        fa[i] = val * ASEC2RAD;
    }
    fa
}

/// Compute the 14 fundamental arguments for planetary nutation in radians
///
/// Arguments 0-4: Delaunay variables (l, l', F, D, Omega)
/// Arguments 5-12: Planetary mean longitudes (Mercury through Neptune)
/// Argument 13: General precession in longitude (multiplied by t)
fn planetary_fundamental_arguments(t: f64) -> [f64; 14] {
    let mut fa = [0.0f64; 14];
    for (i, coeffs) in PLANETARY_FA.iter().enumerate() {
        fa[i] = coeffs[0] + coeffs[1] * t;
    }
    // General precession in longitude gets an extra factor of t
    fa[13] *= t;
    fa
}

/// Compute nutation angles (delta-psi, delta-epsilon) in radians
///
/// Full IAU 2000A model: 678 lunisolar + 687 planetary nutation terms.
///
/// # Arguments
/// * `tt_jd` - TT Julian date
///
/// # Returns
/// * `(d_psi, d_eps)` - nutation in longitude and obliquity, in radians
pub fn iau2000a_nutation(tt_jd: f64) -> (f64, f64) {
    let t = (tt_jd - J2000) / 36525.0;
    let fa = fundamental_arguments(t);

    let mut d_psi = 0.0;
    let mut d_eps = 0.0;

    // Lunisolar nutation (678 terms)
    for i in 0..iau2000a_data::NALS_T.len() {
        let mut arg = 0.0;
        for (j, &fa_val) in fa.iter().enumerate() {
            arg += iau2000a_data::NALS_T[i][j] as f64 * fa_val;
        }

        let sin_arg = arg.sin();
        let cos_arg = arg.cos();

        d_psi += (iau2000a_data::LUNISOLAR_LONGITUDE[i][0]
            + iau2000a_data::LUNISOLAR_LONGITUDE[i][1] * t)
            * sin_arg
            + iau2000a_data::LUNISOLAR_LONGITUDE[i][2] * cos_arg;
        d_eps += (iau2000a_data::LUNISOLAR_OBLIQUITY[i][0]
            + iau2000a_data::LUNISOLAR_OBLIQUITY[i][1] * t)
            * cos_arg
            + iau2000a_data::LUNISOLAR_OBLIQUITY[i][2] * sin_arg;
    }

    // Planetary nutation (687 terms)
    let fa14 = planetary_fundamental_arguments(t);
    for i in 0..iau2000a_data::NAPL_T.len() {
        let mut arg = 0.0;
        for (j, &fa_val) in fa14.iter().enumerate() {
            arg += iau2000a_data::NAPL_T[i][j] as f64 * fa_val;
        }

        let sin_arg = arg.sin();
        let cos_arg = arg.cos();

        d_psi += iau2000a_data::PLANETARY_LONGITUDE[i][0] * sin_arg
            + iau2000a_data::PLANETARY_LONGITUDE[i][1] * cos_arg;
        d_eps += iau2000a_data::PLANETARY_OBLIQUITY[i][0] * sin_arg
            + iau2000a_data::PLANETARY_OBLIQUITY[i][1] * cos_arg;
    }

    (d_psi * TENTH_USEC_2_RAD, d_eps * TENTH_USEC_2_RAD)
}

/// Compute the mean obliquity of the ecliptic in radians
///
/// From Capitaine et al. (2003), Astronomy and Astrophysics 412, 567-586.
///
/// # Arguments
/// * `jd_tdb` - TDB Julian date
pub fn mean_obliquity(jd_tdb: f64) -> f64 {
    let t = (jd_tdb - J2000) / 36525.0;

    let epsilon = ((((-0.0000000434 * t - 0.000000576) * t + 0.00200340) * t - 0.0001831) * t
        - 46.836769)
        * t
        + 84381.406;

    epsilon * ASEC2RAD
}

/// Build the nutation rotation matrix
///
/// Matches Skyfield's construction using mean obliquity, true obliquity,
/// and nutation in longitude (d_psi), all as positive angles.
///
/// # Arguments
/// * `mean_obliquity_rad` - mean obliquity of ecliptic in radians
/// * `d_psi` - nutation in longitude in radians
/// * `d_eps` - nutation in obliquity in radians
pub fn build_nutation_matrix(mean_obliquity_rad: f64, d_psi: f64, d_eps: f64) -> Matrix3<f64> {
    let eps_mean = mean_obliquity_rad;
    let eps_true = eps_mean + d_eps;

    let (sobm, cobm) = eps_mean.sin_cos();
    let (sobt, cobt) = eps_true.sin_cos();
    let (spsi, cpsi) = d_psi.sin_cos();

    #[rustfmt::skip]
    let n = Matrix3::new(
        cpsi,          -spsi * cobm,                      -spsi * sobm,
        spsi * cobt,    cpsi * cobm * cobt + sobm * sobt,  cpsi * sobm * cobt - cobm * sobt,
        spsi * sobt,    cpsi * cobm * sobt - sobm * cobt,  cpsi * sobm * sobt + cobm * cobt,
    );

    n
}

/// Number of complementary terms for equation of equinoxes
const N_EQEQ_TERMS: usize = 33;

/// Complementary terms argument multipliers (14 fundamental arguments each)
#[rustfmt::skip]
const KE0: [[i32; 14]; N_EQEQ_TERMS] = [
    [  0,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,  -2,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,  -2,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,  -2,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   0,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   0,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   2,  -2,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   2,  -2,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   4,  -4,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   1,  -1,   1,   0,  -8,  12,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   2,   0,   3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   2,   0,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,  -2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,  -2,   2,  -3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,  -2,   2,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   0,   0,   0,   0,   8, -13,   0,   0,   0,   0,   0,  -1],
    [  0,   0,   0,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  2,   0,  -2,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   0,  -2,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   1,   2,  -2,   2,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,   0,  -2,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   4,  -2,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  0,   0,   2,  -2,   4,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,  -2,   0,  -3,   0,   0,   0,   0,   0,   0,   0,   0,   0],
    [  1,   0,  -2,   0,  -1,   0,   0,   0,   0,   0,   0,   0,   0,   0],
];

/// Complementary terms sine coefficients (arcseconds)
#[rustfmt::skip]
const SE0_SIN: [f64; N_EQEQ_TERMS] = [
    2.640960e-03, 6.352000e-05, 1.175000e-05, 1.121000e-05, -4.550000e-06,
    2.020000e-06, 1.980000e-06, -1.720000e-06, -1.410000e-06, -1.260000e-06,
    -6.300000e-07, -6.300000e-07, 4.600000e-07, 4.500000e-07, 3.600000e-07,
    -2.400000e-07, 3.200000e-07, 2.800000e-07, 2.700000e-07, 2.600000e-07,
    -2.100000e-07, 1.900000e-07, 1.800000e-07, -1.000000e-07, 1.500000e-07,
    -1.400000e-07, 1.400000e-07, -1.400000e-07, 1.400000e-07, 1.300000e-07,
    -1.100000e-07, 1.100000e-07, 1.100000e-07,
];

/// Complementary terms cosine coefficients (arcseconds)
#[rustfmt::skip]
const SE0_COS: [f64; N_EQEQ_TERMS] = [
    -3.900000e-07, -2.000000e-08, 1.000000e-08, 1.000000e-08, 0.0,
    0.0, 0.0, 0.0, -1.000000e-08, -1.000000e-08,
    0.0, 0.0, 0.0, 0.0, 0.0,
    -1.200000e-07, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 5.000000e-08, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
];

/// Single time-dependent complementary term coefficients
const SE1_SIN: f64 = -0.87e-6;
const SE1_COS: f64 = 0.0;

/// Single time-dependent complementary term argument [l, l', F, D, Omega, ...]
const KE1: [i32; 14] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// Compute the complementary terms of the equation of equinoxes in radians
///
/// From IERS Conventions (2010), Chapter 5, Table 5.2e.
///
/// # Arguments
/// * `tt_jd` - TT Julian date
pub fn equation_of_the_equinoxes_complementary_terms(tt_jd: f64) -> f64 {
    let t = (tt_jd - J2000) / 36525.0;

    // Compute the 14 fundamental arguments
    let mut fa = [0.0f64; 14];

    // Moon and Sun arguments (Delaunay variables)
    fa[0] = (485868.249036
        + (715923.2178 + (31.8792 + (0.051635 + (-0.00024470) * t) * t) * t) * t)
        * ASEC2RAD
        + (1325.0 * t).rem_euclid(1.0) * TAU;

    fa[1] = (1287104.793048
        + (1292581.0481 + (-0.5532 + (0.000136 + (-0.00001149) * t) * t) * t) * t)
        * ASEC2RAD
        + (99.0 * t).rem_euclid(1.0) * TAU;

    fa[2] = (335779.526232
        + (295262.8478 + (-12.7512 + (-0.001037 + (0.00000417) * t) * t) * t) * t)
        * ASEC2RAD
        + (1342.0 * t).rem_euclid(1.0) * TAU;

    fa[3] = (1072260.703692
        + (1105601.2090 + (-6.3706 + (0.006593 + (-0.00003169) * t) * t) * t) * t)
        * ASEC2RAD
        + (1236.0 * t).rem_euclid(1.0) * TAU;

    fa[4] = (450160.398036
        + (-482890.5431 + (7.4722 + (0.007702 + (-0.00005939) * t) * t) * t) * t)
        * ASEC2RAD
        + (-5.0 * t).rem_euclid(1.0) * TAU;

    // Planetary longitudes (Mercury through Neptune)
    fa[5] = 4.402608842 + 2608.7903141574 * t;
    fa[6] = 3.176146697 + 1021.3285546211 * t;
    fa[7] = 1.753470314 + 628.3075849991 * t;
    fa[8] = 6.203480913 + 334.0612426700 * t;
    fa[9] = 0.599546497 + 52.9690962641 * t;
    fa[10] = 0.874016757 + 21.3299104960 * t;
    fa[11] = 5.481293872 + 7.4781598567 * t;
    fa[12] = 5.311886287 + 3.8133035638 * t;

    // General accumulated precession in longitude
    fa[13] = (0.024381750 + 0.00000538691 * t) * t;

    // Reduce to [0, 2π)
    for val in &mut fa {
        *val = val.rem_euclid(TAU);
    }

    // Evaluate the single time-dependent term
    let mut a = 0.0;
    for j in 0..14 {
        a += KE1[j] as f64 * fa[j];
    }
    let mut c_terms = SE1_SIN * a.sin() + SE1_COS * a.cos();
    c_terms *= t;

    // Evaluate the 33 constant terms
    for i in 0..N_EQEQ_TERMS {
        let mut arg = 0.0;
        for j in 0..14 {
            arg += KE0[i][j] as f64 * fa[j];
        }
        c_terms += SE0_SIN[i] * arg.sin() + SE0_COS[i] * arg.cos();
    }

    c_terms * ASEC2RAD
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fundamental_arguments_at_j2000() {
        let fa = fundamental_arguments(0.0);
        assert_relative_eq!(fa[0], FA_COEFFS[0][0] * ASEC2RAD, epsilon = 1e-10);
        assert_relative_eq!(fa[4], FA_COEFFS[4][0] * ASEC2RAD, epsilon = 1e-10);
    }

    #[test]
    fn test_nutation_at_j2000() {
        let (d_psi, d_eps) = iau2000a_nutation(J2000);
        let d_psi_asec = d_psi / ASEC2RAD;
        assert!(d_psi_asec.abs() < 20.0, "d_psi = {d_psi_asec} arcseconds");
        let d_eps_asec = d_eps / ASEC2RAD;
        assert!(d_eps_asec.abs() < 15.0, "d_eps = {d_eps_asec} arcseconds");
    }

    #[test]
    fn test_mean_obliquity_at_j2000() {
        let eps = mean_obliquity(J2000);
        let eps_deg = eps.to_degrees();
        assert_relative_eq!(eps_deg, 23.4393, epsilon = 0.001);
    }

    #[test]
    fn test_nutation_matrix_orthogonality() {
        let eps = mean_obliquity(J2000);
        let (d_psi, d_eps) = iau2000a_nutation(J2000);
        let n = build_nutation_matrix(eps, d_psi, d_eps);
        let product = n.transpose() * n;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_nutation_matrix_determinant() {
        let eps = mean_obliquity(J2000);
        let (d_psi, d_eps) = iau2000a_nutation(J2000);
        let n = build_nutation_matrix(eps, d_psi, d_eps);
        assert_relative_eq!(n.determinant(), 1.0, epsilon = 1e-14);
    }

    #[test]
    fn test_complementary_terms_small() {
        let c = equation_of_the_equinoxes_complementary_terms(J2000);
        let c_asec = c / ASEC2RAD;
        assert!(
            c_asec.abs() < 0.01,
            "complementary terms = {c_asec} arcseconds"
        );
    }
}
