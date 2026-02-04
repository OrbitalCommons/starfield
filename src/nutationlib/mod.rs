//! Nutation computations based on the IAU 2000A model
//!
//! Implements nutation in longitude (delta-psi) and obliquity (delta-epsilon),
//! mean obliquity of the ecliptic, the nutation rotation matrix, and the
//! complementary terms of the equation of equinoxes.
//!
//! Uses the top 77 lunisolar terms of the IAU 2000A model, which captures
//! 99.97% of the lunisolar nutation in longitude.

use crate::constants::{ASEC2RAD, J2000, TAU};
use nalgebra::Matrix3;

/// Conversion factor from 0.1 microarcsecond to radians
const TENTH_USEC_2_RAD: f64 = ASEC2RAD / 1e7;

/// Number of lunisolar nutation terms (top 77 by magnitude)
const N_NUTATION_TERMS: usize = 77;

/// Fundamental argument multipliers [l, l', F, D, Omega] for each term
#[rustfmt::skip]
const NALS: [[i32; 5]; N_NUTATION_TERMS] = [
    [  0,   0,   0,   0,   1], [  0,   0,   2,  -2,   2], [  0,   0,   2,   0,   2],
    [  0,   0,   0,   0,   2], [  0,   1,   0,   0,   0], [  0,   1,   2,  -2,   2],
    [  1,   0,   0,   0,   0], [  0,   0,   2,   0,   1], [  1,   0,   2,   0,   2],
    [  0,  -1,   2,  -2,   2], [  0,   0,   2,  -2,   1], [ -1,   0,   2,   0,   2],
    [ -1,   0,   0,   2,   0], [  1,   0,   0,   0,   1], [ -1,   0,   0,   0,   1],
    [ -1,   0,   2,   2,   2], [  1,   0,   2,   0,   1], [ -2,   0,   2,   0,   1],
    [  0,   0,   0,   2,   0], [  0,   0,   2,   2,   2], [  0,  -2,   2,  -2,   2],
    [ -2,   0,   0,   2,   0], [  2,   0,   2,   0,   2], [  1,   0,   2,  -2,   2],
    [ -1,   0,   2,   0,   1], [  2,   0,   0,   0,   0], [  0,   0,   2,   0,   0],
    [  0,   1,   0,   0,   1], [ -1,   0,   0,   2,   1], [  0,   2,   2,  -2,   2],
    [  0,   0,  -2,   2,   0], [  1,   0,   0,  -2,   1], [  0,  -1,   0,   0,   1],
    [ -1,   0,   2,   2,   1], [  0,   2,   0,   0,   0], [  1,   0,   2,   2,   2],
    [ -2,   0,   2,   0,   0], [  0,   1,   2,   0,   2], [  0,   0,   2,   2,   1],
    [  0,  -1,   2,   0,   2], [  0,   0,   0,   2,   1], [  1,   0,   2,  -2,   1],
    [  2,   0,   2,  -2,   2], [ -2,   0,   0,   2,   1], [  2,   0,   2,   0,   1],
    [  0,  -1,   2,  -2,   1], [  0,   0,   0,  -2,   1], [ -1,  -1,   0,   2,   0],
    [  2,   0,   0,  -2,   1], [  1,   0,   0,   2,   0], [  0,   1,   2,  -2,   1],
    [  1,  -1,   0,   0,   0], [ -2,   0,   2,   0,   2], [  3,   0,   2,   0,   2],
    [  0,  -1,   0,   2,   0], [  1,  -1,   2,   0,   2], [  0,   0,   0,   1,   0],
    [ -1,  -1,   2,   2,   2], [ -1,   0,   2,   0,   0], [  0,  -1,   2,   2,   2],
    [ -2,   0,   0,   0,   1], [  1,   1,   2,   0,   2], [  2,   0,   0,   0,   1],
    [ -1,   1,   0,   1,   0], [  1,   1,   0,   0,   0], [  1,   0,   2,   0,   0],
    [ -1,   0,   2,  -2,   1], [  1,   0,   0,   0,   2], [ -1,   0,   0,   1,   0],
    [  0,   0,   2,   1,   2], [ -1,   0,   2,   4,   2], [ -1,   1,   0,   1,   1],
    [  1,   0,   2,   2,   1], [ -2,   0,   2,   2,   2], [ -1,   0,   0,   0,   2],
    [  3,   0,   0,   0,   0], [ -1,   0,   0,   4,   0],
];

/// Longitude nutation coefficients [S_i, S_i_dot, C_i] in 0.1 microarcseconds
#[rustfmt::skip]
const LSC: [[f64; 3]; N_NUTATION_TERMS] = [
    [-172064161.0, -174666.0, 33386.0], [-13170906.0, -1675.0, -13696.0],
    [-2276413.0, -234.0, 2796.0], [2074554.0, 207.0, -698.0],
    [1475877.0, -3633.0, 11817.0], [-516821.0, 1226.0, -524.0],
    [711159.0, 73.0, -872.0], [-387298.0, -367.0, 380.0],
    [-301461.0, -36.0, 816.0], [215829.0, -494.0, 111.0],
    [128227.0, 137.0, 181.0], [123457.0, 11.0, 19.0],
    [156994.0, 10.0, -168.0], [63110.0, 63.0, 27.0],
    [-57976.0, -63.0, -189.0], [-59641.0, -11.0, 149.0],
    [-51613.0, -42.0, 129.0], [45893.0, 50.0, 31.0],
    [63384.0, 11.0, -150.0], [-38571.0, -1.0, 158.0],
    [32481.0, 0.0, 0.0], [-47722.0, 0.0, -18.0],
    [-31046.0, -1.0, 131.0], [28593.0, 0.0, -1.0],
    [20441.0, 21.0, 10.0], [29243.0, 0.0, -74.0],
    [25887.0, 0.0, -66.0], [-14053.0, -25.0, 79.0],
    [15164.0, 10.0, 11.0], [-15794.0, 72.0, -16.0],
    [21783.0, 0.0, 13.0], [-12873.0, -10.0, -37.0],
    [-12654.0, 11.0, 63.0], [-10204.0, 0.0, 25.0],
    [16707.0, -85.0, -10.0], [-7691.0, 0.0, 44.0],
    [-11024.0, 0.0, -14.0], [7566.0, -21.0, -11.0],
    [-6637.0, -11.0, 25.0], [-7141.0, 21.0, 8.0],
    [-6302.0, -11.0, 2.0], [5800.0, 10.0, 2.0],
    [6443.0, 0.0, -7.0], [-5774.0, -11.0, -15.0],
    [-5350.0, 0.0, 21.0], [-4752.0, -11.0, -3.0],
    [-4940.0, -11.0, -21.0], [7350.0, 0.0, -8.0],
    [4065.0, 0.0, 6.0], [6579.0, 0.0, -24.0],
    [3579.0, 0.0, 5.0], [4725.0, 0.0, -6.0],
    [-3075.0, 0.0, -2.0], [-2904.0, 0.0, 15.0],
    [4348.0, 0.0, -10.0], [-2878.0, 0.0, 8.0],
    [-4230.0, 0.0, 5.0], [-2819.0, 0.0, 7.0],
    [-4056.0, 0.0, 5.0], [-2647.0, 0.0, 11.0],
    [-2294.0, 0.0, -10.0], [2481.0, 0.0, -7.0],
    [2179.0, 0.0, -2.0], [3276.0, 0.0, 1.0],
    [-3389.0, 0.0, 5.0], [3339.0, 0.0, -13.0],
    [-1987.0, 0.0, -6.0], [-1981.0, 0.0, 0.0],
    [4026.0, 0.0, -353.0], [1660.0, 0.0, -5.0],
    [-1521.0, 0.0, 9.0], [1314.0, 0.0, 0.0],
    [-1331.0, 0.0, 8.0], [1383.0, 0.0, -2.0],
    [1405.0, 0.0, 4.0], [1575.0, 0.0, -6.0],
    [1338.0, 0.0, -5.0],
];

/// Obliquity nutation coefficients [C_i, C_i_dot, S_i] in 0.1 microarcseconds
#[rustfmt::skip]
const LOC: [[f64; 3]; N_NUTATION_TERMS] = [
    [92052331.0, 9086.0, 15377.0], [5730336.0, -3015.0, -4587.0],
    [978459.0, -485.0, 1374.0], [-897492.0, 470.0, -291.0],
    [73871.0, -184.0, -1924.0], [224386.0, -677.0, -174.0],
    [-6750.0, 0.0, 358.0], [200728.0, 18.0, 318.0],
    [129025.0, -63.0, 367.0], [-95929.0, 299.0, 132.0],
    [-68982.0, -9.0, 39.0], [-53311.0, 32.0, -4.0],
    [-1235.0, 0.0, 82.0], [-33228.0, 0.0, -9.0],
    [31429.0, 0.0, -75.0], [25543.0, -11.0, 66.0],
    [26366.0, 0.0, 78.0], [-24236.0, -10.0, 20.0],
    [-1220.0, 0.0, 29.0], [16452.0, -11.0, 68.0],
    [-13870.0, 0.0, 0.0], [477.0, 0.0, -25.0],
    [13238.0, -11.0, 59.0], [-12338.0, 10.0, -3.0],
    [-10758.0, 0.0, -3.0], [-609.0, 0.0, 13.0],
    [-550.0, 0.0, 11.0], [8551.0, -2.0, -45.0],
    [-8001.0, 0.0, -1.0], [6850.0, -42.0, -5.0],
    [-167.0, 0.0, 13.0], [6953.0, 0.0, -14.0],
    [6415.0, 0.0, 26.0], [5222.0, 0.0, 15.0],
    [168.0, -1.0, 10.0], [3268.0, 0.0, 19.0],
    [104.0, 0.0, 2.0], [-3250.0, 0.0, -5.0],
    [3353.0, 0.0, 14.0], [3070.0, 0.0, 4.0],
    [3272.0, 0.0, 4.0], [-3045.0, 0.0, -1.0],
    [-2768.0, 0.0, -4.0], [3041.0, 0.0, -5.0],
    [2695.0, 0.0, 12.0], [2719.0, 0.0, -3.0],
    [2720.0, 0.0, -9.0], [-51.0, 0.0, 4.0],
    [-2206.0, 0.0, 1.0], [-199.0, 0.0, 2.0],
    [-1900.0, 0.0, 1.0], [-41.0, 0.0, 3.0],
    [1313.0, 0.0, -1.0], [1233.0, 0.0, 7.0],
    [-81.0, 0.0, 2.0], [1232.0, 0.0, 4.0],
    [-20.0, 0.0, -2.0], [1207.0, 0.0, 3.0],
    [40.0, 0.0, -2.0], [1129.0, 0.0, 5.0],
    [1266.0, 0.0, -4.0], [-1062.0, 0.0, -3.0],
    [-1129.0, 0.0, -2.0], [-9.0, 0.0, 0.0],
    [35.0, 0.0, -2.0], [-107.0, 0.0, 1.0],
    [1073.0, 0.0, -2.0], [854.0, 0.0, 0.0],
    [-553.0, 0.0, -139.0], [-710.0, 0.0, -2.0],
    [647.0, 0.0, 4.0], [-700.0, 0.0, 0.0],
    [663.0, 0.0, 4.0], [-594.0, 0.0, -2.0],
    [-610.0, 0.0, 2.0], [-50.0, 0.0, 0.0],
    [-39.0, 0.0, 0.0],
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

/// Compute nutation angles (delta-psi, delta-epsilon) in radians
///
/// Uses the top 77 terms of the IAU 2000A lunisolar nutation series.
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

    for i in 0..N_NUTATION_TERMS {
        let mut arg = 0.0;
        for j in 0..5 {
            arg += NALS[i][j] as f64 * fa[j];
        }

        let sin_arg = arg.sin();
        let cos_arg = arg.cos();

        d_psi += (LSC[i][0] + LSC[i][1] * t) * sin_arg + LSC[i][2] * cos_arg;
        d_eps += (LOC[i][0] + LOC[i][1] * t) * cos_arg + LOC[i][2] * sin_arg;
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
/// N = R1(-eps_true) * R3(d_psi) * R1(eps_mean)
///
/// (Note: Skyfield constructs N = R1(eps_true) * R3(-d_psi) * R1(-eps_mean)
/// which is the inverse order convention, matching the forward nutation direction.)
///
/// # Arguments
/// * `mean_obliquity_rad` - mean obliquity of ecliptic in radians
/// * `d_psi` - nutation in longitude in radians
/// * `d_eps` - nutation in obliquity in radians
pub fn build_nutation_matrix(mean_obliquity_rad: f64, d_psi: f64, d_eps: f64) -> Matrix3<f64> {
    let eps_mean = mean_obliquity_rad;
    let eps_true = eps_mean + d_eps;

    let (sp, cp) = (-d_psi).sin_cos();
    let (se, ce) = (-eps_mean).sin_cos();
    let (set, cet) = (eps_true).sin_cos();

    // N = R1(eps_true) * R3(-d_psi) * R1(-eps_mean)
    #[rustfmt::skip]
    let n = Matrix3::new(
        cp,                     -sp * ce,                         -sp * se,
        sp * cet,               cp * ce * cet + se * set,         cp * se * cet - ce * set,
        sp * set,               cp * ce * set - se * cet,         cp * se * set + ce * cet,
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
