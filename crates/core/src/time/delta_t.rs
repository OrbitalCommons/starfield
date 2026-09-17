//! Delta-T (TT - UT1) computation using cubic spline interpolation
//!
//! Uses the Morrison, Stephenson, Hohenkerk, and Zawilski Table S15.2020
//! cubic spline segments for years -720 to 2019, with the Stephenson-Morrison-
//! Hohenkerk 2016 long-term parabola for dates outside that range.
//! Transition splines provide smooth connections at the boundaries.

/// Number of segments in Table S15.2020
const N_SEGMENTS: usize = 58;

/// Table S15.2020 segment start years
const S15_X0: [f64; N_SEGMENTS] = [
    -720.0, -100.0, 400.0, 1000.0, 1150.0, 1300.0, 1500.0, 1600.0, 1650.0, 1720.0, 1800.0, 1810.0,
    1820.0, 1830.0, 1840.0, 1850.0, 1855.0, 1860.0, 1865.0, 1870.0, 1875.0, 1880.0, 1885.0, 1890.0,
    1895.0, 1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0,
    1953.0, 1956.0, 1959.0, 1962.0, 1965.0, 1968.0, 1971.0, 1974.0, 1977.0, 1980.0, 1983.0, 1986.0,
    1989.0, 1992.0, 1995.0, 1998.0, 2001.0, 2004.0, 2007.0, 2010.0, 2013.0, 2016.0,
];

/// Table S15.2020 segment end years
const S15_X1: [f64; N_SEGMENTS] = [
    -100.0, 400.0, 1000.0, 1150.0, 1300.0, 1500.0, 1600.0, 1650.0, 1720.0, 1800.0, 1810.0, 1820.0,
    1830.0, 1840.0, 1850.0, 1855.0, 1860.0, 1865.0, 1870.0, 1875.0, 1880.0, 1885.0, 1890.0, 1895.0,
    1900.0, 1905.0, 1910.0, 1915.0, 1920.0, 1925.0, 1930.0, 1935.0, 1940.0, 1945.0, 1950.0, 1953.0,
    1956.0, 1959.0, 1962.0, 1965.0, 1968.0, 1971.0, 1974.0, 1977.0, 1980.0, 1983.0, 1986.0, 1989.0,
    1992.0, 1995.0, 1998.0, 2001.0, 2004.0, 2007.0, 2010.0, 2013.0, 2016.0, 2019.0,
];

/// Table S15.2020 cubic coefficients (a3)
const S15_A3: [f64; N_SEGMENTS] = [
    4.0916000e+02,
    -5.0343300e+02,
    1.0850870e+03,
    -2.5346000e+01,
    -2.4641000e+01,
    -2.9414000e+01,
    1.6197000e+01,
    3.0180000e+00,
    -2.1270000e+00,
    -3.7939000e+01,
    1.9180000e+00,
    -3.8120000e+00,
    3.2500000e+00,
    -9.6000000e-02,
    -5.3900000e-01,
    -8.8300000e-01,
    1.5580000e+00,
    -2.4770000e+00,
    2.7200000e+00,
    -9.1400000e-01,
    -3.9000000e-02,
    5.6300000e-01,
    -1.4380000e+00,
    1.8710000e+00,
    -2.3200000e-01,
    -1.2570000e+00,
    7.2000000e-01,
    -8.2500000e-01,
    2.6200000e-01,
    8.0000000e-03,
    1.2700000e-01,
    1.4200000e-01,
    7.0200000e-01,
    -1.1060000e+00,
    6.1400000e-01,
    -2.7700000e-01,
    6.3100000e-01,
    -7.9900000e-01,
    5.0700000e-01,
    1.9900000e-01,
    -4.1400000e-01,
    2.0200000e-01,
    -2.2900000e-01,
    1.7200000e-01,
    -1.9200000e-01,
    8.1000000e-02,
    -1.6500000e-01,
    4.4800000e-01,
    -2.7600000e-01,
    1.1000000e-01,
    -3.1300000e-01,
    1.0900000e-01,
    1.9900000e-01,
    -1.7000000e-02,
    -8.4000000e-02,
    1.2800000e-01,
    -9.5000000e-02,
    -1.3900000e-01,
];

/// Table S15.2020 quadratic coefficients (a2)
const S15_A2: [f64; N_SEGMENTS] = [
    7.7624700e+02,
    1.3031510e+03,
    -2.9829100e+02,
    1.8481100e+02,
    1.0877100e+02,
    6.1953000e+01,
    -6.5720000e+00,
    1.0505000e+01,
    3.8333000e+01,
    4.1731000e+01,
    -1.1260000e+00,
    4.6290000e+00,
    -6.8060000e+00,
    2.9440000e+00,
    2.6580000e+00,
    2.6100000e-01,
    -2.3890000e+00,
    2.2840000e+00,
    -5.1480000e+00,
    3.0110000e+00,
    2.6900000e-01,
    1.5200000e-01,
    1.8420000e+00,
    -2.4740000e+00,
    3.1380000e+00,
    2.4430000e+00,
    -1.3290000e+00,
    8.3100000e-01,
    -1.6430000e+00,
    -8.5600000e-01,
    -8.3100000e-01,
    -4.4900000e-01,
    -2.2000000e-02,
    2.0860000e+00,
    -1.2320000e+00,
    2.2000000e-01,
    -6.1000000e-01,
    1.2820000e+00,
    -1.1150000e+00,
    4.0600000e-01,
    1.0020000e+00,
    -2.4200000e-01,
    3.6400000e-01,
    -3.2300000e-01,
    1.9300000e-01,
    -3.8400000e-01,
    -1.4000000e-01,
    -6.3700000e-01,
    7.0800000e-01,
    -1.2100000e-01,
    2.1000000e-01,
    -7.2900000e-01,
    -4.0200000e-01,
    1.9400000e-01,
    1.4400000e-01,
    -1.0900000e-01,
    2.7700000e-01,
    -7.0000000e-03,
];

/// Table S15.2020 linear coefficients (a1)
const S15_A1: [f64; N_SEGMENTS] = [
    -9.9995860e+03,
    -5.8222700e+03,
    -5.6715190e+03,
    -7.5321000e+02,
    -4.5962800e+02,
    -4.2134500e+02,
    -1.9284100e+02,
    -7.8697000e+01,
    -6.8089000e+01,
    2.5070000e+00,
    -3.4810000e+00,
    2.1000000e-02,
    -2.1570000e+00,
    -6.0180000e+00,
    -4.1600000e-01,
    1.6420000e+00,
    -4.8600000e-01,
    -5.9100000e-01,
    -3.4560000e+00,
    -5.5930000e+00,
    -2.3140000e+00,
    -1.8930000e+00,
    1.0100000e-01,
    -5.3100000e-01,
    1.3400000e-01,
    5.7150000e+00,
    6.8280000e+00,
    6.3300000e+00,
    5.5180000e+00,
    3.0200000e+00,
    1.3330000e+00,
    5.2000000e-02,
    -4.1900000e-01,
    1.6450000e+00,
    2.4990000e+00,
    1.1270000e+00,
    7.3700000e-01,
    1.4090000e+00,
    1.5770000e+00,
    8.6800000e-01,
    2.2750000e+00,
    3.0350000e+00,
    3.1570000e+00,
    3.1990000e+00,
    3.0690000e+00,
    2.8780000e+00,
    2.3540000e+00,
    1.5770000e+00,
    1.6480000e+00,
    2.2350000e+00,
    2.3240000e+00,
    1.8040000e+00,
    6.7400000e-01,
    4.6600000e-01,
    8.0400000e-01,
    8.3900000e-01,
    1.0070000e+00,
    1.2770000e+00,
];

/// Table S15.2020 constant coefficients (a0)
const S15_A0: [f64; N_SEGMENTS] = [
    2.0371848e+04,
    1.1557668e+04,
    6.5351160e+03,
    1.6503930e+03,
    1.0566470e+03,
    6.8114900e+02,
    2.9234300e+02,
    1.0912700e+02,
    4.3952000e+01,
    1.2068000e+01,
    1.8367000e+01,
    1.5678000e+01,
    1.6516000e+01,
    1.0804000e+01,
    7.6340000e+00,
    9.3380000e+00,
    1.0357000e+01,
    9.0400000e+00,
    8.2550000e+00,
    2.3710000e+00,
    -1.1260000e+00,
    -3.2100000e+00,
    -4.3880000e+00,
    -3.8840000e+00,
    -5.0170000e+00,
    -1.9770000e+00,
    4.9230000e+00,
    1.1142000e+01,
    1.7479000e+01,
    2.1617000e+01,
    2.3789000e+01,
    2.4418000e+01,
    2.4164000e+01,
    2.4426000e+01,
    2.7050000e+01,
    2.8932000e+01,
    3.0002000e+01,
    3.0760000e+01,
    3.2652000e+01,
    3.3621000e+01,
    3.5093000e+01,
    3.7956000e+01,
    4.0951000e+01,
    4.4244000e+01,
    4.7291000e+01,
    5.0361000e+01,
    5.2936000e+01,
    5.4984000e+01,
    5.6373000e+01,
    5.8453000e+01,
    6.0678000e+01,
    6.2898000e+01,
    6.4083000e+01,
    6.4553000e+01,
    6.5197000e+01,
    6.6061000e+01,
    6.6920000e+01,
    6.8109000e+01,
];

/// Long-term parabola coefficients (Stephenson-Morrison-Hohenkerk 2016)
///
/// Represents: delta_t = -320 + 32.5 * ((year - 1825) / 100)^2
/// Stored as spline: [x0=1825, x1=1925, a3=0, a2=32.5, a1=0, a0=-320]
const PARABOLA_X0: f64 = 1825.0;
const PARABOLA_X1: f64 = 1925.0;
const PARABOLA_WIDTH: f64 = PARABOLA_X1 - PARABOLA_X0; // 100.0
const PARABOLA_A2: f64 = 32.5;
const PARABOLA_A0: f64 = -320.0;

/// Width of transition splines connecting S15 table to long-term parabola
const PATCH_WIDTH: f64 = 800.0;

/// Evaluate the long-term parabola at a given year
fn parabola_eval(year: f64) -> f64 {
    let t = (year - PARABOLA_X0) / PARABOLA_WIDTH;
    // Horner's: ((a3*t + a2)*t + a1)*t + a0 with a3=0, a1=0
    (PARABOLA_A2 * t) * t + PARABOLA_A0
}

/// Evaluate the derivative of the long-term parabola at a given year
fn parabola_deriv(year: f64) -> f64 {
    let t = (year - PARABOLA_X0) / PARABOLA_WIDTH;
    // d/dx of ((a2*t)*t + a0) = 2*a2*t / width
    2.0 * PARABOLA_A2 * t / PARABOLA_WIDTH
}

/// Evaluate a single cubic spline segment at year `x`
///
/// Segment is defined by (x0, x1, a3, a2, a1, a0).
/// Uses Horner's method: value = ((a3*t + a2)*t + a1)*t + a0
/// where t = (x - x0) / (x1 - x0)
fn spline_eval(x: f64, x0: f64, x1: f64, a3: f64, a2: f64, a1: f64, a0: f64) -> f64 {
    let t = (x - x0) / (x1 - x0);
    ((a3 * t + a2) * t + a1) * t + a0
}

/// Evaluate the derivative of a cubic spline segment at year `x`
fn spline_deriv(x: f64, x0: f64, x1: f64, a3: f64, a2: f64, a1: f64, _a0: f64) -> f64 {
    let width = x1 - x0;
    let t = (x - x0) / width;
    // Derivative of ((a3*t + a2)*t + a1)*t + a0 w.r.t. x:
    // = (3*a3*t^2 + 2*a2*t + a1) / width
    (3.0 * a3 * t * t + 2.0 * a2 * t + a1) / width
}

/// Build a cubic spline that connects two endpoints with given slopes
///
/// Returns (a3, a2, a1, a0) for the segment [x0, x1]
fn build_spline_given_ends(
    x0: f64,
    y0: f64,
    slope0: f64,
    x1: f64,
    y1: f64,
    slope1: f64,
) -> (f64, f64, f64, f64) {
    let width = x1 - x0;
    let s0 = slope0 * width;
    let s1 = slope1 * width;
    let a0 = y0;
    let a1 = s0;
    let a2 = -2.0 * s0 - s1 - 3.0 * y0 + 3.0 * y1;
    let a3 = s0 + s1 + 2.0 * y0 - 2.0 * y1;
    (a3, a2, a1, a0)
}

/// Find the S15 segment index for a given year
fn find_s15_segment(year: f64) -> Option<usize> {
    if !(S15_X0[0]..=S15_X1[N_SEGMENTS - 1]).contains(&year) {
        return None;
    }
    // Binary search: find i such that S15_X0[i] <= year < S15_X1[i]
    match S15_X0.binary_search_by(|x| x.partial_cmp(&year).unwrap()) {
        Ok(i) => Some(i),
        Err(i) => {
            if i == 0 {
                Some(0)
            } else {
                Some(i - 1)
            }
        }
    }
}

/// Evaluate the S15 spline table at a given year
fn s15_eval(year: f64) -> f64 {
    if let Some(i) = find_s15_segment(year) {
        spline_eval(
            year, S15_X0[i], S15_X1[i], S15_A3[i], S15_A2[i], S15_A1[i], S15_A0[i],
        )
    } else {
        f64::NAN
    }
}

/// Evaluate the derivative of the S15 spline table at a given year
fn s15_deriv(year: f64) -> f64 {
    if let Some(i) = find_s15_segment(year) {
        spline_deriv(
            year, S15_X0[i], S15_X1[i], S15_A3[i], S15_A2[i], S15_A1[i], S15_A0[i],
        )
    } else {
        f64::NAN
    }
}

/// Pre-built composite delta-T evaluator
///
/// Structure mirrors Skyfield's `build_delta_t()`:
/// - Far left: pure long-term parabola (as single spline segment)
/// - Left transition: cubic spline connecting parabola to S15
/// - S15 region: Table S15.2020 splines
/// - Right of S15: polynomial extrapolation matching Skyfield's delta_t_approx
/// - Far right: pure long-term parabola
#[derive(Debug, Clone)]
pub struct DeltaT {
    /// Left transition spline: connects parabola to S15
    left_transition: SplineSegment,
    /// Far left boundary (left of transition)
    far_left: SplineSegment,
    /// Right transition spline: connects S15 end to parabola
    right_transition: SplineSegment,
    /// Far right boundary
    far_right: SplineSegment,
}

/// A single cubic spline segment
#[derive(Debug, Clone)]
struct SplineSegment {
    x0: f64,
    x1: f64,
    a3: f64,
    a2: f64,
    a1: f64,
    a0: f64,
}

impl SplineSegment {
    fn eval(&self, x: f64) -> f64 {
        spline_eval(x, self.x0, self.x1, self.a3, self.a2, self.a1, self.a0)
    }

    fn contains(&self, x: f64) -> bool {
        x >= self.x0 && x <= self.x1
    }
}

impl Default for DeltaT {
    fn default() -> Self {
        Self::new()
    }
}

impl DeltaT {
    /// Build the composite delta-T evaluator
    pub fn new() -> Self {
        // S15 boundaries
        let s15_left = S15_X0[0]; // -720.0
        let s15_right = S15_X1[N_SEGMENTS - 1]; // 2019.0

        // Build left transition: parabola → S15
        let left_x1 = s15_left;
        let left_x0 = left_x1 - PATCH_WIDTH;
        let (la3, la2, la1, la0) = build_spline_given_ends(
            left_x0,
            parabola_eval(left_x0),
            parabola_deriv(left_x0),
            left_x1,
            s15_eval(left_x1),
            s15_deriv(left_x1),
        );
        let left_transition = SplineSegment {
            x0: left_x0,
            x1: left_x1,
            a3: la3,
            a2: la2,
            a1: la1,
            a0: la0,
        };

        // Build far left: pure parabola segment
        let far_left_x1 = left_x0;
        let far_left_x0 = far_left_x1 - PARABOLA_WIDTH;
        let (fla3, fla2, fla1, fla0) = build_spline_given_ends(
            far_left_x0,
            parabola_eval(far_left_x0),
            parabola_deriv(far_left_x0),
            far_left_x1,
            parabola_eval(far_left_x1),
            parabola_deriv(far_left_x1),
        );
        let far_left = SplineSegment {
            x0: far_left_x0,
            x1: far_left_x1,
            a3: fla3,
            a2: fla2,
            a1: fla1,
            a0: fla0,
        };

        // Build right transition: S15 → parabola
        // Skyfield connects end of IERS daily data to parabola.
        // Since we don't bundle IERS daily data, we connect end of S15 to parabola.
        let right_x0 = s15_right;
        let right_x1 = ((right_x0 + PATCH_WIDTH) / 100.0).floor() * 100.0;
        let (ra3, ra2, ra1, ra0) = build_spline_given_ends(
            right_x0,
            s15_eval(right_x0),
            s15_deriv(right_x0),
            right_x1,
            parabola_eval(right_x1),
            parabola_deriv(right_x1),
        );
        let right_transition = SplineSegment {
            x0: right_x0,
            x1: right_x1,
            a3: ra3,
            a2: ra2,
            a1: ra1,
            a0: ra0,
        };

        // Build far right: pure parabola segment
        let far_right_x0 = right_x1;
        let far_right_x1 = far_right_x0 + PARABOLA_WIDTH;
        let (fra3, fra2, fra1, fra0) = build_spline_given_ends(
            far_right_x0,
            parabola_eval(far_right_x0),
            parabola_deriv(far_right_x0),
            far_right_x1,
            parabola_eval(far_right_x1),
            parabola_deriv(far_right_x1),
        );
        let far_right = SplineSegment {
            x0: far_right_x0,
            x1: far_right_x1,
            a3: fra3,
            a2: fra2,
            a1: fra1,
            a0: fra0,
        };

        DeltaT {
            left_transition,
            far_left,
            right_transition,
            far_right,
        }
    }

    /// Compute delta-T in seconds for a given TT Julian date
    pub fn compute(&self, tt_jd: f64) -> f64 {
        let year = (tt_jd - 1_721_045.0) / 365.25;
        self.compute_for_year(year)
    }

    /// Compute delta-T in seconds for a given year
    pub fn compute_for_year(&self, year: f64) -> f64 {
        // Check regions in order from center outward

        // S15 table region: -720 to 2019
        if (S15_X0[0]..=S15_X1[N_SEGMENTS - 1]).contains(&year) {
            return s15_eval(year);
        }

        // Left transition: connects parabola to S15
        if self.left_transition.contains(year) {
            return self.left_transition.eval(year);
        }

        // Right transition: connects S15 to parabola
        if self.right_transition.contains(year) {
            return self.right_transition.eval(year);
        }

        // Far left
        if self.far_left.contains(year) {
            return self.far_left.eval(year);
        }

        // Far right
        if self.far_right.contains(year) {
            return self.far_right.eval(year);
        }

        // Beyond all splines: pure long-term parabola
        parabola_eval(year)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parabola_at_known_points() {
        // At year 1825 (center), delta_t should be -320 + 32.5 * 0^2 = -320
        assert_relative_eq!(parabola_eval(1825.0), -320.0, epsilon = 1e-10);

        // At year 1925 (1 century later), delta_t = -320 + 32.5 * 1^2 = -287.5
        assert_relative_eq!(parabola_eval(1925.0), -287.5, epsilon = 1e-10);

        // At year 2025 (2 centuries later), delta_t = -320 + 32.5 * 4 = -190
        assert_relative_eq!(parabola_eval(2025.0), -190.0, epsilon = 1e-10);
    }

    #[test]
    fn test_s15_at_boundaries() {
        // At the start of the first segment, value should equal a0[0]
        let val = s15_eval(-720.0);
        assert_relative_eq!(val, S15_A0[0], epsilon = 1e-6);

        // At the start of the last segment
        let val = s15_eval(2016.0);
        assert_relative_eq!(val, S15_A0[N_SEGMENTS - 1], epsilon = 1e-6);
    }

    #[test]
    fn test_s15_continuity() {
        // At each segment boundary, the value should be continuous
        for i in 0..N_SEGMENTS - 1 {
            let boundary = S15_X1[i];
            let val_left = spline_eval(
                boundary, S15_X0[i], S15_X1[i], S15_A3[i], S15_A2[i], S15_A1[i], S15_A0[i],
            );
            let val_right = spline_eval(
                boundary,
                S15_X0[i + 1],
                S15_X1[i + 1],
                S15_A3[i + 1],
                S15_A2[i + 1],
                S15_A1[i + 1],
                S15_A0[i + 1],
            );
            assert_relative_eq!(val_left, val_right, epsilon = 0.1,);
        }
    }

    #[test]
    fn test_composite_delta_t_j2000() {
        let dt = DeltaT::new();
        // J2000 is year 2000.0 — within S15 range
        let val = dt.compute_for_year(2000.0);
        // Known delta-T for year 2000 is about 63.83 seconds
        assert_relative_eq!(val, 63.83, epsilon = 1.5);
    }

    #[test]
    fn test_composite_delta_t_recent() {
        let dt = DeltaT::new();
        // Year 2015 — within S15 range
        let val = dt.compute_for_year(2015.0);
        // Known delta-T for 2015 is about 68 seconds
        assert!((65.0..=70.0).contains(&val), "delta_t(2015) = {val}");
    }

    #[test]
    fn test_composite_delta_t_historical() {
        let dt = DeltaT::new();
        // Year 1900 — within S15 range
        let val = dt.compute_for_year(1900.0);
        // Known delta-T for 1900 is about -3 seconds
        assert!((-6.0..=0.0).contains(&val), "delta_t(1900) = {val}");
    }

    #[test]
    fn test_composite_delta_t_far_past() {
        let dt = DeltaT::new();
        // Year -2000 — well beyond S15, should use parabola
        let val = dt.compute_for_year(-2000.0);
        // Should be a large positive value (parabolic growth)
        assert!(val > 10000.0, "delta_t(-2000) = {val}");
    }

    #[test]
    fn test_composite_delta_t_far_future() {
        let dt = DeltaT::new();
        // Year 5000 — well beyond S15, should use parabola
        let val = dt.compute_for_year(5000.0);
        // Should be a large positive value
        assert!(val > 1000.0, "delta_t(5000) = {val}");
    }

    #[test]
    fn test_build_spline_given_ends() {
        // A simple line from (0, 0) to (1, 1) with slope 1 at both ends
        let (a3, a2, a1, a0) = build_spline_given_ends(0.0, 0.0, 1.0, 1.0, 1.0, 1.0);
        // Should be a pure line: a3=0, a2=0, a1=1, a0=0
        assert_relative_eq!(a3, 0.0, epsilon = 1e-10);
        assert_relative_eq!(a2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(a1, 1.0, epsilon = 1e-10);
        assert_relative_eq!(a0, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_jd() {
        let dt = DeltaT::new();
        // J2000 = JD 2451545.0 = year 2000.0
        let val = dt.compute(2_451_545.0);
        let val_year = dt.compute_for_year(2000.0);
        assert_relative_eq!(val, val_year, epsilon = 0.1);
    }
}
