//! Python comparison tests for time module
//!
//! Validates Rust leap-second detection, delta-T spline, sidereal time,
//! and Earth rotation matrices against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn parse_f64_array(result: &str) -> Vec<f64> {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::Array {
                dtype,
                shape: _,
                data,
            } => {
                assert_eq!(dtype, "float64");
                let n = data.len() / 8;
                let mut values = Vec::with_capacity(n);
                for i in 0..n {
                    let bytes: [u8; 8] = data[i * 8..(i + 1) * 8].try_into().unwrap();
                    values.push(f64::from_le_bytes(bytes));
                }
                values
            }
            _ => panic!("Expected Array result, got {:?}", parsed),
        }
    }

    fn fetch_delta_t(bridge: &PyRustBridge, tt_jd: f64) -> f64 {
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({tt_jd})
rust.collect_string(str(t.delta_t))
"#
            ))
            .unwrap_or_else(|e| panic!("Python failed for delta_t at JD {tt_jd}: {e}"));
        parse_f64(&py_result)
    }

    fn year_to_tt_jd(year: f64) -> f64 {
        2451545.0 + (year - 2000.0) * 365.25
    }

    fn fetch_scalar(bridge: &PyRustBridge, jd: f64, prop: &str) -> f64 {
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
rust.collect_string(str(t.{prop}))
"#
            ))
            .unwrap_or_else(|e| panic!("Python failed for {prop} at JD {jd}: {e}"));
        parse_f64(&py_result)
    }

    fn fetch_matrix(bridge: &PyRustBridge, jd: f64, prop: &str) -> Vec<f64> {
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
import numpy as np
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
rust.collect_array(np.array(t.{prop}.flatten(), dtype=np.float64))
"#
            ))
            .unwrap_or_else(|e| panic!("Python failed for {prop} at JD {jd}: {e}"));
        let vals = parse_f64_array(&py_result);
        assert_eq!(vals.len(), 9, "{prop} matrix should have 9 elements");
        vals
    }

    fn assert_matrices_match(
        rust: &nalgebra::Matrix3<f64>,
        py: &[f64],
        label: &str,
        jd: f64,
        epsilon: f64,
    ) {
        for i in 0..3 {
            for j in 0..3 {
                let rust_val = rust[(i, j)];
                let py_val = py[i * 3 + j];
                let diff = (rust_val - py_val).abs();
                assert!(
                    diff < epsilon,
                    "{label}[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff} (tol={epsilon})"
                );
            }
        }
    }

    const TEST_JDS: [f64; 5] = [
        2451545.0, // J2000.0
        2455000.5, // ~2009
        2458000.5, // ~2017
        2458849.5, // ~2020-01-01
        2460000.5, // ~2023
    ];

    // --- Leap second tests ---

    /// Test that second=60 leap second produces the correct TT JD vs Skyfield
    #[test]
    fn test_leap_second_60_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.utc(2016, 12, 31, 23, 59, 60)
rust.collect_string(str(t.tt))
"#,
            )
            .expect("Failed to run Python code");

        let py_tt = parse_f64(&py_result);

        let ts = Timescale::default();
        let t = ts.utc((2016, 12, 31, 23, 59, 60.0));
        let rust_tt = t.tt();

        let diff = (rust_tt - py_tt).abs() * 86400.0;
        assert!(
            diff < 2.0,
            "Leap second TT mismatch: rust={rust_tt} python={py_tt} diff={diff}s"
        );

        assert!(t.is_leap_second());
    }

    /// Test that normal times (non-leap) produce the same TT as Skyfield
    #[test]
    fn test_utc_tt_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.utc(2016, 12, 31, 23, 59, 59)
rust.collect_string(str(t.tt))
"#,
            )
            .expect("Failed to run Python code");

        let py_tt = parse_f64(&py_result);

        let ts = Timescale::default();
        let t = ts.utc((2016, 12, 31, 23, 59, 59.0));
        let rust_tt = t.tt();

        let diff = (rust_tt - py_tt).abs() * 86400.0;
        assert!(
            diff < 2.0,
            "UTC TT mismatch: rust={rust_tt} python={py_tt} diff={diff}s"
        );
        assert!(!t.is_leap_second());
    }

    // --- Delta-T tests ---

    #[test]
    fn test_delta_t_at_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();
        let py_dt = fetch_delta_t(&bridge, 2451545.0);
        let rust_dt = ts.delta_t(2451545.0);
        assert_relative_eq!(rust_dt, py_dt, epsilon = 0.5);
    }

    #[test]
    fn test_delta_t_within_s15_pre_iers_range() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();
        let years = [
            -500.0, 0.0, 500.0, 1000.0, 1500.0, 1700.0, 1800.0, 1850.0, 1900.0, 1950.0, 1970.0,
        ];
        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);
            assert_relative_eq!(rust_dt, py_dt, epsilon = 0.5);
        }
    }

    #[test]
    fn test_delta_t_post_1973_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();
        let years = [1980.0, 1990.0, 2000.0, 2005.0, 2010.0, 2015.0, 2018.0];
        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);
            assert_relative_eq!(rust_dt, py_dt, epsilon = 1.0);
        }
    }

    #[test]
    fn test_delta_t_far_past() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();
        let years = [-2000.0, -1000.0];
        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);
            let max_relative = 0.02;
            let diff_frac = ((rust_dt - py_dt) / py_dt).abs();
            assert!(
                diff_frac < max_relative,
                "delta-T at year {year}: rust={rust_dt:.2} python={py_dt:.2} relative_diff={diff_frac:.4}"
            );
        }
    }

    #[test]
    fn test_delta_t_monotonic_recent() {
        let ts = Timescale::default();
        let mut prev = ts.delta_t(year_to_tt_jd(1970.0));
        for year in (1971..=2019).step_by(1) {
            let dt = ts.delta_t(year_to_tt_jd(year as f64));
            assert!(dt >= prev - 0.1, "delta-T not monotonic at year {year}: {prev:.4} -> {dt:.4}");
            prev = dt;
        }
    }

    #[test]
    fn test_delta_t_positive_modern() {
        let ts = Timescale::default();
        for year in (1970..=2018).step_by(1) {
            let dt = ts.delta_t(year_to_tt_jd(year as f64));
            assert!(dt > 0.0, "delta-T should be positive at year {year}: got {dt:.4}");
        }
    }

    #[test]
    fn test_delta_t_smoothness() {
        let ts = Timescale::default();
        let mut prev = ts.delta_t(year_to_tt_jd(1800.0));
        for year_10x in 18010..=20190 {
            let year = year_10x as f64 / 10.0;
            let dt = ts.delta_t(year_to_tt_jd(year));
            let change_per_year = (dt - prev).abs() * 10.0;
            assert!(change_per_year < 5.0, "delta-T discontinuity near year {year:.1}: prev={prev:.4} curr={dt:.4} rate={change_per_year:.2}s/yr");
            prev = dt;
        }
    }

    // --- GMST / GAST tests ---

    #[test]
    fn test_gmst_at_j2000_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let py_gmst = fetch_scalar(&bridge, 2451545.0, "gmst");
        let rust_gmst = ts.tt_jd(2451545.0, None).gmst();
        assert_relative_eq!(rust_gmst, py_gmst, epsilon = 5e-4);
    }

    #[test]
    fn test_gast_at_j2000_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let py_gast = fetch_scalar(&bridge, 2451545.0, "gast");
        let rust_gast = ts.tt_jd(2451545.0, None).gast();
        assert_relative_eq!(rust_gast, py_gast, epsilon = 5e-4);
    }

    #[test]
    fn test_gmst_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_gmst = fetch_scalar(&bridge, jd, "gmst");
            let rust_gmst = ts.tt_jd(jd, None).gmst();
            assert_relative_eq!(rust_gmst, py_gmst, epsilon = 2e-3);
        }
    }

    #[test]
    fn test_gast_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_gast = fetch_scalar(&bridge, jd, "gast");
            let rust_gast = ts.tt_jd(jd, None).gast();
            assert_relative_eq!(rust_gast, py_gast, epsilon = 2e-3);
        }
    }

    #[test]
    fn test_gmst_in_valid_range() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let gmst = ts.tt_jd(jd, None).gmst();
            assert!((0.0..24.0).contains(&gmst), "GMST={gmst} out of [0, 24) at JD {jd}");
        }
    }

    #[test]
    fn test_gast_in_valid_range() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let gast = ts.tt_jd(jd, None).gast();
            assert!((0.0..24.0).contains(&gast), "GAST={gast} out of [0, 24) at JD {jd}");
        }
    }

    #[test]
    fn test_sidereal_rate() {
        let ts = Timescale::default();
        let jd1 = 2451545.0;
        let jd2 = jd1 + 0.99726957;
        let gmst1 = ts.tt_jd(jd1, None).gmst();
        let gmst2 = ts.tt_jd(jd2, None).gmst();
        let advance = if gmst2 > gmst1 { gmst2 - gmst1 } else { gmst2 + 24.0 - gmst1 };
        assert_relative_eq!(advance, 24.0, epsilon = 0.01);
    }

    #[test]
    fn test_equation_of_equinoxes_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_result = bridge.run_py_to_json(&format!(r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
rust.collect_string(str(t.gast - t.gmst))
"#)).expect("Failed to run Python code");
            let py_eq_eq = parse_f64(&py_result);
            let rust_eq_eq = ts.tt_jd(jd, None).gast() - ts.tt_jd(jd, None).gmst();
            assert_relative_eq!(rust_eq_eq, py_eq_eq, epsilon = 1e-5);
        }
    }

    // --- Matrix tests ---

    #[test]
    fn test_m_matrix_at_j2000_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let jd = 2451545.0;
        let py_m = fetch_matrix(&bridge, jd, "M");
        let m = ts.tt_jd(jd, None).m_matrix();
        assert_matrices_match(&m, &py_m, "M", jd, 1e-8);
    }

    #[test]
    fn test_m_matrix_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_m = fetch_matrix(&bridge, jd, "M");
            let m = ts.tt_jd(jd, None).m_matrix();
            assert_matrices_match(&m, &py_m, "M", jd, 1e-8);
        }
    }

    #[test]
    fn test_mt_matrix_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_mt = fetch_matrix(&bridge, jd, "MT");
            let mt = ts.tt_jd(jd, None).mt_matrix();
            assert_matrices_match(&mt, &py_mt, "MT", jd, 1e-8);
        }
    }

    #[test]
    fn test_c_matrix_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_c = fetch_matrix(&bridge, jd, "C");
            let c = ts.tt_jd(jd, None).c_matrix();
            assert_matrices_match(&c, &py_c, "C", jd, 1e-5);
        }
    }

    #[test]
    fn test_ct_matrix_at_multiple_dates_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let py_ct = fetch_matrix(&bridge, jd, "CT");
            let ct = ts.tt_jd(jd, None).ct_matrix();
            assert_matrices_match(&ct, &py_ct, "CT", jd, 1e-5);
        }
    }

    // --- Structural properties ---

    #[test]
    fn test_m_matrix_orthogonality() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let m = ts.tt_jd(jd, None).m_matrix();
            let product = m.transpose() * m;
            for i in 0..3 { for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-14);
            }}
            assert_relative_eq!(m.determinant(), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_c_matrix_orthogonality() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let c = ts.tt_jd(jd, None).c_matrix();
            let product = c.transpose() * c;
            for i in 0..3 { for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(product[(i, j)], expected, epsilon = 1e-14);
            }}
            assert_relative_eq!(c.determinant(), 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_mt_is_transpose_of_m() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let t = ts.tt_jd(jd, None);
            let m = t.m_matrix();
            let mt = t.mt_matrix();
            for i in 0..3 { for j in 0..3 {
                assert_relative_eq!(mt[(i, j)], m[(j, i)], epsilon = 1e-15);
            }}
        }
    }

    #[test]
    fn test_ct_is_transpose_of_c() {
        let ts = Timescale::default();
        for &jd in &TEST_JDS {
            let t = ts.tt_jd(jd, None);
            let c = t.c_matrix();
            let ct = t.ct_matrix();
            for i in 0..3 { for j in 0..3 {
                assert_relative_eq!(ct[(i, j)], c[(j, i)], epsilon = 1e-15);
            }}
        }
    }
}
