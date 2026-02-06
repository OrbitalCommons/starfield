//! Python comparison tests for polar motion
//!
//! Validates polar motion angles, W matrix, and ITRS rotation
//! against Python Skyfield with IERS polar motion data.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

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

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    const TEST_JDS: [f64; 5] = [
        2451545.0, // J2000.0
        2455000.5, // ~2009
        2458000.5, // ~2017
        2458849.5, // ~2020-01-01
        2460000.5, // ~2023
    ];

    /// Test polar motion angles (sprime) without table matches Skyfield
    #[test]
    fn test_sprime_without_table_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        for &jd in &TEST_JDS {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
sprime, x, y = t.polar_motion_angles()
rust.collect_string(str(sprime))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for sprime at JD {jd}: {e}"));

            let py_sprime = parse_f64(&py_result);
            let t = ts.tt_jd(jd, None);
            let (rust_sprime, _, _) = t.polar_motion_angles();

            assert_relative_eq!(rust_sprime, py_sprime, epsilon = 1e-6,);
        }
    }

    /// Test that without polar motion table, x and y are zero (matches Skyfield)
    #[test]
    fn test_polar_motion_angles_zero_without_table() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2451545.0)
sprime, x, y = t.polar_motion_angles()
rust.collect_string(f"{x},{y}")
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).unwrap();
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected string"),
        };
        let parts: Vec<f64> = s.split(',').map(|p| p.parse().unwrap()).collect();
        assert_eq!(parts[0], 0.0);
        assert_eq!(parts[1], 0.0);

        let t = ts.tt_jd(2451545.0, None);
        let (_, x, y) = t.polar_motion_angles();
        assert_eq!(x, 0.0);
        assert_eq!(y, 0.0);
    }

    /// Test polar motion matrix (W) without table is near-identity
    /// (only sprime contributes a tiny rotation)
    #[test]
    fn test_polar_motion_matrix_near_identity_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        for &jd in &TEST_JDS {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
import numpy as np
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
W = t.polar_motion_matrix()
rust.collect_array(np.array(W.flatten(), dtype=np.float64))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for W at JD {jd}: {e}"));

            let py_w = parse_f64_array(&py_result);
            let t = ts.tt_jd(jd, None);
            let w = t.polar_motion_matrix();

            for i in 0..3 {
                for j in 0..3 {
                    let rust_val = w[(i, j)];
                    let py_val = py_w[i * 3 + j];
                    let diff = (rust_val - py_val).abs();
                    assert!(
                        diff < 1e-12,
                        "W[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff}"
                    );
                }
            }
        }
    }

    /// Test ITRS rotation with polar motion table installed
    /// Uses a synthetic table with constant x=0.1, y=0.3 arcsec
    #[test]
    fn test_itrs_with_polar_motion_table_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let jd = 2458849.5; // 2020-01-01

        // Get Skyfield ITRS rotation with a polar motion table
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
import numpy as np
from skyfield.api import load

ts = load.timescale()
# Install a simple polar motion table
# Dates in TT JD, x and y in arcseconds
tt_dates = np.array([2458800.0, 2458850.0, 2458900.0])
x_vals = np.array([0.1, 0.1, 0.1])
y_vals = np.array([0.3, 0.3, 0.3])
ts.polar_motion_table = (tt_dates, x_vals, y_vals)

t = ts.tt_jd({jd})
from skyfield.framelib import itrs
R = itrs.rotation_at(t)
rust.collect_array(np.array(R.flatten(), dtype=np.float64))
"#
            ))
            .unwrap_or_else(|e| panic!("Python ITRS with polar motion failed: {e}"));

        let py_itrs = parse_f64_array(&py_result);

        // Create Rust Timescale with same polar motion table
        let mut ts = Timescale::default();
        ts.set_polar_motion_table(
            vec![2458800.0, 2458850.0, 2458900.0],
            vec![0.1, 0.1, 0.1],
            vec![0.3, 0.3, 0.3],
        );

        let t = ts.tt_jd(jd, None);
        let c = t.c_matrix();

        // Compare (tolerance 1e-4 for UT1-UTC approximation differences)
        for i in 0..3 {
            for j in 0..3 {
                let rust_val = c[(i, j)];
                let py_val = py_itrs[i * 3 + j];
                let diff = (rust_val - py_val).abs();
                assert!(
                    diff < 1e-4,
                    "ITRS_W[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff}"
                );
            }
        }
    }

    /// Test that ITRS without polar motion table still matches Skyfield
    #[test]
    fn test_itrs_without_polar_motion_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        for &jd in &TEST_JDS {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs
ts = load.timescale()
t = ts.tt_jd({jd})
R = itrs.rotation_at(t)
rust.collect_array(np.array(R.flatten(), dtype=np.float64))
"#
                ))
                .unwrap_or_else(|e| panic!("Python ITRS failed at JD {jd}: {e}"));

            let py_c = parse_f64_array(&py_result);
            let c = ts.tt_jd(jd, None).c_matrix();

            for i in 0..3 {
                for j in 0..3 {
                    let rust_val = c[(i, j)];
                    let py_val = py_c[i * 3 + j];
                    let diff = (rust_val - py_val).abs();
                    assert!(
                        diff < 1e-4,
                        "C[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff}"
                    );
                }
            }
        }
    }

    /// Test polar motion matrix with table installed vs Skyfield
    #[test]
    fn test_polar_motion_matrix_with_table_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let jd = 2458849.5;

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
import numpy as np
from skyfield.api import load

ts = load.timescale()
tt_dates = np.array([2458800.0, 2458850.0, 2458900.0])
x_vals = np.array([0.2, 0.2, 0.2])
y_vals = np.array([0.4, 0.4, 0.4])
ts.polar_motion_table = (tt_dates, x_vals, y_vals)

t = ts.tt_jd({jd})
W = t.polar_motion_matrix()
rust.collect_array(np.array(W.flatten(), dtype=np.float64))
"#
            ))
            .unwrap_or_else(|e| panic!("Python W matrix with table failed: {e}"));

        let py_w = parse_f64_array(&py_result);

        let mut ts = Timescale::default();
        ts.set_polar_motion_table(
            vec![2458800.0, 2458850.0, 2458900.0],
            vec![0.2, 0.2, 0.2],
            vec![0.4, 0.4, 0.4],
        );

        let t = ts.tt_jd(jd, None);
        let w = t.polar_motion_matrix();

        for i in 0..3 {
            for j in 0..3 {
                let rust_val = w[(i, j)];
                let py_val = py_w[i * 3 + j];
                let diff = (rust_val - py_val).abs();
                assert!(
                    diff < 1e-10,
                    "W[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff}"
                );
            }
        }
    }
}
