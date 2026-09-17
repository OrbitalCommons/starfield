//! Python comparison tests for reference frames
//!
//! Validates `ItrsFrame` against Skyfield's `skyfield.framelib.itrs`.

#[cfg(test)]
mod tests {
    use crate::framelib::{Frame, ItrsFrame};
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;

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

    /// `ItrsFrame::rotation_at` must reproduce `skyfield.framelib.itrs.rotation_at`.
    ///
    /// Both sides are built from the same UT1 — the Skyfield time is created
    /// with `ut1_jd` from the Rust time's own UT1 — so the comparison measures
    /// the rotation chain rather than the difference between the two delta-T
    /// models. (Feeding both `tt_jd` instead leaves a residual of ~1e-5 in the
    /// elements that carry the daily rotation, which is what the existing
    /// `c_matrix` tests see with their 1e-4 tolerance.) The half-second of TT
    /// that delta-T disagreement moves has no visible effect on precession or
    /// nutation at this tolerance.
    #[test]
    fn test_itrs_frame_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        for jd in [2451545.0, 2458849.5, 2460000.5] {
            let t = ts.tt_jd(jd, None);
            let ut1 = t.ut1();
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
import numpy as np
from skyfield.api import load
from skyfield.framelib import itrs
ts = load.timescale()
t = ts.ut1_jd({ut1:.17})
rust.collect_array(np.array(itrs.rotation_at(t).flatten(), dtype=np.float64))
"#
                ))
                .unwrap_or_else(|e| panic!("Python ITRS failed at JD {jd}: {e}"));

            let py_r = parse_f64_array(&py_result);
            let r = ItrsFrame.rotation_at(&t);

            for i in 0..3 {
                for j in 0..3 {
                    let rust_val = r[(i, j)];
                    let py_val = py_r[i * 3 + j];
                    let diff = (rust_val - py_val).abs();
                    assert!(
                        diff < 1e-9,
                        "ITRS[{i},{j}] at JD {jd}: rust={rust_val} python={py_val} diff={diff}"
                    );
                }
            }
        }
    }
}
