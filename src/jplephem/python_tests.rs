//! Python comparison tests for jplephem
//!
//! These tests validate our SPK computation against Python's jplephem/Skyfield.

#[cfg(test)]
mod tests {
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem::spk::SPK;
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;

    fn test_data_path(filename: &str) -> String {
        format!("src/jplephem/test_data/{filename}")
    }

    fn parse_f64_array(result: &str) -> Vec<f64> {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::Array { dtype, shape, data } => {
                assert_eq!(dtype, "float64");
                let n = shape.iter().product::<usize>();
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

    /// Test that our segment computation matches Python jplephem for Earth Barycenter at J2000
    #[test]
    fn test_earth_barycenter_position_at_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let bsp_path = test_data_path("de421.bsp");

        // Get Python's answer
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
import numpy as np
from jplephem.spk import SPK
kernel = SPK.open('{bsp_path}')
segment = kernel[0, 3]  # SSB -> Earth Barycenter
position, velocity = segment.compute_and_differentiate(0.0, 0.0)
rust.collect_array(np.array([position[0], position[1], position[2], velocity[0], velocity[1], velocity[2]], dtype=np.float64))
"#
            ))
            .expect("Failed to run Python code");

        let py_values = parse_f64_array(&py_result);
        assert_eq!(py_values.len(), 6);

        // Get Rust's answer
        let mut spk = SPK::open(&bsp_path).expect("Failed to open SPK");
        let seg = spk.get_segment_mut(0, 3).expect("Failed to get segment");
        let (pos, vel) = seg
            .compute_and_differentiate(0.0, 0.0)
            .expect("Failed to compute");

        // Compare position (km) — should match to ~1e-8 km (0.01 mm)
        let pos_tol = 1e-8;
        assert!(
            (pos.x - py_values[0]).abs() < pos_tol,
            "X position mismatch: rust={} python={} diff={}",
            pos.x,
            py_values[0],
            (pos.x - py_values[0]).abs()
        );
        assert!(
            (pos.y - py_values[1]).abs() < pos_tol,
            "Y position mismatch: rust={} python={} diff={}",
            pos.y,
            py_values[1],
            (pos.y - py_values[1]).abs()
        );
        assert!(
            (pos.z - py_values[2]).abs() < pos_tol,
            "Z position mismatch: rust={} python={} diff={}",
            pos.z,
            py_values[2],
            (pos.z - py_values[2]).abs()
        );

        // Compare velocity (km/s) — should match to ~1e-12 km/s
        let vel_tol = 1e-12;
        assert!(
            (vel.x - py_values[3]).abs() < vel_tol,
            "VX mismatch: rust={} python={} diff={}",
            vel.x,
            py_values[3],
            (vel.x - py_values[3]).abs()
        );
        assert!(
            (vel.y - py_values[4]).abs() < vel_tol,
            "VY mismatch: rust={} python={} diff={}",
            vel.y,
            py_values[4],
            (vel.y - py_values[4]).abs()
        );
        assert!(
            (vel.z - py_values[5]).abs() < vel_tol,
            "VZ mismatch: rust={} python={} diff={}",
            vel.z,
            py_values[5],
            (vel.z - py_values[5]).abs()
        );
    }

    /// Test all 15 DE421 segments at J2000
    #[test]
    fn test_all_segments_at_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let bsp_path = test_data_path("de421.bsp");

        let expected_pairs: Vec<(i32, i32)> = vec![
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (3, 301),
            (3, 399),
            (1, 199),
            (2, 299),
            (4, 499),
        ];

        let mut spk = SPK::open(&bsp_path).expect("Failed to open SPK");

        for (center, target) in &expected_pairs {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
import numpy as np
from jplephem.spk import SPK
kernel = SPK.open('{bsp_path}')
segment = kernel[{center}, {target}]
position, velocity = segment.compute_and_differentiate(0.0, 0.0)
rust.collect_array(np.array([position[0], position[1], position[2]], dtype=np.float64))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for ({center},{target}): {e}"));

            let py_pos = parse_f64_array(&py_result);

            let seg = spk
                .get_segment_mut(*center, *target)
                .unwrap_or_else(|_| panic!("Missing segment ({center},{target})"));
            let (pos, _vel) = seg
                .compute_and_differentiate(0.0, 0.0)
                .unwrap_or_else(|e| panic!("Compute failed for ({center},{target}): {e}"));

            let tol = 1e-8;
            assert!(
                (pos.x - py_pos[0]).abs() < tol
                    && (pos.y - py_pos[1]).abs() < tol
                    && (pos.z - py_pos[2]).abs() < tol,
                "Position mismatch for ({center},{target}): rust=({},{},{}) python=({},{},{})",
                pos.x,
                pos.y,
                pos.z,
                py_pos[0],
                py_pos[1],
                py_pos[2]
            );
        }
    }

    /// Test segment computation at multiple times
    #[test]
    fn test_earth_barycenter_at_multiple_times() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let bsp_path = test_data_path("de421.bsp");

        // Test at J2000, J2000+1yr, J2000+10yr (in TDB seconds)
        let times_seconds = [0.0, 365.25 * 86400.0, 3652.5 * 86400.0];

        let mut spk = SPK::open(&bsp_path).expect("Failed to open SPK");

        for &tdb in &times_seconds {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
import numpy as np
from jplephem.spk import SPK
kernel = SPK.open('{bsp_path}')
segment = kernel[0, 3]
position, velocity = segment.compute_and_differentiate({tdb}, 0.0)
rust.collect_array(np.array([position[0], position[1], position[2]], dtype=np.float64))
"#
                ))
                .expect("Failed to run Python code");

            let py_pos = parse_f64_array(&py_result);

            let seg = spk.get_segment_mut(0, 3).unwrap();
            let (pos, _vel) = seg.compute_and_differentiate(tdb, 0.0).unwrap();

            let tol = 1e-8;
            assert!(
                (pos.x - py_pos[0]).abs() < tol
                    && (pos.y - py_pos[1]).abs() < tol
                    && (pos.z - py_pos[2]).abs() < tol,
                "Position mismatch at tdb={tdb}: rust=({},{},{}) python=({},{},{})",
                pos.x,
                pos.y,
                pos.z,
                py_pos[0],
                py_pos[1],
                py_pos[2]
            );
        }
    }

    /// Test the high-level kernel API against Skyfield
    #[test]
    fn test_kernel_earth_position_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let bsp_path = test_data_path("de421.bsp");

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
import numpy as np
from skyfield.api import Loader
load = Loader('.')
planets = load('{bsp_path}')
earth = planets['earth']
ts = load.timescale()
t = ts.tdb_jd(2451545.0)
pos = earth.at(t)
rust.collect_array(np.array(pos.position.km, dtype=np.float64))
"#
            ))
            .expect("Failed to run Python code");

        let py_pos_km = parse_f64_array(&py_result);

        // Compute with our kernel API
        let mut kernel = SpiceKernel::open(&bsp_path).expect("Failed to open kernel");
        let ts = crate::time::Timescale::default();
        let t = ts.tdb_jd(2451545.0);
        let (pos_km, _vel) = kernel.compute_km("earth", &t).expect("Failed to compute");

        let tol = 1e-5; // 0.01 mm
        assert!(
            (pos_km.x - py_pos_km[0]).abs() < tol
                && (pos_km.y - py_pos_km[1]).abs() < tol
                && (pos_km.z - py_pos_km[2]).abs() < tol,
            "Earth position (km) mismatch at J2000:\n  rust=({},{},{})\n  python=({},{},{})",
            pos_km.x,
            pos_km.y,
            pos_km.z,
            py_pos_km[0],
            py_pos_km[1],
            py_pos_km[2]
        );
    }
}
