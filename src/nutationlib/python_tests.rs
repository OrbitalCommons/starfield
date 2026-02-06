//! Python comparison tests for nutation calculations

#[cfg(test)]
mod tests {
    use crate::constants::{ASEC2RAD, J2000};
    use crate::nutationlib::iau2000a_nutation;
    use crate::pybridge::{PyRustBridge, PythonResult};

    fn unwrap_py_string(raw: &str) -> String {
        match PythonResult::try_from(raw).expect("Failed to parse Python result") {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        }
    }

    /// Full IAU2000A nutation matches Skyfield at J2000
    #[test]
    fn test_nutation_matches_skyfield_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.nutationlib import iau2000a
import json

jd_tt = 2451545.0
dpsi, deps = iau2000a(jd_tt)

rust.collect_string(json.dumps({
    "dpsi": float(dpsi),
    "deps": float(deps),
}))
"#,
            )
            .expect("Python nutation failed");

        let inner = unwrap_py_string(&py_result);
        let parsed: serde_json::Value = serde_json::from_str(&inner).expect("JSON parse failed");
        let py_dpsi = parsed["dpsi"].as_f64().unwrap();
        let py_deps = parsed["deps"].as_f64().unwrap();

        let (rust_dpsi, rust_deps) = iau2000a_nutation(J2000);

        // Convert Rust radians to 0.1 µas (Skyfield's native units)
        let rust_dpsi_tenth_usec = rust_dpsi / (ASEC2RAD / 1e7);
        let rust_deps_tenth_usec = rust_deps / (ASEC2RAD / 1e7);

        assert!(
            (rust_dpsi_tenth_usec - py_dpsi).abs() < 1.0,
            "dpsi: rust={rust_dpsi_tenth_usec} python={py_dpsi}"
        );
        assert!(
            (rust_deps_tenth_usec - py_deps).abs() < 1.0,
            "deps: rust={rust_deps_tenth_usec} python={py_deps}"
        );
    }

    /// Full IAU2000A nutation matches Skyfield at a modern date
    #[test]
    fn test_nutation_matches_skyfield_2020() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.nutationlib import iau2000a
import json

jd_tt = 2459062.5
dpsi, deps = iau2000a(jd_tt)

rust.collect_string(json.dumps({
    "dpsi": float(dpsi),
    "deps": float(deps),
}))
"#,
            )
            .expect("Python nutation failed");

        let inner = unwrap_py_string(&py_result);
        let parsed: serde_json::Value = serde_json::from_str(&inner).expect("JSON parse failed");
        let py_dpsi = parsed["dpsi"].as_f64().unwrap();
        let py_deps = parsed["deps"].as_f64().unwrap();

        let (rust_dpsi, rust_deps) = iau2000a_nutation(2459062.5);
        let rust_dpsi_tenth_usec = rust_dpsi / (ASEC2RAD / 1e7);
        let rust_deps_tenth_usec = rust_deps / (ASEC2RAD / 1e7);

        assert!(
            (rust_dpsi_tenth_usec - py_dpsi).abs() < 1.0,
            "dpsi: rust={rust_dpsi_tenth_usec} python={py_dpsi}"
        );
        assert!(
            (rust_deps_tenth_usec - py_deps).abs() < 1.0,
            "deps: rust={rust_deps_tenth_usec} python={py_deps}"
        );
    }

    /// Full nutation matches Skyfield at several dates
    #[test]
    fn test_nutation_matches_skyfield_multiple_dates() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.nutationlib import iau2000a
import json

dates = [2451545.0, 2455000.5, 2458000.5, 2460000.5]
results = []
for jd in dates:
    dpsi, deps = iau2000a(jd)
    results.append({"jd": jd, "dpsi": float(dpsi), "deps": float(deps)})

rust.collect_string(json.dumps(results))
"#,
            )
            .expect("Python nutation multi-date failed");

        let inner = unwrap_py_string(&py_result);
        let parsed: Vec<serde_json::Value> =
            serde_json::from_str(&inner).expect("JSON parse failed");

        for entry in &parsed {
            let jd = entry["jd"].as_f64().unwrap();
            let py_dpsi = entry["dpsi"].as_f64().unwrap();
            let py_deps = entry["deps"].as_f64().unwrap();

            let (rust_dpsi, rust_deps) = iau2000a_nutation(jd);
            let rust_dpsi_tenth_usec = rust_dpsi / (ASEC2RAD / 1e7);
            let rust_deps_tenth_usec = rust_deps / (ASEC2RAD / 1e7);

            assert!(
                (rust_dpsi_tenth_usec - py_dpsi).abs() < 1.0,
                "JD {jd} dpsi: rust={rust_dpsi_tenth_usec} python={py_dpsi}"
            );
            assert!(
                (rust_deps_tenth_usec - py_deps).abs() < 1.0,
                "JD {jd} deps: rust={rust_deps_tenth_usec} python={py_deps}"
            );
        }
    }
}
