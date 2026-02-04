//! Python comparison tests for delta-T spline interpolation
//!
//! Validates our Table S15.2020 spline implementation against
//! Skyfield's delta-T computation.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    /// Compare delta-T values at several historical dates against Skyfield
    #[test]
    fn test_delta_t_vs_skyfield_historical() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        // Test years spanning the S15 table range
        let test_years: Vec<f64> = vec![
            -500.0, 0.0, 500.0, 1000.0, 1500.0, 1700.0, 1800.0, 1850.0, 1900.0, 1950.0, 1970.0,
            1980.0, 1990.0, 2000.0, 2005.0, 2010.0, 2015.0, 2018.0,
        ];

        let years_str = format!("{:?}", test_years);
        let python_code = format!(
            r#"
from skyfield.api import load
import json

ts = load.timescale()
years = {}
results = []
for y in years:
    t = ts.tt_jd(2451545.0 + (y - 2000.0) * 365.25)
    dt = t.delta_t
    results.append({{"year": y, "delta_t": float(dt), "tt_jd": float(t.tt)}})
rust.collect_string(json.dumps(results))
"#,
            years_str
        );

        let result = bridge
            .run_py_to_json(&python_code)
            .expect("Failed to run Python code");
        let parsed = PythonResult::try_from(result.as_str()).expect("Failed to parse result");

        if let PythonResult::String(json_str) = parsed {
            let results: Vec<serde_json::Value> =
                serde_json::from_str(&json_str).expect("Failed to parse JSON");

            for entry in &results {
                let year = entry["year"].as_f64().unwrap();
                let py_delta_t = entry["delta_t"].as_f64().unwrap();
                let tt_jd = entry["tt_jd"].as_f64().unwrap();

                let rust_delta_t = ts.delta_t(tt_jd);

                // For dates within S15 range (-720 to 2019), we should match closely
                // since both use the same S15 spline data.
                // For dates outside that range, tolerance is wider.
                let tolerance = if (-720.0..=2019.0).contains(&year) {
                    // Within S15 range: Skyfield may also blend with IERS daily
                    // data for recent dates, so allow wider tolerance there
                    if year > 1973.0 {
                        1.0 // Skyfield uses IERS daily data for recent dates
                    } else {
                        0.5
                    }
                } else {
                    5.0 // Outside S15 range
                };

                assert_relative_eq!(rust_delta_t, py_delta_t, epsilon = tolerance,);
            }
        } else {
            panic!("Expected String result from Python");
        }
    }

    /// Compare delta-T at J2000 epoch specifically
    #[test]
    fn test_delta_t_at_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2451545.0)
rust.collect_string(str(t.delta_t))
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(result.as_str()).expect("Failed to parse result");
        if let PythonResult::String(s) = parsed {
            let py_delta_t: f64 = s.parse().expect("Failed to parse delta_t");
            let rust_delta_t = ts.delta_t(2_451_545.0);
            // At J2000 (year 2000), Skyfield uses IERS daily data
            // while we use S15 splines, so allow ~0.5s tolerance
            assert_relative_eq!(rust_delta_t, py_delta_t, epsilon = 0.5);
        } else {
            panic!("Expected String result");
        }
    }

    /// Compare delta-T for far-past dates (long-term parabola)
    #[test]
    fn test_delta_t_far_past() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
import json
ts = load.timescale()
results = []
for year in [-2000, -1000, -500]:
    tt_jd = 2451545.0 + (year - 2000.0) * 365.25
    t = ts.tt_jd(tt_jd)
    results.append({"year": year, "delta_t": float(t.delta_t)})
rust.collect_string(json.dumps(results))
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(result.as_str()).expect("Failed to parse result");
        if let PythonResult::String(json_str) = parsed {
            let results: Vec<serde_json::Value> =
                serde_json::from_str(&json_str).expect("Failed to parse JSON");

            for entry in &results {
                let year = entry["year"].as_f64().unwrap();
                let py_delta_t = entry["delta_t"].as_f64().unwrap();
                let tt_jd = 2_451_545.0 + (year - 2000.0) * 365.25;
                let rust_delta_t = ts.delta_t(tt_jd);

                // For far-past dates both should use the same parabola
                // but transition splines differ, so tolerance scales with distance
                let tolerance = (year.abs() / 100.0).max(10.0);
                assert_relative_eq!(rust_delta_t, py_delta_t, epsilon = tolerance);
            }
        } else {
            panic!("Expected String result");
        }
    }

    /// Verify that delta-T is monotonically increasing in recent decades
    #[test]
    fn test_delta_t_monotonic_recent() {
        let ts = Timescale::default();

        let mut prev = ts.delta_t(2_451_545.0 - 30.0 * 365.25); // ~1970
        for year_offset in -29..=19 {
            let tt_jd = 2_451_545.0 + year_offset as f64 * 365.25;
            let dt = ts.delta_t(tt_jd);
            assert!(
                dt >= prev - 0.5,
                "delta-T not monotonic near year {}: {prev} -> {dt}",
                2000.0 + year_offset as f64,
            );
            prev = dt;
        }
    }
}
