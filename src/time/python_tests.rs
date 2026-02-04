//! Python comparison tests for the time module
//!
//! Validates Rust time caching behavior against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64 from string"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    /// Test that cached TDB value matches Skyfield
    #[test]
    fn test_cached_tdb_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2451545.0)
rust.collect_string(str(t.tdb))
"#,
            )
            .expect("Failed to run Python code");

        let py_tdb = parse_f64(&py_result);

        let ts = Timescale::default();
        let t = ts.tt_jd(2451545.0, None);

        // First call computes and caches
        let tdb1 = t.tdb();
        // Second call should return same cached value
        let tdb2 = t.tdb();
        assert_eq!(tdb1, tdb2);

        let diff = (tdb1 - py_tdb).abs();
        assert!(
            diff < 1e-8,
            "TDB mismatch: rust={tdb1} python={py_tdb} diff={diff}"
        );
    }

    /// Test that cached delta_t matches Skyfield
    #[test]
    fn test_cached_delta_t_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2451545.0)
rust.collect_string(str(t.delta_t))
"#,
            )
            .expect("Failed to run Python code");

        let py_dt = parse_f64(&py_result);

        let ts = Timescale::default();
        let t = ts.tt_jd(2451545.0, None);

        // First call computes and caches
        let dt1 = t.delta_t();
        // Second call returns cached value
        let dt2 = t.delta_t();
        assert_eq!(dt1, dt2);

        // Our polynomial approximation may differ from Skyfield's table-based approach,
        // but should be in the right ballpark (within ~1 second for J2000)
        let diff = (dt1 - py_dt).abs();
        assert!(
            diff < 2.0,
            "delta_t mismatch: rust={dt1} python={py_dt} diff={diff}s"
        );
    }
}
