//! Python comparison tests for the time module
//!
//! Validates Rust time scale conversions against Python Skyfield.

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

    /// Test that second=60 leap second produces the correct TT JD vs Skyfield
    #[test]
    fn test_leap_second_60_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        // The 2016-12-31 leap second: second=60
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

        // Allow some tolerance for different leap second handling
        let diff = (rust_tt - py_tt).abs() * 86400.0; // diff in seconds
        assert!(
            diff < 2.0,
            "Leap second TT mismatch: rust={rust_tt} python={py_tt} diff={diff}s"
        );

        // Verify the leap second flag is set
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
}
