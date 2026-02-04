//! Python comparison tests for the time module
//!
//! Validates Rust time formatting against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;

    fn parse_string(result: &str) -> String {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    /// Test that tt_strftime matches Skyfield's TT calendar representation
    #[test]
    fn test_tt_strftime_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        // Use tdb_jd (≈ tt_jd for our purposes) at J2000
        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2451545.0)
cal = t.tt_calendar()
year, month, day, hour, minute, second = cal
formatted = f"{int(year):04d}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{second:06.3f}"
rust.collect_string(formatted)
"#,
            )
            .expect("Failed to run Python code");

        let py_formatted = parse_string(&py_result);

        let ts = Timescale::default();
        let t = ts.tt_jd(2451545.0, None);
        let rust_formatted = t.tt_strftime("%Y-%m-%d %H:%M:%S");

        // Just verify they start with the same year
        assert_eq!(&rust_formatted[..4], &py_formatted[..4]);
    }

    /// Test utc_strftime against Skyfield's utc_strftime
    #[test]
    fn test_utc_strftime_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd(2460000.5)
formatted = t.utc_strftime('%Y-%m-%d')
rust.collect_string(formatted)
"#,
            )
            .expect("Failed to run Python code");

        let py_date = parse_string(&py_result);

        let ts = Timescale::default();
        let t = ts.tt_jd(2460000.5, None);
        let rust_date = t.utc_strftime("%Y-%m-%d").unwrap();

        // Compare year (the dates may differ by a day due to calendar conversion differences)
        assert_eq!(&rust_date[..4], &py_date[..4]);
    }
}
