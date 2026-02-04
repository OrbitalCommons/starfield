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

    /// Test that our leap second offset matches Skyfield at various epochs
    #[test]
    fn test_leap_seconds_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let ts = Timescale::default();

        // Test at several epochs where the leap second count is well-known
        let test_cases = [
            (2451545.0, "J2000"),      // 2000-01-01: 32 leap seconds
            (2457754.5, "2017-01-01"), // 37 leap seconds
            (2460310.5, "2024-01-01"), // Still 37
            (2444239.5, "1980-01-01"), // 19 leap seconds
        ];

        for (jd, label) in test_cases {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({jd})
# Get TAI-UTC offset: TAI = TT - 32.184, UTC = TAI - leap_seconds
# So leap_seconds = TAI - UTC
# In Skyfield: t.tai - t.utc_jd() gives the offset in days
tai_jd = t.tai
tt_jd = t.tt
# TAI-UTC in seconds: (TAI - TT) = -32.184, (TT-UTC) = delta_T when close enough
# Simpler: just get the dut1 and compute
# Actually the cleanest way: check what Skyfield thinks
import skyfield.timelib
leap_dates = ts._leap_dates
leap_offsets = ts._leap_offsets
import numpy as np
idx = np.searchsorted(leap_dates, {jd}, side='right') - 1
if idx < 0:
    ls = 0
else:
    ls = int(leap_offsets[idx])
rust.collect_string(str(ls))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for {label}: {e}"));

            let py_ls = parse_f64(&py_result);

            let t = ts.tt_jd(jd, None);
            let rust_ls = t.leap_seconds();

            assert!(
                (rust_ls - py_ls).abs() < 0.5,
                "{label} (JD {jd}): leap seconds mismatch: rust={rust_ls} python={py_ls}"
            );
        }
    }
}
