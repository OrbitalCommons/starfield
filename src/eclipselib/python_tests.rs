//! Python comparison tests for eclipse detection
//!
//! Validates lunar eclipse detection against Skyfield's `lunar_eclipses()`.
//! Skyfield does not implement solar eclipses, so those are validated
//! against NASA eclipse catalog dates.

#[cfg(test)]
mod tests {
    use crate::eclipselib::{lunar_eclipses, LunarEclipseType};
    use crate::jplephem::kernel::SpiceKernel;
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

    fn test_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("de421.bsp required")
    }

    /// Compare lunar eclipse count and types for 2015 against Skyfield
    #[test]
    fn test_lunar_eclipses_2015_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        // Get Skyfield's lunar eclipse results for 2015
        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.eclipselib import lunar_eclipses
ts = load.timescale()
eph = load('de421.bsp')
t0 = ts.tt_jd(2457023.5)  # 2015-01-01
t1 = ts.tt_jd(2457388.5)  # 2016-01-01
times, types, details = lunar_eclipses(t0, t1, eph)
results = []
for i in range(len(times)):
    results.append(f"{times.tt[i]:.4f}:{int(types[i])}")
rust.collect_string('|'.join(results))
"#,
            )
            .expect("Failed to run Python code");

        let py_str = parse_string(&py_result);

        // Parse Skyfield results
        let py_eclipses: Vec<(f64, i32)> = py_str
            .split('|')
            .filter(|s| !s.is_empty())
            .map(|s| {
                let parts: Vec<&str> = s.split(':').collect();
                (parts[0].parse().unwrap(), parts[1].parse().unwrap())
            })
            .collect();

        // Get our results
        let t0 = ts.tt_jd(2457023.5, None).tdb();
        let t1 = ts.tt_jd(2457388.5, None).tdb();
        let rust_eclipses = lunar_eclipses(&mut kernel, t0, t1);

        // Should find the same number of eclipses
        assert_eq!(
            rust_eclipses.len(),
            py_eclipses.len(),
            "Eclipse count mismatch: rust={} python={}",
            rust_eclipses.len(),
            py_eclipses.len()
        );

        // Check that dates match within 1 day (our search is approximate)
        for (i, (py_jd, py_type)) in py_eclipses.iter().enumerate() {
            let rust_jd = rust_eclipses[i].jd_tdb;
            let diff_days = (rust_jd - py_jd).abs();
            assert!(
                diff_days < 1.0,
                "Eclipse {i}: date mismatch: rust={rust_jd:.4} python={py_jd:.4} diff={diff_days:.4} days"
            );

            // Check type matches
            let rust_type = rust_eclipses[i].eclipse_type as i32;
            assert_eq!(
                rust_type, *py_type,
                "Eclipse {i}: type mismatch: rust={rust_type} ({}) python={py_type}",
                rust_eclipses[i].eclipse_type
            );
        }
    }

    /// Compare lunar eclipse count for 2019 against Skyfield
    #[test]
    fn test_lunar_eclipses_2019_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = test_kernel();
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.eclipselib import lunar_eclipses
ts = load.timescale()
eph = load('de421.bsp')
t0 = ts.tt_jd(2458484.5)  # 2019-01-01
t1 = ts.tt_jd(2458849.5)  # 2020-01-01
times, types, details = lunar_eclipses(t0, t1, eph)
results = []
for i in range(len(times)):
    results.append(f"{times.tt[i]:.4f}:{int(types[i])}")
rust.collect_string('|'.join(results))
"#,
            )
            .expect("Failed to run Python code");

        let py_str = parse_string(&py_result);
        let py_count = py_str.split('|').filter(|s| !s.is_empty()).count();

        let t0 = ts.tt_jd(2458484.5, None).tdb();
        let t1 = ts.tt_jd(2458849.5, None).tdb();
        let rust_eclipses = lunar_eclipses(&mut kernel, t0, t1);

        // 2019 had a total (Jan 21) and partial (Jul 16)
        assert!(
            rust_eclipses.len() >= 2,
            "Expected at least 2 lunar eclipses in 2019"
        );

        assert_eq!(
            rust_eclipses.len(),
            py_count,
            "Eclipse count: rust={} python={}",
            rust_eclipses.len(),
            py_count
        );

        // Check there's a total eclipse (Jan 21, 2019)
        let has_total = rust_eclipses
            .iter()
            .any(|e| e.eclipse_type == LunarEclipseType::Total);
        assert!(has_total, "Expected a total lunar eclipse in Jan 2019");
    }
}
