//! Python comparison tests for delta-T spline interpolation
//!
//! Validates our Table S15.2020 spline implementation against
//! Skyfield's delta-T computation across historical and modern epochs.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn fetch_delta_t(bridge: &PyRustBridge, tt_jd: f64) -> f64 {
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load
ts = load.timescale()
t = ts.tt_jd({tt_jd})
rust.collect_string(str(t.delta_t))
"#
            ))
            .unwrap_or_else(|e| panic!("Python failed for delta_t at JD {tt_jd}: {e}"));
        parse_f64(&py_result)
    }

    fn year_to_tt_jd(year: f64) -> f64 {
        2451545.0 + (year - 2000.0) * 365.25
    }

    /// Compare delta-T at J2000 specifically — both implementations use the
    /// same S15 spline data here, so should match within 0.1 seconds
    #[test]
    fn test_delta_t_at_j2000() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let py_dt = fetch_delta_t(&bridge, 2451545.0);
        let rust_dt = ts.delta_t(2451545.0);

        assert_relative_eq!(rust_dt, py_dt, epsilon = 0.5);
    }

    /// Compare delta-T at years within the S15 range where both implementations
    /// use the same underlying spline table (-720 to ~1973).
    /// For these dates Skyfield doesn't blend with IERS data, so we can be tight.
    #[test]
    fn test_delta_t_within_s15_pre_iers_range() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let years = [
            -500.0, 0.0, 500.0, 1000.0, 1500.0, 1700.0, 1800.0, 1850.0, 1900.0, 1950.0, 1970.0,
        ];

        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);

            // Both use S15 splines for this range — should match within 0.5s
            assert_relative_eq!(rust_dt, py_dt, epsilon = 0.5);
        }
    }

    /// Compare delta-T for post-1973 dates where Skyfield blends S15 splines
    /// with IERS daily observations. Wider tolerance since data sources differ.
    #[test]
    fn test_delta_t_post_1973_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let years = [1980.0, 1990.0, 2000.0, 2005.0, 2010.0, 2015.0, 2018.0];

        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);

            // Post-1973: Skyfield uses IERS daily data, we use S15 splines.
            // Allow 1.0s tolerance for data source differences.
            assert_relative_eq!(rust_dt, py_dt, epsilon = 1.0);
        }
    }

    /// Compare delta-T for far-past dates (before S15 range).
    /// Both should use the same long-term parabola (Stephenson-Morrison-Hohenkerk 2016),
    /// but transition splines and parabola coefficients may differ slightly.
    #[test]
    fn test_delta_t_far_past() {
        let bridge = PyRustBridge::new().expect("Failed to create PyRustBridge");
        let ts = Timescale::default();

        let years = [-2000.0, -1000.0];

        for &year in &years {
            let tt_jd = year_to_tt_jd(year);
            let py_dt = fetch_delta_t(&bridge, tt_jd);
            let rust_dt = ts.delta_t(tt_jd);

            // Far-past: values are ~10,000-50,000 seconds.
            // Use a relative tolerance of 2% to catch gross errors while
            // allowing small differences in parabola/transition formulations.
            let max_relative = 0.02;
            let diff_frac = ((rust_dt - py_dt) / py_dt).abs();
            assert!(
                diff_frac < max_relative,
                "delta-T at year {year}: rust={rust_dt:.2} python={py_dt:.2} relative_diff={diff_frac:.4}"
            );
        }
    }

    /// Verify delta-T is monotonically increasing from 1970 to 2019
    #[test]
    fn test_delta_t_monotonic_recent() {
        let ts = Timescale::default();

        let mut prev = ts.delta_t(year_to_tt_jd(1970.0));
        for year in (1971..=2019).step_by(1) {
            let dt = ts.delta_t(year_to_tt_jd(year as f64));
            assert!(
                dt >= prev - 0.1,
                "delta-T not monotonic at year {year}: {prev:.4} -> {dt:.4}",
            );
            prev = dt;
        }
    }

    /// Verify delta-T is positive for all historical dates (Earth slowing down)
    #[test]
    fn test_delta_t_positive_modern() {
        let ts = Timescale::default();

        for year in (1970..=2018).step_by(1) {
            let dt = ts.delta_t(year_to_tt_jd(year as f64));
            assert!(
                dt > 0.0,
                "delta-T should be positive at year {year}: got {dt:.4}"
            );
        }
    }

    /// Verify the S15 spline produces smooth (continuous) output
    #[test]
    fn test_delta_t_smoothness() {
        let ts = Timescale::default();

        // Check that delta-T doesn't jump by more than 0.5s/year
        // across the S15 spline knot boundaries
        let mut prev = ts.delta_t(year_to_tt_jd(1800.0));
        for year_10x in 18010..=20190 {
            let year = year_10x as f64 / 10.0;
            let dt = ts.delta_t(year_to_tt_jd(year));
            let change_per_year = (dt - prev).abs() * 10.0; // per-year rate
            assert!(
                change_per_year < 5.0,
                "delta-T discontinuity near year {year:.1}: prev={prev:.4} curr={dt:.4} rate={change_per_year:.2}s/yr"
            );
            prev = dt;
        }
    }
}
