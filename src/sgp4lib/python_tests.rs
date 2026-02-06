//! Python comparison tests for SGP4 satellite tracking
//!
//! Validates Rust SGP4 propagation and TEME→GCRS transformation against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::test_utils::parse_f64_triple;
    use crate::sgp4lib::EarthSatellite;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    // ISS TLE used for testing (from Sep 2008)
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    /// Test that TLE epoch parsing matches Skyfield
    #[test]
    fn test_tle_epoch_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
epoch_jd = sat.epoch.tt
rust.collect_string(f"{{epoch_jd}}")
"#,
                ISS_LINE1, ISS_LINE2
            ))
            .expect("Failed to run Python code");

        let py_epoch: f64 = py_result.trim_matches('"').parse().unwrap();

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let rust_epoch = sat.epoch_jd();

        // Epochs should match to high precision (< 1 second)
        assert_relative_eq!(rust_epoch, py_epoch, epsilon = 1e-5);
    }

    /// Test TEME position matches Skyfield's _position_and_velocity_TEME_km
    #[test]
    fn test_teme_position_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        // Use a specific time for comparison
        let jd = 2454729.5; // 2008-09-20 00:00 TDB

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
t = ts.tt_jd({})
r, v, error = sat._position_and_velocity_TEME_km(t)
rust.collect_string(f"{{r[0]}},{{r[1]}},{{r[2]}}")
"#,
                ISS_LINE1, ISS_LINE2, jd
            ))
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let t = ts.tt_jd(jd, None);
        let (pos_teme, _vel_teme) = sat
            .position_and_velocity_teme_km(&t)
            .expect("Failed to get TEME position");

        // TEME positions should match to ~1 meter (1e-3 km)
        assert_relative_eq!(pos_teme.x, py_x, epsilon = 1e-3);
        assert_relative_eq!(pos_teme.y, py_y, epsilon = 1e-3);
        assert_relative_eq!(pos_teme.z, py_z, epsilon = 1e-3);
    }

    /// Test TEME velocity matches Skyfield
    #[test]
    fn test_teme_velocity_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let jd = 2454729.5;

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
t = ts.tt_jd({})
r, v, error = sat._position_and_velocity_TEME_km(t)
rust.collect_string(f"{{v[0]}},{{v[1]}},{{v[2]}}")
"#,
                ISS_LINE1, ISS_LINE2, jd
            ))
            .expect("Failed to run Python code");

        let (py_vx, py_vy, py_vz) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let t = ts.tt_jd(jd, None);
        let (_pos_teme, vel_teme) = sat
            .position_and_velocity_teme_km(&t)
            .expect("Failed to get TEME position");

        // Velocities should match to ~1e-6 km/s
        assert_relative_eq!(vel_teme.x, py_vx, epsilon = 1e-6);
        assert_relative_eq!(vel_teme.y, py_vy, epsilon = 1e-6);
        assert_relative_eq!(vel_teme.z, py_vz, epsilon = 1e-6);
    }

    /// Test GCRS position (after frame transformation) matches Skyfield
    #[test]
    fn test_gcrs_position_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let jd = 2454729.5;

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
t = ts.tt_jd({})
pos = sat.at(t)
x, y, z = pos.position.au
rust.collect_string(f"{{x}},{{y}},{{z}}")
"#,
                ISS_LINE1, ISS_LINE2, jd
            ))
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let t = ts.tt_jd(jd, None);
        let pos = sat.at(&t).expect("Failed to propagate");

        // GCRS positions should match to ~1e-8 AU (~1.5 km)
        // Frame transformation may introduce small differences
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-7);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-7);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-7);
    }

    /// Test propagation at epoch (should be close to TLE reference)
    #[test]
    fn test_propagation_at_epoch_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
t = sat.epoch
pos = sat.at(t)
x, y, z = pos.position.au
rust.collect_string(f"{{x}},{{y}},{{z}}")
"#,
                ISS_LINE1, ISS_LINE2
            ))
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let pos = sat.at(&sat.epoch).expect("Failed to propagate");

        // At epoch, positions should be very close
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-7);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-7);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-7);
    }

    /// Test position 1 day after epoch
    #[test]
    fn test_propagation_one_day_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
t = ts.tt_jd(sat.epoch.tt + 1.0)
pos = sat.at(t)
x, y, z = pos.position.au
rust.collect_string(f"{{x}},{{y}},{{z}}")
"#,
                ISS_LINE1, ISS_LINE2
            ))
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let t = ts.tt_jd(sat.epoch_jd() + 1.0, None);
        let pos = sat.at(&t).expect("Failed to propagate");

        // After 1 day, positions should still match closely
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-7);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-7);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-7);
    }
}
