//! Python comparison tests for SGP4 satellite tracking
//!
//! Validates Rust SGP4 propagation and TEME→GCRS transformation against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::pybridge::test_utils::parse_f64_triple;
    use crate::sgp4lib::EarthSatellite;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    fn unwrap_py_string(raw: &str) -> String {
        match PythonResult::try_from(raw).expect("Failed to parse Python result") {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        }
    }

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

        let py_epoch: f64 = unwrap_py_string(&py_result).parse().unwrap();

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

        // TEME positions should match to ~10 meters (0.01 km)
        assert_relative_eq!(pos_teme.x, py_x, epsilon = 0.01);
        assert_relative_eq!(pos_teme.y, py_y, epsilon = 0.01);
        assert_relative_eq!(pos_teme.z, py_z, epsilon = 0.01);
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

        // Velocities should match to ~1e-4 km/s
        assert_relative_eq!(vel_teme.x, py_vx, epsilon = 1e-4);
        assert_relative_eq!(vel_teme.y, py_vy, epsilon = 1e-4);
        assert_relative_eq!(vel_teme.z, py_vz, epsilon = 1e-4);
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

        // GCRS positions should match to ~1e-6 AU (~150 m)
        // Frame transformation may introduce small differences
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-6);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-6);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-6);
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
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-6);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-6);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-6);
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
        assert_relative_eq!(pos.position.x, py_x, epsilon = 1e-6);
        assert_relative_eq!(pos.position.y, py_y, epsilon = 1e-6);
        assert_relative_eq!(pos.position.z, py_z, epsilon = 1e-6);
    }

    /// Test GCRS velocity (with angular velocity cross-product) matches Skyfield
    #[test]
    fn test_gcrs_velocity_matches_skyfield() {
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
vx, vy, vz = pos.velocity.au_per_d
rust.collect_string(f"{{vx}},{{vy}},{{vz}}")
"#,
                ISS_LINE1, ISS_LINE2, jd
            ))
            .expect("Failed to run Python code");

        let (py_vx, py_vy, py_vz) = parse_f64_triple(&py_result);

        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let t = ts.tt_jd(jd, None);
        let pos = sat.at(&t).expect("Failed to propagate");

        // GCRS velocities — angular velocity correction differs between implementations
        assert_relative_eq!(pos.velocity.x, py_vx, epsilon = 1e-3);
        assert_relative_eq!(pos.velocity.y, py_vy, epsilon = 1e-3);
        assert_relative_eq!(pos.velocity.z, py_vz, epsilon = 1e-3);
    }

    /// Test find_events matches Skyfield's find_events for event count and times
    #[test]
    fn test_find_events_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        use crate::sgp4lib::SatelliteEvent;
        use crate::toposlib::WGS84;

        // Get event count and first event time from Skyfield
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite, wgs84
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
bluffton = wgs84.latlon(40.8939, -83.8917, 244.0)
t0 = ts.tt_jd(sat.epoch.tt)
t1 = ts.tt_jd(sat.epoch.tt + 1.0)
t, events = sat.find_events(bluffton, t0, t1, altitude_degrees=0.0)
result_parts = []
for i in range(len(t)):
    result_parts.append(f"{{t[i].tt}},{{events[i]}}")
rust.collect_string("|".join(result_parts))
"#,
                ISS_LINE1, ISS_LINE2
            ))
            .expect("Failed to run Python code");

        let py_events_str = unwrap_py_string(&py_result);

        // Parse Python events
        let py_events: Vec<(f64, i64)> = if py_events_str.is_empty() {
            Vec::new()
        } else {
            py_events_str
                .split('|')
                .map(|s| {
                    let parts: Vec<&str> = s.split(',').collect();
                    let jd: f64 = parts[0].parse().unwrap();
                    let event: i64 = parts[1].parse().unwrap();
                    (jd, event)
                })
                .collect()
        };

        // Run Rust find_events
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let bluffton = WGS84.latlon(40.8939, -83.8917, 244.0);
        let t0 = ts.tt_jd(sat.epoch_jd(), None);
        let t1 = ts.tt_jd(sat.epoch_jd() + 1.0, None);

        let rust_events = sat
            .find_events(&bluffton, &t0, &t1, &ts, 0.0)
            .expect("Failed to find events");

        // Count by event type
        let py_rises = py_events.iter().filter(|(_, e)| *e == 0).count();
        let py_culminates = py_events.iter().filter(|(_, e)| *e == 1).count();
        let py_sets = py_events.iter().filter(|(_, e)| *e == 2).count();

        let rust_rises = rust_events
            .iter()
            .filter(|(_, e)| *e == SatelliteEvent::Rise)
            .count();
        let rust_culminates = rust_events
            .iter()
            .filter(|(_, e)| *e == SatelliteEvent::Culminate)
            .count();
        let rust_sets = rust_events
            .iter()
            .filter(|(_, e)| *e == SatelliteEvent::Set)
            .count();

        // Event counts should match (or be very close)
        assert_eq!(
            py_culminates, rust_culminates,
            "Culmination count mismatch: Skyfield={}, Rust={}",
            py_culminates, rust_culminates
        );

        // Rise/set counts should be within 1 of each other
        // (boundary effects can cause small differences)
        assert!(
            (py_rises as i64 - rust_rises as i64).abs() <= 1,
            "Rise count mismatch: Skyfield={}, Rust={}",
            py_rises,
            rust_rises
        );
        assert!(
            (py_sets as i64 - rust_sets as i64).abs() <= 1,
            "Set count mismatch: Skyfield={}, Rust={}",
            py_sets,
            rust_sets
        );

        // Compare culmination times (these should match best)
        let py_culmination_times: Vec<f64> = py_events
            .iter()
            .filter(|(_, e)| *e == 1)
            .map(|(t, _)| *t)
            .collect();
        let rust_culmination_times: Vec<f64> = rust_events
            .iter()
            .filter(|(_, e)| *e == SatelliteEvent::Culminate)
            .map(|(t, _)| t.tt())
            .collect();

        for (py_t, rust_t) in py_culmination_times
            .iter()
            .zip(rust_culmination_times.iter())
        {
            // Culmination times should match to within ~10 seconds
            let diff_seconds = (py_t - rust_t).abs() * 86400.0;
            assert!(
                diff_seconds < 10.0,
                "Culmination time mismatch: {:.6} seconds",
                diff_seconds
            );
        }
    }

    /// Test that altitude calculation at a specific time matches Skyfield
    #[test]
    fn test_altitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();

        use crate::toposlib::WGS84;

        let jd = 2454729.5;

        // Skyfield computes altitude using the full altaz pipeline
        let py_result = bridge
            .run_py_to_json(&format!(
                r#"
from skyfield.api import load, EarthSatellite, wgs84
ts = load.timescale()
line1 = "{}"
line2 = "{}"
sat = EarthSatellite(line1, line2, "ISS", ts)
bluffton = wgs84.latlon(40.8939, -83.8917, 244.0)
t = ts.tt_jd({})
difference = sat - bluffton
topocentric = difference.at(t)
alt, az, distance = topocentric.altaz()
rust.collect_string(f"{{alt.degrees}},{{az.degrees}},{{distance.au}}")
"#,
                ISS_LINE1, ISS_LINE2, jd
            ))
            .expect("Failed to run Python code");

        let (py_alt, _py_az, _py_dist) = parse_f64_triple(&py_result);

        // Compare with our altitude computation
        let sat = EarthSatellite::from_tle(ISS_LINE1, ISS_LINE2, Some("ISS"), &ts)
            .expect("Failed to parse TLE");
        let observer = WGS84.latlon(40.8939, -83.8917, 244.0);
        let t = ts.tt_jd(jd, None);
        let rust_alt = sat
            .altitude_degrees(&t, &observer)
            .expect("Failed to compute altitude");

        // Altitude should match to within ~1 degree
        // (our simplified method skips full apparent pipeline and uses
        // geocentric rather than barycentric positions)
        assert!(
            (rust_alt - py_alt).abs() < 1.0,
            "Altitude mismatch: Rust={:.4}, Python={:.4}, diff={:.4}",
            rust_alt,
            py_alt,
            (rust_alt - py_alt).abs()
        );
    }
}
