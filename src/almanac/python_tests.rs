//! Python comparison tests for almanac
//!
//! Validates Rust seasons, moon phases, sunrise/sunset, twilight, and
//! opposition/conjunction calculations against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::almanac::*;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::searchlib::{find_discrete, DEFAULT_NUM, EPSILON_DISCRETE};
    use crate::time::Timescale;

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn parse_f64_list(result: &str) -> Vec<f64> {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.split(',').map(|v| v.trim().parse().unwrap()).collect(),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn parse_i64_list(result: &str) -> Vec<i64> {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.split(',').map(|v| v.trim().parse().unwrap()).collect(),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    // --- Season event times ---

    /// Compare 2005 season event JD times against Skyfield
    #[test]
    fn test_seasons_2005_times_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2453371.5)  # 2005-Jan-01
t1 = ts.tt_jd(2453736.5)  # 2006-Jan-01

f = almanac.seasons(eph)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
rust.collect_string(jds)
"#,
            )
            .expect("Failed to run Python code");

        let py_jds = parse_f64_list(&py_result);

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Different number of season events: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, _), &py_jd)) in events.iter().zip(py_jds.iter()).enumerate() {
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 60.0,
                "Season event {} time diff: {:.1}s (rust={} py={})",
                i,
                diff_sec,
                rust_jd,
                py_jd
            );
        }
    }

    /// Compare season event types against Skyfield
    #[test]
    fn test_seasons_2005_types_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2453371.5)
t1 = ts.tt_jd(2453736.5)

f = almanac.seasons(eph)
times, events = almanac.find_discrete(t0, t1, f)

rust.collect_string(','.join(str(e) for e in events))
"#,
            )
            .expect("Failed to run Python code");

        let py_types = parse_i64_list(&py_result);

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2006, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);
        let rust_types: Vec<i64> = events.iter().map(|e| e.1).collect();

        assert_eq!(
            rust_types, py_types,
            "Season event types differ: rust={:?} python={:?}",
            rust_types, py_types
        );
    }

    /// Seasons in 2010 — different year for robustness
    #[test]
    fn test_seasons_2010_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2455197.5)  # 2010-Jan-01
t1 = ts.tt_jd(2455562.5)  # 2011-Jan-01

f = almanac.seasons(eph)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(e) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2010, 1, 1)).tdb();
        let t1 = ts.tt((2011, 1, 1)).tdb();

        let mut f = seasons(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 90.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(events.len(), py_jds.len());
        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Season {} type mismatch: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 60.0,
                "Season {} time diff {:.1}s too large",
                i,
                diff_sec
            );
        }
    }

    // --- Moon phase angle ---

    /// Compare continuous moon phase angle at specific times
    #[test]
    fn test_moon_phase_angle_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let test_jds = [2451545.0, 2453371.5, 2455000.5, 2456000.5];

        for jd in test_jds {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t = ts.tdb_jd({jd})
angle = almanac.moon_phase(eph, t)
rust.collect_string(str(angle.degrees))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for jd={jd}: {e}"));

            let py_deg = parse_f64(&py_result);

            let mut kernel = de421_kernel();
            let angles = moon_phase_angle(&mut kernel, &[jd]);
            let rust_deg = angles[0];

            let diff = (rust_deg - py_deg).abs().min(
                (rust_deg - py_deg + 360.0)
                    .abs()
                    .min((rust_deg - py_deg - 360.0).abs()),
            );
            assert!(
                diff < 1.0,
                "Moon phase angle at JD {}: rust={:.2}° py={:.2}° diff={:.2}°",
                jd,
                rust_deg,
                py_deg,
                diff
            );
        }
    }

    // --- Moon phase events ---

    /// Compare moon phase event times for Jan-Mar 2005
    #[test]
    fn test_moon_phases_2005_times_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2453371.5)  # 2005-Jan-01
t1 = ts.tt_jd(2453461.5)  # 2005-Apr-01

f = almanac.moon_phases(eph)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(e) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2005, 4, 1)).tdb();

        let mut f = moon_phases(&mut kernel);
        let events = find_discrete(t0, t1, &mut f, 7.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Moon phase event count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Moon phase {} type mismatch: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 120.0,
                "Moon phase {} time diff {:.1}s (rust={} py={})",
                i,
                diff_sec,
                rust_jd,
                py_jd
            );
        }
    }

    // --- Sun ecliptic longitude ---

    /// Compare Sun ecliptic longitude at several dates
    #[test]
    fn test_sun_ecliptic_longitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let test_jds = [2451545.0, 2453371.5, 2455000.5];

        for jd in test_jds {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load
from skyfield.framelib import ecliptic_frame

ts = load.timescale()
eph = load('de421.bsp')

t = ts.tdb_jd({jd})
e = eph['earth'].at(t)
_, slon, _ = e.observe(eph['sun']).apparent().frame_latlon(ecliptic_frame)
rust.collect_string(str(slon.degrees))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for jd={jd}: {e}"));

            let py_deg = parse_f64(&py_result);

            let mut kernel = de421_kernel();
            let lons = sun_ecliptic_longitude(&mut kernel, &[jd]);
            let rust_deg = lons[0].to_degrees();

            let diff = (rust_deg - py_deg).abs().min(
                (rust_deg - py_deg + 360.0)
                    .abs()
                    .min((rust_deg - py_deg - 360.0).abs()),
            );
            assert!(
                diff < 0.01,
                "Sun ecl lon at JD {}: rust={:.4}° py={:.4}° diff={:.4}°",
                jd,
                rust_deg,
                py_deg,
                diff
            );
        }
    }

    // --- Moon ecliptic longitude ---

    /// Compare Moon ecliptic longitude at several dates
    #[test]
    fn test_moon_ecliptic_longitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let test_jds = [2451545.0, 2453371.5, 2455000.5];

        for jd in test_jds {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load
from skyfield.framelib import ecliptic_frame

ts = load.timescale()
eph = load('de421.bsp')

t = ts.tdb_jd({jd})
e = eph['earth'].at(t)
_, mlon, _ = e.observe(eph['moon']).apparent().frame_latlon(ecliptic_frame)
rust.collect_string(str(mlon.degrees))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for jd={jd}: {e}"));

            let py_deg = parse_f64(&py_result);

            let mut kernel = de421_kernel();
            let lons = moon_ecliptic_longitude(&mut kernel, &[jd]);
            let rust_deg = lons[0].to_degrees();

            let diff = (rust_deg - py_deg).abs().min(
                (rust_deg - py_deg + 360.0)
                    .abs()
                    .min((rust_deg - py_deg - 360.0).abs()),
            );
            assert!(
                diff < 0.05,
                "Moon ecl lon at JD {}: rust={:.4}° py={:.4}° diff={:.4}°",
                jd,
                rust_deg,
                py_deg,
                diff
            );
        }
    }

    // --- Opposition / conjunction ---

    /// Mars opposition in 2003 — should match Skyfield's event time
    #[test]
    fn test_mars_opposition_2003_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2452640.5)  # 2003-Jan-01
t1 = ts.tt_jd(2453005.5)  # 2004-Jan-01

f = almanac.oppositions_conjunctions(eph, eph['mars'])
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(e) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2003, 1, 1)).tdb();
        let t1 = ts.tt((2004, 1, 1)).tdb();

        let mut f = oppositions_conjunctions(&mut kernel, "mars");
        let events = find_discrete(t0, t1, &mut f, 30.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Mars opposition/conj count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, _), &py_jd)) in events.iter().zip(py_jds.iter()).enumerate() {
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 300.0,
                "Mars event {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    /// Jupiter opposition/conjunction events 2005-2007
    #[test]
    fn test_jupiter_opposition_2005_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

t0 = ts.tt_jd(2453371.5)  # 2005-Jan-01
t1 = ts.tt_jd(2454101.5)  # 2007-Jan-01

f = almanac.oppositions_conjunctions(eph, eph['jupiter barycenter'])
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(e) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 1, 1)).tdb();
        let t1 = ts.tt((2007, 1, 1)).tdb();

        let mut f = oppositions_conjunctions(&mut kernel, "jupiter barycenter");
        let events = find_discrete(t0, t1, &mut f, 30.0, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Jupiter event count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, _), &py_jd)) in events.iter().zip(py_jds.iter()).enumerate() {
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 300.0,
                "Jupiter event {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    // --- Sunrise / sunset ---

    /// Compare sunrise/sunset events at Boston for June 2005
    #[test]
    fn test_sunrise_sunset_boston_june_2005_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
t0 = ts.tt_jd(2453541.5)  # 2005-Jun-20
t1 = ts.tt_jd(2453544.5)  # 2005-Jun-23

f = almanac.sunrise_sunset(eph, boston)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = sunrise_sunset(&mut kernel, 42.3583, -71.0603, 43.0);
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Sunrise/sunset count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Event {} type mismatch: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 120.0,
                "Sunrise/sunset {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    /// Sunrise/sunset at Sydney (southern hemisphere) December 2009
    #[test]
    fn test_sunrise_sunset_sydney_dec_2009_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

sydney = wgs84.latlon(-33.8688, 151.2093, elevation_m=58.0)
t0 = ts.tt_jd(2455175.5)  # 2009-Dec-10
t1 = ts.tt_jd(2455178.5)  # 2009-Dec-13

f = almanac.sunrise_sunset(eph, sydney)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2009, 12, 10)).tdb();
        let t1 = ts.tt((2009, 12, 13)).tdb();

        let mut f = sunrise_sunset(&mut kernel, -33.8688, 151.2093, 58.0);
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Sunrise/sunset count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Sydney event {} type: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 120.0,
                "Sydney sunrise/sunset {} diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    // --- Twilight ---

    /// Compare twilight transitions at Boston
    #[test]
    fn test_twilight_boston_june_2005_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
t0 = ts.tt_jd(2453541.5)  # 2005-Jun-20
t1 = ts.tt_jd(2453543.5)  # 2005-Jun-22

f = almanac.dark_twilight_day(eph, boston)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 22)).tdb();

        let mut f = dark_twilight_day(&mut kernel, 42.3583, -71.0603, 43.0);
        let events = find_discrete(t0, t1, &mut f, 0.25, EPSILON_DISCRETE, DEFAULT_NUM);

        // We should get the same number of transitions
        assert_eq!(
            events.len(),
            py_jds.len(),
            "Twilight transition count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Twilight {} type: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 120.0,
                "Twilight {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    // --- Meridian transits ---

    /// Compare Sun meridian transits at Boston
    #[test]
    fn test_meridian_transits_sun_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
t0 = ts.tt_jd(2453541.5)  # 2005-Jun-20
t1 = ts.tt_jd(2453544.5)  # 2005-Jun-23

f = almanac.meridian_transits(eph, eph['sun'], boston)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = meridian_transits(&mut kernel, "sun", 42.3583, -71.0603, 43.0);
        let events = find_discrete(t0, t1, &mut f, 0.5, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Transit count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Transit {} type: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 120.0,
                "Transit {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    // --- Rising and setting ---

    /// Compare Moon risings/settings at Boston
    #[test]
    fn test_moon_risings_settings_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
t0 = ts.tt_jd(2453541.5)  # 2005-Jun-20
t1 = ts.tt_jd(2453544.5)  # 2005-Jun-23

f = almanac.risings_and_settings(eph, eph['moon'], boston)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = risings_and_settings(
            &mut kernel,
            "moon",
            42.3583,
            -71.0603,
            43.0,
            REFRACTION_DEGREES,
        );
        let events = find_discrete(t0, t1, &mut f, 0.5, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Moon rise/set count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, rust_type), (&py_jd, &py_type))) in events
            .iter()
            .zip(py_jds.iter().zip(py_types.iter()))
            .enumerate()
        {
            assert_eq!(
                *rust_type, py_type,
                "Moon event {} type: rust={} py={}",
                i, rust_type, py_type
            );
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 300.0,
                "Moon rise/set {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }

    /// Compare Mars risings/settings at Greenwich
    #[test]
    fn test_mars_risings_settings_greenwich_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
from skyfield import almanac

ts = load.timescale()
eph = load('de421.bsp')

greenwich = wgs84.latlon(51.4769, -0.0005, elevation_m=11.0)
t0 = ts.tt_jd(2453541.5)  # 2005-Jun-20
t1 = ts.tt_jd(2453544.5)  # 2005-Jun-23

f = almanac.risings_and_settings(eph, eph['mars'], greenwich)
times, events = almanac.find_discrete(t0, t1, f)

jds = ','.join(str(ti.tdb) for ti in times)
evts = ','.join(str(int(e)) for e in events)
rust.collect_string(jds + '|' + evts)
"#,
            )
            .expect("Failed to run Python code");

        let parsed = PythonResult::try_from(py_result.as_str()).expect("parse");
        let s = match parsed {
            PythonResult::String(s) => s,
            _ => panic!("Expected String"),
        };
        let parts: Vec<&str> = s.split('|').collect();
        let py_jds: Vec<f64> = parts[0]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();
        let py_types: Vec<i64> = parts[1]
            .split(',')
            .map(|v| v.trim().parse().unwrap())
            .collect();

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t0 = ts.tt((2005, 6, 20)).tdb();
        let t1 = ts.tt((2005, 6, 23)).tdb();

        let mut f = risings_and_settings(
            &mut kernel,
            "mars",
            51.4769,
            -0.0005,
            11.0,
            REFRACTION_DEGREES,
        );
        let events = find_discrete(t0, t1, &mut f, 0.5, EPSILON_DISCRETE, DEFAULT_NUM);

        assert_eq!(
            events.len(),
            py_jds.len(),
            "Mars rise/set count: rust={} python={}",
            events.len(),
            py_jds.len()
        );

        for (i, ((rust_jd, _), &py_jd)) in events.iter().zip(py_jds.iter()).enumerate() {
            let diff_sec = (rust_jd - py_jd).abs() * 86400.0;
            assert!(
                diff_sec < 300.0,
                "Mars rise/set {} time diff: {:.1}s",
                i,
                diff_sec
            );
        }
    }
}
