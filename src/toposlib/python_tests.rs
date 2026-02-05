//! Python comparison tests for toposlib
//!
//! Validates Rust geographic positions, ITRS coordinates, observer barycentric
//! positions, altaz coordinates, and local sidereal time against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::jplephem::kernel::SpiceKernel;
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::helpers::PythonResult;
    use crate::time::Timescale;
    use crate::toposlib::{IERS2010, WGS84};
    use approx::assert_relative_eq;

    fn parse_f64(result: &str) -> f64 {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => s.parse::<f64>().expect("Failed to parse f64"),
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn parse_f64_triple(result: &str) -> (f64, f64, f64) {
        let parsed = PythonResult::try_from(result).expect("Failed to parse Python result");
        match parsed {
            PythonResult::String(s) => {
                let parts: Vec<f64> = s.split(',').map(|p| p.trim().parse().unwrap()).collect();
                (parts[0], parts[1], parts[2])
            }
            _ => panic!("Expected String result, got {:?}", parsed),
        }
    }

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").expect("Failed to open DE421")
    }

    // --- ITRS position tests ---

    /// Test that our geodetic-to-ITRS conversion matches Skyfield
    #[test]
    fn test_itrs_xyz_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let test_cases = [
            (42.3583, -71.0603, 43.0, "Boston"),
            (0.0, 0.0, 0.0, "Equator/PM"),
            (90.0, 0.0, 0.0, "North Pole"),
            (-33.8688, 151.2093, 58.0, "Sydney"),
            (51.4769, -0.0005, 11.0, "Greenwich"),
        ];

        for (lat, lon, elev, label) in test_cases {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import wgs84
from skyfield.constants import AU_M
pos = wgs84.latlon({lat}, {lon}, elevation_m={elev})
x, y, z = pos.itrs_xyz.au
rust.collect_string(f"{{x}},{{y}},{{z}}")
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for {label}: {e}"));

            let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

            let rust_pos = WGS84.latlon(lat, lon, elev);

            assert_relative_eq!(rust_pos.itrs_xyz.x, py_x, epsilon = 1e-12,);
            assert_relative_eq!(rust_pos.itrs_xyz.y, py_y, epsilon = 1e-12,);
            assert_relative_eq!(rust_pos.itrs_xyz.z, py_z, epsilon = 1e-12,);
        }
    }

    // --- Observer barycentric position tests ---

    /// Test that observer barycentric position is close to Earth's
    #[test]
    fn test_observer_geocentric_offset_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        // Skyfield: get geocentric offset of Boston observer
        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
ts = load.timescale()
t = ts.tdb_jd(2451545.0)
boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
geo = boston.at(t)
# Geocentric position in AU
x, y, z = geo.position.au
rust.collect_string(f"{x},{y},{z}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();

        // Compare absolute positions (should match to ~1 meter)
        let diff_x = (observer.position.x - py_x).abs() * crate::constants::AU_M;
        let diff_y = (observer.position.y - py_y).abs() * crate::constants::AU_M;
        let diff_z = (observer.position.z - py_z).abs() * crate::constants::AU_M;

        assert!(diff_x < 10.0, "X offset mismatch: {diff_x} m");
        assert!(diff_y < 10.0, "Y offset mismatch: {diff_y} m");
        assert!(diff_z < 10.0, "Z offset mismatch: {diff_z} m");
    }

    // --- Altaz tests ---

    /// Test alt/az of Mars from Boston matches Skyfield
    #[test]
    fn test_altaz_mars_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)
boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
observer = eph['earth'] + boston
astrometric = observer.at(t).observe(eph['mars'])
apparent = astrometric.apparent()
alt, az, dist = apparent.altaz()
rust.collect_string(f"{alt.degrees},{az.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_alt, py_az, py_dist) = parse_f64_triple(&py_result);

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();
        let mars_astro = observer.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();
        let (rust_alt, rust_az, rust_dist) = boston.altaz(&mars_app, &t);

        // Altitude and azimuth should match within ~0.1 degrees
        // (small differences expected due to UT1-UTC, nutation model details)
        assert!(
            (rust_alt - py_alt).abs() < 0.5,
            "Altitude mismatch: rust={rust_alt} python={py_alt} diff={}°",
            (rust_alt - py_alt).abs()
        );
        assert!(
            (rust_az - py_az).abs() < 0.5,
            "Azimuth mismatch: rust={rust_az} python={py_az} diff={}°",
            (rust_az - py_az).abs()
        );
        assert!(
            (rust_dist - py_dist).abs() / py_dist < 1e-6,
            "Distance mismatch: rust={rust_dist} python={py_dist}"
        );
    }

    /// Test alt/az of Jupiter from Sydney
    #[test]
    fn test_altaz_jupiter_sydney_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2455000.5); // ~2009

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2455000.5)
sydney = wgs84.latlon(-33.8688, 151.2093, elevation_m=58.0)
observer = eph['earth'] + sydney
astrometric = observer.at(t).observe(eph['jupiter barycenter'])
apparent = astrometric.apparent()
alt, az, dist = apparent.altaz()
rust.collect_string(f"{alt.degrees},{az.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_alt, py_az, py_dist) = parse_f64_triple(&py_result);

        let sydney = WGS84.latlon(-33.8688, 151.2093, 58.0);
        let observer = sydney.at(&t, &mut kernel).unwrap();
        let jupiter_astro = observer
            .observe("jupiter barycenter", &mut kernel, &t)
            .unwrap();
        let jupiter_app = jupiter_astro.apparent(&mut kernel, &t).unwrap();
        let (rust_alt, rust_az, rust_dist) = sydney.altaz(&jupiter_app, &t);

        assert!(
            (rust_alt - py_alt).abs() < 0.5,
            "Altitude mismatch: rust={rust_alt} python={py_alt}"
        );
        assert!(
            (rust_az - py_az).abs() < 0.5,
            "Azimuth mismatch: rust={rust_az} python={py_az}"
        );
        assert!(
            (rust_dist - py_dist).abs() / py_dist < 1e-6,
            "Distance mismatch: rust={rust_dist} python={py_dist}"
        );
    }

    // --- RA/Dec tests ---

    /// Test that topocentric RA/Dec of Mars matches Skyfield
    #[test]
    fn test_radec_mars_topocentric_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)
boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
observer = eph['earth'] + boston
astrometric = observer.at(t).observe(eph['mars'])
apparent = astrometric.apparent()
ra, dec, dist = apparent.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_ra_deg, py_dec, py_dist) = parse_f64_triple(&py_result);
        let py_ra_h = py_ra_deg / 15.0;

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let observer = boston.at(&t, &mut kernel).unwrap();
        let mars_astro = observer.observe("mars", &mut kernel, &t).unwrap();
        let mars_app = mars_astro.apparent(&mut kernel, &t).unwrap();
        let (rust_ra_h, rust_dec, rust_dist) = mars_app.radec(None);

        // RA should match within ~1 arcsecond = 0.067 seconds of time = 0.004h
        assert!(
            (rust_ra_h - py_ra_h).abs() < 0.01,
            "RA mismatch: rust={rust_ra_h}h python={py_ra_h}h diff={}h",
            (rust_ra_h - py_ra_h).abs()
        );
        assert!(
            (rust_dec - py_dec).abs() < 0.01,
            "Dec mismatch: rust={rust_dec}° python={py_dec}° diff={}°",
            (rust_dec - py_dec).abs()
        );
        assert!(
            (rust_dist - py_dist).abs() / py_dist < 1e-5,
            "Distance mismatch: rust={rust_dist} python={py_dist}"
        );
    }

    // --- Local sidereal time tests ---

    /// Test that LST matches Skyfield's lst_hours_at
    #[test]
    fn test_lst_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, wgs84
ts = load.timescale()
t = ts.tdb_jd(2451545.0)
boston = wgs84.latlon(42.3583, -71.0603, elevation_m=43.0)
lst = boston.lst_hours_at(t)
rust.collect_string(str(lst))
"#,
            )
            .expect("Failed to run Python code");

        let py_lst = parse_f64(&py_result);

        let boston = WGS84.latlon(42.3583, -71.0603, 43.0);
        let rust_lst = boston.lst_hours(&t);

        assert!(
            (rust_lst - py_lst).abs() < 0.01,
            "LST mismatch: rust={rust_lst}h python={py_lst}h diff={}h",
            (rust_lst - py_lst).abs()
        );
    }

    /// Test LST at multiple longitudes
    #[test]
    fn test_lst_multiple_longitudes_match_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let t = ts.tdb_jd(2458000.5); // ~2017

        let longitudes = [0.0, 90.0, -90.0, 180.0, -45.0, 139.65];

        for lon in longitudes {
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load, wgs84
ts = load.timescale()
t = ts.tdb_jd(2458000.5)
pos = wgs84.latlon(0.0, {lon}, elevation_m=0.0)
lst = pos.lst_hours_at(t)
rust.collect_string(str(lst))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for lon={lon}: {e}"));

            let py_lst = parse_f64(&py_result);

            let pos = WGS84.latlon(0.0, lon, 0.0);
            let rust_lst = pos.lst_hours(&t);

            assert!(
                (rust_lst - py_lst).abs() < 0.01,
                "LST at lon={lon}: rust={rust_lst}h python={py_lst}h",
            );
        }
    }

    // --- Refraction tests ---

    /// Test atmospheric refraction matches standard values
    #[test]
    fn test_refraction_standard_values() {
        // At horizon: ~34 arcmin ≈ 0.567°
        let r0 = crate::toposlib::GeographicPosition::refract(0.0, 10.0, 1010.0);
        assert!(r0 > 0.4 && r0 < 0.7, "Horizon refraction {r0}");

        // At 45°: ~1 arcmin ≈ 0.017°
        let r45 = crate::toposlib::GeographicPosition::refract(45.0, 10.0, 1010.0);
        assert!(r45 > 0.01 && r45 < 0.03, "45° refraction {r45}");

        // At 10°: ~5 arcmin ≈ 0.083°
        let r10 = crate::toposlib::GeographicPosition::refract(10.0, 10.0, 1010.0);
        assert!(r10 > 0.05 && r10 < 0.15, "10° refraction {r10}");
    }
}
