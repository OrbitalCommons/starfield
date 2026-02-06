//! Python comparison tests for constellation identification
//!
//! Validates constellation lookup against Python Skyfield's
//! `load_constellation_map()` function.

#[cfg(test)]
mod tests {
    use crate::constellationlib::ConstellationMap;
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

    /// Test constellation identification for well-known stars vs Skyfield
    #[test]
    fn test_known_stars_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let map = ConstellationMap::new();

        // Test cases: (name, RA hours ICRS, Dec degrees ICRS, expected constellation)
        // We'll verify both our result and Skyfield's agree
        let test_stars = [
            ("Polaris", 2.5302, 89.2641),   // Ursa Minor
            ("Sirius", 6.7525, -16.7161),   // Canis Major
            ("Betelgeuse", 5.9195, 7.4071), // Orion
            ("Vega", 18.6156, 38.7837),     // Lyra
            ("Rigel", 5.2423, -8.2016),     // Orion
            ("Antares", 16.4901, -26.4320), // Scorpius
            ("Canopus", 6.3992, -52.6957),  // Carina
            ("Deneb", 20.6905, 45.2803),    // Cygnus
        ];

        for (name, ra_hours, dec_deg) in &test_stars {
            // Get Skyfield's answer
            let py_result = bridge
                .run_py_to_json(&format!(
                    r#"
from skyfield.api import load_constellation_map, position_of_radec
constellation_at = load_constellation_map()
pos = position_of_radec({ra_hours}, {dec_deg})
rust.collect_string(constellation_at(pos))
"#
                ))
                .unwrap_or_else(|e| panic!("Python failed for {name}: {e}"));

            let py_constellation = parse_string(&py_result);

            // Get our answer using the same approach (Position -> precess to B1875)
            let pos = crate::positions::Position::barycentric(
                direction_from_ra_dec(*ra_hours, *dec_deg),
                nalgebra::Vector3::zeros(),
                0,
            );
            let rust_constellation = map.constellation_of(&pos, &ts);

            assert_eq!(
                rust_constellation, py_constellation,
                "{name}: rust={rust_constellation} python={py_constellation}"
            );
        }
    }

    /// Test grid of positions across the sky vs Skyfield
    #[test]
    fn test_sky_grid_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let map = ConstellationMap::new();

        // Test a grid of positions across the sky
        let ra_values = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0];
        let dec_values = [-60.0, -30.0, 0.0, 30.0, 60.0];

        for &ra in &ra_values {
            for &dec in &dec_values {
                let py_result = bridge
                    .run_py_to_json(&format!(
                        r#"
from skyfield.api import load_constellation_map, position_of_radec
constellation_at = load_constellation_map()
pos = position_of_radec({ra}, {dec})
rust.collect_string(constellation_at(pos))
"#
                    ))
                    .unwrap_or_else(|e| panic!("Python failed for RA={ra} Dec={dec}: {e}"));

                let py_constellation = parse_string(&py_result);

                let pos = crate::positions::Position::barycentric(
                    direction_from_ra_dec(ra, dec),
                    nalgebra::Vector3::zeros(),
                    0,
                );
                let rust_constellation = map.constellation_of(&pos, &ts);

                assert_eq!(
                    rust_constellation, py_constellation,
                    "RA={ra}h Dec={dec}°: rust={rust_constellation} python={py_constellation}"
                );
            }
        }
    }

    /// Test planets at a specific date vs Skyfield
    #[test]
    fn test_planet_constellations_vs_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let ts = Timescale::default();
        let map = ConstellationMap::new();

        // Get planet positions and constellations from Skyfield at a specific date
        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, load_constellation_map
constellation_at = load_constellation_map()
ts = load.timescale()
t = ts.tt_jd(2458849.5)  # 2020-01-01
eph = load('de421.bsp')
earth = eph['earth']

results = []
for name in ['mars', 'jupiter barycenter', 'saturn barycenter']:
    astrometric = earth.at(t).observe(eph[name])
    const_name = constellation_at(astrometric)
    ra, dec, _ = astrometric.radec()
    results.append(f"{name}:{const_name}:{ra.hours:.6f}:{dec.degrees:.6f}")

rust.collect_string('|'.join(results))
"#,
            )
            .expect("Failed to run Python code");

        let result_str = parse_string(&py_result);

        for entry in result_str.split('|') {
            let parts: Vec<&str> = entry.split(':').collect();
            let planet_name = parts[0];
            let py_constellation = parts[1];
            let ra_hours: f64 = parts[2].parse().unwrap();
            let dec_deg: f64 = parts[3].parse().unwrap();

            let pos = crate::positions::Position::barycentric(
                direction_from_ra_dec(ra_hours, dec_deg),
                nalgebra::Vector3::zeros(),
                0,
            );
            let rust_constellation = map.constellation_of(&pos, &ts);

            assert_eq!(
                rust_constellation, py_constellation,
                "{planet_name}: rust={rust_constellation} python={py_constellation}"
            );
        }
    }

    /// Convert RA (hours) and Dec (degrees) to a unit direction vector
    fn direction_from_ra_dec(ra_hours: f64, dec_deg: f64) -> nalgebra::Vector3<f64> {
        let ra_rad = ra_hours * std::f64::consts::PI / 12.0;
        let dec_rad = dec_deg * std::f64::consts::PI / 180.0;
        let cos_dec = dec_rad.cos();
        nalgebra::Vector3::new(
            cos_dec * ra_rad.cos(),
            cos_dec * ra_rad.sin(),
            dec_rad.sin(),
        )
    }
}
