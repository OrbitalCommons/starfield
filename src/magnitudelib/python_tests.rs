//! Python comparison tests for planetary magnitude calculations

#[cfg(test)]
mod tests {
    use crate::jplephem::SpiceKernel;
    use crate::magnitudelib::planetary_magnitude;
    use crate::pybridge::{PyRustBridge, PythonResult};
    use crate::time::Timescale;

    fn unwrap_py_json(raw: &str) -> serde_json::Value {
        let inner = match PythonResult::try_from(raw).expect("Failed to parse Python result") {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        serde_json::from_str(&inner).expect("JSON parse failed")
    }

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").unwrap()
    }

    /// Jupiter magnitude matches Skyfield at 2020-Jul-31
    #[test]
    fn test_jupiter_magnitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2459062.5); // 2020-Jul-31

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.magnitudelib import planetary_magnitude
import json

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2459062.5)

earth = eph['earth'].at(t)
jupiter = earth.observe(eph['jupiter barycenter'])
mag = planetary_magnitude(jupiter)

rust.collect_string(json.dumps({"mag": float(mag)}))
"#,
            )
            .expect("Python Jupiter magnitude failed");

        let parsed = unwrap_py_json(&py_result);
        let py_mag = parsed["mag"].as_f64().unwrap();

        let earth = kernel.at("earth", &t).unwrap();
        let jupiter = earth
            .observe("jupiter barycenter", &mut kernel, &t)
            .unwrap();
        let rust_mag = planetary_magnitude(&jupiter, &t).unwrap();

        assert!(
            (rust_mag - py_mag).abs() < 0.05,
            "Jupiter mag: rust={rust_mag} python={py_mag}"
        );
    }

    /// Mars magnitude matches Skyfield at J2000
    #[test]
    fn test_mars_magnitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.magnitudelib import planetary_magnitude
import json

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)

earth = eph['earth'].at(t)
mars = earth.observe(eph['mars'])
mag = planetary_magnitude(mars)

rust.collect_string(json.dumps({"mag": float(mag)}))
"#,
            )
            .expect("Python Mars magnitude failed");

        let parsed = unwrap_py_json(&py_result);
        let py_mag = parsed["mag"].as_f64().unwrap();

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let rust_mag = planetary_magnitude(&mars, &t).unwrap();

        assert!(
            (rust_mag - py_mag).abs() < 0.1,
            "Mars mag: rust={rust_mag} python={py_mag}"
        );
    }

    /// Venus magnitude matches Skyfield
    #[test]
    fn test_venus_magnitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.magnitudelib import planetary_magnitude
import json

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)

earth = eph['earth'].at(t)
venus = earth.observe(eph['venus'])
mag = planetary_magnitude(venus)

rust.collect_string(json.dumps({"mag": float(mag)}))
"#,
            )
            .expect("Python Venus magnitude failed");

        let parsed = unwrap_py_json(&py_result);
        let py_mag = parsed["mag"].as_f64().unwrap();

        let earth = kernel.at("earth", &t).unwrap();
        let venus = earth.observe("venus", &mut kernel, &t).unwrap();
        let rust_mag = planetary_magnitude(&venus, &t).unwrap();

        assert!(
            (rust_mag - py_mag).abs() < 0.05,
            "Venus mag: rust={rust_mag} python={py_mag}"
        );
    }

    /// Saturn magnitude matches Skyfield (with ring tilt)
    #[test]
    fn test_saturn_magnitude_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.magnitudelib import planetary_magnitude
import json, math

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)

earth = eph['earth'].at(t)
saturn = earth.observe(eph['saturn barycenter'])
mag = planetary_magnitude(saturn)

rust.collect_string(json.dumps({"mag": float(mag) if not math.isnan(mag) else "nan"}))
"#,
            )
            .expect("Python Saturn magnitude failed");

        let parsed = unwrap_py_json(&py_result);

        let earth = kernel.at("earth", &t).unwrap();
        let saturn = earth.observe("saturn barycenter", &mut kernel, &t).unwrap();
        let rust_mag = planetary_magnitude(&saturn, &t).unwrap();

        if let Some(py_mag) = parsed["mag"].as_f64() {
            assert!(
                (rust_mag - py_mag).abs() < 0.1,
                "Saturn mag: rust={rust_mag} python={py_mag}"
            );
        } else {
            // Both should be NaN
            assert!(rust_mag.is_nan(), "Expected NaN for Saturn, got {rust_mag}");
        }
    }

    /// Multiple planets at a single date match Skyfield
    #[test]
    fn test_all_planets_magnitude_batch() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2458000.5); // 2017-Sep-04

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.magnitudelib import planetary_magnitude
import json, math

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2458000.5)

earth = eph['earth'].at(t)
results = {}
for name in ['venus', 'mars', 'jupiter barycenter', 'uranus barycenter', 'neptune barycenter']:
    obs = earth.observe(eph[name])
    mag = planetary_magnitude(obs)
    key = name.replace(' barycenter', '')
    results[key] = float(mag) if not math.isnan(mag) else None

rust.collect_string(json.dumps(results))
"#,
            )
            .expect("Python batch magnitude failed");

        let parsed = unwrap_py_json(&py_result);

        let planets = [
            ("venus", "venus"),
            ("mars", "mars"),
            ("jupiter", "jupiter barycenter"),
            ("uranus", "uranus barycenter"),
            ("neptune", "neptune barycenter"),
        ];

        let earth = kernel.at("earth", &t).unwrap();

        for (key, spice_name) in planets {
            let obs = earth.observe(spice_name, &mut kernel, &t).unwrap();
            let rust_mag = planetary_magnitude(&obs, &t).unwrap();

            if let Some(py_mag) = parsed[key].as_f64() {
                assert!(
                    (rust_mag - py_mag).abs() < 0.1,
                    "{key} mag: rust={rust_mag} python={py_mag}"
                );
            }
        }
    }
}
