//! Python comparison tests for reference frame conversions

#[cfg(test)]
mod tests {
    use crate::jplephem::SpiceKernel;
    use crate::pybridge::{PyRustBridge, PythonResult};
    use crate::time::Timescale;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("src/jplephem/test_data/de421.bsp").unwrap()
    }

    /// Ecliptic longitude/latitude of Mars matches Skyfield's frame_latlon
    #[test]
    fn test_ecliptic_latlon_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.framelib import ecliptic_frame
import json

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)

earth = eph['earth'].at(t)
mars = earth.observe(eph['mars'])
lat, lon, dist = mars.frame_latlon(ecliptic_frame)

rust.collect_string(json.dumps({
    "lon_deg": lon.degrees,
    "lat_deg": lat.degrees,
    "dist_au": dist.au,
}))
"#,
            )
            .expect("Python ecliptic failed");

        let inner_str = match PythonResult::try_from(py_result.as_str())
            .expect("Failed to parse Python result")
        {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(&inner_str).expect("JSON parse failed");

        let py_lon = parsed["lon_deg"].as_f64().unwrap();
        let py_lat = parsed["lat_deg"].as_f64().unwrap();
        let py_dist = parsed["dist_au"].as_f64().unwrap();

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let (lon_rad, lat_rad, dist) = mars.ecliptic_latlon(&t);
        let rust_lon = lon_rad.to_degrees();
        let rust_lat = lat_rad.to_degrees();

        assert!(
            (rust_lon - py_lon).abs() < 0.01,
            "Ecliptic lon: rust={rust_lon} python={py_lon}"
        );
        assert!(
            (rust_lat - py_lat).abs() < 0.01,
            "Ecliptic lat: rust={rust_lat} python={py_lat}"
        );
        assert!(
            (dist - py_dist).abs() < 0.001,
            "Distance: rust={dist} python={py_dist}"
        );
    }

    /// Galactic latitude/longitude of Mars matches Skyfield
    #[test]
    fn test_galactic_frame_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load
from skyfield.framelib import galactic_frame
import json

ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)

earth = eph['earth'].at(t)
mars = earth.observe(eph['mars'])
lat, lon, dist = mars.frame_latlon(galactic_frame)

rust.collect_string(json.dumps({
    "lon_deg": lon.degrees,
    "lat_deg": lat.degrees,
    "dist_au": dist.au,
}))
"#,
            )
            .expect("Python galactic failed");

        let inner_str = match PythonResult::try_from(py_result.as_str())
            .expect("Failed to parse Python result")
        {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        let parsed: serde_json::Value =
            serde_json::from_str(&inner_str).expect("JSON parse failed");

        let py_lon = parsed["lon_deg"].as_f64().unwrap();
        let py_lat = parsed["lat_deg"].as_f64().unwrap();

        let earth = kernel.at("earth", &t).unwrap();
        let mars = earth.observe("mars", &mut kernel, &t).unwrap();
        let (lon_rad, lat_rad, _) = mars.frame_latlon(&crate::framelib::GALACTIC, &t);
        let rust_lon = lon_rad.to_degrees();
        let rust_lat = lat_rad.to_degrees();

        assert!(
            (rust_lon - py_lon).abs() < 0.01,
            "Galactic lon: rust={rust_lon} python={py_lon}"
        );
        assert!(
            (rust_lat - py_lat).abs() < 0.01,
            "Galactic lat: rust={rust_lat} python={py_lat}"
        );
    }
}
