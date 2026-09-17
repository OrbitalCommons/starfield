//! Python comparison tests for reference frame conversions

#[cfg(test)]
mod tests {
    use crate::jplephem::SpiceKernel;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::pybridge::{PyRustBridge, PythonResult};
    use crate::time::Timescale;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").unwrap()
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
eph = load('test_data/de421.bsp')
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
eph = load('test_data/de421.bsp')
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

    /// The three epochs the illumination comparison uses, as TDB Julian dates:
    /// J2000, the HiRISE Earth portrait of 2007-10-03, and the Mars opposition
    /// of 2020-10-13.
    const ILLUMINATION_EPOCHS: [f64; 3] = [2451545.0, 2454376.5, 2459136.5];

    /// `Position::phase_angle` and `illuminated_fraction` match Skyfield's
    /// `phase_angle` and `fraction_illuminated`, both for a planet seen from
    /// the Earth and for the Earth seen from that planet.
    #[test]
    fn test_phase_angle_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, load_file
import json

ts = load.timescale()
eph = load_file('test_data/de421.bsp')
sun = eph[10]

rows = []
for jd in (2451545.0, 2454376.5, 2459136.5):
    t = ts.tdb_jd(jd)
    mars_from_earth = eph[399].at(t).observe(eph[499])
    earth_from_mars = eph[499].at(t).observe(eph[399])
    rows.append({
        "mars_phase": mars_from_earth.phase_angle(sun).radians,
        "mars_fraction": float(mars_from_earth.fraction_illuminated(sun)),
        "earth_phase": earth_from_mars.phase_angle(sun).radians,
        "earth_fraction": float(earth_from_mars.fraction_illuminated(sun)),
    })

rust.collect_string(json.dumps(rows))
"#,
            )
            .expect("Python phase angle failed");

        let inner_str = match PythonResult::try_from(py_result.as_str())
            .expect("Failed to parse Python result")
        {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        let rows: serde_json::Value = serde_json::from_str(&inner_str).expect("JSON parse failed");

        for (i, &jd) in ILLUMINATION_EPOCHS.iter().enumerate() {
            let t = ts.tdb_jd(jd);
            let row = &rows[i];

            let earth = kernel.at("399", &t).unwrap();
            let mars_from_earth = earth.observe("499", &mut kernel, &t).unwrap();
            let mars = kernel.at("499", &t).unwrap();
            let earth_from_mars = mars.observe("399", &mut kernel, &t).unwrap();

            for (label, position, py_phase, py_fraction) in [
                (
                    "Mars from Earth",
                    &mars_from_earth,
                    row["mars_phase"].as_f64().unwrap(),
                    row["mars_fraction"].as_f64().unwrap(),
                ),
                (
                    "Earth from Mars",
                    &earth_from_mars,
                    row["earth_phase"].as_f64().unwrap(),
                    row["earth_fraction"].as_f64().unwrap(),
                ),
            ] {
                let phase = position.phase_angle(&mut kernel, &t).unwrap();
                let fraction = position.illuminated_fraction(&mut kernel, &t).unwrap();
                assert!(
                    (phase - py_phase).abs() < 1e-8,
                    "{label} phase at JD {jd}: rust={phase} python={py_phase}"
                );
                assert!(
                    (fraction - py_fraction).abs() < 1e-8,
                    "{label} fraction at JD {jd}: rust={fraction} python={py_fraction}"
                );
            }
        }
    }
}
