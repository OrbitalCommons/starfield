//! Python comparison tests for osculating orbital elements

use crate::constants::{AU_KM, GM_SUN};
use crate::elementslib::OsculatingElements;
use crate::pybridge::bridge::PyRustBridge;
use crate::pybridge::helpers::PythonResult;
use nalgebra::Vector3;

/// Test osculating elements against Skyfield for a planet from DE421
#[test]
fn test_elements_earth_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    let python_code = r#"
from skyfield.api import load
from skyfield.elementslib import osculating_elements_of
from skyfield.data.spice import inertial_frames

ts = load.timescale()
t = ts.tdb_jd(2451545.0)

eph = load('de421.bsp')
sun = eph['sun']
earth = eph['earth']

pos = (earth - sun).at(t)
elements = osculating_elements_of(pos)

results = [
    float(elements.semi_major_axis.au),
    float(elements.eccentricity),
    float(elements.inclination.radians),
    float(elements.longitude_of_ascending_node.radians),
    float(elements.argument_of_periapsis.radians),
    float(elements.true_anomaly.radians),
    float(elements.mean_anomaly.radians),
    float(elements.periapsis_distance.au),
    float(elements.period_in_days),
]
rust.collect_string(",".join(f"{v:.15f}" for v in results))
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let py: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    // Get Earth's heliocentric state from DE421
    let mut kernel =
        crate::jplephem::kernel::SpiceKernel::open("src/jplephem/test_data/de421.bsp").unwrap();
    let ts = crate::time::Timescale::default();
    let t = ts.tdb_jd(2451545.0);

    let earth = kernel.at("earth", &t).unwrap();
    let sun = kernel.at("sun", &t).unwrap();

    let pos_au = earth.position - sun.position;
    let vel_au_day = earth.velocity - sun.velocity;

    // Use Sun's GM for heliocentric elements
    let elem = OsculatingElements::from_au(&pos_au, &vel_au_day, GM_SUN);

    // Compare elements — tolerance reflects different state vector extraction paths
    let tol = 1e-5;
    assert!(
        (elem.semi_major_axis_au() - py[0]).abs() < tol,
        "a: rust={} py={} diff={}",
        elem.semi_major_axis_au(),
        py[0],
        (elem.semi_major_axis_au() - py[0]).abs()
    );
    assert!(
        (elem.eccentricity() - py[1]).abs() < 1e-5,
        "e: rust={} py={} diff={}",
        elem.eccentricity(),
        py[1],
        (elem.eccentricity() - py[1]).abs()
    );
    assert!(
        (elem.inclination_rad() - py[2]).abs() < 1e-4,
        "i: rust={} py={} diff={}",
        elem.inclination_rad(),
        py[2],
        (elem.inclination_rad() - py[2]).abs()
    );
}

/// Test osculating elements for Mars against Skyfield
#[test]
fn test_elements_mars_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    let python_code = r#"
from skyfield.api import load
from skyfield.elementslib import osculating_elements_of

ts = load.timescale()
t = ts.tdb_jd(2451545.0)

eph = load('de421.bsp')
sun = eph['sun']
mars = eph['mars']

pos = (mars - sun).at(t)
elements = osculating_elements_of(pos)

results = [
    float(elements.semi_major_axis.au),
    float(elements.eccentricity),
    float(elements.inclination.radians),
    float(elements.periapsis_distance.au),
    float(elements.period_in_days),
    float(elements.mean_anomaly.radians),
]
rust.collect_string(",".join(f"{v:.15f}" for v in results))
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let py: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    let mut kernel =
        crate::jplephem::kernel::SpiceKernel::open("src/jplephem/test_data/de421.bsp").unwrap();
    let ts = crate::time::Timescale::default();
    let t = ts.tdb_jd(2451545.0);

    let mars = kernel.at("mars", &t).unwrap();
    let sun = kernel.at("sun", &t).unwrap();

    let pos_au = mars.position - sun.position;
    let vel_au_day = mars.velocity - sun.velocity;

    let elem = OsculatingElements::from_au(&pos_au, &vel_au_day, GM_SUN);

    assert!(
        (elem.semi_major_axis_au() - py[0]).abs() < 1e-6,
        "a: rust={} py={}",
        elem.semi_major_axis_au(),
        py[0]
    );
    assert!(
        (elem.eccentricity() - py[1]).abs() < 1e-6,
        "e: rust={} py={}",
        elem.eccentricity(),
        py[1]
    );
    assert!(
        (elem.inclination_rad() - py[2]).abs() < 1e-4,
        "i: rust={} py={}",
        elem.inclination_rad(),
        py[2]
    );
    assert!(
        (elem.periapsis_distance_au() - py[3]).abs() < 1e-6,
        "q: rust={} py={}",
        elem.periapsis_distance_au(),
        py[3]
    );
    assert!(
        (elem.period_days() - py[4]).abs() < 0.01,
        "P: rust={} py={}",
        elem.period_days(),
        py[4]
    );
}

/// Test round-trip: elements → state vector → elements
#[test]
fn test_roundtrip_elements_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    // Create an orbit from elements in Python, get the state vector, then recompute elements
    let python_code = r#"
from skyfield.api import load
from skyfield.elementslib import osculating_elements_of
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN

ts = load.timescale()
t = ts.tdb_jd(2451545.0)

eph = load('de421.bsp')
jupiter = eph['jupiter barycenter']
sun = eph['sun']

pos = (jupiter - sun).at(t)
el = osculating_elements_of(pos, gm_km3_s2=GM_SUN)

# Get state vector in km and km/s
px, py_val, pz = pos.position.km
vx, vy, vz = pos.velocity.km_per_s

# Also get the elements for comparison
results = [
    float(px), float(py_val), float(pz),
    float(vx), float(vy), float(vz),
    float(el.semi_major_axis.au),
    float(el.eccentricity),
    float(el.inclination.radians),
    float(el.longitude_of_ascending_node.radians),
    float(el.argument_of_periapsis.radians),
    float(el.true_anomaly.radians),
]
rust.collect_string(",".join(f"{v:.15f}" for v in results))
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let vals: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    let pos_km = Vector3::new(vals[0], vals[1], vals[2]);
    let vel_km_s = Vector3::new(vals[3], vals[4], vals[5]);

    let elem = OsculatingElements::new(pos_km, vel_km_s, GM_SUN);

    // Compare with Skyfield's elements
    assert!(
        (elem.semi_major_axis_au() - vals[6]).abs() < 1e-8,
        "a: rust={} py={}",
        elem.semi_major_axis_au(),
        vals[6]
    );
    assert!(
        (elem.eccentricity() - vals[7]).abs() < 1e-10,
        "e: rust={} py={}",
        elem.eccentricity(),
        vals[7]
    );
    assert!(
        (elem.inclination_rad() - vals[8]).abs() < 1e-10,
        "i: rust={} py={}",
        elem.inclination_rad(),
        vals[8]
    );
    assert!(
        (elem.longitude_of_ascending_node_rad() - vals[9]).abs() < 1e-8,
        "Ω: rust={} py={}",
        elem.longitude_of_ascending_node_rad(),
        vals[9]
    );
    assert!(
        (elem.argument_of_periapsis_rad() - vals[10]).abs() < 1e-8,
        "ω: rust={} py={}",
        elem.argument_of_periapsis_rad(),
        vals[10]
    );
    assert!(
        (elem.true_anomaly_rad() - vals[11]).abs() < 1e-8,
        "ν: rust={} py={}",
        elem.true_anomaly_rad(),
        vals[11]
    );
}

use crate::constants::DAY_S;
