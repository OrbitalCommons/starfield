//! Python comparison tests for Kepler orbit propagation

use crate::constants::GM_SUN;
use crate::keplerlib::{comet_orbit, mpcorb_orbit};
use crate::pybridge::bridge::PyRustBridge;
use crate::pybridge::helpers::PythonResult;
use crate::time::Timescale;

/// Test comet orbit propagation against Skyfield
///
/// Creates a comet with known orbital elements, propagates in both
/// Rust and Python, and compares the resulting positions.
#[test]
fn test_comet_orbit_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    // Comet parameters (approximate Halley)
    let q = 0.586; // perihelion distance AU
    let e = 0.967;
    let i = 162.26;
    let om = 58.42;
    let w = 111.33;
    let perihelion_year = 1986;
    let perihelion_month = 2;
    let perihelion_day: f64 = 9.4589;

    let ts = Timescale::default();
    // Use tt() with integer day, then add fractional day via tt_jd
    let t_whole = ts.tt((
        perihelion_year,
        perihelion_month as u32,
        perihelion_day as u32,
    ));
    let perihelion_jd = t_whole.tt() + (perihelion_day - perihelion_day.floor());
    let t_perihelion = ts.tt_jd(perihelion_jd, None);

    let orbit = comet_orbit(q, e, i, om, w, &t_perihelion, GM_SUN, Some("1P/Halley"));

    // Propagate to a test date (2000-01-01)
    let t_test = ts.tt_jd(2451545.0, None);
    let pos = orbit.at(&t_test);

    // Get Skyfield's result
    let python_code = format!(
        r#"
from skyfield.api import load
from skyfield.data import mpc
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.keplerlib import _KeplerOrbit

ts = load.timescale()

t_perihelion = ts.tt({perihelion_year}, {perihelion_month}, {perihelion_day})

q = {q}
e = {e}
p = q / (1.0 - e) * (1.0 - e*e) if e != 1.0 else q * 2.0

comet = _KeplerOrbit._from_periapsis(
    p, e, {i}, {om}, {w},
    t_perihelion, GM_SUN, 10, '1P/Halley',
)
from skyfield.data.spice import inertial_frames
comet._rotation = inertial_frames['ECLIPJ2000'].T

t = ts.tt_jd(2451545.0)
pos, vel, _, _ = comet._at(t)

rust.collect_string(f"{{pos[0]:.12f}},{{pos[1]:.12f}},{{pos[2]:.12f}}")
"#
    );

    let result = bridge.run_py_to_json(&python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let coords_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let coords: Vec<f64> = coords_str.split(',').map(|s| s.parse().unwrap()).collect();

    let py_x = coords[0];
    let py_y = coords[1];
    let py_z = coords[2];

    // Compare positions (tolerance ~0.01 AU for a comet at ~35 AU)
    let tol = 0.1; // AU — generous tolerance for distant comet
    assert!(
        (pos.position.x - py_x).abs() < tol,
        "X mismatch: rust={} py={} diff={}",
        pos.position.x,
        py_x,
        (pos.position.x - py_x).abs()
    );
    assert!(
        (pos.position.y - py_y).abs() < tol,
        "Y mismatch: rust={} py={} diff={}",
        pos.position.y,
        py_y,
        (pos.position.y - py_y).abs()
    );
    assert!(
        (pos.position.z - py_z).abs() < tol,
        "Z mismatch: rust={} py={} diff={}",
        pos.position.z,
        py_z,
        (pos.position.z - py_z).abs()
    );
}

/// Test minor planet orbit against Skyfield using Ceres-like elements
#[test]
fn test_mpcorb_orbit_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    // Ceres approximate elements
    let a = 2.7691651;
    let e = 0.0760091;
    let i_deg = 10.59351;
    let om_deg = 80.30553;
    let w_deg = 73.59764;
    let m_deg = 95.98917;
    let epoch_jd = 2458600.5; // 2019 Apr 27.0 TT

    let ts = Timescale::default();
    let epoch = ts.tt_jd(epoch_jd, None);

    let orbit = mpcorb_orbit(
        a,
        e,
        i_deg,
        om_deg,
        w_deg,
        m_deg,
        &epoch,
        GM_SUN,
        Some("Ceres"),
    );

    // Propagate 100 days forward
    let t_test = ts.tt_jd(epoch_jd + 100.0, None);
    let pos = orbit.at(&t_test);

    // Get Skyfield's result
    let python_code = format!(
        r#"
from skyfield.api import load
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.keplerlib import _KeplerOrbit
from skyfield.data.spice import inertial_frames
import numpy as np

ts = load.timescale()

a = {a}
e = {e}
p = a * (1.0 - e*e)

epoch = ts.tt_jd({epoch_jd})

mp = _KeplerOrbit._from_mean_anomaly(
    p, e, {i_deg}, {om_deg}, {w_deg}, {m_deg},
    epoch, GM_SUN, 10, 'Ceres',
)
mp._rotation = inertial_frames['ECLIPJ2000'].T

t = ts.tt_jd({epoch_jd} + 100.0)
pos, vel, _, _ = mp._at(t)

rust.collect_string(f"{{pos[0]:.12f}},{{pos[1]:.12f}},{{pos[2]:.12f}}")
"#
    );

    let result = bridge.run_py_to_json(&python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let coords_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let coords: Vec<f64> = coords_str.split(',').map(|s| s.parse().unwrap()).collect();

    let py_x = coords[0];
    let py_y = coords[1];
    let py_z = coords[2];

    // Ceres at ~2.8 AU: tolerance 0.001 AU = ~150,000 km
    let tol = 0.001;
    assert!(
        (pos.position.x - py_x).abs() < tol,
        "X mismatch: rust={} py={} diff={}",
        pos.position.x,
        py_x,
        (pos.position.x - py_x).abs()
    );
    assert!(
        (pos.position.y - py_y).abs() < tol,
        "Y mismatch: rust={} py={} diff={}",
        pos.position.y,
        py_y,
        (pos.position.y - py_y).abs()
    );
    assert!(
        (pos.position.z - py_z).abs() < tol,
        "Z mismatch: rust={} py={} diff={}",
        pos.position.z,
        py_z,
        (pos.position.z - py_z).abs()
    );
}

/// Test eccentric anomaly solver against Skyfield
#[test]
fn test_eccentric_anomaly_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    let test_cases = [(0.1, 1.0), (0.5, 0.5), (0.9, 2.0), (0.01, 3.0), (0.99, 0.1)];

    let python_code = r#"
from skyfield.keplerlib import eccentric_anomaly
import numpy as np

cases = [(0.1, 1.0), (0.5, 0.5), (0.9, 2.0), (0.01, 3.0), (0.99, 0.1)]
results = []
for e, M in cases:
    E = eccentric_anomaly(float(e), float(M))
    results.append(f"{float(E):.15f}")
rust.collect_string(",".join(results))
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let py_values: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    for ((e, m), py_ea) in test_cases.iter().zip(py_values.iter()) {
        let rust_ea = crate::keplerlib::eccentric_anomaly(*e, *m);
        assert!(
            (rust_ea - py_ea).abs() < 1e-12,
            "e={e} M={m}: rust={rust_ea} py={py_ea} diff={}",
            (rust_ea - py_ea).abs()
        );
    }
}

/// Test ele_to_vec against Skyfield
#[test]
fn test_ele_to_vec_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    let python_code = r#"
from skyfield.keplerlib import ele_to_vec
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.constants import AU_KM, DAY_S
import numpy as np

CONVERT_GM = DAY_S * DAY_S / AU_KM / AU_KM / AU_KM
gm = GM_SUN * CONVERT_GM

# Test case: Earth-like orbit
p = 1.0
e = 0.0167
i = 0.0  # face-on for simplicity
Om = 0.0
w = 0.0
v = 0.5  # true anomaly = 0.5 radians

pos, vel = ele_to_vec(float(p), float(e), float(i), float(Om), float(w), float(v), float(gm))

rust.collect_string(f"{float(pos[0]):.15f},{float(pos[1]):.15f},{float(pos[2]):.15f},{float(vel[0]):.15f},{float(vel[1]):.15f},{float(vel[2]):.15f}")
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let vals: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    let convert_gm = DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);
    let gm = GM_SUN as f64 * convert_gm;

    let (pos, vel) = crate::keplerlib::ele_to_vec(1.0, 0.0167, 0.0, 0.0, 0.0, 0.5, gm);

    assert!(
        (pos.x - vals[0]).abs() < 1e-12,
        "pos.x: rust={} py={}",
        pos.x,
        vals[0]
    );
    assert!(
        (pos.y - vals[1]).abs() < 1e-12,
        "pos.y: rust={} py={}",
        pos.y,
        vals[1]
    );
    assert!(
        (pos.z - vals[2]).abs() < 1e-12,
        "pos.z: rust={} py={}",
        pos.z,
        vals[2]
    );
    assert!(
        (vel.x - vals[3]).abs() < 1e-10,
        "vel.x: rust={} py={}",
        vel.x,
        vals[3]
    );
    assert!(
        (vel.y - vals[4]).abs() < 1e-10,
        "vel.y: rust={} py={}",
        vel.y,
        vals[4]
    );
    assert!(
        (vel.z - vals[5]).abs() < 1e-12,
        "vel.z: rust={} py={}",
        vel.z,
        vals[5]
    );
}

/// Test propagation of a simple orbit against Skyfield
#[test]
fn test_propagate_vs_skyfield() {
    let bridge = PyRustBridge::new().unwrap();

    let python_code = r#"
from skyfield.keplerlib import propagate
from skyfield.constants import GM_SUN_Pitjeva_2005_km3_s2 as GM_SUN
from skyfield.constants import AU_KM, DAY_S
import numpy as np

CONVERT_GM = DAY_S * DAY_S / AU_KM / AU_KM / AU_KM
gm = GM_SUN * CONVERT_GM

# Circular orbit at 1 AU
pos0 = np.array([1.0, 0.0, 0.0])
v_circ = np.sqrt(gm)
vel0 = np.array([0.0, v_circ, 0.0])

# Propagate 100 days
t0 = 2451545.0
t1 = np.array([2451645.0])

pos1, vel1 = propagate(pos0, vel0, t0, t1, gm)

rust.collect_string(f"{float(pos1[0]):.15f},{float(pos1[1]):.15f},{float(pos1[2]):.15f},{float(vel1[0]):.15f},{float(vel1[1]):.15f},{float(vel1[2]):.15f}")
"#;

    let result = bridge.run_py_to_json(python_code).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    let values_str = match parsed {
        PythonResult::String(s) => s,
        _ => panic!("Expected string result"),
    };
    let vals: Vec<f64> = values_str.split(',').map(|s| s.parse().unwrap()).collect();

    use crate::constants::AU_KM;
    let convert_gm = DAY_S * DAY_S / (AU_KM * AU_KM * AU_KM);
    let gm = GM_SUN as f64 * convert_gm;
    let v_circ = gm.sqrt();

    let pos0 = nalgebra::Vector3::new(1.0, 0.0, 0.0);
    let vel0 = nalgebra::Vector3::new(0.0, v_circ, 0.0);

    let (pos1, vel1) = crate::keplerlib::propagate(&pos0, &vel0, 2451545.0, 2451645.0, gm);

    let tol = 1e-10;
    assert!(
        (pos1.x - vals[0]).abs() < tol,
        "pos.x: rust={} py={} diff={}",
        pos1.x,
        vals[0],
        (pos1.x - vals[0]).abs()
    );
    assert!(
        (pos1.y - vals[1]).abs() < tol,
        "pos.y: rust={} py={} diff={}",
        pos1.y,
        vals[1],
        (pos1.y - vals[1]).abs()
    );
    assert!(
        (pos1.z - vals[2]).abs() < tol,
        "pos.z: rust={} py={} diff={}",
        pos1.z,
        vals[2],
        (pos1.z - vals[2]).abs()
    );
    assert!(
        (vel1.x - vals[3]).abs() < tol,
        "vel.x: rust={} py={} diff={}",
        vel1.x,
        vals[3],
        (vel1.x - vals[3]).abs()
    );
    assert!(
        (vel1.y - vals[4]).abs() < tol,
        "vel.y: rust={} py={} diff={}",
        vel1.y,
        vals[4],
        (vel1.y - vals[4]).abs()
    );
    assert!(
        (vel1.z - vals[5]).abs() < tol,
        "vel.z: rust={} py={} diff={}",
        vel1.z,
        vals[5],
        (vel1.z - vals[5]).abs()
    );
}

// Make `ele_to_vec` and `propagate` accessible for tests
use crate::constants::{AU_KM, DAY_S};
