//! Python comparison tests for `planetarylib`.
//!
//! These compare [`PlanetaryConstants::read_text`] against Skyfield's
//! `skyfield.planetarylib.PlanetaryConstants.read_text` on the checked-in
//! excerpt of `pck00011.tpc`, so both parsers see exactly the same bytes;
//! [`PckFrame`](crate::planetarylib::PckFrame) against Skyfield's
//! `planetarylib.Frame` on the lunar principal-axes kernel; and
//! [`IauFrame`](crate::planetarylib::IauFrame) against SpiceyPy `pxform` and
//! `sxform`. The kernel tests are `#[ignore]`d because they need a download;
//! the same numbers are checked in as golden matrices next to each frame.

#[cfg(test)]
mod tests {
    use crate::framelib::Frame;
    use crate::planetarylib::{KernelValue, PlanetaryConstants};
    use crate::pybridge::{PyRustBridge, PythonResult};
    use std::collections::HashMap;

    /// The three epochs both implementations are asked about, as TDB Julian
    /// dates: 2000-01-01, 2010-06-15 and 2025-03-01.
    const EPOCHS: [f64; 3] = [2451544.5, 2455362.5, 2460735.5];

    /// The excerpt both parsers read, relative to the crate root.
    const EXCERPT_PATH: &str = "src/planetarylib/pck00011_excerpt.tpc";

    fn unwrap_py_json(raw: &str) -> serde_json::Value {
        let inner = match PythonResult::try_from(raw).expect("Failed to parse Python result") {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        serde_json::from_str(&inner).expect("JSON parse failed")
    }

    /// Every variable Skyfield reads out of the excerpt, as a flat list of
    /// numbers per name.
    fn skyfield_variables() -> HashMap<String, Vec<f64>> {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let code = format!(
            r#"
import json
from skyfield.planetarylib import PlanetaryConstants

pc = PlanetaryConstants()
with open('{path}', 'rb') as f:
    pc.read_text(f)

out = {{}}
for name, value in pc.variables.items():
    if not isinstance(value, list):
        value = [value]
    out[name] = [float(x) for x in value]

rust.collect_string(json.dumps(out))
"#,
            path = EXCERPT_PATH,
        );

        let py_result = bridge
            .run_py_to_json(&code)
            .expect("Skyfield text kernel parse failed");

        let parsed = unwrap_py_json(&py_result);
        let object = parsed.as_object().expect("expected a JSON object");
        object
            .iter()
            .map(|(name, value)| {
                let numbers = value
                    .as_array()
                    .expect("expected a JSON array")
                    .iter()
                    .map(|x| x.as_f64().expect("expected a JSON number"))
                    .collect();
                (name.clone(), numbers)
            })
            .collect()
    }

    /// The parser reproduces Skyfield's variables dict name for name.
    #[test]
    fn test_variables_match_skyfield() {
        let expected = skyfield_variables();
        assert!(!expected.is_empty(), "Skyfield read no variables");

        let mut pc = PlanetaryConstants::new();
        pc.open_text(EXCERPT_PATH).unwrap();

        let mut names: Vec<&String> = pc.variables.keys().collect();
        names.sort();
        let mut expected_names: Vec<&String> = expected.keys().collect();
        expected_names.sort();
        assert_eq!(names, expected_names, "variable names differ");

        for (name, want) in &expected {
            let got = pc.variables[name]
                .to_numbers()
                .unwrap_or_else(|| panic!("{} did not parse as numbers", name));
            assert_eq!(&got, want, "{} differs from Skyfield", name);
        }
    }

    /// Skyfield unboxes a lone value to a scalar, and so do we.
    #[test]
    fn test_scalars_are_unboxed_like_skyfield() {
        let mut pc = PlanetaryConstants::new();
        pc.open_text(EXCERPT_PATH).unwrap();

        assert!(matches!(
            pc.variables["BODY4_MAX_PHASE_DEGREE"],
            KernelValue::Number(_)
        ));
        assert!(matches!(
            pc.variables["BODY499_RADII"],
            KernelValue::Numbers(_)
        ));

        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let code = format!(
            r#"
import json
from skyfield.planetarylib import PlanetaryConstants

pc = PlanetaryConstants()
with open('{path}', 'rb') as f:
    pc.read_text(f)

rust.collect_string(json.dumps({{
    "scalar": isinstance(pc.variables['BODY4_MAX_PHASE_DEGREE'], list),
    "vector": isinstance(pc.variables['BODY499_RADII'], list),
}}))
"#,
            path = EXCERPT_PATH,
        );
        let parsed = unwrap_py_json(&bridge.run_py_to_json(&code).expect("Skyfield parse failed"));
        assert_eq!(parsed["scalar"].as_bool(), Some(false));
        assert_eq!(parsed["vector"].as_bool(), Some(true));
    }

    /// `PckFrame` reproduces Skyfield's `MOON_PA_DE421` rotation matrices.
    ///
    /// Both sides read the same two kernels from the data directory, so the
    /// only thing under test is the interpolation and the assembly of the
    /// matrix.
    #[test]
    #[ignore = "downloads moon_pa_de421_1900-2050.bpc and moon_080317.tf"]
    fn test_moon_pa_frame_matches_skyfield() {
        let loader = crate::Loader::new();
        let text_path = loader.ensure_file("moon_080317.tf").unwrap();
        let binary_path = loader.ensure_file("moon_pa_de421_1900-2050.bpc").unwrap();

        let mut pc = PlanetaryConstants::new();
        pc.open_text(&text_path).unwrap();
        pc.open_binary(&binary_path).unwrap();
        let frame = pc.build_frame_named("MOON_PA_DE421").unwrap();

        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let code = format!(
            r#"
import json
from skyfield.api import load
from skyfield.planetarylib import PlanetaryConstants

ts = load.timescale()
pc = PlanetaryConstants()
pc.read_text(open('{text}', 'rb'))
pc.read_binary(open('{binary}', 'rb'))
frame = pc.build_frame_named('MOON_PA_DE421')

out = []
for jd in {epochs:?}:
    out.append([float(x) for x in frame.rotation_at(ts.tdb_jd(jd)).flatten()])

rust.collect_string(json.dumps(out))
"#,
            text = text_path.display(),
            binary = binary_path.display(),
            epochs = EPOCHS,
        );

        let parsed = unwrap_py_json(&bridge.run_py_to_json(&code).expect("Skyfield frame failed"));
        let rows = parsed.as_array().expect("expected a JSON array");
        assert_eq!(rows.len(), EPOCHS.len());

        let ts = crate::time::Timescale::default();
        for (jd, expected) in EPOCHS.iter().zip(rows) {
            let rotation = frame.rotation_at(&ts.tdb_jd(*jd));
            let expected = expected.as_array().expect("expected a JSON array");
            for (i, want) in expected.iter().enumerate() {
                let want = want.as_f64().expect("expected a number");
                let got = rotation[(i / 3, i % 3)];
                assert!(
                    (got - want).abs() < 1e-9,
                    "JD {jd}: element {i} is {got}, Skyfield says {want}"
                );
            }
        }
    }
}

/// Live comparison of [`IauFrame`](crate::planetarylib::IauFrame) against
/// SpiceyPy.
///
/// SPICE is the reference nothing else in the environment can replace: no
/// other package in the venv evaluates the IAU polynomial elements. The tests
/// need `pck00011.tpc` and `naif0012.tls`, so they download them into
/// `~/.cache/starfield/` and are `#[ignore]`d like the other network tests.
/// The matrices they check are also checked in, as the golden constants of
/// `planetarylib::iau_frame`, so CI covers the same comparison offline.
#[cfg(test)]
mod spiceypy_tests {
    use crate::data::get_cache_dir;
    use crate::framelib::Frame;
    use crate::planetarylib::{IauFrame, PlanetaryConstants};
    use crate::planetlib::Body;
    use crate::pybridge::{PyRustBridge, PythonResult};
    use crate::time::Timescale;
    use nalgebra::{Matrix3, Vector3};

    /// The generic kernels the comparison needs, and where NAIF keeps them.
    const KERNELS: [(&str, &str); 2] = [
        (
            "pck00011.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/",
        ),
        (
            "naif0012.tls",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/",
        ),
    ];

    /// The same three TDB Julian dates as the checked-in golden matrices.
    const EPOCHS: [f64; 3] = [2451545.0, 2455362.5, 2460735.5];

    /// One arcsecond in radians, the acceptance of the comparison.
    const ONE_ARCSEC: f64 = 4.84813681109536e-6;

    /// The Python prologue that furnishes the kernels, downloading them first
    /// if the cache has not got them.
    fn furnsh() -> String {
        let cache = get_cache_dir();
        let mut code = String::from(
            "import json, os, urllib.request\nimport spiceypy as sp\n\
             J2000 = 2451545.0\n",
        );
        for (name, url) in KERNELS {
            code.push_str(&format!(
                "path = os.path.join({cache:?}, {name:?})\n\
                 os.makedirs({cache:?}, exist_ok=True)\n\
                 if not os.path.exists(path):\n\
                 \x20   urllib.request.urlretrieve({url:?} + {name:?}, path)\n\
                 sp.furnsh(path)\n",
                cache = cache.to_string_lossy(),
                name = name,
                url = url,
            ));
        }
        code
    }

    /// The angle, in radians, of the rotation that carries `b` onto `a`.
    ///
    /// Taken from the sine and cosine together, so that it stays accurate for
    /// the small angles this comparison is looking for.
    fn angle_between(a: &Matrix3<f64>, b: &Matrix3<f64>) -> f64 {
        let d = a * b.transpose();
        let axis = Vector3::new(
            d[(2, 1)] - d[(1, 2)],
            d[(0, 2)] - d[(2, 0)],
            d[(1, 0)] - d[(0, 1)],
        ) * 0.5;
        axis.norm().atan2(0.5 * (d.trace() - 1.0))
    }

    /// Run `code` and parse its JSON payload.
    fn run(code: &str) -> serde_json::Value {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let raw = bridge.run_py_to_json(code).expect("SpiceyPy call failed");
        let inner = match PythonResult::try_from(raw.as_str()).expect("bad Python result") {
            PythonResult::String(s) => s,
            other => panic!("Expected String result, got {:?}", other),
        };
        serde_json::from_str(&inner).expect("JSON parse failed")
    }

    /// Read a nine-element JSON array as a row-major matrix.
    fn matrix(value: &serde_json::Value) -> Matrix3<f64> {
        let values: Vec<f64> = value
            .as_array()
            .expect("expected a JSON array")
            .iter()
            .map(|x| x.as_f64().expect("expected a JSON number"))
            .collect();
        Matrix3::from_row_slice(&values)
    }

    /// `IauFrame::rotation_at` reproduces `pxform('J2000', 'IAU_<BODY>', et)`.
    ///
    /// SPICE ET is TDB seconds past J2000, so the epoch conversion is exact on
    /// both sides: `et = (jd_tdb − 2451545.0) × 86400`. The test asserts that
    /// against `str2et` before it compares anything.
    #[test]
    #[ignore]
    fn test_pxform_matches_spiceypy() {
        let mut code = furnsh();
        code.push_str(
            r#"
out = {}
out['str2et'] = [sp.str2et('JD %r TDB' % jd) for jd in EPOCHS]
for name, body in [('MARS', 499), ('EARTH', 399), ('MOON', 301), ('JUPITER', 599)]:
    rotations, rates = [], []
    for jd in EPOCHS:
        et = (jd - J2000) * 86400.0
        rotations.append([float(x) for row in sp.pxform('J2000', 'IAU_' + name, et)
                          for x in row])
        state = sp.sxform('J2000', 'IAU_' + name, et)
        rates.append([float(state[i][j]) * 86400.0 for i in range(3, 6) for j in range(3)])
    out[name] = {'rotation': rotations, 'rate': rates}
rust.collect_string(json.dumps(out))
"#,
        );
        let code = code.replace("EPOCHS", &format!("{:?}", EPOCHS));
        let spice = run(&code);

        // ET is TDB seconds past J2000, nothing more.
        for (i, jd) in EPOCHS.iter().enumerate() {
            let expected = (jd - 2451545.0) * 86400.0;
            let actual = spice["str2et"][i].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 1e-6,
                "str2et gave {actual} for JD {jd} TDB, expected {expected}"
            );
        }

        let ts = Timescale::default();
        for (name, body) in [
            ("MARS", Body::Mars),
            ("EARTH", Body::Earth),
            ("MOON", Body::Moon),
            ("JUPITER", Body::Jupiter),
        ] {
            let frame = IauFrame::from_body(body);
            let mut worst = 0.0_f64;
            for (i, jd) in EPOCHS.iter().enumerate() {
                let t = ts.tdb_jd(*jd);
                let (rotation, rate) = frame.rotation_and_rate_at(&t);

                let expected = matrix(&spice[name]["rotation"][i]);
                let angle = angle_between(&rotation, &expected);
                worst = worst.max(angle);
                assert!(
                    angle < ONE_ARCSEC,
                    "{name} at JD {jd}: {} arcsec from pxform",
                    angle / ONE_ARCSEC
                );

                let expected_rate = matrix(&spice[name]["rate"][i]);
                let error = (rate - expected_rate).abs().max() / expected_rate.abs().max();
                assert!(
                    error < 1e-9,
                    "{name} at JD {jd}: rate is {error} (relative) from sxform"
                );
            }
            println!(
                "{name}: worst rotation error {:.3e} rad = {:.3e} arcsec",
                worst,
                worst / ONE_ARCSEC
            );
        }
    }

    /// The quadratic phase angles of the Mars system match SPICE.
    ///
    /// Mars itself has zero amplitude on the one angle that carries a
    /// deg/century² term, so only Phobos and Deimos exercise it. Their
    /// elements are not in the embedded table, which makes this a test of
    /// `PlanetaryConstants` on the full downloaded kernel as well.
    #[test]
    #[ignore]
    fn test_phobos_quadratic_angles_match_spiceypy() {
        let mut code = furnsh();
        code.push_str(
            r#"
out = {}
for name, body in [('PHOBOS', 401), ('DEIMOS', 402)]:
    out[name] = [[float(x) for row in sp.pxform('J2000', 'IAU_' + name,
                                                (jd - J2000) * 86400.0)
                  for x in row] for jd in EPOCHS]
rust.collect_string(json.dumps(out))
"#,
        );
        let code = code.replace("EPOCHS", &format!("{:?}", EPOCHS));
        let spice = run(&code);

        let mut constants = PlanetaryConstants::new();
        constants
            .open_text(get_cache_dir().join("pck00011.tpc"))
            .expect("the kernel was downloaded above");

        let ts = Timescale::default();
        for (name, body) in [("PHOBOS", 401), ("DEIMOS", 402)] {
            let frame = IauFrame::new(body, &constants).unwrap();
            assert!(frame
                .elements
                .nut_prec_angle_accel
                .iter()
                .any(|&a| a != 0.0));

            for (i, jd) in EPOCHS.iter().enumerate() {
                let expected = matrix(&spice[name][i]);
                let angle = angle_between(&frame.rotation_at(&ts.tdb_jd(*jd)), &expected);
                assert!(
                    angle < ONE_ARCSEC,
                    "{name} at JD {jd}: {} arcsec from pxform",
                    angle / ONE_ARCSEC
                );
                println!("{name} at JD {jd}: {:.3e} arcsec", angle / ONE_ARCSEC);
            }
        }
    }
}
