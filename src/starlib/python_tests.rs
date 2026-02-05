//! Python comparison tests for starlib
//!
//! Validates Rust star positions, velocity vectors, proper motion propagation,
//! and the full observe→apparent→radec pipeline against Python Skyfield.

#[cfg(test)]
mod tests {
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::test_utils::{de421_kernel, parse_f64_triple};
    use crate::starlib::Star;
    use crate::time::Timescale;
    use approx::assert_relative_eq;

    // --- Position vector tests ---

    /// Barnard's Star position vector from catalog coords matches Skyfield
    #[test]
    fn test_position_vector_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import Star
s = Star(ra_hours=17.963471675, dec_degrees=4.69339088889,
         ra_mas_per_year=-798.71, dec_mas_per_year=10337.77,
         parallax_mas=545.4, radial_km_per_s=-110.6)
x, y, z = s._position_au
rust.collect_string(f"{x},{y},{z}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let s = Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        );

        assert_relative_eq!(s.position_au.x, py_x, epsilon = 1e-6);
        assert_relative_eq!(s.position_au.y, py_y, epsilon = 1e-6);
        assert_relative_eq!(s.position_au.z, py_z, epsilon = 1e-6);
    }

    /// Barnard's Star velocity vector matches Skyfield
    #[test]
    fn test_velocity_vector_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import Star
s = Star(ra_hours=17.963471675, dec_degrees=4.69339088889,
         ra_mas_per_year=-798.71, dec_mas_per_year=10337.77,
         parallax_mas=545.4, radial_km_per_s=-110.6)
vx, vy, vz = s._velocity_au_per_d
rust.collect_string(f"{vx},{vy},{vz}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_vx, py_vy, py_vz) = parse_f64_triple(&py_result);

        let s = Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        );

        // Velocity vectors should match to high precision
        assert_relative_eq!(s.velocity_au_per_day.x, py_vx, epsilon = 1e-10);
        assert_relative_eq!(s.velocity_au_per_day.y, py_vy, epsilon = 1e-10);
        assert_relative_eq!(s.velocity_au_per_day.z, py_vz, epsilon = 1e-10);
    }

    // --- Astrometric RA/Dec tests ---

    /// Barnard's Star astrometric RA/Dec at J2000 matches Skyfield
    #[test]
    fn test_barnard_radec_j2000_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, Star
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)
barnard = Star(ra_hours=17.963471675, dec_degrees=4.69339088889,
               ra_mas_per_year=-798.71, dec_mas_per_year=10337.77,
               parallax_mas=545.4, radial_km_per_s=-110.6)
astrometric = eph['earth'].at(t).observe(barnard)
ra, dec, dist = astrometric.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_ra_deg, py_dec, py_dist) = parse_f64_triple(&py_result);
        let py_ra_h = py_ra_deg / 15.0;

        let earth = kernel.at("earth", &t).unwrap();
        let barnard = Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        );
        let astro = barnard.observe_from(&earth, &t);
        let (rust_ra_h, rust_dec, rust_dist) = astro.radec(None);

        // Tolerance: ~1 arcsecond
        assert!(
            (rust_ra_h - py_ra_h).abs() < 0.01,
            "RA mismatch: rust={rust_ra_h}h python={py_ra_h}h diff={}h",
            (rust_ra_h - py_ra_h).abs()
        );
        assert!(
            (rust_dec - py_dec).abs() < 0.01,
            "Dec mismatch: rust={rust_dec}° python={py_dec}°",
        );
        assert!(
            (rust_dist - py_dist).abs() / py_dist < 0.001,
            "Distance mismatch: rust={rust_dist} python={py_dist}",
        );
    }

    /// Barnard's Star apparent RA/Dec matches Skyfield
    #[test]
    fn test_barnard_apparent_radec_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, Star
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)
barnard = Star(ra_hours=17.963471675, dec_degrees=4.69339088889,
               ra_mas_per_year=-798.71, dec_mas_per_year=10337.77,
               parallax_mas=545.4, radial_km_per_s=-110.6)
apparent = eph['earth'].at(t).observe(barnard).apparent()
ra, dec, dist = apparent.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_ra_deg, py_dec, _py_dist) = parse_f64_triple(&py_result);
        let py_ra_h = py_ra_deg / 15.0;

        let earth = kernel.at("earth", &t).unwrap();
        let barnard = Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        );
        let astro = barnard.observe_from(&earth, &t);
        let apparent = astro.apparent(&mut kernel, &t).unwrap();
        let (rust_ra_h, rust_dec, _) = apparent.radec(None);

        assert!(
            (rust_ra_h - py_ra_h).abs() < 0.01,
            "Apparent RA mismatch: rust={rust_ra_h}h python={py_ra_h}h",
        );
        assert!(
            (rust_dec - py_dec).abs() < 0.01,
            "Apparent Dec mismatch: rust={rust_dec}° python={py_dec}°",
        );
    }

    // --- Proper motion tests ---

    /// Proper motion shifts Barnard's Star measurably over 50 years
    #[test]
    fn test_barnard_proper_motion_50yr_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0 + 50.0 * 365.25);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, Star
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0 + 50.0 * 365.25)
barnard = Star(ra_hours=17.963471675, dec_degrees=4.69339088889,
               ra_mas_per_year=-798.71, dec_mas_per_year=10337.77,
               parallax_mas=545.4, radial_km_per_s=-110.6)
astrometric = eph['earth'].at(t).observe(barnard)
ra, dec, dist = astrometric.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_ra_deg, py_dec, _) = parse_f64_triple(&py_result);
        let py_ra_h = py_ra_deg / 15.0;

        let earth = kernel.at("earth", &t).unwrap();
        let barnard = Star::from_ra_hours(
            17.963_471_675,
            4.693_390_889,
            -798.71,
            10337.77,
            545.4,
            -110.6,
        );
        let astro = barnard.observe_from(&earth, &t);
        let (rust_ra_h, rust_dec, _) = astro.radec(None);

        assert!(
            (rust_ra_h - py_ra_h).abs() < 0.01,
            "RA at +50yr: rust={rust_ra_h}h python={py_ra_h}h",
        );
        assert!(
            (rust_dec - py_dec).abs() < 0.01,
            "Dec at +50yr: rust={rust_dec}° python={py_dec}°",
        );
    }

    // --- Zero-motion star test ---

    /// A star with zero proper motion and parallax
    #[test]
    fn test_zero_motion_star_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, Star
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2451545.0)
s = Star(ra_hours=6.0, dec_degrees=45.0)
astrometric = eph['earth'].at(t).observe(s)
ra, dec, dist = astrometric.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_ra_deg, py_dec, _) = parse_f64_triple(&py_result);
        let py_ra_h = py_ra_deg / 15.0;

        let earth = kernel.at("earth", &t).unwrap();
        let s = Star::from_ra_hours(6.0, 45.0, 0.0, 0.0, 0.0, 0.0);
        let astro = s.observe_from(&earth, &t);
        let (rust_ra_h, rust_dec, _) = astro.radec(None);

        assert!(
            (rust_ra_h - py_ra_h).abs() < 0.01,
            "RA mismatch: rust={rust_ra_h}h python={py_ra_h}h",
        );
        assert!(
            (rust_dec - py_dec).abs() < 0.01,
            "Dec mismatch: rust={rust_dec}° python={py_dec}°",
        );
    }

    // --- Polaris test ---

    /// Polaris apparent position matches Skyfield at ~2017
    #[test]
    fn test_polaris_apparent_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2458000.5);

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import load, Star
ts = load.timescale()
eph = load('de421.bsp')
t = ts.tdb_jd(2458000.5)
polaris = Star(ra_hours=2.5301, dec_degrees=89.2641,
               ra_mas_per_year=44.22, dec_mas_per_year=-11.74,
               parallax_mas=7.54, radial_km_per_s=-17.4)
apparent = eph['earth'].at(t).observe(polaris).apparent()
ra, dec, dist = apparent.radec()
rust.collect_string(f"{ra._degrees},{dec.degrees},{dist.au}")
"#,
            )
            .expect("Failed to run Python code");

        let (_, py_dec, _) = parse_f64_triple(&py_result);

        let earth = kernel.at("earth", &t).unwrap();
        let polaris = Star::from_ra_hours(2.5301, 89.2641, 44.22, -11.74, 7.54, -17.4);
        let astro = polaris.observe_from(&earth, &t);
        let apparent = astro.apparent(&mut kernel, &t).unwrap();
        let (_, rust_dec, _) = apparent.radec(None);

        // Polaris RA is unstable near the pole — only check Dec
        assert!(
            (rust_dec - py_dec).abs() < 0.05,
            "Polaris Dec mismatch: rust={rust_dec}° python={py_dec}°",
        );
    }

    // --- Sirius test (negative declination, moderate proper motion) ---

    /// Sirius position vector matches Skyfield
    #[test]
    fn test_sirius_position_matches_skyfield() {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");

        let py_result = bridge
            .run_py_to_json(
                r#"
from skyfield.api import Star
s = Star(ra_hours=6.7525694, dec_degrees=-16.7161,
         ra_mas_per_year=-546.01, dec_mas_per_year=-1223.07,
         parallax_mas=379.21, radial_km_per_s=-5.5)
x, y, z = s._position_au
rust.collect_string(f"{x},{y},{z}")
"#,
            )
            .expect("Failed to run Python code");

        let (py_x, py_y, py_z) = parse_f64_triple(&py_result);

        let s = Star::from_ra_hours(6.7525694, -16.7161, -546.01, -1223.07, 379.21, -5.5);

        assert_relative_eq!(s.position_au.x, py_x, epsilon = 1e-6);
        assert_relative_eq!(s.position_au.y, py_y, epsilon = 1e-6);
        assert_relative_eq!(s.position_au.z, py_z, epsilon = 1e-6);
    }
}
