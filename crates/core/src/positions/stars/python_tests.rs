//! Python comparison tests for `Position::observe_star`
//!
//! Checks the apparent places of twenty Hipparcos stars against Skyfield's
//! `observer.at(t).observe(star).apparent().radec()`, for an Earth observer
//! and for an observer riding Mars — the case the Rust method exists for,
//! where the observer's velocity contributes ~17″ of aberration.

#[cfg(test)]
mod tests {
    use crate::catalogs::StarData;
    use crate::framelib::inertial::ProperMotion;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::pybridge::bridge::PyRustBridge;
    use crate::pybridge::test_utils::{de421_kernel, parse_f64_list};
    use crate::time::Timescale;

    /// Twenty Hipparcos stars: (HIP, RA°, Dec°, pmRA mas/yr, pmDec mas/yr,
    /// parallax mas), ICRS at epoch J2000.0.
    ///
    /// The astrometry is the published Hipparcos solution rounded to the
    /// precision printed in the usual bright-star tables; six of the rows are
    /// the entries already embedded in `catalogs::hipparcos`. The comparison
    /// below feeds identical numbers to Rust and to Skyfield, so what is under
    /// test is the propagation and the light-time/aberration/deflection
    /// pipeline, not the catalog values themselves.
    const STARS: [(u64, f64, f64, f64, f64, f64); 20] = [
        (32349, 101.2874, -16.7161, -546.01, -1223.07, 379.21), // Sirius
        (91262, 279.2347, 38.7837, 200.94, 286.23, 130.23),     // Vega
        (27989, 88.7929, 7.4070, 26.40, 9.56, 5.95),            // Betelgeuse
        (26727, 85.1897, -1.9426, 3.19, 2.03, 3.99),            // Alnitak
        (26311, 84.0534, -1.2019, 1.49, -1.06, 2.43),           // Alnilam
        (25930, 83.0016, -0.2991, 0.92, -1.20, 3.56),           // Mintaka
        (24608, 79.1723, 45.9980, 75.52, -427.11, 77.29),       // Capella
        (24436, 78.6345, -8.2017, 1.87, -0.56, 4.22),           // Rigel
        (37279, 114.8255, 5.2250, -716.57, -1034.58, 285.93),   // Procyon
        (69673, 213.9153, 19.1824, -1093.45, -1999.40, 88.85),  // Arcturus
        (71683, 219.9021, -60.8340, -3678.19, 481.84, 742.12),  // Alpha Cen A
        (87937, 269.4521, 4.6934, -798.71, 10337.77, 549.01),   // Barnard's Star
        (11767, 37.9529, 89.2641, 44.22, -11.74, 7.54),         // Polaris
        (65474, 201.2983, -11.1613, -42.50, -31.73, 12.44),     // Spica
        (80763, 247.3519, -26.4320, -10.16, -23.21, 5.40),      // Antares
        (97649, 297.6958, 8.8683, 536.82, 385.54, 194.44),      // Altair
        (21421, 68.9802, 16.5093, 63.45, -188.94, 50.09),       // Aldebaran
        (30438, 95.9880, -52.6957, 19.93, 23.24, 10.43),        // Canopus
        (49669, 152.0930, 11.9672, -249.40, 4.91, 42.09),       // Regulus
        (102098, 310.3580, 45.2803, 1.56, 1.55, 1.01),          // Deneb
    ];

    /// Render `STARS` as a Python list of tuples.
    fn python_star_table() -> String {
        STARS
            .iter()
            .map(|(_, ra, dec, pmra, pmdec, plx)| {
                format!("    ({ra:?}, {dec:?}, {pmra:?}, {pmdec:?}, {plx:?}),")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Skyfield apparent RA/Dec in degrees for every star in `STARS`, in
    /// order, at each of the given TDB Julian dates, as seen from `body`.
    fn skyfield_apparent(body: &str, jds: &[f64]) -> Vec<f64> {
        let bridge = PyRustBridge::new().expect("Failed to create Python bridge");
        let code = format!(
            r#"
from skyfield.api import load, load_file, Star

ts = load.timescale()
eph = load_file('test_data/de421.bsp')
rows = [
{table}
]
out = []
for jd in [{jds}]:
    t = ts.tdb_jd(jd)
    observer = eph['{body}'].at(t)
    for ra, dec, pmra, pmdec, plx in rows:
        star = Star(ra_hours=ra / 15.0, dec_degrees=dec,
                    ra_mas_per_year=pmra, dec_mas_per_year=pmdec,
                    parallax_mas=plx)
        apparent = observer.observe(star).apparent()
        ra_a, dec_a, _ = apparent.radec()
        out.append(ra_a._degrees)
        out.append(dec_a.degrees)
# float() first: NumPy 2 reprs a scalar as np.float64(...), which the
# Rust side cannot parse.
rust.collect_string(','.join(repr(float(v)) for v in out))
"#,
            table = python_star_table(),
            jds = jds
                .iter()
                .map(|jd| format!("{jd:?}"))
                .collect::<Vec<_>>()
                .join(", "),
            body = body,
        );
        let result = bridge.run_py_to_json(&code).expect("Skyfield run failed");
        parse_f64_list(&result)
    }

    /// Angular separation of two RA/Dec pairs (degrees in, mas out).
    fn separation_mas(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
        let d_dec = dec2 - dec1;
        let mut d_ra = ra2 - ra1;
        if d_ra > 180.0 {
            d_ra -= 360.0;
        } else if d_ra < -180.0 {
            d_ra += 360.0;
        }
        let d_ra = d_ra * dec1.to_radians().cos();
        (d_ra * d_ra + d_dec * d_dec).sqrt() * 3_600_000.0
    }

    /// Compare `Position::observe_star(...).apparent()` with Skyfield for the
    /// twenty stars, from `body`, at every epoch in `jds`. Returns the worst
    /// separation in milliarcseconds.
    fn max_error_mas(body: &str, jds: &[f64]) -> f64 {
        let python = skyfield_apparent(body, jds);
        assert_eq!(python.len(), 2 * STARS.len() * jds.len());

        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let mut worst = 0.0f64;
        let mut index = 0;

        for &jd in jds {
            let t = ts.tdb_jd(jd);
            let observer = kernel.at(body, &t).unwrap();
            for &(hip, ra, dec, pmra, pmdec, plx) in STARS.iter() {
                let star = StarData::new(hip, ra, dec, 0.0, None);
                let pm = ProperMotion::new(pmra, pmdec);
                let apparent = observer
                    .observe_star(&star, Some(&pm), Some(plx), &t)
                    .apparent(&mut kernel, &t)
                    .unwrap();
                let (ra_h, dec_deg, _) = apparent.radec(None);

                let error = separation_mas(ra_h * 15.0, dec_deg, python[index], python[index + 1]);
                assert!(
                    error < 1.0,
                    "HIP {hip} at JD {jd} from {body}: {error:.4} mas \
                     (rust {ra_deg:.9} {dec_deg:.9}, skyfield {py_ra:.9} {py_dec:.9})",
                    ra_deg = ra_h * 15.0,
                    py_ra = python[index],
                    py_dec = python[index + 1],
                );
                worst = worst.max(error);
                index += 2;
            }
        }
        worst
    }

    /// Twenty Hipparcos stars from Earth at two epochs, under 1 mas.
    #[test]
    fn test_observe_star_matches_skyfield_from_earth() {
        // J2000.0 and 2025-01-01, both inside the DE421 span.
        let worst = max_error_mas("earth", &[2451545.0, 2460676.5]);
        println!("observe_star from Earth: max error {worst:.6} mas");
        assert!(worst < 1.0, "max error {worst} mas");
    }

    /// The same stars from Mars, where the observer's velocity supplies a
    /// different ~17″ aberration than Earth's.
    #[test]
    fn test_observe_star_matches_skyfield_from_mars() {
        let worst = max_error_mas("mars", &[2451545.0]);
        println!("observe_star from Mars: max error {worst:.6} mas");
        assert!(worst < 1.0, "max error {worst} mas");
    }
}
