//! An observer on a Mars orbit
//!
//! Builds a circular 17 000 km orbit about Mars, turns it into a barycentric
//! observer with `KeplerOrbit::barycentric_at`, and observes Earth from it —
//! the geometry a Mars-orbiting camera needs. The apparent direction differs
//! from a Mars-centre observer by the orbit radius over the range, ~47″ at the
//! 0.5 AU close approach of 2003.
//!
//! `Position::from_spk_target` does the same job for a spacecraft that has a
//! real SPK kernel, addressed by its (negative) NAIF id.
//!
//! Usage: cargo run --example mars_orbiter

use nalgebra::Vector3;
use starfield::constants::{AU_KM, DAY_S, GM_KM3_S2_TO_AU3_D2, GM_MARS};
use starfield::jplephem::kernel::SpiceKernel;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::keplerlib::KeplerOrbit;
use starfield::magnitudelib::planetary_magnitude;
use starfield::positions::Position;
use starfield::Timescale;

const ORBIT_RADIUS_KM: f64 = 17_000.0;

fn main() {
    let mut kernel = match SpiceKernel::open("test_data/de421.bsp") {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Could not open test_data/de421.bsp: {e}");
            return;
        }
    };

    let ts = Timescale::default();
    // 2003-10-02, shortly after the closest Mars opposition in 60 000 years.
    let t = ts.tdb_jd(2452923.0);

    let mars = kernel.at("mars", &t).expect("Mars missing from DE421");
    let earth = kernel.at("earth", &t).expect("Earth missing from DE421");

    // Place the orbiter a quarter turn from the Mars-Earth line, so its whole
    // offset shows as a shift in the apparent direction of Earth.
    let to_earth = (earth.position - mars.position).normalize();
    let radial = to_earth.cross(&Vector3::z()).normalize();
    let along_track = to_earth.cross(&radial).normalize();

    let radius_au = ORBIT_RADIUS_KM / AU_KM;
    let mu = GM_MARS * GM_KM3_S2_TO_AU3_D2;
    let speed = (mu / radius_au).sqrt();

    let orbit = KeplerOrbit::new(
        radial * radius_au,
        along_track * speed,
        &t,
        mu,
        Some(499),
        Some("Mars orbiter"),
    );

    let period_hours = 2.0 * std::f64::consts::PI * (radius_au.powi(3) / mu).sqrt() * 24.0;
    println!("Circular Mars orbit");
    println!("===================\n");
    println!(
        "radius {ORBIT_RADIUS_KM:.0} km, speed {:.3} km/s, period {period_hours:.2} h\n",
        speed * AU_KM / DAY_S
    );

    let orbiter = orbit
        .barycentric_at(&mut kernel, "mars", &t)
        .expect("orbiter state");

    for (label, observer) in [("Mars centre", &mars), ("orbiter", &orbiter)] {
        let astrometric = observer
            .observe("earth", &mut kernel, &t)
            .expect("observe Earth");
        let apparent = astrometric.apparent(&mut kernel, &t).expect("apparent");
        let (ra_hours, dec_deg, distance) = apparent.radec(None);
        let magnitude = planetary_magnitude(&astrometric, &t).expect("magnitude");

        println!(
            "Earth from {label:<12} RA {:>11.6}°  Dec {:>11.6}°  range {distance:.5} AU  V {magnitude:.2}",
            ra_hours * 15.0,
            dec_deg
        );
    }

    let from_mars = mars.observe("earth", &mut kernel, &t).unwrap();
    let from_orbiter = orbiter.observe("earth", &mut kernel, &t).unwrap();
    let offset_arcsec = from_orbiter.separation_from(&from_mars).to_degrees() * 3600.0;
    let expected = radius_au / from_mars.distance() * (180.0 / std::f64::consts::PI) * 3600.0;
    println!("\nOffset from the Mars-centre direction: {offset_arcsec:.2}″ (radius/range = {expected:.2}″)");

    // A flown mission would come from its own kernel instead; DE421 has no
    // spacecraft, so this reports that nothing reaches NAIF id -74 (MRO).
    match Position::from_spk_target(&mut kernel, -74, &t) {
        Ok(p) => println!("\nMRO from the kernel: {p}"),
        Err(e) => println!("\nPosition::from_spk_target(-74): {e}"),
    }
}
