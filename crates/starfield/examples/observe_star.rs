//! Apparent place of a catalog star from two different observers
//!
//! `Position::observe_star` gives the direction to a catalog star from any
//! barycentric observer, so star and planet directions come out in the same
//! apparent frame. Sirius is shown from Earth and from Mars on the same date:
//! the two differ mostly by the difference of the observers' aberration, a few
//! arcseconds, which is what a star tracker at Mars has to allow for.
//!
//! Usage: cargo run --example observe_star

use starfield::catalogs::StarData;
use starfield::jplephem::kernel::SpiceKernel;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::{ProperMotion, Timescale};

fn main() {
    let mut kernel = match SpiceKernel::open("test_data/de421.bsp") {
        Ok(k) => k,
        Err(e) => {
            eprintln!("Could not open test_data/de421.bsp: {e}");
            return;
        }
    };

    let ts = Timescale::default();
    let t = ts.utc((2024, 1, 1, 0, 0, 0.0));

    // Sirius (HIP 32349), Hipparcos astrometry at epoch J2000.0.
    let sirius = StarData::new(32349, 101.2874, -16.7161, -1.46, None);
    let pm = ProperMotion::new(-546.01, -1223.07);
    let parallax_mas = 379.21;

    println!("Apparent place of Sirius on 2024-01-01 00:00 UTC");
    println!("================================================\n");
    println!(
        "Catalog (J2000): RA {:>12.6}°  Dec {:>12.6}°",
        sirius.ra_deg(),
        sirius.dec_deg()
    );

    let mut places = Vec::new();
    for body in ["earth", "mars"] {
        let observer = kernel.at(body, &t).expect("body missing from DE421");
        let apparent = observer
            .observe_star(&sirius, Some(&pm), Some(parallax_mas), &t)
            .apparent(&mut kernel, &t)
            .expect("apparent place");

        let (ra_hours, dec_deg, distance_au) = apparent.radec(None);
        let speed_km_s = observer.velocity.norm() * 149_597_870.7 / 86_400.0;

        println!(
            "\nFrom {body:<6} (barycentric speed {speed_km_s:.2} km/s)\n  \
             RA {:>12.6}°  Dec {:>12.6}°  distance {:.0} AU",
            ra_hours * 15.0,
            dec_deg,
            distance_au
        );
        places.push(apparent);
    }

    let separation_arcsec = places[0].separation_from(&places[1]).to_degrees() * 3600.0;
    println!("\nEarth vs Mars apparent direction: {separation_arcsec:.2}″");
    println!("(aberration difference plus the small parallax of a 2.6 pc star)");
}
