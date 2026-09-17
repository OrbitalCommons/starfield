//! Illumination geometry of the planets, and of the Earth seen from Mars.
//!
//! Run with:
//!
//! ```text
//! cargo run --example illumination
//! ```

use starfield::jplephem::kernel::SpiceKernel;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::time::Timescale;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = SpiceKernel::open("test_data/de421.bsp")?;
    let ts = Timescale::default();

    // The planets as an Earth-bound observer sees them.
    let t = ts.utc((2007, 10, 3, 0, 0, 0.0));
    println!("Illumination geometry from the Earth on 2007-10-03");
    println!(
        "{:<10} {:>10} {:>10} {:>12}",
        "body", "phase(°)", "lit", "elong(°)"
    );

    let earth = kernel.at("earth", &t)?;
    for name in [
        "mercury",
        "venus",
        "mars",
        "jupiter barycenter",
        "saturn barycenter",
        "moon",
    ] {
        let body = earth.observe(name, &mut kernel, &t)?;
        println!(
            "{:<10} {:>10.2} {:>10.3} {:>12.2}",
            name.split(' ').next().unwrap(),
            body.phase_angle(&mut kernel, &t)?.to_degrees(),
            body.illuminated_fraction(&mut kernel, &t)?,
            body.solar_elongation(&mut kernel, &t)?.to_degrees(),
        );
    }

    // The view the other way round: the Earth as a gibbous disc over Mars,
    // at the epoch of the HiRISE image PSP_005558_9040.
    println!();
    println!("The Earth seen from Mars at the HiRISE epoch");
    let mars = kernel.at("mars", &t)?;
    let seen_from_mars = mars.observe("earth", &mut kernel, &t)?;
    println!(
        "  phase angle          {:.2}°",
        seen_from_mars.phase_angle(&mut kernel, &t)?.to_degrees()
    );
    println!(
        "  illuminated fraction {:.3}",
        seen_from_mars.illuminated_fraction(&mut kernel, &t)?
    );
    println!(
        "  solar elongation     {:.2}°",
        seen_from_mars
            .solar_elongation(&mut kernel, &t)?
            .to_degrees()
    );

    // Mars near its 2020 opposition: a nearly full disc opposite the Sun.
    println!();
    println!("Mars around its opposition of 2020-10-13");
    for hour in [0, 12, 23] {
        let t = ts.utc((2020, 10, 13, hour, 0, 0.0));
        let earth = kernel.at("earth", &t)?;
        let mars = earth.observe("mars", &mut kernel, &t)?;
        println!(
            "  {:02}h  phase {:.3}°  lit {:.5}  elongation {:.3}°",
            hour,
            mars.phase_angle(&mut kernel, &t)?.to_degrees(),
            mars.illuminated_fraction(&mut kernel, &t)?,
            mars.solar_elongation(&mut kernel, &t)?.to_degrees(),
        );
    }

    Ok(())
}
