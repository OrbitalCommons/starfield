//! Print the shape and orientation constants of Mars.
//!
//! Run it with no arguments to read the embedded IAU WGCCRE 2015 table:
//!
//! ```text
//! cargo run --example planetary_constants
//! ```
//!
//! Give it the path of a NAIF text kernel to read that instead, and to see the
//! two compared:
//!
//! ```text
//! cargo run --example planetary_constants -- ~/.cache/starfield/pck00011.tpc
//! ```

use starfield::planetarylib::{body_constants, PlanetaryConstants, RotationalElements};
use starfield::planetlib::Body;

/// Print one body's radii, mean radius and flattening.
fn print_shape(label: &str, radii: [f64; 3]) {
    let mean = (radii[0] + radii[1] + radii[2]) / 3.0;
    let flattening = (radii[0] - radii[2]) / radii[0];
    println!("{label}");
    println!(
        "  radii            a = {:.4} km  b = {:.4} km  c = {:.4} km",
        radii[0], radii[1], radii[2]
    );
    println!("  mean radius      {mean:.4} km");
    println!("  flattening       {flattening:.7}");
}

/// Print one body's pole, prime meridian and periodic terms.
fn print_elements(elements: &RotationalElements) {
    println!(
        "  pole RA          {:.6} deg  {:+.8} deg/cy  {:+.3e} deg/cy^2",
        elements.pole_ra[0], elements.pole_ra[1], elements.pole_ra[2]
    );
    println!(
        "  pole Dec         {:.6} deg  {:+.8} deg/cy  {:+.3e} deg/cy^2",
        elements.pole_dec[0], elements.pole_dec[1], elements.pole_dec[2]
    );
    println!(
        "  prime meridian   {:.6} deg  {:+.12} deg/day  {:+.3e} deg/day^2",
        elements.pm[0], elements.pm[1], elements.pm[2]
    );
    println!(
        "  periodic terms   {} in RA, {} in Dec, {} in W",
        elements.nut_prec_ra.len(),
        elements.nut_prec_dec.len(),
        elements.nut_prec_pm.len()
    );
    println!(
        "  barycentre angles {} available",
        elements.nut_prec_angles.len()
    );
    for (i, (phase, rate)) in elements.nut_prec_angles.iter().enumerate().take(5) {
        println!(
            "    M{:<2} {:>14.8} deg {:>18.8} deg/cy {:>14.9} deg/cy^2",
            i + 1,
            phase,
            rate,
            elements.nut_prec_angle_accel[i]
        );
    }
    if elements.nut_prec_angles.len() > 5 {
        println!("    ...");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mars = body_constants(Body::Mars.naif_id()).expect("Mars is in the embedded table");

    println!("Embedded IAU WGCCRE 2015 table");
    println!("==============================");
    print_shape(
        &format!("{} (NAIF {})", mars.name, mars.naif_id),
        mars.radii,
    );
    print_elements(&mars.elements);

    println!();
    println!("Through planetlib::Body");
    println!("=======================");
    println!("  Body::Mars.radii_km()        {:?}", Body::Mars.radii_km());
    println!(
        "  Body::Mars.mean_radius_km()  {:.4} km",
        Body::Mars.mean_radius_km()
    );
    println!(
        "  Body::Mars.flattening()      {:.7}",
        Body::Mars.flattening()
    );

    let Some(path) = std::env::args().nth(1) else {
        println!();
        println!("Pass the path of a text PCK kernel to compare against a kernel, for example");
        println!("  cargo run --example planetary_constants -- pck00011.tpc");
        return Ok(());
    };

    println!();
    println!("From {path}");
    println!("==============================");

    let mut constants = PlanetaryConstants::new();
    constants.open_text(&path)?;
    println!("  {} variables read", constants.variables.len());

    let radii = constants
        .radii(Body::Mars.naif_id())
        .ok_or("the kernel carries no BODY499_RADII")?;
    print_shape("Mars from the kernel", radii);

    let elements = constants
        .rotational_elements(Body::Mars.naif_id())
        .ok_or("the kernel carries no orientation for Mars")?;
    print_elements(&elements);

    println!();
    if radii == mars.radii && elements == mars.elements {
        println!("The kernel agrees with the embedded table.");
    } else {
        println!("The kernel differs from the embedded table.");
    }

    Ok(())
}
