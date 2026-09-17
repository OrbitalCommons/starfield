//! Disc orientation: which hemisphere of a body faces the observer, which
//! faces the Sun, and how the axis and the bright limb lie on the sky.
//!
//! Run with:
//!
//! ```text
//! cargo run --example sub_points
//! ```

use starfield::framelib::{Frame, ItrsFrame};
use starfield::jplephem::kernel::SpiceKernel;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::planetarylib::subpoint::{LongitudeSense, SubPoint};
use starfield::planetarylib::{body_constants, IauFrame};
use starfield::time::Timescale;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut kernel = SpiceKernel::open("test_data/de421.bsp")?;
    let ts = Timescale::default();

    // The epoch of the HiRISE image PSP_005558_9040, in which Mars
    // Reconnaissance Orbiter photographed a gibbous Earth.
    const EPOCH: &str = "2007-10-03 00:00 UTC";
    let t = ts.utc((2007, 10, 3, 0, 0, 0.0));

    // Mars as an Earth-bound observer sees it, then the Earth seen from Mars.
    for (target, center) in [(499, 399), (399, 499)] {
        let constants = body_constants(target).unwrap();
        let radii = constants.radii;

        // The Earth is the one body with a frame better than its IAU
        // elements; everything else uses the WGCCRE polynomials.
        let frame: Box<dyn Frame> = if target == 399 {
            Box::new(ItrsFrame)
        } else {
            Box::new(IauFrame::from_naif_id(target).unwrap())
        };

        let observer = kernel.at(&center.to_string(), &t)?;
        let position = observer.observe(&target.to_string(), &mut kernel, &t)?;

        let sub_observer = position.sub_observer_point(frame.as_ref(), radii, &t);
        let sub_solar = position.sub_solar_point(frame.as_ref(), radii, &mut kernel, &t)?;

        println!(
            "{} seen from {} at {EPOCH}",
            constants.name,
            body_constants(center).unwrap().name,
        );
        report("sub-observer", &sub_observer, target, radii);
        report("sub-solar", &sub_solar, target, radii);

        // Where to put the axis and the crescent when the disc is drawn.
        println!(
            "  north pole    {:9.4}° east of celestial north",
            position
                .north_pole_position_angle(frame.as_ref(), &t)
                .to_degrees()
        );
        println!(
            "  bright limb   {:9.4}° east of celestial north, {:.1}% lit",
            position
                .bright_limb_position_angle(&mut kernel, &t)?
                .to_degrees(),
            100.0 * position.illuminated_fraction(&mut kernel, &t)?,
        );
        println!();
    }

    Ok(())
}

/// Print one sub-point in every form: the reported planetographic longitude,
/// the east longitude of the underlying frame, and both latitudes.
fn report(label: &str, point: &SubPoint, target: i32, radii: [f64; 3]) {
    let sense = LongitudeSense::for_body(target);
    let direction = match sense {
        LongitudeSense::East => "E",
        LongitudeSense::West => "W",
    };
    println!(
        "  {label:<13} {:9.4}° {direction}  {:8.4}° planetographic   \
         ({:9.4}° E, {:8.4}° planetocentric)",
        point.longitude_in(sense).to_degrees(),
        point.lat_rad.to_degrees(),
        point.lon_rad.to_degrees(),
        point.to_planetocentric(radii).lat_rad.to_degrees(),
    );
}
