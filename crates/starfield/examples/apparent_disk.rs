//! The Earth and the Moon as resolved discs, seen from Mars.
//!
//! Prints the angular diameter and the projected ellipse of both bodies at the
//! epoch of the HiRISE Earth-and-Moon portrait, 2007 October 3, then flies an
//! observer once around Mars on a 17 000 km circular orbit and finds the
//! interval in which Mars hides the Earth from it.
//!
//! Run with `cargo run --example apparent_disk`.

use nalgebra::Vector3;

use starfield::constants::{AU_KM, DAY_S};
use starfield::framelib::Frame;
use starfield::jplephem::SpiceKernel;
use starfield::jplephem_ext::SpiceKernelExt;
use starfield::planetarylib::occult::{occultation, Occultation};
use starfield::planetarylib::{IauFrame, PlanetaryConstants};
use starfield::planetlib::Body;
use starfield::positions::Position;
use starfield::searchlib::{find_discrete, DEFAULT_NUM, EPSILON_DISCRETE};
use starfield::time::{Time, Timescale};

/// Radians to arcseconds.
const RAD2ASEC: f64 = 206_264.806_247_096_36;

/// Gravitational parameter of Mars, km³/s², from DE440.
const GM_MARS: f64 = 42_828.375_214;

/// Radius of the circular orbit the observer flies, in km.
const ORBIT_RADIUS_KM: f64 = 17_000.0;

fn main() {
    let mut kernel = SpiceKernel::open("test_data/de421.bsp").expect("test_data/de421.bsp");
    let ts = Timescale::default();
    let t = ts.utc((2007, 10, 3, 8, 30, 0.0));

    println!("Seen from Mars on 2007 October 3 at 08:30 UTC\n");

    let mars = kernel.at("mars", &t).unwrap();
    let earth = mars.observe("earth", &mut kernel, &t).unwrap();
    let moon = mars.observe("moon", &mut kernel, &t).unwrap();

    // The Earth is best served by the ITRS; the Moon by its IAU elements.
    let earth_frame = PlanetaryConstants::new().frame_for(399).unwrap();
    let moon_frame = IauFrame::from_body(Body::Moon);
    report("Earth", &earth, Body::Earth, earth_frame.as_ref(), &t);
    report("Moon", &moon, Body::Moon, &moon_frame, &t);

    let state = occultation(
        &moon,
        &earth,
        Body::Moon.radii_km()[0],
        Body::Earth.radii_km()[0],
    );
    println!("\nThe Moon hidden by the Earth, seen from Mars: {}", state);

    occultation_from_mars_orbit(&mut kernel, &ts, &t);
}

/// Print the apparent size and shape of one body.
fn report(name: &str, position: &Position, body: Body, frame: &dyn Frame, t: &Time) {
    let radii = body.radii_km();
    let diameter = 2.0 * position.angular_semi_diameter(radii) * RAD2ASEC;
    let (major, minor, pa) = position.apparent_ellipse(frame, radii, t);

    println!("{name}");
    println!("  distance          {:.6} AU", position.distance());
    println!("  angular diameter  {:.3} arcseconds", diameter);
    println!(
        "  apparent ellipse  {:.3} × {:.3} arcseconds, axis ratio {:.5}",
        2.0 * major * RAD2ASEC,
        2.0 * minor * RAD2ASEC,
        minor / major
    );
    println!("  major axis at     {:.2}° east of north", pa.to_degrees());
}

/// Fly one orbit about Mars and find when Mars hides the Earth.
fn occultation_from_mars_orbit(kernel: &mut SpiceKernel, ts: &Timescale, t: &Time) {
    let mars = kernel.at("mars", t).unwrap();
    let earth = mars.observe("earth", kernel, t).unwrap();

    // An orbit plane containing the direction to the Earth, so the observer
    // passes behind Mars once per revolution.
    let toward_earth = earth.position.normalize();
    let across = toward_earth.cross(&Vector3::z()).normalize();
    let period_days = std::f64::consts::TAU * (ORBIT_RADIUS_KM.powi(3) / GM_MARS).sqrt() / DAY_S;
    let epoch_tt = t.tt();

    let state = |kernel: &mut SpiceKernel, t: &Time| -> Occultation {
        let phase = std::f64::consts::TAU * (t.tt() - epoch_tt) / period_days;
        let radius_au = ORBIT_RADIUS_KM / AU_KM;
        let speed = std::f64::consts::TAU * radius_au / period_days;
        let (sin, cos) = phase.sin_cos();

        let mars = kernel.at("mars", t).unwrap();
        let observer = Position::barycentric(
            mars.position + radius_au * (cos * toward_earth + sin * across),
            mars.velocity + speed * (-sin * toward_earth + cos * across),
            -1,
        );
        let earth = observer.observe("earth", kernel, t).unwrap();
        let mars = observer.observe("mars", kernel, t).unwrap();
        occultation(
            &earth,
            &mars,
            Body::Earth.radii_km()[0],
            Body::Mars.radii_km()[0],
        )
    };

    let mut sample = |jd: &[f64]| -> Vec<i64> {
        jd.iter()
            .map(|&jd| {
                let t = ts.tt_jd(jd, None);
                match state(kernel, &t) {
                    Occultation::None => 0,
                    Occultation::Partial => 1,
                    Occultation::Full => 2,
                }
            })
            .collect()
    };

    let events = find_discrete(
        epoch_tt,
        epoch_tt + period_days,
        &mut sample,
        0.005,
        EPSILON_DISCRETE,
        DEFAULT_NUM,
    );

    println!(
        "\nFrom a {:.0} km circular Mars orbit of {:.0} s, over one revolution:",
        ORBIT_RADIUS_KM,
        period_days * DAY_S
    );
    for (jd, value) in &events {
        let label = match value {
            0 => "clear of the limb",
            1 => "partly hidden",
            _ => "wholly hidden",
        };
        println!(
            "  {}  {}",
            ts.tt_jd(*jd, None).utc_iso('T', 0).unwrap(),
            label
        );
    }
    if events.len() == 4 {
        println!(
            "  total occultation lasted {:.1} s, partial phases {:.2} s and {:.2} s",
            (events[2].0 - events[1].0) * DAY_S,
            (events[1].0 - events[0].0) * DAY_S,
            (events[3].0 - events[2].0) * DAY_S,
        );
    }
}
