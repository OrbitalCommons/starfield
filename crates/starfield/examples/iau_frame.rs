//! Body-fixed frames from the IAU WGCCRE rotational elements.
//!
//! Prints the pole, the prime meridian and the ICRF → body-fixed rotation of
//! Mars, the Moon and Jupiter at one epoch, all from the embedded IAU 2015
//! table, then shows what `PlanetaryConstants::frame_for` picks for the Earth.
//!
//! Usage: cargo run --example iau_frame [TDB Julian date]

use nalgebra::Matrix3;
use starfield::framelib::Frame;
use starfield::planetarylib::IauFrame;
use starfield::planetlib::Body;
use starfield::Timescale;

/// Print a rotation matrix, row by row.
fn print_matrix(matrix: &Matrix3<f64>) {
    for row in 0..3 {
        println!(
            "    [{:>12.8}, {:>12.8}, {:>12.8}]",
            matrix[(row, 0)],
            matrix[(row, 1)],
            matrix[(row, 2)]
        );
    }
}

/// Wrap an angle in degrees into `[0, 360)`.
fn wrap_degrees(degrees: f64) -> f64 {
    degrees.rem_euclid(360.0)
}

fn main() {
    let jd: f64 = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(2455362.5);

    let ts = Timescale::default();
    let t = ts.tdb_jd(jd);

    println!("IAU body-fixed frames at TDB JD {jd}");
    println!("=====================================\n");

    for body in [Body::Mars, Body::Moon, Body::Jupiter] {
        let frame = IauFrame::from_body(body);
        let (ra, dec, w) = frame.pole_and_meridian(&t);
        let (rotation, rate) = frame.rotation_and_rate_at(&t);

        println!("{} (NAIF {})", body.name(), frame.body);
        println!(
            "  pole RA          {:>12.6} deg",
            wrap_degrees(ra.to_degrees())
        );
        println!("  pole Dec         {:>12.6} deg", dec.to_degrees());
        println!(
            "  prime meridian W {:>12.6} deg",
            wrap_degrees(w.to_degrees())
        );
        println!(
            "  rotation period  {:>12.6} days",
            360.0 / body.rotational_elements().pm[1]
        );
        println!("  ICRF -> body-fixed rotation:");
        print_matrix(&rotation);
        println!("  and its rate, per day:");
        print_matrix(&rate);
        println!(
            "  det R = {:.15}, max |R.Rt - I| = {:.3e}",
            rotation.determinant(),
            (rotation * rotation.transpose() - Matrix3::identity())
                .abs()
                .max()
        );
        println!();
    }

    // The x axis of the frame is the prime meridian, so this is where east
    // longitude zero on the equator points in the ICRF. Most bodies, Mars
    // among them, report planetographic longitude west-positive instead.
    let mars = IauFrame::from_body(Body::Mars);
    let prime_meridian = mars.rotation_at(&t).row(0).transpose();
    println!("Mars prime meridian direction in ICRF");
    println!("=====================================");
    println!(
        "  ({:>12.8}, {:>12.8}, {:>12.8})",
        prime_meridian[0], prime_meridian[1], prime_meridian[2]
    );
    println!("  east longitude 30 deg is planetographic longitude 330 deg W\n");

    // The Earth is the one body served by a better frame.
    println!("The Earth");
    println!("=========");
    let iau_earth = IauFrame::from_body(Body::Earth).rotation_at(&t);
    let itrs = starfield::framelib::ItrsFrame.rotation_at(&t);
    let difference = iau_earth * itrs.transpose();
    let angle = (0.5 * (difference.trace() - 1.0)).clamp(-1.0, 1.0).acos();
    println!(
        "  IauFrame(399) is {:.4} deg from ItrsFrame, which is why",
        angle.to_degrees()
    );
    println!("  PlanetaryConstants::frame_for(399) returns ItrsFrame.");
}
