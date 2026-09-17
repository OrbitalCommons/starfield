//! ITRS (Earth-fixed) frame rotations
//!
//! Demonstrates `framelib::ItrsFrame`, the ICRF → ITRS rotation of the IERS
//! conventions: precession, nutation, the daily sidereal rotation and polar
//! motion. Rotating a ground site's Earth-fixed vector by its transpose gives
//! the geocentric (GCRS) offset that `GeographicPosition::at` adds to Earth.
//!
//! Usage: cargo run --example itrs_frame

use starfield::framelib::{Frame, ItrsFrame};
use starfield::toposlib::WGS84;
use starfield::Timescale;

fn main() {
    let ts = Timescale::default();
    let boston = WGS84.latlon(42.3583, -71.0603, 43.0);

    println!("ITRS frame rotation for a site in Boston");
    println!("========================================\n");
    println!(
        "ITRS position: ({:.9}, {:.9}, {:.9}) AU\n",
        boston.itrs_xyz.x, boston.itrs_xyz.y, boston.itrs_xyz.z
    );

    // Six hours of Earth rotation, sampled every two hours.
    for step in 0..4 {
        let jd = 2451545.0 + step as f64 * 2.0 / 24.0;
        let t = ts.tt_jd(jd, None);

        let icrf_to_itrs = ItrsFrame.rotation_at(&t);
        let gcrs = icrf_to_itrs.transpose() * boston.itrs_xyz;

        println!("TT JD {jd:.5}  (GAST {:.6} h)", t.gast());
        println!(
            "  GCRS offset: ({:>12.9}, {:>12.9}, {:>12.9}) AU",
            gcrs.x, gcrs.y, gcrs.z
        );
    }

    // The rotation is orthonormal, so the transpose is the inverse.
    let t = ts.tt_jd(2451545.0, None);
    let round_trip = ItrsFrame.rotation_at(&t) * (ItrsFrame.rotation_at(&t).transpose());
    println!(
        "\nR·Rᵀ deviation from identity: {:.3e}",
        (round_trip - nalgebra::Matrix3::identity()).abs().max()
    );
}
