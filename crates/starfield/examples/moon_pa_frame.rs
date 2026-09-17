//! Print the lunar principal-axes rotation from a binary PCK kernel.
//!
//! Downloads `moon_pa_de421_1900-2050.bpc` and `moon_080317.tf` into the data
//! directory (`~/.cache/starfield/` unless one is set) the first time it runs,
//! then prints the `MOON_PA_DE421` rotation matrix at a few epochs, along with
//! the Euler angles it is built from and the rate at which it turns.
//!
//! ```text
//! cargo run --example moon_pa_frame
//! ```
//!
//! `MOON_PA_DE421` is the principal-axes frame that comes with the DE421
//! ephemeris, and is the frame lunar libration is expressed in. `MOON_ME_DE421`
//! — the mean-Earth/mean-rotation frame that lunar maps use — is defined in the
//! same text kernel as a fixed offset from it, and is printed too.

use starfield::framelib::Frame;
use starfield::Loader;

/// The epochs to report, as TDB Julian dates.
const EPOCHS: [(&str, f64); 3] = [
    ("2000-01-01", 2451544.5),
    ("2010-06-15", 2455362.5),
    ("2025-03-01", 2460735.5),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let loader = Loader::new();

    println!("Reading moon_080317.tf ...");
    let mut pc = loader.open_text_pck("moon_080317.tf")?;

    println!("Reading moon_pa_de421_1900-2050.bpc ...");
    let pck = loader.open_binary_pck("moon_pa_de421_1900-2050.bpc")?;
    println!("\n{pck}");
    pc.read_binary(pck);

    let frame = pc.build_frame_named("MOON_PA_DE421")?;
    let mean_earth = pc.build_frame_named("MOON_ME_DE421")?;
    println!(
        "MOON_PA_DE421 is centred on body {} and reads frame {} of the kernel\n",
        frame.center(),
        frame.segment().body
    );

    let ts = loader.timescale();
    for (label, jd) in EPOCHS {
        let t = ts.tdb_jd(jd);
        let (angles, rates) = frame.segment().compute(jd)?;
        let (rotation, rate) = frame.rotation_and_rate_at(&t)?;

        println!("{label}  (TDB JD {jd})");
        println!(
            "  Euler angles     {:+.9} {:+.9} {:+.9} rad",
            angles[0], angles[1], angles[2]
        );
        println!(
            "  rates            {:+.9} {:+.9} {:+.9} rad/day",
            rates[0], rates[1], rates[2]
        );
        println!("  ICRF -> MOON_PA_DE421");
        for row in 0..3 {
            println!(
                "    {:+.12} {:+.12} {:+.12}",
                rotation[(row, 0)],
                rotation[(row, 1)],
                rotation[(row, 2)]
            );
        }
        println!(
            "  d/dt (per day)   norm {:.9}, largest element {:.9}",
            rate.norm(),
            rate.abs().max()
        );

        // The mean-Earth frame differs from the principal axes by less than a
        // milliradian, so report the difference rather than the whole matrix.
        let difference = mean_earth.rotation_at(&t) - rotation;
        println!(
            "  MOON_ME_DE421 differs by at most {:.3e} in any element\n",
            difference.abs().max()
        );
    }

    Ok(())
}
