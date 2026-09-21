//! Observe Io from Mars through a combined set of local SPK files.
//!
//! First obtain the kernels through `Loader::open_many` or the datastore.
//! Run: `cargo run --example observe_satellite -- de440s.bsp mar099.bsp jup365.bsp`
//! The paths are local; this example does not download the large satellite files.

use starfield::{jplephem::SpiceKernel, positions::Position, time::Timescale};

fn main() -> starfield::Result<()> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    let mut kernel = SpiceKernel::open_many(&paths)?;
    let time = Timescale::default().tdb_jd(2451545.0);
    let mars = Position::from_spk_target(&mut kernel, 499, &time)?;
    let io = mars
        .observe("501", &mut kernel, &time)?
        .apparent(&mut kernel, &time)?;
    println!(
        "Apparent Mars–Io distance at J2000: {:.9} AU",
        io.position.norm()
    );
    Ok(())
}
