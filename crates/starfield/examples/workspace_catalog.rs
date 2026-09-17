//! Use a datasource with core traits through one dependency, without network I/O.
//!
//! Run with `cargo run --example workspace_catalog --features hipparcos,horizons`.

use starfield::catalogs::{hipparcos::catalog::HipparcosCatalog, StarCatalog};
use starfield::jpl::horizons::{Center, Command, EphemerisRequest, TimeSpec};

fn main() {
    let catalog = HipparcosCatalog::create_synthetic();
    println!("Synthetic Hipparcos catalog: {} stars", catalog.len());

    let mut request = EphemerisRequest::observer(
        Command::MajorBody(499),
        Center::BodyCenter(399),
        TimeSpec::JulianDayList(vec![2451545.0]),
    );
    request.cal_format = Some("JD".into());
    println!("Mars request (no query sent):\n{}", request.to_input_file());
}
