//! Exercise the public facade across crate boundaries.

use starfield::catalogs::StarCatalog;

fn accepts_catalog<C: StarCatalog>(_: &C) {}

#[test]
fn existing_catalog_paths_remain_available() {
    let catalog = starfield::catalogs::hipparcos::HipparcosCatalog::new();
    accepts_catalog(&catalog);
}

#[cfg(feature = "hipparcos")]
#[test]
fn datasource_catalog_implements_the_facade_trait() {
    let catalog = starfield::catalogs::hipparcos::catalog::HipparcosCatalog::new();
    accepts_catalog(&catalog);
}

#[cfg(feature = "horizons")]
#[test]
fn horizons_paths_share_types_and_preserve_calendar_format() {
    use starfield::jpl::horizons::{Center, Command, EphemerisRequest, TimeSpec};
    let mut request = EphemerisRequest::observer(
        Command::MajorBody(499),
        Center::BodyCenter(399),
        TimeSpec::JulianDayList(vec![2451545.0]),
    );
    request.cal_format = Some("JD".into());
    let compatibility_path: starfield::horizons::EphemerisRequest = request;
    assert!(compatibility_path
        .to_input_file()
        .contains("CAL_FORMAT='JD'"));
}

#[cfg(feature = "sbdb")]
#[test]
fn snapshot_support_survives_the_move() {
    let delta = starfield::jpl::sbdb::snapshot::angle_delta_deg(1.0, 359.0);
    assert_eq!(delta.abs(), 2.0);
}
