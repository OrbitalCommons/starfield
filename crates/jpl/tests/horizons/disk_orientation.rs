//! Live canary for the golden disk-orientation values also used by core tests.

use starfield_jpl::horizons::{
    parser, Center, Command, EphemerisRequest, HorizonsClient, TimeSpec,
};

#[test]
#[ignore = "queries the JPL Horizons API over the network"]
fn test_horizons_fixture_is_current() {
    let rows: Vec<Vec<f64>> = include_str!("disk_orientation.csv")
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty() && !line.starts_with("target"))
        .map(|line| line.split(',').map(|s| s.trim().parse().unwrap()).collect())
        .collect();
    let client = HorizonsClient::new().unwrap();
    for (target, center) in [(499, 399), (399, 499)] {
        let rows: Vec<_> = rows
            .iter()
            .filter(|r| r[0] == target as f64 && r[1] == center as f64)
            .collect();
        let mut request = EphemerisRequest::observer(
            Command::MajorBody(target),
            Center::BodyCenter(center),
            TimeSpec::JulianDayList(rows.iter().map(|r| r[2]).collect()),
        );
        request.quantities = Some("14,15,16,17".into());
        request.cal_format = Some("JD".into());
        let result = client.query(&request).unwrap().result.unwrap();
        let names = parser::extract_column_names(&result);
        let block = parser::extract_ephemeris_block(&result).unwrap();
        let fetched = parser::parse_observer_rows(block, &names).unwrap();
        assert_eq!(fetched.len(), rows.len());
        for (expected, actual) in rows.iter().zip(fetched) {
            assert!((actual.jd - expected[2]).abs() <= 1e-6);
            for (index, field) in [
                "ObsSub-LON",
                "ObsSub-LAT",
                "SunSub-LON",
                "SunSub-LAT",
                "SN.ang",
                "NP.ang",
            ]
            .iter()
            .enumerate()
            {
                let value: f64 = actual
                    .fields
                    .iter()
                    .find(|(key, _)| key == field)
                    .unwrap()
                    .1
                    .parse()
                    .unwrap();
                let tolerance = if index < 4 { 1e-4 } else { 1e-2 };
                assert!(
                    (value - expected[index + 3]).abs() <= tolerance,
                    "{field}: {value} vs {}",
                    expected[index + 3]
                );
            }
        }
    }
}
