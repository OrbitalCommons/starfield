//! Validates the embedded archive against known spectroscopy.
//!
//! These assertions are the real guard on this crate. A parser that reads
//! plausible-looking floats out of the wrong columns, or off by a row, still
//! produces a table that passes shape and range checks. It does not reproduce
//! the methane band structure of the giant planets at the right wavelengths.

use starfield_planet_spectra::{KarkoschkaTable, SpectralBody};

/// Centres of the strong visible/near-IR methane bands, in nm.
const CH4_727: f64 = 727.6;
const CH4_887: f64 = 887.2;

#[test]
fn strongest_methane_absorption_is_the_887_nm_band() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    let (peak_nm, peak_k) = t
        .vacuum_nm()
        .iter()
        .zip(t.methane_absorption())
        .map(|(&w, &k)| (w, k))
        .fold((0.0, f64::MIN), |acc, x| if x.1 > acc.1 { x } else { acc });

    assert!(
        (peak_nm - CH4_887).abs() < 1.0,
        "strongest CH4 band at {peak_nm} nm, expected {CH4_887}"
    );
    assert!(peak_k > 40.0, "peak coefficient {peak_k} (km-amagat)^-1");
}

#[test]
fn ice_giants_are_swallowed_by_the_619_nm_methane_band() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    for body in [SpectralBody::Uranus, SpectralBody::Neptune] {
        let s = t.albedo(body).unwrap();
        let continuum = s.at_nm(550.0).unwrap();
        let band = s.at_nm(619.2).unwrap();
        assert!(
            continuum > 4.0 * band,
            "{}: 550 nm continuum {continuum} vs 619 nm band {band}",
            body.name()
        );
    }
}

#[test]
fn every_body_absorbs_in_the_727_nm_band() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    for s in t.albedos() {
        let band = s.at_nm(CH4_727).unwrap();
        let blue_shoulder = s.at_nm(700.0).unwrap();
        let red_shoulder = s.at_nm(755.0).unwrap();
        assert!(
            band < blue_shoulder && band < red_shoulder,
            "{}: 727 nm {band} is not a minimum between {blue_shoulder} and {red_shoulder}",
            s.body().name()
        );
    }
}

#[test]
fn ice_giants_are_blue_and_titan_is_red() {
    let t = KarkoschkaTable::load_embedded().unwrap();

    // Uranus and Neptune reflect strongly in the blue and are dark past 600 nm.
    // They are the only two bodies here whose albedo falls from blue to red.
    for body in [SpectralBody::Uranus, SpectralBody::Neptune] {
        let s = t.albedo(body).unwrap();
        let blue = s.mean_over(400.0, 500.0).unwrap();
        let red = s.mean_over(600.0, 700.0).unwrap();
        assert!(
            blue > 1.5 * red,
            "{}: blue {blue} vs red {red}",
            body.name()
        );
    }

    // Neptune is the bluer of the two — it has the deeper methane column.
    let uranus = t.albedo(SpectralBody::Uranus).unwrap();
    let neptune = t.albedo(SpectralBody::Neptune).unwrap();
    let uranus_slope =
        uranus.mean_over(400.0, 500.0).unwrap() / uranus.mean_over(600.0, 700.0).unwrap();
    let neptune_slope =
        neptune.mean_over(400.0, 500.0).unwrap() / neptune.mean_over(600.0, 700.0).unwrap();
    assert!(
        neptune_slope > uranus_slope,
        "Neptune blue/red {neptune_slope} should exceed Uranus {uranus_slope}"
    );

    // Titan's haze does the opposite: very dark in the blue, rising to the red.
    let titan = t.albedo(SpectralBody::Titan).unwrap();
    let blue = titan.mean_over(400.0, 500.0).unwrap();
    let red = titan.mean_over(600.0, 700.0).unwrap();
    assert!(red > 2.0 * blue, "Titan: blue {blue} vs red {red}");
}

#[test]
fn titan_brightens_monotonically_through_the_visible() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    let titan = t.albedo(SpectralBody::Titan).unwrap();
    let mut previous = titan.at_nm(350.0).unwrap();
    for nm in [400.0, 450.0, 500.0, 550.0, 600.0] {
        let a = titan.at_nm(nm).unwrap();
        assert!(a > previous, "Titan dipped at {nm} nm: {a} <= {previous}");
        previous = a;
    }
}

#[test]
fn all_five_bodies_are_tabulated_across_the_silicon_response() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    assert_eq!(t.albedos().len(), 5);
    for s in t.albedos() {
        let (lo, hi) = s.range_nm();
        assert!(lo <= 350.0, "{} starts at {lo} nm", s.body().name());
        assert!(hi >= 1000.0, "{} ends at {hi} nm", s.body().name());
        // A typical silicon QE band integrates cleanly.
        assert!(s.mean_over(400.0, 900.0).is_some());
    }
}

#[test]
fn jupiter_outshines_the_saturn_globe_in_the_blue() {
    let t = KarkoschkaTable::load_embedded().unwrap();
    let jupiter = t.albedo(SpectralBody::Jupiter).unwrap();
    let saturn = t.albedo(SpectralBody::Saturn).unwrap();
    // Saturn's globe is the more strongly reddened of the two; past ~640 nm it
    // overtakes Jupiter, so this only holds on the blue side.
    let j = jupiter.mean_over(400.0, 550.0).unwrap();
    let s = saturn.mean_over(400.0, 550.0).unwrap();
    assert!(j > s, "Jupiter {j} vs Saturn globe {s}");
}
