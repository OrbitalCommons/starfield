//! Galaxy-candidates loader tests against synthetic CSV fixtures.

use std::io::Write;

use starfield_gaia::Cone;
use starfield_gaia_extended::Dr3GalaxyCatalog;

/// (source_id, ra, dec, radius_sersic, n_sersic) per row.
type GalaxyRow = (u64, f64, f64, Option<f32>, Option<f32>);

/// Build a minimal CSV that exercises the columns we model. The published
/// `galaxy_candidates` files have ~50 columns; we include only the ones
/// the parser looks up by name and let the rest be implicitly absent.
fn write_galaxy_fixture(rows: &[GalaxyRow]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(
        f,
        "source_id,ra,dec,classlabel_dsc,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_quasar,radius_sersic,n_sersic,ellipticity_sersic,pa_sersic,gof_galaxy,vari_best_class_name,vari_best_class_score"
    )
    .unwrap();
    for (i, (id, ra, dec, radius_sersic, n_sersic)) in rows.iter().enumerate() {
        let radius = radius_sersic.map(|v| v.to_string()).unwrap_or_default();
        let nser = n_sersic.map(|v| v.to_string()).unwrap_or_default();
        let label = if i % 2 == 0 { "galaxy" } else { "" };
        writeln!(
            f,
            "{id},{ra},{dec},{label},0.97,0.02,{radius},{nser},0.3,42.0,1.5,VARIABLE,0.8"
        )
        .unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn parses_minimal_fixture() {
    let fixture = write_galaxy_fixture(&[
        (1001, 10.0, 20.0, Some(2.5), Some(4.0)),
        (1002, 11.0, 21.0, Some(1.2), Some(1.0)),
        (1003, 12.0, 22.0, None, None),
    ]);
    let cat = Dr3GalaxyCatalog::from_csv_file(fixture.path()).unwrap();
    assert_eq!(cat.len(), 3);
    let g = cat.get(1001).unwrap();
    assert_eq!(g.ra, 10.0);
    assert_eq!(g.dec, 20.0);
    assert_eq!(g.radius_sersic, Some(2.5));
    assert_eq!(g.n_sersic, Some(4.0));
    assert_eq!(g.classlabel_dsc.as_deref(), Some("galaxy"));
    assert_eq!(g.classprob_dsc_combmod_galaxy, Some(0.97));

    let g3 = cat.get(1003).unwrap();
    assert_eq!(g3.radius_sersic, None);
    assert_eq!(g3.n_sersic, None);

    // 1002 is the odd-indexed row in the fixture and has an empty
    // `classlabel_dsc` cell; the parser should turn that into None.
    let g2 = cat.get(1002).unwrap();
    assert_eq!(g2.classlabel_dsc, None);
}

#[test]
fn cone_filter_returns_only_inside_galaxies() {
    let fixture = write_galaxy_fixture(&[
        (1, 10.0, 20.0, None, None),   // inside
        (2, 10.5, 20.5, None, None),   // inside
        (3, 200.0, -45.0, None, None), // outside
        (4, 11.0, 19.0, None, None),   // inside
    ]);
    let cat = Dr3GalaxyCatalog::from_csv_file(fixture.path()).unwrap();
    let cone = Cone::from_degrees(10.0, 20.0, 2.0);
    let inside = cat.in_cone(&cone);
    let mut ids: Vec<u64> = inside.iter().map(|g| g.source_id).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2, 4]);
}

#[test]
fn missing_required_column_errors() {
    // Drop `dec` — the parser must reject this loudly.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,ra,radius_sersic").unwrap();
    writeln!(f, "1,10.0,2.5").unwrap();
    f.flush().unwrap();
    let err = Dr3GalaxyCatalog::from_csv_file(f.path()).expect_err("dec missing");
    let msg = err.to_string();
    assert!(
        msg.contains("dec"),
        "error should mention missing dec column, got: {}",
        msg
    );
}

#[test]
fn unknown_columns_are_ignored() {
    // Add a bunch of columns we don't model — parser should ignore them.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(
        f,
        "source_id,ra,dec,n_sersic,some_fake_extra_col,another_extra"
    )
    .unwrap();
    writeln!(f, "1,10.0,20.0,4.0,foo,bar").unwrap();
    f.flush().unwrap();
    let cat = Dr3GalaxyCatalog::from_csv_file(f.path()).unwrap();
    assert_eq!(cat.len(), 1);
    assert_eq!(cat.get(1).unwrap().n_sersic, Some(4.0));
}

#[test]
fn empty_file_after_header_yields_empty_catalog() {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,ra,dec").unwrap();
    f.flush().unwrap();
    let cat = Dr3GalaxyCatalog::from_csv_file(f.path()).unwrap();
    assert_eq!(cat.len(), 0);
    assert!(cat.is_empty());
}

#[test]
fn extended_source_returns_some_when_all_sersic_fields_present() {
    use starfield_core::catalogs::ExtendedSource;

    let fixture = write_galaxy_fixture(&[(1001, 10.0, 20.0, Some(2.5), Some(4.0))]);
    let cat = Dr3GalaxyCatalog::from_csv_file(fixture.path()).unwrap();
    let g = cat.get(1001).unwrap();
    let p = g.sersic_profile().expect("all four shape fields populated");
    assert_eq!(p.theta_half_arcsec, 2.5);
    assert_eq!(p.n, 4.0);
    // Fixture writes ellipticity_sersic = 0.3 and pa_sersic = 42.0.
    assert!(
        (p.axis_ratio - 0.7).abs() < 1e-6,
        "axis_ratio should be 1 - 0.3, got {}",
        p.axis_ratio
    );
    assert_eq!(p.position_angle_deg, 42.0);
}

#[test]
fn extended_source_returns_none_when_any_sersic_field_missing() {
    use starfield_core::catalogs::ExtendedSource;

    // The fixture writer leaves radius_sersic and n_sersic empty when
    // the option is None, but still emits ellipticity (0.3) and PA (42)
    // unconditionally. The impl must require all four — entry 1003
    // returns None because radius and n are blank.
    let fixture = write_galaxy_fixture(&[(1003, 12.0, 22.0, None, None)]);
    let cat = Dr3GalaxyCatalog::from_csv_file(fixture.path()).unwrap();
    let g = cat.get(1003).unwrap();
    assert!(
        g.sersic_profile().is_none(),
        "missing radius/n should mean no sersic_profile"
    );
}

#[test]
fn duplicate_source_id_collides() {
    // Same source_id twice → second wins (HashMap insert semantics).
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,ra,dec,n_sersic").unwrap();
    writeln!(f, "42,10.0,20.0,1.0").unwrap();
    writeln!(f, "42,30.0,40.0,4.0").unwrap();
    f.flush().unwrap();
    let cat = Dr3GalaxyCatalog::from_csv_file(f.path()).unwrap();
    assert_eq!(cat.len(), 1);
    let g = cat.get(42).unwrap();
    assert_eq!(g.ra, 30.0);
    assert_eq!(g.n_sersic, Some(4.0));
}
