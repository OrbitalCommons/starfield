//! QSO-candidates loader tests against synthetic CSV fixtures.

use std::io::Write;

use starfield_gaia::Cone;
use starfield_gaia_extended::Dr3QsoCatalog;

/// (source_id, ra, dec, redshift_qsoc, gaia_crf_source) per row.
type QsoRow = (u64, f64, f64, Option<f32>, Option<bool>);

fn write_qso_fixture(rows: &[QsoRow]) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(
        f,
        "source_id,ra,dec,classlabel_dsc,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,redshift_qsoc,redshift_qsoc_lower,redshift_qsoc_upper,gaia_crf_source,host_galaxy_flag,vari_best_class_name,vari_best_class_score"
    )
    .unwrap();
    for (id, ra, dec, redshift, in_crf) in rows {
        let z = redshift.map(|v| v.to_string()).unwrap_or_default();
        let crf = in_crf
            .map(|b| if b { "true" } else { "false" }.to_string())
            .unwrap_or_default();
        writeln!(
            f,
            "{id},{ra},{dec},quasar,0.99,0.005,{z},,,{crf},false,QSO,0.85"
        )
        .unwrap();
    }
    f.flush().unwrap();
    f
}

#[test]
fn parses_minimal_fixture() {
    let fixture = write_qso_fixture(&[
        (2001, 100.0, -10.0, Some(2.5), Some(true)),
        (2002, 101.0, -11.0, Some(0.8), Some(false)),
        (2003, 102.0, -12.0, None, None),
    ]);
    let cat = Dr3QsoCatalog::from_csv_file(fixture.path()).unwrap();
    assert_eq!(cat.len(), 3);
    let q = cat.get(2001).unwrap();
    assert_eq!(q.ra, 100.0);
    assert_eq!(q.dec, -10.0);
    assert_eq!(q.redshift_qsoc, Some(2.5));
    assert_eq!(q.gaia_crf_source, Some(true));
    assert_eq!(q.classlabel_dsc.as_deref(), Some("quasar"));
    assert_eq!(q.classprob_dsc_combmod_quasar, Some(0.99));

    let q3 = cat.get(2003).unwrap();
    assert_eq!(q3.redshift_qsoc, None);
    assert_eq!(q3.gaia_crf_source, None);
}

#[test]
fn cone_filter_returns_only_inside_quasars() {
    let fixture = write_qso_fixture(&[
        (1, 100.0, -10.0, None, None), // inside
        (2, 100.5, -10.5, None, None), // inside
        (3, 50.0, 50.0, None, None),   // outside
    ]);
    let cat = Dr3QsoCatalog::from_csv_file(fixture.path()).unwrap();
    let cone = Cone::from_degrees(100.0, -10.0, 2.0);
    let inside = cat.in_cone(&cone);
    let mut ids: Vec<u64> = inside.iter().map(|q| q.source_id).collect();
    ids.sort();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn missing_required_column_errors() {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,dec,redshift_qsoc").unwrap(); // ra missing
    writeln!(f, "1,20.0,1.5").unwrap();
    f.flush().unwrap();
    let err = Dr3QsoCatalog::from_csv_file(f.path()).expect_err("ra missing");
    let msg = err.to_string();
    assert!(
        msg.contains("ra"),
        "error should mention missing ra column, got: {}",
        msg
    );
}

#[test]
fn boolean_parsing_handles_t_f_and_numeric() {
    // gaia_crf_source can show up as true/false, t/f, or 1/0 in different
    // exports; the parser should accept all three.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,ra,dec,gaia_crf_source").unwrap();
    writeln!(f, "1,10.0,20.0,true").unwrap();
    writeln!(f, "2,11.0,21.0,t").unwrap();
    writeln!(f, "3,12.0,22.0,1").unwrap();
    writeln!(f, "4,13.0,23.0,false").unwrap();
    writeln!(f, "5,14.0,24.0,f").unwrap();
    writeln!(f, "6,15.0,25.0,0").unwrap();
    writeln!(f, "7,16.0,26.0,").unwrap(); // empty -> None
    f.flush().unwrap();
    let cat = Dr3QsoCatalog::from_csv_file(f.path()).unwrap();
    assert_eq!(cat.get(1).unwrap().gaia_crf_source, Some(true));
    assert_eq!(cat.get(2).unwrap().gaia_crf_source, Some(true));
    assert_eq!(cat.get(3).unwrap().gaia_crf_source, Some(true));
    assert_eq!(cat.get(4).unwrap().gaia_crf_source, Some(false));
    assert_eq!(cat.get(5).unwrap().gaia_crf_source, Some(false));
    assert_eq!(cat.get(6).unwrap().gaia_crf_source, Some(false));
    assert_eq!(cat.get(7).unwrap().gaia_crf_source, None);
}

#[test]
fn empty_file_after_header_yields_empty_catalog() {
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "source_id,ra,dec").unwrap();
    f.flush().unwrap();
    let cat = Dr3QsoCatalog::from_csv_file(f.path()).unwrap();
    assert_eq!(cat.len(), 0);
    assert!(cat.is_empty());
}
