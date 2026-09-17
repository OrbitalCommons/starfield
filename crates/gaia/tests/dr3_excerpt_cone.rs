//! End-to-end test for the cone-search loader: build a HEALPix-sharded
//! excerpt directory from a synthetic fixture, then call
//! `Dr3Catalog::from_excerpt_dir_for_cone` and check that the returned
//! catalog contains exactly the stars inside the cone (no false negatives,
//! no false positives because the loader post-filters HEALPix's
//! conservative covering).
#![cfg(feature = "dr3")]

use std::collections::HashMap;
use std::io::Write;

use starfield_core::catalogs::StarCatalog;
use starfield_gaia::excerpt::{excerpt_csv_file, HashIdShard, HealpixShard};
use starfield_gaia::{
    Cone, Dr3, Dr3Catalog, GaiaCatalog, LazyLoadingCatalog, MemoryResidentCatalog,
};

const HEADER: &str = "source_id,solution_id,ref_epoch,random_index,designation,ra,ra_error,dec,dec_error,ra_dec_corr,parallax,parallax_error,parallax_over_error,pm,pmra,pmra_error,pmdec,pmdec_error,l,b,ecl_lon,ecl_lat,phot_g_mean_mag,phot_g_mean_flux,phot_g_mean_flux_error,phot_g_n_obs,phot_variable_flag,astrometric_n_obs_al,astrometric_excess_noise,astrometric_excess_noise_sig,astrometric_primary_flag,duplicated_source,matched_observations,astrometric_n_good_obs_al,astrometric_n_bad_obs_al,astrometric_gof_al,astrometric_chi2_al,astrometric_params_solved,visibility_periods_used,astrometric_sigma5d_max,nu_eff_used_in_astrometry,pseudocolour,pseudocolour_error,ruwe,ipd_gof_harmonic_amplitude,ipd_gof_harmonic_phase,ipd_frac_multi_peak,ipd_frac_odd_win,phot_bp_mean_mag,phot_bp_mean_flux,phot_bp_mean_flux_error,phot_bp_n_obs,phot_rp_mean_mag,phot_rp_mean_flux,phot_rp_mean_flux_error,phot_rp_n_obs,bp_rp,bp_g,g_rp,phot_bp_rp_excess_factor,phot_proc_mode,radial_velocity,radial_velocity_error,rv_method_used,rv_nb_transits,rv_expected_sig_to_noise,rv_amplitude_robust,rv_template_teff,rv_template_logg,rv_template_fe_h,rv_atm_param_origin,teff_gspphot,teff_gspphot_lower,teff_gspphot_upper,logg_gspphot,mh_gspphot,distance_gspphot,distance_gspphot_lower,distance_gspphot_upper,azero_gspphot,ag_gspphot,ebpminrp_gspphot,libname_gspphot,has_xp_continuous,has_xp_sampled,has_rvs,has_epoch_photometry,has_epoch_rv,has_mcmc_gspphot,has_mcmc_msc,in_qso_candidates,in_galaxy_candidates,in_andromeda_survey,non_single_star,classprob_dsc_combmod_quasar,classprob_dsc_combmod_galaxy,classprob_dsc_combmod_star";

fn row_from(overrides: &HashMap<&str, &str>) -> String {
    HEADER
        .split(',')
        .map(|c| overrides.get(c).copied().unwrap_or(""))
        .collect::<Vec<_>>()
        .join(",")
}

/// Write a row with the given source_id / RA / Dec / mag.
fn write_row(file: &mut tempfile::NamedTempFile, id: u64, ra_deg: f64, dec_deg: f64, mag: f64) {
    let row = row_from(&HashMap::from([
        ("source_id", id.to_string().as_str()),
        ("solution_id", "1635721458409799680"),
        ("ref_epoch", "2016.0"),
        ("ra", ra_deg.to_string().as_str()),
        ("ra_error", "0.04"),
        ("dec", dec_deg.to_string().as_str()),
        ("dec_error", "0.04"),
        ("l", "0.0"),
        ("b", "0.0"),
        ("ecl_lon", "0.0"),
        ("ecl_lat", "0.0"),
        ("phot_g_mean_mag", mag.to_string().as_str()),
        ("phot_variable_flag", "NOT_AVAILABLE"),
    ]));
    writeln!(file, "{}", row).unwrap();
}

/// Returns (fixture_file, set_of_inside_ids) for a fixture with `inside`
/// stars packed within `inside_radius_deg` of (`centre_ra`, `centre_dec`)
/// and `outside` stars sprinkled across the sphere far from the centre.
/// The `id` for inside stars uses the high bit so callers can partition
/// results without keeping the full set.
fn fixture_with_cone(
    inside: usize,
    outside: usize,
    centre_ra: f64,
    centre_dec: f64,
    inside_radius_deg: f64,
) -> (tempfile::NamedTempFile, std::collections::HashSet<u64>) {
    const INSIDE_BIT: u64 = 1 << 62;
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    let mut inside_ids = std::collections::HashSet::new();

    // Inside cone: place each star at a small offset from the centre by
    // shifting RA only — keeps the math trivial and makes intent obvious.
    // For dec ≈ -30 the cosine-correction is significant, so the test
    // exercises the great-circle distance path rather than naive
    // angular coordinates.
    for i in 0..inside {
        let id = INSIDE_BIT | (i as u64);
        let frac = (i as f64) / (inside.max(1) as f64); // 0..1
        let dra = (frac - 0.5) * 2.0 * (inside_radius_deg * 0.5);
        // Project the RA offset through the cosine of dec so the actual
        // angular separation is dra * cos(dec).
        let ra = (centre_ra + dra / centre_dec.to_radians().cos()).rem_euclid(360.0);
        write_row(&mut f, id, ra, centre_dec, 10.0);
        inside_ids.insert(id);
    }

    // Outside cone: scatter on the opposite hemisphere so distance from
    // centre is always > 90°.
    let opp_ra = (centre_ra + 180.0).rem_euclid(360.0);
    let opp_dec = -centre_dec;
    for i in 0..outside {
        let id = i as u64; // no INSIDE_BIT
        let frac = (i as f64) / (outside.max(1) as f64);
        let ra = (opp_ra + (frac - 0.5) * 60.0).rem_euclid(360.0);
        let dec = (opp_dec + (frac - 0.5) * 30.0).clamp(-89.0, 89.0);
        write_row(&mut f, id, ra, dec, 10.0);
    }

    f.flush().unwrap();
    (f, inside_ids)
}

#[test]
fn cone_loader_returns_only_inside_stars() {
    let centre_ra = 120.0;
    let centre_dec = -30.0;
    let inside_radius = 0.5; // deg — well inside one HEALPix-3 cell (~14.7°)

    let (fixture, inside_ids) = fixture_with_cone(50, 200, centre_ra, centre_dec, inside_radius);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .expect("excerpt");

    // Ask for the cone — radius generous enough to include every "inside"
    // star but small enough that no opposite-hemisphere "outside" star
    // could possibly land in the result.
    let cone = Cone::from_degrees(centre_ra, centre_dec, inside_radius * 2.0);
    let cat =
        Dr3Catalog::from_excerpt_dir_for_cone(out.path(), cone, f64::INFINITY).expect("cone load");

    let returned: std::collections::HashSet<u64> = cat.stars().map(|s| s.core.source_id).collect();
    assert_eq!(returned.len(), inside_ids.len(), "wrong row count");
    assert_eq!(returned, inside_ids, "returned rows ≠ inside-cone rows");
}

#[test]
fn cone_loader_post_filter_drops_boundary_cell_rows() {
    // HEALPix gives a *conservative* covering: cells at the boundary of
    // the cone include stars outside the cone. Those must be dropped.
    // Construct a tight fixture with stars right at the edge of a level-3
    // cell so the covering pulls the whole cell, then verify the loader
    // drops the ones outside the requested radius.
    let centre_ra = 0.0;
    let centre_dec = 0.0;
    // Place stars on a 3°-wide ring; with a 0.5° cone, only the ones
    // within 0.5° of centre should be returned.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for i in 0..200u64 {
        let theta = (i as f64) * std::f64::consts::TAU / 200.0;
        let r = 0.1 + (i as f64 / 200.0) * 3.0; // 0.1° → 3.1°
        let ra = (centre_ra + r * theta.cos()).rem_euclid(360.0);
        let dec = (centre_dec + r * theta.sin()).clamp(-89.0, 89.0);
        write_row(&mut f, i, ra, dec, 10.0);
    }
    f.flush().unwrap();

    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .expect("excerpt");

    let cone = Cone::from_degrees(centre_ra, centre_dec, 0.5);
    let cat =
        Dr3Catalog::from_excerpt_dir_for_cone(out.path(), cone, f64::INFINITY).expect("cone load");

    // Brute-force ground truth from the same fixture.
    let truth = Dr3Catalog::from_csv_file(f.path(), f64::INFINITY).unwrap();
    let truth_inside: std::collections::HashSet<u64> = truth
        .stars()
        .filter(|s| cone.contains_radec_deg(s.core.ra, s.core.dec))
        .map(|s| s.core.source_id)
        .collect();
    let returned: std::collections::HashSet<u64> = cat.stars().map(|s| s.core.source_id).collect();
    assert_eq!(returned, truth_inside, "post-filter mismatch");
}

#[test]
fn cone_loader_honours_mag_limit() {
    // Build a fixture with 50 stars at mag 8 (bright) and 50 at mag 18
    // (faint), all inside the cone. Loading with mag_limit=10 should
    // return only the 50 bright ones.
    let centre_ra = 200.0;
    let centre_dec = 45.0;
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for i in 0..50u64 {
        let ra = centre_ra + 0.01 * (i as f64);
        write_row(&mut f, i, ra, centre_dec, 8.0);
    }
    for i in 50..100u64 {
        let ra = centre_ra + 0.01 * (i as f64);
        write_row(&mut f, i, ra, centre_dec, 18.0);
    }
    f.flush().unwrap();

    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .expect("excerpt");

    let cone = Cone::from_degrees(centre_ra, centre_dec, 5.0);
    let cat = Dr3Catalog::from_excerpt_dir_for_cone(out.path(), cone, 10.0).expect("cone load");
    assert_eq!(cat.len(), 50, "mag_limit=10 should drop the mag-18 cohort");
}

#[test]
fn cone_loader_rejects_non_healpix_dir() {
    // Hash-sharded dir has no spatial coherence; the loader must refuse it.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for i in 0..20u64 {
        write_row(&mut f, i, 10.0, 20.0, 10.0);
    }
    f.flush().unwrap();

    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HashIdShard { num_shards: 4 },
        |_| true,
    )
    .expect("excerpt");

    let cone = Cone::from_degrees(10.0, 20.0, 5.0);
    let err = Dr3Catalog::from_excerpt_dir_for_cone(out.path(), cone, f64::INFINITY)
        .expect_err("should refuse non-healpix dir");
    let msg = err.to_string();
    assert!(
        msg.contains("HEALPix") || msg.contains("healpix"),
        "error should mention HEALPix requirement, got: {}",
        msg
    );
}

#[test]
fn cone_loader_rejects_mod_collapsed_healpix_dir() {
    // A pre-PR-#42 mod-collapsed dir claims kind="healpix" but has
    // num_shards < cell_count(level) — the loader would otherwise silently
    // skip ~99% of the cone's cells. Forge such a manifest and check the
    // loader bails with a clear message.
    let out = tempfile::tempdir().unwrap();
    let manifest = serde_json::json!({
        "version": 1,
        "release": "Dr3",
        "mag_limit": 20.0,
        "sharder": {
            "kind": "healpix",
            "num_shards": 128,
            "healpix_level": 5,
        },
        "shard_sizes": vec![0u64; 128],
        "shard_rows": vec![0u64; 128],
        "processed_files": Vec::<String>::new(),
        "kept_rows": 0,
    });
    std::fs::write(
        out.path().join(".gaia-excerpt-manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let cone = Cone::from_degrees(0.0, 0.0, 1.0);
    let err = Dr3Catalog::from_excerpt_dir_for_cone(out.path(), cone, f64::INFINITY)
        .expect_err("should refuse mod-collapsed healpix dir");
    let msg = err.to_string();
    assert!(
        msg.contains("one-file-per-cell") || msg.contains("mod-collapsed"),
        "error should explain mod-collapsed layout, got: {}",
        msg
    );
}

#[test]
fn cone_loader_errors_on_missing_dir() {
    let cone = Cone::from_degrees(0.0, 0.0, 1.0);
    let err = Dr3Catalog::from_excerpt_dir_for_cone(
        "/this/path/does/not/exist/excerpt-dir",
        cone,
        f64::INFINITY,
    )
    .expect_err("should error on missing manifest");
    let msg = err.to_string();
    assert!(
        msg.contains("manifest") || msg.contains("excerpt"),
        "error should mention missing manifest, got: {}",
        msg
    );
}

// ====================================================================
// Trait-level coverage: same fixture, exercised through both
// implementations of `GaiaCatalog<Dr3>` to confirm callers can hold a
// `&dyn GaiaCatalog<Dr3>` and swap backends.
// ====================================================================

/// Run the same cone query through `&dyn GaiaCatalog<Dr3>` and count
/// the rows it produces. Used by the trait-level tests below.
fn cone_count_via_trait(cat: &dyn GaiaCatalog<Dr3>, cone: Cone, mag_limit: f64) -> usize {
    let mut n = 0;
    for row in cat
        .entries_in_cone(cone, mag_limit)
        .expect("entries_in_cone")
    {
        let _ = row.expect("row");
        n += 1;
    }
    n
}

#[test]
fn lazy_loading_streams_entries_without_full_materialization() {
    // Build a fixture with 30 inside / 70 outside, shard healpix-3, then
    // open the dir as a LazyLoadingCatalog and stream the cone — no
    // materialization, just count.
    let centre_ra = 80.0;
    let centre_dec = -10.0;
    let (fixture, inside_ids) = fixture_with_cone(30, 70, centre_ra, centre_dec, 0.5);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).expect("open lazy");
    let cone = Cone::from_degrees(centre_ra, centre_dec, 1.0);
    let returned: std::collections::HashSet<u64> = lazy
        .entries_in_cone(cone, f64::INFINITY)
        .unwrap()
        .map(|r| r.unwrap().core.source_id)
        .collect();
    assert_eq!(returned, inside_ids);
}

#[test]
fn memory_resident_and_lazy_agree_on_same_cone() {
    // Same fixture, same cone, two backends — the result sets must be
    // identical (modulo ordering, hence the HashSet compare).
    let centre_ra = 12.3;
    let centre_dec = 45.6;
    let (fixture, _inside_ids) = fixture_with_cone(40, 80, centre_ra, centre_dec, 0.5);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let mem: MemoryResidentCatalog<Dr3> =
        MemoryResidentCatalog::from_csv_file(fixture.path(), f64::INFINITY).unwrap();
    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();

    let cone = Cone::from_degrees(centre_ra, centre_dec, 1.0);
    let mem_ids: std::collections::HashSet<u64> = (&mem as &dyn GaiaCatalog<Dr3>)
        .entries_in_cone(cone, f64::INFINITY)
        .unwrap()
        .map(|r| r.unwrap().core.source_id)
        .collect();
    let lazy_ids: std::collections::HashSet<u64> = (&lazy as &dyn GaiaCatalog<Dr3>)
        .entries_in_cone(cone, f64::INFINITY)
        .unwrap()
        .map(|r| r.unwrap().core.source_id)
        .collect();

    assert_eq!(mem_ids, lazy_ids);
    // And both agree with the trait-object count helper.
    assert_eq!(
        cone_count_via_trait(&mem, cone, f64::INFINITY),
        cone_count_via_trait(&lazy, cone, f64::INFINITY)
    );
}

#[test]
fn materialize_cone_round_trips_through_lazy() {
    // `materialize_cone` is the trait default that drains the iterator
    // into a MemoryResidentCatalog. Verify it returns the same set as the
    // direct iterator.
    let centre_ra = 250.0;
    let centre_dec = 12.0;
    let (fixture, inside_ids) = fixture_with_cone(25, 75, centre_ra, centre_dec, 0.5);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    let cone = Cone::from_degrees(centre_ra, centre_dec, 1.0);
    let materialized = lazy.materialize_cone(cone, f64::INFINITY).unwrap();
    let ids: std::collections::HashSet<u64> =
        materialized.stars().map(|s| s.core.source_id).collect();
    assert_eq!(ids, inside_ids);
}

#[test]
fn lazy_len_reads_from_manifest() {
    // Build a fixture with 60 rows, ingest with the healpix sharder, then
    // assert `LazyLoadingCatalog::len()` returns 60 from the manifest
    // without us having to scan any shard files.
    let (fixture, _ids) = fixture_with_cone(40, 20, 100.0, 25.0, 0.5);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();
    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    assert_eq!(<_ as GaiaCatalog<Dr3>>::len(&lazy), 60);
    assert!(!<_ as GaiaCatalog<Dr3>>::is_empty(&lazy));
}

#[test]
fn memory_resident_len_matches_hashmap_size() {
    let (fixture, _ids) = fixture_with_cone(15, 35, 50.0, -15.0, 0.5);
    let mem: MemoryResidentCatalog<Dr3> =
        MemoryResidentCatalog::from_csv_file(fixture.path(), f64::INFINITY).unwrap();
    assert_eq!(<_ as GaiaCatalog<Dr3>>::len(&mem), 50);
}

#[test]
fn empty_lazy_dir_is_empty() {
    // No input files committed → manifest.kept_rows == 0.
    let out = tempfile::tempdir().unwrap();
    // Initialize an empty manifest by opening + immediately finishing a writer.
    use starfield_gaia::excerpt::ShardedCsvWriter;
    let writer = ShardedCsvWriter::<Dr3, _>::new_or_resume(
        out.path(),
        HealpixShard { level: 3 },
        f64::INFINITY,
    )
    .unwrap();
    let _ = writer.finish().unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    assert_eq!(<_ as GaiaCatalog<Dr3>>::len(&lazy), 0);
    assert!(<_ as GaiaCatalog<Dr3>>::is_empty(&lazy));
}

#[test]
fn lazy_open_validates_release() {
    // A manifest written by a Dr2 run shouldn't open as a LazyLoadingCatalog<Dr3>.
    let out = tempfile::tempdir().unwrap();
    let manifest = serde_json::json!({
        "version": 1,
        "release": "Dr2",
        "mag_limit": 20.0,
        "sharder": {
            "kind": "healpix",
            "num_shards": 12_288,
            "healpix_level": 5,
        },
        "shard_sizes": vec![0u64; 12_288],
        "shard_rows": vec![0u64; 12_288],
        "processed_files": Vec::<String>::new(),
        "kept_rows": 0,
    });
    std::fs::write(
        out.path().join(".gaia-excerpt-manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = LazyLoadingCatalog::<Dr3>::open(out.path()).expect_err("release mismatch");
    let msg = err.to_string();
    assert!(
        msg.contains("Dr2") && msg.contains("Dr3"),
        "error should name both releases, got: {}",
        msg
    );
}

// ====================================================================
// Per-cell access: drives the cell-at-a-time iteration pattern needed
// by downstream index builders (zodiacal#65). Empty cells return an
// empty iterator (no I/O); out-of-range cell ids error.
// ====================================================================

#[test]
fn entries_in_cell_returns_only_that_cell() {
    // Build a fixture spread across the sky, ingest healpix-3 (768 cells),
    // then iterate cells and check that the union of every cell's entries
    // equals the whole catalog and no entry appears in more than one cell.
    let (fixture, _ids) = fixture_with_cone(80, 120, 30.0, 10.0, 0.5);
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        fixture.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    assert_eq!(lazy.cell_count(), 12 << (2 * 3)); // 768 for level-3

    let mut union: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut total_via_cells = 0u64;
    for cell in 0..lazy.cell_count() {
        let cell_ids: Vec<u64> = lazy
            .entries_in_cell(cell, f64::INFINITY)
            .unwrap()
            .map(|r| r.unwrap().core.source_id)
            .collect();
        for id in &cell_ids {
            assert!(
                union.insert(*id),
                "id {} appeared in cell {} after already appearing in another cell",
                id,
                cell
            );
        }
        total_via_cells += cell_ids.len() as u64;
    }
    // Every committed row should be accounted for exactly once.
    assert_eq!(total_via_cells, <_ as GaiaCatalog<Dr3>>::len(&lazy));
}

#[test]
fn entries_in_cell_empty_cell_yields_empty_iterator() {
    // Single-row fixture forces all but one cell to be empty. Pick any
    // cell other than the one we know got data and assert the iterator
    // yields nothing.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    write_row(&mut f, 1, 0.0, 0.0, 10.0); // (ra=0, dec=0) lands in some level-3 cell
    f.flush().unwrap();

    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    // Find the cell that got the row, then walk every other cell and
    // confirm an empty iterator.
    let occupied: u32 = (0..lazy.cell_count())
        .find(|&c| {
            lazy.entries_in_cell(c, f64::INFINITY)
                .unwrap()
                .next()
                .is_some()
        })
        .expect("at least one cell got the row");
    let mut empty_cells_checked = 0;
    for cell in 0..lazy.cell_count() {
        if cell == occupied {
            continue;
        }
        let any = lazy
            .entries_in_cell(cell, f64::INFINITY)
            .unwrap()
            .next()
            .is_some();
        assert!(!any, "cell {} unexpectedly returned a row", cell);
        empty_cells_checked += 1;
    }
    assert_eq!(empty_cells_checked, lazy.cell_count() - 1);
}

#[test]
fn entries_in_cell_rejects_out_of_range_id() {
    // Empty fixture, but a level-3 layout still has 768 cells.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    write_row(&mut f, 1, 0.0, 0.0, 10.0);
    f.flush().unwrap();
    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    let bad_id = lazy.cell_count(); // first invalid id
                                    // `expect_err` would require `Debug` on the `Ok` variant — and
                                    // `Box<dyn Iterator<...>>` doesn't impl `Debug` — so match instead.
    match lazy.entries_in_cell(bad_id, f64::INFINITY) {
        Ok(_) => panic!("out-of-range cell_id should error"),
        Err(e) => assert!(
            e.to_string().contains("out of range"),
            "error should mention out of range, got: {}",
            e
        ),
    };
}

#[test]
fn entries_in_cell_honours_mag_limit() {
    // Single cell with both bright and faint stars; mag_limit gates them.
    let mut f = tempfile::Builder::new().suffix(".csv").tempfile().unwrap();
    writeln!(f, "{}", HEADER).unwrap();
    for i in 0..10u64 {
        write_row(&mut f, i, 100.0, 30.0, 8.0);
    }
    for i in 10..20u64 {
        write_row(&mut f, i, 100.001, 30.001, 18.0);
    }
    f.flush().unwrap();

    let out = tempfile::tempdir().unwrap();
    excerpt_csv_file::<Dr3, _, _>(
        f.path(),
        f64::INFINITY,
        out.path(),
        HealpixShard { level: 3 },
        |_| true,
    )
    .unwrap();

    let lazy = LazyLoadingCatalog::<Dr3>::open(out.path()).unwrap();
    // Find the cell holding our 20 stars (they're all within ~0.001° of
    // each other so they'll land together).
    let occupied = (0..lazy.cell_count())
        .find(|&c| {
            lazy.entries_in_cell(c, f64::INFINITY)
                .unwrap()
                .next()
                .is_some()
        })
        .unwrap();
    let n_bright = lazy
        .entries_in_cell(occupied, 10.0)
        .unwrap()
        .filter_map(|r| r.ok())
        .count();
    assert_eq!(n_bright, 10, "mag_limit=10 should drop the mag-18 cohort");
}
