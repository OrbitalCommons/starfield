//! Real-network integration tests against MAST. All `#[ignore]`d so
//! `cargo test` doesn't hit the network in CI; run them locally with:
//!
//! ```sh
//! cargo test -p starfield-mast --test network -- --ignored --nocapture
//! ```
//!
//! These tests download real HST data products (small ones, but
//! still — expect a few MB per run). Each run uses a fresh mirror-free store
//! so a cached product cannot hide upstream archive rot.

use starfield_datasource_utils::datastore::Datastore;
use starfield_gaia::Cone;
use starfield_mast::{MastClient, Wcs};

#[test]
#[ignore = "live upstream; must bypass the mirror and cache. Detects archive rot, so a warm or mirrored resolve would defeat it"]
fn end_to_end_m101_cone_search_and_first_drz_download() {
    // 3' cone around the M101 nucleus — guaranteed dense HST coverage.
    let cone = Cone::from_degrees(210.802, 54.349, 0.05);
    let client = MastClient::new().expect("build client");

    let hst = client
        .hst_observations_in_cone(&cone)
        .expect("CAOM cone search");
    eprintln!("M101 nucleus 3' cone → {} HST observations", hst.len());
    assert!(!hst.is_empty(), "M101 should have HST imaging");

    // Pick the first observation, list its products, find a SCIENCE
    // .fits file, download it, parse the WCS.
    let obs = &hst[0];
    eprintln!(
        "first obs: id={} instrument={:?} filters={:?}",
        obs.obs_id, obs.instrument_name, obs.filters
    );
    let obsid = obs.obsid_string().expect("CAOM rows must carry an obsid");
    let products = client.data_products(&obsid).expect("data products");
    eprintln!("  {} data products listed", products.len());

    let science = products
        .iter()
        .find(|p| p.is_science_fits())
        .expect("at least one SCIENCE .fits product");
    eprintln!(
        "  downloading: {} ({} bytes)",
        science.filename,
        science.size_bytes.unwrap_or(0)
    );
    let cache = tempfile::tempdir().unwrap();
    let store = Datastore::builder()
        .cache_root(cache.path().to_path_buf())
        .without_mirror()
        .allow_upstream(true)
        .build()
        .unwrap();
    let path = client.download_product_with(&store, science).expect("download from live upstream; STARFIELD_ALLOW_UPSTREAM=1 is required outside this explicit canary");
    eprintln!("  cached at: {}", path.display());

    // Parse the WCS and dump pointing + footprint.
    let wcs = Wcs::read_from_fits(&path).expect("read WCS");
    eprintln!(
        "  WCS: CRVAL=({:.4}, {:.4}) CTYPE=({}, {}) NAXIS={}x{} scale={:.3} \"/pix",
        wcs.crval1,
        wcs.crval2,
        wcs.ctype1,
        wcs.ctype2,
        wcs.naxis1,
        wcs.naxis2,
        wcs.pixel_scale_arcsec()
    );
    if wcs.is_tan() {
        let corners = wcs.footprint().expect("TAN footprint");
        for (i, (ra, dec)) in corners.iter().enumerate() {
            eprintln!("  corner[{}] = ({:.4}°, {:.4}°)", i, ra, dec);
        }
    }
}
