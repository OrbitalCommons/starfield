//! Synthetic-JSON tests for the CaomObservation and DataProduct serde
//! shapes. We don't have a `MashupResponse` parser exposed publicly
//! (the client wraps it), so test the row deserialisation directly.

use starfield_mast::{CaomObservation, DataProduct, ProductType};

#[test]
fn caom_observation_parses_minimal_row() {
    let row = serde_json::json!({
        "obs_id": "j8c0a1011",
        "obsid": 1234567,
        "obs_collection": "HST",
        "instrument_name": "ACS/WFC",
        "filters": "F814W",
        "s_ra": 210.802,
        "s_dec": 54.349,
        "t_exptime": 720.0,
        "t_min": 52800.0,
        "t_max": 52800.01,
        "proposal_id": "9405",
        "proposal_pi": "Holtzman",
        "target_name": "NGC-5457-FIELD",
        "dataURL": "mast:HST/product/j8c0a1011_drz.fits",
        "dataRights": "PUBLIC",
        "s_region": "POLYGON ICRS 210.7 54.3 210.9 54.3 210.9 54.4 210.7 54.4"
    });
    let obs: CaomObservation = serde_json::from_value(row).unwrap();
    assert_eq!(obs.obs_id, "j8c0a1011");
    assert_eq!(obs.obs_collection, "HST");
    assert_eq!(obs.instrument_name.as_deref(), Some("ACS/WFC"));
    assert_eq!(obs.filters.as_deref(), Some("F814W"));
    assert_eq!(obs.s_ra, Some(210.802));
    assert_eq!(obs.s_dec, Some(54.349));
    assert_eq!(obs.t_exptime, Some(720.0));
    assert_eq!(obs.proposal_id.as_deref(), Some("9405"));
    assert_eq!(obs.target_name.as_deref(), Some("NGC-5457-FIELD"));
    assert!(obs.data_url.as_deref().unwrap().starts_with("mast:HST"));
    assert_eq!(obs.data_rights.as_deref(), Some("PUBLIC"));
    assert!(obs.s_region.as_deref().unwrap().starts_with("POLYGON"));
}

#[test]
fn caom_observation_unknown_columns_land_in_extras() {
    let row = serde_json::json!({
        "obs_id": "abc",
        "obs_collection": "HST",
        "wavelength_min": 1.2e-7,
        "calib_level": 3
    });
    let obs: CaomObservation = serde_json::from_value(row).unwrap();
    assert_eq!(obs.obs_id, "abc");
    assert_eq!(obs.obs_collection, "HST");
    // We don't model wavelength_min / calib_level — they should land in extras.
    assert!(obs.extras.contains_key("wavelength_min"));
    assert!(obs.extras.contains_key("calib_level"));
}

#[test]
fn data_product_parses_minimal_row() {
    let row = serde_json::json!({
        "obs_id": "j8c0a1011",
        "productFilename": "j8c0a1011_drz.fits",
        "productType": "SCIENCE",
        "productSubGroupDescription": "DRZ",
        "dataURI": "mast:HST/product/j8c0a1011_drz.fits",
        "size": 12345678
    });
    let p: DataProduct = serde_json::from_value(row).unwrap();
    assert_eq!(p.filename, "j8c0a1011_drz.fits");
    assert_eq!(p.product_type.as_deref(), Some("SCIENCE"));
    assert_eq!(p.product_type_enum(), ProductType::Science);
    assert!(p.is_science_fits());
    assert_eq!(p.size_bytes, Some(12_345_678));
    assert_eq!(p.obs_id.as_deref(), Some("j8c0a1011"));
}

#[test]
fn data_product_resolve_url_translates_mast_uri() {
    let p: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "x.fits",
        "productType": "SCIENCE",
        "dataURI": "mast:HST/product/x.fits"
    }))
    .unwrap();
    let url = p.resolve_url().unwrap();
    assert!(
        url.contains("mast.stsci.edu/api/v0.1/Download/file"),
        "expected MAST download endpoint, got: {}",
        url
    );
    let parsed = reqwest::Url::parse(&url).unwrap();
    let pairs: Vec<_> = parsed.query_pairs().collect();
    assert_eq!(
        pairs,
        vec![("uri".into(), "mast:HST/product/x.fits".into())]
    );
}

#[test]
fn data_product_resolve_url_prefers_explicit_dataurl() {
    let p: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "y.fits",
        "productType": "SCIENCE",
        "dataURI": "mast:HST/product/y.fits",
        "dataURL": "https://example.com/y.fits"
    }))
    .unwrap();
    assert_eq!(
        p.resolve_url().as_deref(),
        Some("https://example.com/y.fits")
    );
}

#[test]
fn data_product_is_science_fits_only_for_fits() {
    let preview: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "preview.jpg",
        "productType": "PREVIEW"
    }))
    .unwrap();
    assert!(!preview.is_science_fits());

    let aux_fits: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "aux.fits",
        "productType": "AUXILIARY"
    }))
    .unwrap();
    assert!(!aux_fits.is_science_fits()); // wrong product type

    let sci_jpg: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "sci.jpg",
        "productType": "SCIENCE"
    }))
    .unwrap();
    assert!(!sci_jpg.is_science_fits()); // wrong filename
}
