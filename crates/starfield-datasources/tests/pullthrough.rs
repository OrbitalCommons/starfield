#[path = "support/stub.rs"]
mod stub;

use starfield_datasource_utils::{adopt_legacy, artifact::materialize, sha256_bytes};
use starfield_datastore::{Datastore, Mirror};
use starfield_gaia::{Downloader, Dr1, Dr2, Dr3};
use starfield_mast::{DataProduct, MastClient};
use starfield_planet_maps::PRODUCTS;
use stub::{Response, Stub};

fn store(root: &std::path::Path, mirror: &Stub) -> Datastore {
    Datastore::builder()
        .cache_root(root.to_path_buf())
        .mirror(Mirror::Http {
            base_url: mirror.url(),
            writable: false,
        })
        .allow_upstream(false)
        .build()
        .unwrap()
}

#[test]
fn map_uses_mirror_key_rejects_html_and_honors_digest_on_hits() {
    let mirror = Stub::start("127.0.0.1");
    let root = tempfile::tempdir().unwrap();
    let store = store(root.path(), &mirror);
    let product = &PRODUCTS[0];
    let artifact = product.artifact(None).unwrap();
    let route = format!("/artifact/{}", artifact.key);
    mirror.route(&route, Response::ok(b"<html>login</html>".to_vec()));
    assert!(product.download_with(&store, None).is_err());
    assert!(!store.contains(&artifact.key));

    let mut bytes = vec![0u8; 2048];
    bytes[..4].copy_from_slice(b"II+\0");
    mirror.route(&route, Response::ok(bytes.clone()));
    let digest = sha256_bytes(&bytes);
    let path = product.download_with(&store, Some(&digest)).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), bytes);
    assert!(product
        .download_with(&store, Some(&"0".repeat(64)))
        .is_err());
    assert_eq!(product.download_with(&store, Some(&digest)).unwrap(), path);
    let requests = mirror.requests();
    // A conflicting pin rejects the local hit and retries the mirror before
    // reporting the rejection; the subsequent correct pin is a local hit.
    assert_eq!(requests.len(), 3);
    assert!(requests
        .iter()
        .all(|r| r.path == route && r.method == "GET" && !r.headers.contains_key("authorization")));
}

#[test]
fn explicit_legacy_adoption_is_validated_and_alias_cleanup_preserves_blob() {
    let root = tempfile::tempdir().unwrap();
    let store = Datastore::builder()
        .cache_root(root.path().join("cache"))
        .offline(true)
        .build()
        .unwrap();
    let product = &PRODUCTS[0];
    let artifact = product.artifact(None).unwrap();
    let legacy = root.path().join(product.file_name);
    std::fs::write(&legacy, b"<html>bad legacy</html>").unwrap();
    adopt_legacy(&store, &artifact, &legacy).unwrap();
    assert!(!store.contains(&artifact.key));
    let mut bytes = vec![0u8; 2048];
    bytes[..4].copy_from_slice(b"II*\0");
    std::fs::write(&legacy, &bytes).unwrap();
    // A caller-supplied offline store never discovers the legacy file itself.
    assert!(product.download_with(&store, None).is_err());
    adopt_legacy(&store, &artifact, &legacy).unwrap();
    let blob = product.download_with(&store, None).unwrap();
    let alias = root.path().join("alias.tif");
    materialize(&blob, &alias).unwrap();
    std::fs::remove_file(alias).unwrap();
    assert_eq!(std::fs::read(blob).unwrap(), bytes);
}

#[test]
fn gaia_checksums_and_shards_both_use_mirror_and_bad_md5_is_evicted() {
    let mirror = Stub::start("127.0.0.1");
    let root = tempfile::tempdir().unwrap();
    let store = store(root.path(), &mirror);
    let filename = "GaiaSource_test.csv.gz";
    let artifact = Downloader::<Dr3>::artifact(filename).unwrap();
    let mut bytes = vec![0u8; 2048];
    bytes[..2].copy_from_slice(&[0x1f, 0x8b]);
    let checksum = format!("{:x}  {filename}\n", md5::compute(&bytes));
    mirror.route(
        "/artifact/gaia/dr3/gaia_source/_MD5SUM.txt",
        Response::ok(checksum),
    );
    mirror.route(
        &format!("/artifact/{}", artifact.key),
        Response::ok(bytes.clone()),
    );
    let path = Downloader::<Dr3>::download_file_with(&store, filename).unwrap();
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    assert_eq!(mirror.requests().len(), 2);
    assert_eq!(
        Downloader::<Dr3>::list_remote_with(&store).unwrap(),
        vec![filename]
    );
    assert!(Downloader::<Dr3>::download_file_with(&store, "absent.csv.gz").is_err());
    assert_eq!(
        mirror.requests().len(),
        2,
        "discovery is cached; unlisted shards never fetch"
    );
    store.remove(&artifact.key).unwrap();
    bytes[20] = 1;
    mirror.route(&format!("/artifact/{}", artifact.key), Response::ok(bytes));
    assert!(Downloader::<Dr3>::download_file_with(&store, filename).is_err());
    assert!(!store.contains(&artifact.key));
    assert_eq!(
        Downloader::<Dr1>::artifact("MD5SUM.txt")
            .unwrap()
            .key
            .as_str(),
        "gaia/dr1/gaia_source/MD5SUM.txt"
    );
    assert_eq!(
        Downloader::<Dr2>::artifact("MD5SUM.txt")
            .unwrap()
            .key
            .as_str(),
        "gaia/dr2/gaia_source/MD5SUM.txt"
    );
    assert!(Downloader::<Dr3>::artifact("../escape.gz").is_err());
}

#[test]
fn mast_product_identity_and_mirror_fit_validation() {
    let mirror = Stub::start("127.0.0.1");
    let root = tempfile::tempdir().unwrap();
    let store = store(root.path(), &mirror);
    let product: DataProduct = serde_json::from_value(serde_json::json!({
        "productFilename": "test.fits", "dataURI": "mast:HST/product/test.fits", "size": 2880
    }))
    .unwrap();
    let artifact = product.artifact().unwrap();
    assert_eq!(artifact.key.as_str(), "mast/HST/product/test.fits");
    let mut bytes = vec![b' '; 2880];
    bytes[..9].copy_from_slice(b"SIMPLE  =");
    mirror.route(
        &format!("/artifact/{}", artifact.key),
        Response::ok(bytes.clone()),
    );
    let client = MastClient::new().unwrap();
    let path = client.download_product_with(&store, &product).unwrap();
    assert_eq!(std::fs::read(path).unwrap(), bytes);
    assert_eq!(mirror.requests().len(), 1);

    let mut opaque = product.clone();
    opaque.data_uri = Some("mast:HST/product/test:1&part=2.fits".into());
    let opaque_artifact = opaque.artifact().unwrap();
    assert!(opaque_artifact.key.as_str().starts_with("url/"));
    let source =
        starfield_datasource_utils::artifact_from_url(&opaque.resolve_url().unwrap()).unwrap();
    assert_eq!(opaque_artifact.key, source.key);
    assert!(opaque_artifact.sources[0].url.contains("%26part%3D2"));
    opaque.data_uri = Some("mast:HST/product/test_1&part=2.fits".into());
    assert_ne!(opaque_artifact.key, opaque.artifact().unwrap().key);
}

#[test]
fn repeated_and_concurrent_alias_publication_leaves_no_scratch_links() {
    let root = tempfile::tempdir().unwrap();
    let blob = root.path().join("blob");
    let alias = root.path().join("alias.gz");
    std::fs::write(&blob, vec![42u8; 2048]).unwrap();
    materialize(&blob, &alias).unwrap();
    materialize(&blob, &alias).unwrap();
    std::thread::scope(|scope| {
        for _ in 0..8 {
            scope.spawn(|| materialize(&blob, &alias).unwrap());
        }
    });
    assert_eq!(
        std::fs::read(&alias).unwrap(),
        std::fs::read(&blob).unwrap()
    );
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 2);
}

#[test]
fn temporary_gaia_reader_uses_cache_filesystem_and_cleans_up_on_drop() {
    const CHILD: &str = "SFD_TEMPORARY_READER_CHILD";
    const FILE: &str = "GaiaSource_temporary.csv.gz";
    if std::env::var_os(CHILD).is_some() {
        use std::collections::BTreeSet;
        use std::io::Read;
        let store = Datastore::from_env().unwrap();
        let contents = || -> BTreeSet<_> {
            std::fs::read_dir(store.cache_root())
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .collect()
        };
        let before = contents();
        let mut reader = Downloader::<Dr3>::stream_file(FILE).unwrap();
        let during = contents();
        let scratch: Vec<_> = during.difference(&before).collect();
        assert_eq!(scratch.len(), 1, "scratch lives under cache root");
        assert!(scratch[0].is_dir());
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes.len(), 2048);
        assert_eq!(&bytes[..2], &[0x1f, 0x8b]);
        assert!(
            store.keys().unwrap().is_empty(),
            "no persistent shard or alias"
        );
        drop(reader);
        assert_eq!(contents(), before, "temporary store removed on drop");
        return;
    }
    let mirror = Stub::start("127.0.0.1");
    let root = tempfile::tempdir().unwrap();
    let mut bytes = vec![0u8; 2048];
    bytes[..2].copy_from_slice(&[0x1f, 0x8b]);
    mirror.route(
        "/artifact/gaia/dr3/gaia_source/_MD5SUM.txt",
        Response::ok(format!("{:x}  {FILE}\n", md5::compute(&bytes))),
    );
    mirror.route(
        &format!("/artifact/gaia/dr3/gaia_source/{FILE}"),
        Response::ok(bytes),
    );
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "temporary_gaia_reader_uses_cache_filesystem_and_cleans_up_on_drop",
            "--nocapture",
        ])
        .env_clear()
        .env(CHILD, "1")
        .env("HOME", root.path())
        .env("STARFIELD_CACHE_DIR", root.path().join("cache"))
        .env("STARFIELD_MIRROR", mirror.url())
        .env("STARFIELD_ALLOW_UPSTREAM", "0")
        .env("TMPDIR", root.path().join("nonexistent-os-temp"))
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(mirror.requests().len(), 2);
}

#[test]
fn shared_url_identity_and_typed_errors_match_starfield() {
    use starfield_datasource_utils::{artifact_from_url, datastore_error};
    let kernel_url =
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de421.bsp";
    assert_eq!(
        artifact_from_url(kernel_url).unwrap().key,
        starfield_core::data::kernel_artifact("de421.bsp")
            .unwrap()
            .key
    );
    let url = "https://example.org/archive.gz?product=1";
    assert_eq!(
        artifact_from_url(url).unwrap().key,
        artifact_from_url(&format!("{url}#section")).unwrap().key
    );
    assert_eq!(
        artifact_from_url(url).unwrap().key,
        starfield_core::data::url_artifact(url).unwrap().key
    );
    assert!(artifact_from_url("https://user:secret@example.org/archive.gz").is_err());
    let error = datastore_error(starfield_datastore::DatastoreError::Config(
        "example".into(),
    ));
    assert!(matches!(
        error,
        starfield_core::StarfieldError::Datastore(starfield_datastore::DatastoreError::Config(_))
    ));
}

#[test]
#[ignore = "live mirror; requires STARFIELD_MIRROR and bypasses local cache; upstream disabled"]
fn live_mirror_serves_core_pds_and_gaia_from_empty_client_caches() {
    use starfield_planet_spectra::{KarkoschkaTable, Product};
    let mirror = std::env::var("STARFIELD_MIRROR")
        .expect("set STARFIELD_MIRROR to run the live mirror integration test");
    let mut previous = None;
    for _ in 0..2 {
        let root = tempfile::tempdir().unwrap();
        let store = Datastore::builder()
            .cache_root(root.path().to_path_buf())
            .mirror(Mirror::Http {
                base_url: mirror.clone(),
                writable: false,
            })
            .allow_upstream(false)
            .progress(false)
            .build()
            .unwrap();
        let product = Product::High1995;
        let table = KarkoschkaTable::download_with(&store, product).unwrap();
        assert_eq!(table.len(), product.expected_rows());
        let path = KarkoschkaTable::cached_path(&store, product).unwrap();
        let bytes = std::fs::read(path).unwrap();
        if let Some(previous) = previous.replace(bytes.clone()) {
            assert_eq!(
                bytes, previous,
                "separate cold clients get identical PDS bytes"
            );
        }
        let lsk = starfield_core::data::download_or_cache_with(&store, "naif0012.tls").unwrap();
        assert_eq!(
            sha256_bytes(&std::fs::read(lsk).unwrap()),
            "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b"
        );
        let shards = Downloader::<Dr3>::list_remote_with(&store).unwrap();
        assert_eq!(shards.len(), 3386);
        for key in store.keys().unwrap() {
            assert_eq!(
                store.entry(&key).unwrap().layer,
                starfield_datastore::Layer::Mirror
            );
        }
    }
}
