//! The data files starfield fetches, described as `starfield-datastore`
//! artifacts (rollout step 5 of the datastore spec).
//!
//! Every file has a stable, archive-shaped key that is the same in every
//! consumer and on the ephemeris server, so a relocated upstream changes a
//! `Source` here and nothing else. The content checks reject the wrong kind
//! of thing — an HTML login page under a `.bsp` key — using starfield's own
//! magic numbers; the kernel parser validates the format afterwards.
//!
//! The constructors are public so other consumers (`starfield-datasources`)
//! and the server's manifest tooling build byte-identical artifacts.

use std::path::{Path, PathBuf};

use starfield_datastore::{
    Artifact, ArtifactKey, ContentCheck, Datastore, DatastoreBuilder, Provenance, Source,
};

use crate::data::downloader::{
    HIPPARCOS_URL, JPL_BSP_URL, NAIF_FK_SATELLITES_URL, NAIF_LSK_URL, NAIF_PCK_URL,
    NAIF_PLANETS_OLD_URL, NAIF_PLANETS_URL, NAIF_SATELLITES_URL,
};
use crate::planetarylib::TEXT_MAGIC_NUMBERS;
use crate::{Result, StarfieldError};

/// Licence recorded for NAIF/JPL generic kernels: US Government works.
pub const KERNEL_LICENSE: &str = "public-domain";

/// Licence recorded for the Hipparcos main catalogue as served by CDS.
pub const HIPPARCOS_LICENSE: &str =
    "ESA Hipparcos catalogue I/239 via CDS; free for scientific use with acknowledgement";

/// Key for the Hipparcos main catalogue, `hip_main.dat`.
pub const HIPPARCOS_KEY: &str = "cds/I/239/hip_main.dat";

/// Leading bytes of a DAF-based SPK. `NAIF/DAF` is the pre-N0052 ID word
/// still found in archived kernels; the reader accepts it, so the cache must.
const SPK_MAGIC: [&[u8]; 2] = [b"DAF/SPK", b"NAIF/DAF"];
const BINARY_PCK_MAGIC: [&[u8]; 2] = [b"DAF/PCK", b"NAIF/DAF"];
const LEAP_SECONDS_MAGIC: [&[u8]; 1] = [b"KPL/LSK"];
const GZIP_MAGIC: [&[u8]; 1] = [&[0x1f, 0x8b]];

fn magic(prefixes: &[&[u8]], trim_leading_whitespace: bool) -> ContentCheck {
    ContentCheck::magic(
        prefixes.iter().map(|p| p.to_vec()).collect(),
        trim_leading_whitespace,
    )
}

fn text_kernel_check() -> ContentCheck {
    ContentCheck::magic(
        TEXT_MAGIC_NUMBERS
            .iter()
            .map(|m| m.as_bytes().to_vec())
            .collect(),
        true,
    )
}

fn kernel_provenance(description: String) -> Provenance {
    Provenance {
        description,
        license: KERNEL_LICENSE.into(),
        citation: None,
    }
}

fn key(text: String) -> Result<ArtifactKey> {
    ArtifactKey::new(text).map_err(StarfieldError::from)
}

/// The artifact for a SPICE kernel named the way `Loader::open` and friends
/// name it: `de421.bsp`, `pck00011.tpc`, `moon_080317.tf`, `naif0012.tls`.
///
/// Keys are `naif/spk/<file>` (`naif/spk/satellites/<file>` for the `jup*`
/// satellite ephemerides), `naif/pck/<file>`, `naif/fk/<file>` and
/// `naif/lsk/<file>`; sources follow [`crate::data::resolve_url`]. `None`
/// for an extension starfield does not recognise.
pub fn kernel_artifact(filename: &str) -> Option<Artifact> {
    if filename.contains("://") || filename.contains('/') {
        return None;
    }
    let (subdir, base_url, check, kind) = if filename.ends_with(".bsp") {
        if filename.starts_with("jup") {
            (
                "spk/satellites",
                NAIF_SATELLITES_URL,
                magic(&SPK_MAGIC, false),
                "SPK satellite ephemeris",
            )
        } else {
            (
                "spk",
                JPL_BSP_URL,
                magic(&SPK_MAGIC, false),
                "SPK planetary ephemeris",
            )
        }
    } else if filename.ends_with(".bpc") {
        (
            "pck",
            NAIF_PCK_URL,
            magic(&BINARY_PCK_MAGIC, false),
            "binary PCK",
        )
    } else if filename.ends_with(".tpc") {
        ("pck", NAIF_PCK_URL, text_kernel_check(), "text PCK")
    } else if filename.ends_with(".tf") {
        (
            "fk",
            NAIF_FK_SATELLITES_URL,
            text_kernel_check(),
            "frame kernel",
        )
    } else if filename.ends_with(".tls") {
        (
            "lsk",
            NAIF_LSK_URL,
            magic(&LEAP_SECONDS_MAGIC, true),
            "leap seconds kernel",
        )
    } else {
        return None;
    };
    let key = ArtifactKey::new(format!("naif/{subdir}/{filename}")).ok()?;
    Some(
        Artifact::new(key, vec![Source::new(format!("{base_url}{filename}"))])
            .with_check(check)
            .with_provenance(kernel_provenance(format!("NAIF/JPL {kind} {filename}"))),
    )
}

/// The artifact for the Hipparcos main catalogue.
pub fn hipparcos_artifact() -> Artifact {
    Artifact::new(
        ArtifactKey::new(HIPPARCOS_KEY).expect("constant key is valid"),
        vec![Source::new(HIPPARCOS_URL)],
    )
    .with_check(ContentCheck::default_binary())
    .with_provenance(Provenance {
        description: "Hipparcos main catalogue (ESA 1997), hip_main.dat".into(),
        license: HIPPARCOS_LICENSE.into(),
        citation: Some(
            "ESA (1997). The Hipparcos and Tycho Catalogues, ESA SP-1200; CDS I/239".into(),
        ),
    })
}

/// A Gaia data release whose `gaia_source` shards are named artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GaiaRelease {
    Dr1,
    Dr2,
    Dr3,
}

impl GaiaRelease {
    /// Directory the shards are served from, with trailing slash.
    pub fn base_url(self) -> &'static str {
        match self {
            Self::Dr1 => "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/",
            Self::Dr2 => "https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/",
            Self::Dr3 => "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/",
        }
    }

    /// The archive's MD5 manifest for the release: one line per shard. Part
    /// of the release and immutable, so it is an artifact too
    /// ([`gaia_md5sums_artifact`]) and the source of the shard list.
    pub fn md5sums_url(self) -> &'static str {
        match self {
            Self::Dr1 => "https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/MD5SUM.txt",
            Self::Dr2 => "https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/MD5SUM.txt",
            Self::Dr3 => "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/_MD5SUM.txt",
        }
    }

    /// Key prefix shared by every consumer: `gaia/<release>/gaia_source`.
    pub fn key_prefix(self) -> &'static str {
        match self {
            Self::Dr1 => "gaia/dr1/gaia_source",
            Self::Dr2 => "gaia/dr2/gaia_source",
            Self::Dr3 => "gaia/dr3/gaia_source",
        }
    }
}

/// The artifact for a release's MD5 manifest. The per-shard checksums are
/// part of the release and immutable, so they are cached like any artifact;
/// only the HTML directory listing is fetched live.
pub fn gaia_md5sums_artifact(release: GaiaRelease) -> Artifact {
    let url = release.md5sums_url();
    let filename = url.rsplit('/').next().expect("md5sums URL has a file name");
    Artifact::new(
        ArtifactKey::new(format!("{}/{filename}", release.key_prefix())).expect("constant key"),
        vec![Source::new(url)],
    )
    .with_check(ContentCheck::default_binary())
    .with_provenance(Provenance {
        description: format!("Gaia {release:?} gaia_source MD5 manifest {filename}"),
        license: "ESA Gaia data: free use with the DPAC acknowledgement".into(),
        citation: None,
    })
}

/// The artifact for one `gaia_source` shard, e.g.
/// `GaiaSource_000-000-000.csv.gz`, under `gaia/<release>/gaia_source/<file>`.
///
/// Gzipped shards are checked for the gzip signature; the archive's MD5 is
/// verified by the caller after the fetch, as before.
pub fn gaia_artifact(release: GaiaRelease, filename: &str) -> Result<Artifact> {
    if filename.is_empty() || filename.contains('/') || filename.contains("://") {
        return Err(StarfieldError::DataError(format!(
            "Gaia shard name must be a bare filename, got {filename:?}"
        )));
    }
    let check = if filename.ends_with(".gz") {
        ContentCheck::All(vec![
            magic(&GZIP_MAGIC, false),
            ContentCheck::MinBytes(1024),
        ])
    } else {
        ContentCheck::default_binary()
    };
    Ok(Artifact::new(
        key(format!("{}/{filename}", release.key_prefix()))?,
        vec![Source::new(format!("{}{filename}", release.base_url()))],
    )
    .with_check(check)
    .with_provenance(Provenance {
        description: format!("Gaia {release:?} gaia_source shard {filename}"),
        license: "ESA Gaia data: free use with the DPAC acknowledgement".into(),
        citation: Some("Gaia Collaboration; ESA/Gaia/DPAC".into()),
    }))
}

/// The artifact for a file named by a full URL.
///
/// A kernel URL under one of the archive directories starfield knows maps
/// to the same canonical key as the bare filename would, so `Loader::open`
/// with a URL and with a name hit the same object; the caller's exact URL is
/// kept as the source. Any other `http(s)` URL gets the key
/// `url/<sha256 of the URL without its fragment>` and the default content
/// check, so strict mode still holds: nothing is fetched behind the mirror's
/// back. URLs carrying userinfo, or with any other scheme, are refused.
pub fn url_artifact(url: &str) -> Result<Artifact> {
    // The input is not echoed: a malformed URL may still carry a credential.
    let parsed = url::Url::parse(url)
        .map_err(|e| StarfieldError::DataError(format!("not a resolvable URL: {e}")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(StarfieldError::DataError(format!(
            "only http(s) URLs can be resolved, got {}",
            parsed.scheme()
        )));
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(StarfieldError::DataError(
            "a data URL must not carry userinfo; configure credentials for the host instead".into(),
        ));
    }
    let canonical = [
        JPL_BSP_URL,
        NAIF_PLANETS_URL,
        NAIF_PLANETS_OLD_URL,
        NAIF_SATELLITES_URL,
        NAIF_PCK_URL,
        NAIF_FK_SATELLITES_URL,
        NAIF_LSK_URL,
    ]
    .iter()
    .filter_map(|base| url.strip_prefix(base))
    .find(|rest| !rest.is_empty() && !rest.contains(['/', '?', '#']))
    .and_then(kernel_artifact);
    if let Some(mut artifact) = canonical {
        artifact.sources = vec![Source::new(url)];
        return Ok(artifact);
    }
    if url == HIPPARCOS_URL {
        return Ok(hipparcos_artifact());
    }
    let mut without_fragment = parsed.clone();
    without_fragment.set_fragment(None);
    let digest = {
        use sha2::{Digest, Sha256};
        format!("{:x}", Sha256::digest(without_fragment.as_str().as_bytes()))
    };
    let mut origin = without_fragment.clone();
    origin.set_query(None);
    Ok(
        Artifact::new(key(format!("url/{digest}"))?, vec![Source::new(url)]).with_provenance(
            Provenance {
                description: format!("File fetched from {origin}"),
                license: String::new(),
                citation: None,
            },
        ),
    )
}

/// The artifact for anything `Loader::open` accepts: a kernel filename, the
/// Hipparcos catalogue name, or a full URL.
pub fn artifact_for(filename: &str) -> Result<Artifact> {
    if filename.contains("://") {
        return url_artifact(filename);
    }
    if filename == "hip_main.dat" {
        return Ok(hipparcos_artifact());
    }
    kernel_artifact(filename).ok_or_else(|| {
        StarfieldError::DataError(format!(
            "Unknown file '{}'. Provide a recognized filename (e.g. de421.bsp) or a full URL.",
            filename
        ))
    })
}

/// Resolve a file through a caller-configured store. Hermetic: reads no
/// environment and adopts no legacy cache; what the store holds or can reach
/// is all there is.
pub fn download_or_cache_with(store: &Datastore, filename: &str) -> Result<PathBuf> {
    Ok(store.get(&artifact_for(filename)?)?)
}

/// The store the convenience entry points use: configured from the
/// environment and `~/.config/starfield/datastore.toml`, rooted at `data_dir`
/// when the caller gave one.
pub(crate) fn store_for(data_dir: Option<&Path>) -> Result<Datastore> {
    let builder = DatastoreBuilder::from_env()?;
    let builder = match data_dir {
        Some(dir) => builder.cache_root(dir.to_path_buf()),
        None => builder,
    };
    Ok(builder.build()?)
}

/// Adopt a file the pre-datastore downloader left at `legacy` into the store
/// under `artifact`'s key, validated like a download. True when the store now
/// holds the key because of this call. Never touches the network.
pub(crate) fn adopt_legacy_cache_from(
    store: &Datastore,
    artifact: &Artifact,
    legacy: &Path,
) -> bool {
    if store.contains(&artifact.key) {
        return false;
    }
    let present = std::fs::metadata(legacy)
        .map(|m| m.is_file() && m.len() > 0)
        .unwrap_or(false);
    present && store.import(artifact, legacy).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use starfield_datastore::{DatastoreError, Mirror};
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A loopback HTTP server with canned routes, standing in for the
    /// ephemeris server. Never the network.
    struct Stub {
        base: String,
        routes: Arc<Mutex<HashMap<String, (u16, Vec<u8>)>>>,
    }

    impl Stub {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let routes: Arc<Mutex<HashMap<String, (u16, Vec<u8>)>>> = Arc::default();
            let served = routes.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming() {
                    let Ok(mut stream) = stream else { continue };
                    let mut reader = BufReader::new(stream.try_clone().unwrap());
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        continue;
                    }
                    let path = line.split_whitespace().nth(1).unwrap_or("/").to_string();
                    loop {
                        line.clear();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                    }
                    let (status, body) = served
                        .lock()
                        .unwrap()
                        .get(&path)
                        .cloned()
                        .unwrap_or((404, Vec::new()));
                    let _ = write!(
                        stream,
                        "HTTP/1.1 {status} Stub\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(&body);
                }
            });
            Self { base, routes }
        }

        fn serve(&self, key: &str, body: Vec<u8>) {
            self.routes
                .lock()
                .unwrap()
                .insert(format!("/artifact/{key}"), (200, body));
        }

        fn mirror(&self) -> Mirror {
            Mirror::Http {
                base_url: self.base.clone(),
                writable: false,
            }
        }
    }

    fn spk_bytes(seed: u8) -> Vec<u8> {
        let mut bytes = b"DAF/SPK ".to_vec();
        bytes.extend((0..4096).map(|i| (i as u8).wrapping_mul(seed)));
        bytes
    }

    fn store_with_mirror(root: &Path, stub: &Stub) -> Datastore {
        Datastore::builder()
            .cache_root(root.to_path_buf())
            .mirror(stub.mirror())
            .progress(false)
            .build()
            .unwrap()
    }

    #[test]
    fn kernel_artifacts_use_the_shared_key_layout() {
        let cases = [
            ("de421.bsp", "naif/spk/de421.bsp", JPL_BSP_URL),
            (
                "jup365.bsp",
                "naif/spk/satellites/jup365.bsp",
                NAIF_SATELLITES_URL,
            ),
            ("pck00011.tpc", "naif/pck/pck00011.tpc", NAIF_PCK_URL),
            (
                "moon_pa_de421_1900-2050.bpc",
                "naif/pck/moon_pa_de421_1900-2050.bpc",
                NAIF_PCK_URL,
            ),
            (
                "moon_080317.tf",
                "naif/fk/moon_080317.tf",
                NAIF_FK_SATELLITES_URL,
            ),
            ("naif0012.tls", "naif/lsk/naif0012.tls", NAIF_LSK_URL),
        ];
        for (file, key, base) in cases {
            let artifact = kernel_artifact(file).unwrap();
            assert_eq!(artifact.key.as_str(), key);
            assert_eq!(artifact.sources[0].url, format!("{base}{file}"));
            assert_eq!(artifact.provenance.license, KERNEL_LICENSE);
        }
        assert!(kernel_artifact("hip_main.dat").is_none());
        assert!(kernel_artifact("../de421.bsp").is_none());
        assert!(kernel_artifact("https://x/de421.bsp").is_none());
    }

    #[test]
    fn checks_reject_the_wrong_kind_and_accept_both_daf_id_words() {
        let html = "<html><body>login</body></html>".repeat(100);
        let spk = kernel_artifact("de421.bsp").unwrap().check;
        assert!(spk.check(&spk_bytes(1)).is_ok());
        assert!(spk
            .check(&[b"NAIF/DAF".as_slice(), &[0u8; 4096]].concat())
            .is_ok());
        assert!(spk.check(html.as_bytes()).is_err());
        assert!(spk
            .check(&[b" DAF/SPK".as_slice(), &[0u8; 4096]].concat())
            .is_err());
        let tpc = kernel_artifact("pck00011.tpc").unwrap().check;
        assert!(tpc.check(b"\n\nKPL/PCK\n\\begindata\n").is_ok());
        assert!(tpc.check(html.as_bytes()).is_err());
        let tls = kernel_artifact("naif0012.tls").unwrap().check;
        assert!(tls.check(b"KPL/LSK\n").is_ok());
        assert!(tls.check(b"KPL/PCK\n").is_err());
        let gz = gaia_artifact(GaiaRelease::Dr1, "GaiaSource_000-000-000.csv.gz")
            .unwrap()
            .check;
        assert!(gz
            .check(&[[0x1f, 0x8b].as_slice(), &[0u8; 2048]].concat())
            .is_ok());
        assert!(gz.check(html.as_bytes()).is_err());
    }

    #[test]
    fn hipparcos_and_gaia_keys_are_archive_shaped() {
        assert_eq!(hipparcos_artifact().key.as_str(), HIPPARCOS_KEY);
        assert_eq!(hipparcos_artifact().sources[0].url, HIPPARCOS_URL);
        let shard = gaia_artifact(GaiaRelease::Dr3, "GaiaSource_000000-003111.csv.gz").unwrap();
        assert_eq!(
            shard.key.as_str(),
            "gaia/dr3/gaia_source/GaiaSource_000000-003111.csv.gz"
        );
        assert_eq!(
            shard.sources[0].url,
            "https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/GaiaSource_000000-003111.csv.gz"
        );
        assert!(gaia_artifact(GaiaRelease::Dr1, "../x.csv.gz").is_err());
        assert_eq!(
            gaia_md5sums_artifact(GaiaRelease::Dr1).key.as_str(),
            "gaia/dr1/gaia_source/MD5SUM.txt"
        );
        assert_eq!(
            gaia_md5sums_artifact(GaiaRelease::Dr3).key.as_str(),
            "gaia/dr3/gaia_source/_MD5SUM.txt"
        );
    }

    #[test]
    fn urls_map_to_canonical_keys_when_known_and_digest_keys_otherwise() {
        for base in [JPL_BSP_URL, NAIF_PLANETS_URL, NAIF_PLANETS_OLD_URL] {
            let url = format!("{base}de421.bsp");
            let canonical = url_artifact(&url).unwrap();
            assert_eq!(canonical.key.as_str(), "naif/spk/de421.bsp");
            assert_eq!(
                canonical.sources[0].url, url,
                "the caller's URL is the source"
            );
        }
        assert_eq!(
            url_artifact(HIPPARCOS_URL).unwrap().key.as_str(),
            HIPPARCOS_KEY
        );
        assert!(url_artifact(&format!("{JPL_BSP_URL}sub/de421.bsp"))
            .unwrap()
            .key
            .as_str()
            .starts_with("url/"));

        let other = url_artifact("https://example.org/kernels/custom.bsp").unwrap();
        assert!(other.key.as_str().starts_with("url/"), "{}", other.key);
        assert_eq!(other.key.as_str().len(), "url/".len() + 64);
        assert_eq!(
            other.sources[0].url,
            "https://example.org/kernels/custom.bsp"
        );
        assert!(other
            .provenance
            .description
            .contains("https://example.org/kernels/custom.bsp"));
        let different = [
            "https://example.org/kernels/custom.bsp?v=2",
            "http://example.org/kernels/custom.bsp",
            "https://example.org:8443/kernels/custom.bsp",
        ];
        for url in different {
            assert_ne!(other.key, url_artifact(url).unwrap().key, "{url}");
        }
        assert_ne!(
            url_artifact("https://example.org/a:b").unwrap().key,
            url_artifact("https://example.org/a_b").unwrap().key
        );
        assert_eq!(
            other.key,
            url_artifact("https://example.org/kernels/custom.bsp#frag")
                .unwrap()
                .key,
            "a fragment is not part of identity"
        );
        let queried = url_artifact("https://example.org/get?file=custom.bsp&v=2").unwrap();
        assert!(
            !queried.provenance.description.contains("file="),
            "no query in provenance"
        );

        assert!(url_artifact("not a url").is_err());
        assert!(url_artifact("ftp://example.org/x.bsp").is_err());
        assert!(url_artifact("https://user:pw@example.org/x.bsp").is_err());
        assert!(matches!(
            artifact_for("mystery.xyz"),
            Err(StarfieldError::DataError(_))
        ));
    }

    #[test]
    fn resolves_through_the_mirror_and_then_from_disk_without_network() {
        let root = tempfile::tempdir().unwrap();
        let stub = Stub::start();
        stub.serve("naif/spk/de421.bsp", spk_bytes(3));
        let store = store_with_mirror(root.path(), &stub);
        let path = download_or_cache_with(&store, "de421.bsp").unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), spk_bytes(3));

        let offline = Datastore::builder()
            .cache_root(root.path().to_path_buf())
            .offline(true)
            .progress(false)
            .build()
            .unwrap();
        assert_eq!(download_or_cache_with(&offline, "de421.bsp").unwrap(), path);
        assert_eq!(
            download_or_cache_with(&offline, &format!("{JPL_BSP_URL}de421.bsp")).unwrap(),
            path,
            "a URL naming the same kernel is the same object"
        );
    }

    #[test]
    fn an_html_page_under_a_kernel_key_is_refused_and_not_cached() {
        let root = tempfile::tempdir().unwrap();
        let stub = Stub::start();
        stub.serve(
            "naif/pck/pck00011.tpc",
            format!(
                "<!DOCTYPE html><html><body>{}</body></html>",
                "x".repeat(4096)
            )
            .into_bytes(),
        );
        let store = store_with_mirror(root.path(), &stub);
        let err = download_or_cache_with(&store, "pck00011.tpc").unwrap_err();
        assert!(
            matches!(
                err,
                StarfieldError::Datastore(DatastoreError::ContentRejected { .. })
            ),
            "{err}"
        );
        assert!(store
            .peek(&kernel_artifact("pck00011.tpc").unwrap().key)
            .is_none());
    }

    #[test]
    fn a_miss_in_strict_mode_names_the_opt_in() {
        let root = tempfile::tempdir().unwrap();
        let stub = Stub::start();
        let store = store_with_mirror(root.path(), &stub);
        let err = download_or_cache_with(&store, "de440.bsp").unwrap_err();
        assert!(
            err.to_string().contains("STARFIELD_ALLOW_UPSTREAM=1"),
            "{err}"
        );
    }

    #[test]
    fn a_pre_datastore_flat_cache_file_is_adopted_not_redownloaded() {
        let root = tempfile::tempdir().unwrap();
        let legacy = root.path().join("de421.bsp");
        std::fs::write(&legacy, spk_bytes(5)).unwrap();
        let store = Datastore::builder()
            .cache_root(root.path().to_path_buf())
            .offline(true)
            .progress(false)
            .build()
            .unwrap();
        let artifact = kernel_artifact("de421.bsp").unwrap();
        assert!(adopt_legacy_cache_from(&store, &artifact, &legacy));
        assert!(
            !adopt_legacy_cache_from(&store, &artifact, &legacy),
            "second call is a no-op"
        );
        let path = download_or_cache_with(&store, "de421.bsp").unwrap();
        assert_eq!(std::fs::read(path).unwrap(), spk_bytes(5));
        assert!(
            legacy.exists(),
            "adoption copies; the flat file is left alone"
        );

        let corrupt = root.path().join("pck00011.tpc");
        std::fs::write(&corrupt, "<html>not a kernel</html>").unwrap();
        assert!(!adopt_legacy_cache_from(
            &store,
            &kernel_artifact("pck00011.tpc").unwrap(),
            &corrupt
        ));
    }

    #[test]
    #[ignore = "live upstream; must bypass the mirror and cache. Detects archive rot, so a warm or mirrored resolve would defeat it"]
    fn naif_leap_seconds_kernel_resolves_cold_from_upstream() {
        assert_eq!(
            std::env::var("STARFIELD_ALLOW_UPSTREAM").as_deref(),
            Ok("1"),
            "live upstream canary requires STARFIELD_ALLOW_UPSTREAM=1"
        );
        let root = tempfile::tempdir().unwrap();
        let store = Datastore::builder()
            .cache_root(root.path().to_path_buf())
            .allow_upstream(true)
            .progress(false)
            .build()
            .unwrap();
        let path = download_or_cache_with(&store, "naif0012.tls")
            .expect("NAIF upstream failed with STARFIELD_ALLOW_UPSTREAM=1");
        assert!(std::fs::read(path).unwrap().starts_with(b"KPL/LSK"));
    }
}
