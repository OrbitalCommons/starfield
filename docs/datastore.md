# Data files through the datastore

Since 0.16.2 every named data file starfield fetches — SPICE kernels, the
Hipparcos catalogue, Gaia `gaia_source` shards and their MD5 manifests —
resolves through [`starfield-datastore`](https://github.com/OrbitalCommons/starfield-datastore),
the organisation's pull-through cache. The chain is:

```
local disk  →  ephemeris server (STARFIELD_MIRROR)  →  upstream archive
                                                        (only with STARFIELD_ALLOW_UPSTREAM=1)
```

Nothing that fails validation is ever cached: each kernel type is checked
against starfield's own magic numbers (`TEXT_MAGIC_NUMBERS`, the DAF ID words)
before the bytes land, so an HTML login page can never sit under a `.bsp`
key. The parser validates the format afterwards, as before.

## Configuration

| Setting | Env | Meaning |
|---|---|---|
| Mirror | `STARFIELD_MIRROR=http://cf-services.tail944341.ts.net:8080` | The ephemeris server. On the tailnet this is all a client needs. |
| Upstream | `STARFIELD_ALLOW_UPSTREAM=1` | Permit direct archive downloads: outside contributors, GitHub-hosted CI. |
| Offline | `STARFIELD_OFFLINE=1` | Local disk only. What CI sets once the cache is warm. |
| Cache root | `STARFIELD_CACHE_DIR` | Default `~/.cache/starfield`, shared with the old downloader. |

A file the pre-0.16.2 downloader left under its plain name in the cache
directory (`~/.cache/starfield/de421.bsp`, `hip_main.dat`, a Gaia shard) is
adopted on first use — validated, then copied into the content-addressed
store — so nobody re-downloads anything on the day of the switch.

`Loader::with_data_dir(dir)` roots the store at `dir` instead; a flat file
already there is adopted the same way.

## Keys

Keys are archive-shaped and identical in every consumer and on the server, so
a relocated upstream changes a source URL, never a key:

| File | Key |
|---|---|
| `de421.bsp`, `de440.bsp`, … | `naif/spk/<file>` |
| `jup365.bsp` (satellite SPKs) | `naif/spk/satellites/<file>` |
| `pck00011.tpc`, `moon_pa_de421_1900-2050.bpc` | `naif/pck/<file>` |
| `moon_080317.tf` | `naif/fk/<file>` |
| `naif0012.tls` | `naif/lsk/<file>` |
| `hip_main.dat` | `cds/I/239/hip_main.dat` |
| `GaiaSource_….csv.gz`, `MD5SUM.txt` | `gaia/<dr1|dr2|dr3>/gaia_source/<file>` |
| any other `http(s)` URL | `url/<sha256 of the URL without fragment>` |

`starfield::data::{kernel_artifact, hipparcos_artifact, gaia_artifact,
gaia_md5sums_artifact, url_artifact, artifact_for}` build these; use them
rather than spelling a key by hand.

## Supplying the store

`download_or_cache` and the `Loader` methods configure a store from the
environment. Tests and embedding applications pass their own:

```rust,no_run
use starfield::data::download_or_cache_with;
use starfield_datastore::{Datastore, Mirror};

let store = Datastore::builder()
    .cache_root("/tmp/kernels".into())
    .mirror(Mirror::Http { base_url: "http://cf-services:8080".into(), writable: false })
    .build()?;
let path = download_or_cache_with(&store, "de421.bsp")?;
# Ok::<(), starfield::StarfieldError>(())
```

`download_or_cache_with` is hermetic: it reads no environment and adopts no
legacy cache.

## What stays direct

Query-shaped services — HORIZONS, SBDB — are not artifacts and are unchanged.
The Gaia archive's HTML directory listing is fetched live to discover shard
names; the shards and MD5 manifests themselves are artifacts. A resolved
shard is exposed under its archive name in `~/.cache/starfield/gaia/` (a hard
link to the validated blob), so readers that detect gzip from the `.gz`
suffix keep working. A build with
`default-features = false` keeps the old direct downloader.

## Live tests

The default suite touches no network. The upstream-rot canaries stay pointed
at the archives (`#[ignore = "live upstream; must bypass the mirror and
cache…"]`) and fail, never skip, when `STARFIELD_ALLOW_UPSTREAM=1` is unset:

```
STARFIELD_ALLOW_UPSTREAM=1 cargo test -p starfield-core -- --ignored naif_leap_seconds_kernel_resolves_cold_from_upstream
```
