# Pull-through artifact cache

On the tailnet:

```sh
export STARFIELD_MIRROR=http://cf-services.tail944341.ts.net:8080
```

This is runtime configuration, never a compiled-in endpoint. Named downloads
resolve local cache, then mirror, then upstream only with
`STARFIELD_ALLOW_UPSTREAM=1`. `STARFIELD_OFFLINE=1` disables both network layers.
`STARFIELD_CACHE_DIR` selects the datastore root. With no mirror and no opt-in,
a fresh miss is an error naming the opt-in variable; it never silently bypasses
the configured service.

| Consumer | Archive key or path |
|---|---|
| Hipparcos catalogue and kernel helper | Delegates to starfield's constructors and Loader path; `cds/I/239/hip_main.dat`, `naif/spk/...`, `naif/pck/...`, `naif/fk/...`, `naif/lsk/...` |
| Planet spectra | `pds/gbat_0001/{1993,1995low,1995high}.tab` |
| Planet maps | `usgs/mosaic/<archive filename>` or `nasa/blue-marble/<filename>` |
| Gaia DR1/DR2/DR3 | `gaia/<dr>/gaia_source/<filename>` including release MD5 lists |
| Gaia tools / gaia-extended | Use the same Gaia downloader; derived excerpts and crossmatches stay local |
| MAST FITS products | `mast/<archive URI after mast:>`; URL-only products use a URL digest |
| Spectral-library regeneration | `usgs/splib07/ASCIIdata_splib07a.zip`, from `manifests/build-inputs.toml` |
| Generic `download_to_file` | A URL-derived identity; prefer named artifact APIs for shared server keys |

Convenience download functions validate and import existing flat cache files.
Explicit `_with(&Datastore, ...)` variants resolve only through that store, so
tests can create an empty, mirror-free store without a global file satisfying
the request. Existing legacy files are not deleted on rejection.

Returned paths are store-managed blobs unless the API promises a filename alias.
Gaia's ordinary downloader retains extension-bearing aliases because parsers
use `.csv.gz` to select decompression. It hard-links when possible, otherwise
streams a copy across filesystems. Do not modify a returned raw file in place.
`Downloader::remove_cached` removes the alias and datastore entry for explicit
cleanup; `gaia-excerpt --clean-after-excerpt` uses it for downloaded inputs.
`stream_file` uses one temporary artifact store on the configured cache
filesystem, validates the full shard before returning a reader, and removes
the temporary directory on drop. Budget disk for a complete shard even without
`--cache-raw`; memory is bounded independently of file size. This path creates
no filename alias; `remove_cached` applies to the ordinary downloader's cache.

## Server registration

The server accepts only registered keys. An unregistered product returns a
mirror miss; the client will not grant the server arbitrary upstream URLs.
Export the same artifact descriptions used by clients:

```sh
cargo run -p starfield-datasources --all-features --example datastore_manifest -- \
  --base path/to/ephemeris.toml --base manifests/build-inputs.toml \
  --kernel de430t.bsp --kernel de405.bsp --kernel jup365.bsp \
  --kernel moon_pa_de440_200625.bpc --hipparcos \
  --gaia dr1 path/to/dr1-MD5SUM.txt \
  --gaia dr2 path/to/dr2-MD5SUM.txt \
  --gaia dr3 path/to/dr3-_MD5SUM.txt > serve.toml
```

Gaia checksum lists come from each release's `gaia_source` archive directory;
the exporter reads them as local input and downloads no shard bodies. For MAST,
serialize the selected `Vec<DataProduct>` from `data_products` to JSON and add
`--mast products.json 'reviewed redistribution license'`. A product's access
and redistribution terms must be reviewed before registration; an empty
license blocks server redistribution. Archive kind checks remain mandatory
even when a SHA256 pin has not yet been measured.

Deploy the generated allow-list as the **served manifest**, for `serve` only.
Never pass this manifest to `mirror`: that would download terabytes. Keep the **prewarm
manifest** limited to the curated small kernels: never schedule the full Gaia
catalogue or multi-GB mosaics for nightly prewarming. New MAST observations
require a reviewed registration; they are not automatically authorized by a
query result on a client. Derived data and arbitrary URLs are not automatically
added to the server.

The Python spectral-library builder needs the CLI:

```sh
cargo install starfield-datastore --version 0.1.1 --features cli
python3 crates/reflectance-library/scripts/build_table.py > regenerated.csv
```

## Direct network calls retained after the audit

- HORIZONS, SBDB, MPC, Rubin brokers, MAST search/product listing and Gaia
  TAP queries are query APIs, not immutable artifacts. Gaia file discovery uses
  the cached release MD5 manifest, so it works offline after that manifest is cached.
- The solar-spectrum regeneration script queries LASP by wavelength range.
  Its embedded output needs no runtime downloads. Embedded reflectance,
  spectra, map tiers, bright-galaxy tables and local transformations likewise
  need no runtime fetch path.
- NSA's NYU endpoint has an incomplete TLS chain. Its existing one-off direct
  downloader remains the explicit exception; no datastore client or mirror
  weakens TLS. Remove this exception when the archive serves a valid chain.
- Upstream-rot canaries deliberately build cold, mirror-free stores (or issue
  direct HEAD checks). They must fail if the archive is unavailable. Tests of
  live query behavior remain direct. Neither is a mirror health check.

CI runs offline workspace tests, all features, consumer lint gates and the
minimal facade build. Loopback-stub integration tests exercise actual map,
Gaia and MAST download methods with upstream disabled; live archive tests are
explicitly ignored, never conditionally skipped.
