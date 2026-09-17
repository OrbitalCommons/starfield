# Changelog

## Pull-through cache rollout — facade 0.13.1

- Rubin 0.1.1 updates AMPEL endpoints to the hostname in the project's
  [current service documentation](https://ampelproject.github.io/astronomy/ztf/index).
  Live broker URL checks continue to fail on unavailable services.

- Gaia 0.2.1; datasource-utils, gaia-tools, Hipparcos, MAST, planet-maps and
  planet-spectra 0.1.1 resolve named archive files through starfield-datastore.
  Set `STARFIELD_MIRROR=http://cf-services.tail944341.ts.net:8080`; upstream
  downloads require an explicit `STARFIELD_ALLOW_UPSTREAM=1` opt-in.
- Existing files are validated before adoption; caller-configured `_with`
  methods never discover the global legacy cache. Wrong HTML, signatures,
  size pins and optional SHA256 pins are rejected. Gaia checks MD5 on hits too.
- Gaia's ordinary downloader retains `.csv.gz` filename aliases for parsers;
  explicit cleanup removes both alias and datastore entry. `stream_file` now
  validates a complete file on the cache filesystem before reading, then
  removes temporary storage when the reader closes. It no longer exposes an
  unvalidated HTTP response; memory remains bounded for large shards.
- `MapProduct::cached_path` now errors on a cache miss rather than predicting
  a flat filename. `KarkoschkaTable::cached_path` now takes `&Datastore` and
  returns `Option<PathBuf>`; downloaded paths can be content-addressed.
- The USGS spectral-library regeneration script uses the datastore CLI and
  checked-in build-input manifest instead of buffering an archive download.
- A manifest exporter shares the client artifact constructors, supports Gaia
  checksum lists and explicit MAST product registrations. A served allow-list
  must remain separate from the small scheduled prewarm manifest.
- Live query APIs, upstream-rot canaries, and the documented NSA broken-TLS
  downloader stay direct. See `docs/pullthrough-cache.md` for the full audit.

## Unreleased — datastore seam

### starfield-planet-spectra (breaking)

Step 5 of the `starfield-datastore` rollout begins here.

- `KarkoschkaTable::download` resolves local cache → organisation mirror →
  upstream, reaching upstream only under `STARFIELD_ALLOW_UPSTREAM=1`
- **Breaking:** `cached_path(product) -> Result<PathBuf>` becomes
  `cached_path(&Datastore, product) -> Option<PathBuf>`. It now *peeks* at the
  store rather than computing a path, so it answers "is this cached" instead of
  "where would it go". The crate is unpublished, so no released consumer breaks
- New `download_with(&Datastore, product)` for a caller-configured store
- A pre-seam cache at `~/.cache/starfield/karkoschka/` is **adopted, not
  orphaned**: the first call after upgrading imports it rather than
  re-downloading a file the user already has. Import runs the content check, so
  a corrupt or wrong-kind legacy file is refused rather than promoted
- Artifact keys are archive-shaped (`pds/gbat_0001/1995low.tab`), so a relocated
  upstream changes a `Source` and not the cache layout or a pinned digest

### Test conventions

- Upstream-rot canary ignore reasons now read "must bypass the mirror and cache"
  rather than "must not be re-pointed at the datastore". They *do* use the
  datastore now — in a cold, mirrorless configuration with `allow_upstream` set
  in code — so the old wording had become misleading

## 0.14.0 — Earth composition tier and a shared sampling interface

### starfield-planet-maps

- `AbundanceTier`: per-texel endmember abundances, so colour varies across the
  disk rather than only brightness. The embedded Earth tier is MODIS MCD12C1
  sub-pixel IGBP fractions collapsed onto the shipped endmembers, 1440x720 x 9
  endmembers at 0.25 deg, 0.91 MB
- The tier file is **self-describing**: grid geometry, registration, NAIF id and
  provenance live in its header, so a loader that has never heard of MCD12C1
  cannot misregister it by half a cell or mirror it. Payload length is declared
  and checked, so a truncated file fails at load rather than rendering as a
  dark stripe
- `SurfaceSampler` gives one interface over both backings — a consumer should
  not branch on whether a body happens to have real composition data. Both use
  the same level-selection policy, so a tier-backed and a map-backed body do not
  blur differently at the same geometry
- `AlbedoMap` takes its endmember band mean at construction rather than per
  sample: it is a property of (endmember, band), and taking it once means
  `sample_area` cannot fail for a reason unrelated to the position asked about

### starfield-reflectance-library

- `Endmember::AridSoil` — a real arid field sample for barren terrain. Mapping
  barren onto beach sand would bias Earth's integrated colour blue by roughly
  the Rayleigh term, since barren is a large fraction of the illuminated disk
- `Endmember::Shade` — identically zero reflectance, the standard
  spectral-mixture-analysis shade term and the only synthetic member of the
  enum. splib07's vegetation is leaf-level, and a leaf is far brighter than a
  canopy: pure green vegetation integrates to 0.218 shortwave against published
  forest means of 0.10-0.15, so without it Earth's vegetated surfaces cannot be
  matched to measured albedos at all
- `Endmember::Asphalt`, `Endmember::Concrete` — urban cover is strongly bimodal
  in brightness, so two endmembers rather than one average that represents
  neither

## 0.13.0 — Planetary albedo maps

### starfield-planet-maps 0.1.0 (new)

focalplane D1.3. The spatial half of resolved-body appearance: where the light
comes from on the disk.

- `MapGrid` stores each product's longitude sense, latitude definition, `lon0`
  and row order as data and converts internally. Callers pass east-positive
  planetocentric radians — what `SubPoint::to_planetocentric` gives — and never
  convert. A map registered with the wrong handedness is mirrored, and a
  mirrored Mars looks plausible until checked against an ephemeris
- Landmark tests name real features rather than checking algebra, because the
  algebra is symmetric under exactly that error: Olympus Mons and Tycho must
  land on specific pixels
- `AlbedoMap` with a box-filtered mip pyramid; `sample_area` takes the **sky**
  footprint and the emission cosine separately and does the projection
  internally, so the `1/mu` clamp and the level choice stay together
- `MU_FLOOR` bounds the limb singularity at cos 87°; `SampleFootprint` exposes
  the chosen level, the footprint in texels, the anisotropy and whether the
  clamp fired, so a consumer can assert on terminator behaviour
- Level selection uses the stretched axis: never aliases, over-blurs across the
  minor axis near the limb. True anisotropic filtering is a follow-up rather
  than something faked with a scalar level
- `PhotometricBand` is required, not optional. A scalar mosaic is a reflectance
  *in some band*, and converting it to an `EndmemberMix` is only defined against
  that band
- `PRODUCTS` is a pinned catalogue — USGS's CKAN endpoint returns HTML, not
  JSON, so there is no discovery API, and pinning is right for a data source
  anyway. Five products, URLs and sizes verified by fetch
- Jupiter is deliberately absent and `require_product_for` says so: USGS has no
  controlled global mosaic for it and will not, since there is no solid surface
  to control a photogrammetric network to

### Facade crate

- New `planet-maps` feature (not default)

## 0.12.0 — Endmember reflectance library

### starfield-reflectance-library 0.1.0 (new)

focalplane D1.2. Multiplying a monochrome map by one disk-integrated spectrum
gives a body whose colour is uniform and only its brightness varies. Endmembers
are what make colour vary across the disk.

- Nine curated endmembers from USGS Spectral Library 7 (Kokaly et al. 2017,
  doi:10.5066/F7RR1WDJ): open ocean, coastal water, green and dry vegetation,
  sand, snow, water ice, fresh and weathered basalt
- `scripts/build_table.py` regenerates the table from the ScienceBase archive,
  pairing each spectrum with its spectrometer's wavelength grid and dropping
  splib07's deleted-channel sentinel
- `Endmember` is the stable id shared with the map crate; `EndmemberMix` is what
  the sampler will return, with a uniform surface as a one-endmember mix
- Mix weights are not renormalised — a partially covered texel is legitimately
  short, and renormalising would hide the missing fraction
- `EndmemberMix` evaluation returns `None` if *any* component lacks coverage,
  rather than silently averaging the subset that happens to have data
- `Endmember::WaterIce` covers 859 nm and longward only; splib07 has no
  visible water-ice spectrum. Not padded — the accessors report the gap
- Tests assert real spectroscopy: the vegetation red edge (12.4x from 680 to
  750 nm, absent in dry vegetation), snow bright in the visible and 50x darker
  in the SWIR, ocean dark and blue, basalt dark and flat

### starfield-datasource-utils

- `SampledCurve::weighted_mean(lo, hi, step, response)` — midpoint-rule integral
  against a filter or detector response, normalised so a flat response reproduces
  `mean_over`. The step is explicit because the right value depends on the
  response, not on the curve: a narrow filter needs a fine step even across a
  coarsely sampled curve
- New `SampledCurve`: shared wavelength-sampled curve with one implementation of
  interpolation and band means. `SpectralAlbedo` and `Reflectance` now both wrap
  it, so their band-mean semantics cannot drift apart — a renderer combines
  values from both in a single expression, and a disagreement about partially
  covered bands would surface as an unexplained photometric error

### Facade crate

- New `reflectance-library` feature (not default)

## 0.11.0 — Solar reference spectrum

### starfield-solar-spectrum 0.1.0 (new)

focalplane D1.1. The illumination half of resolved-body rendering: a reflectance
is dimensionless and becomes photons only once multiplied by the solar spectrum.

- TSIS-1 Hybrid Solar Reference Spectrum v2 (Coddington et al. 2023,
  doi:10.1029/2022EA002637), the recognised international reference standard
- Embedded at 1 nm box means over the full archive range, 202–2730 nm (2528 bins,
  86 KB), with per-bin archive uncertainty
- `scripts/build_table.py` regenerates it from LASP LISIRD, streaming the native
  0.001–0.01 nm product in chunks and reducing as it goes
- Resampled from the **native** product, not LISIRD's pre-smoothed 0.1 nm sibling:
  averaging the smoothed product differs from the true box mean by up to 1.1%
  across strong Fraunhofer lines, larger than the archive's own 0.3% uncertainty
- `at_nm` returns the containing bin rather than interpolating, because
  interpolating between box means of a line-blanketed spectrum implies a
  resolution the data does not have
- `irradiance_over` / `mean_over` / `weighted_irradiance` weight partial end bins
  proportionally; `weighted_irradiance` takes a filter or QE response
- `scale_factor_at_au` keeps the 1/r² conversion visible at call sites
- Integrating the table recovers 1325.8 W m⁻² — 97.4% of the solar constant,
  matching the archive's stated coverage, and asserted in the tests

### Facade crate

- New `solar-spectrum` feature (not default)

## 0.10.0 — Planetary spectral albedos

### starfield-planet-spectra 0.1.0 (new)

First crate of the resolved-planetary-appearance effort described in
`docs/planetary-textures-plan.md`. Ships the spectral geometric albedos a
focal-plane renderer needs to give a resolved planet the right colour.

- Karkoschka (1994, 1998) full-disk albedo spectra from PDS Atmospheres volume
  `gbat_0001` (`ESO-J/S/N/U-SPECTROPHOTOMETER-4-V2.0`, DOI `10.17189/2bp8-k793`)
- `1995low.tab` embedded verbatim — 300–1050 nm at 0.4 nm sampling, spanning the
  whole silicon detector response, for Jupiter, Saturn, Uranus, Neptune and Titan
- `KarkoschkaTable::download` fetches the 0.1 nm-sampled `1995high` and the 1993
  reduction into the starfield cache
- `SpectralAlbedo` with linear interpolation and exact piecewise-linear band means;
  refuses to extrapolate or to average a partially covered band
- `AlbedoKind` distinguishes geometric albedo from full-disk albedo at non-zero
  phase — the archive tabulates Jupiter at 6.8° and Saturn at 5.7°, and conflating
  the two misstates the brightness of the two most prominent targets
- `SpectralBody` is keyed on NAIF id so moons (Titan, 606) are representable
- Methane absorption coefficients exposed alongside the albedos
- Regression tests assert the 727/887 nm methane band structure and the blue/red
  slopes of the ice giants and Titan, not just the table shape

### Facade crate

- New `planet-spectra` feature (not default)

## 0.5.0 — Gaia DR1/DR2/DR3 multi-release support

### starfield-gaia 0.1.0 → 0.2.0 (breaking)

The crate now ships first-class loaders for **DR1, DR2, and DR3**. Each release has
its own `Entry` type exposing every field that release publishes, organized into
coherent sub-structs (BP/RP photometry, radial velocity, IPD quality, GSP-Phot,
datalink flags, etc.). The legacy top-level `GaiaCatalog` / `GaiaEntry` are gone.

- Per-release types: `Dr1Catalog` / `Dr1Entry`, `Dr2Catalog` / `Dr2Entry`, `Dr3Catalog` / `Dr3Entry`
- Shared `GaiaCore` (embedded in every entry) for astrometry + G-band photometry
- Generic `GaiaRelease` trait parameterizes the reader, downloader, and in-memory catalog
- Arrow CSV reader with typed per-release schema — missing/renamed columns become typed errors
- Automatic ECSV (`# %ECSV 1.0`) preamble stripping for DR3 files
- DR1 TGAS cross-ids supported via `load_tgas_block_map` + `Dr1Catalog::attach_tgas`
- Cargo features: `dr1`, `dr2`, `dr3` (default), `all-releases`

### Facade crate

- New `gaia-all` feature: forwards to `starfield-gaia/all-releases`
- Default `gaia` feature remains DR3-only

### Migration

| Old API (0.1.0) | New API (0.2.0) |
|---|---|
| `GaiaCatalog::from_file(path, mag)` | `Dr1Catalog::from_csv_file(path, mag)` (or `Dr2Catalog` / `Dr3Catalog`) |
| `GaiaEntry { ra, dec, phot_g_mean_mag, parallax, pmra, pmdec, ... }` | `entry.core.{ra, dec, phot_g_mean_mag, parallax, pmra, pmdec, ...}` |
| `GaiaEntry::unit_vector()`, `cartesian_position()` | `entry.core.unit_vector()`, `entry.core.cartesian_position()` |
| `download_gaia_file(name)` / `download_gaia_catalog(max)` | `dr3::download_file(name)` / `dr3::download_all(max)` (or `dr1::` / `dr2::`) |
| `list_cached_gaia_files()` | `dr3::list_cached()` (per release) |

## 0.4.0

### Removed

- Removed `starfield-jplephem` crate — jplephem has been reintegrated into the main `starfield` crate
- `starfield-horizons` now uses `starfield::jplephem::SpiceKernel` directly
- Removed `jplephem` feature flag from the `starfield-datasources` facade crate

## 0.3.0 — Rubin Observatory Alert Brokers

Added `starfield-rubin` crate with typed Rust clients for all seven Vera C. Rubin Observatory LSST community alert brokers.

### Brokers

| Broker | Auth | Module |
|--------|------|--------|
| ALeRCE | None | `alerce` |
| ANTARES | None (search) | `antares` |
| Fink | None | `fink` |
| Lasair | API token | `lasair` |
| Pitt-Google | GCP credentials | `pitt_google` |
| AMPEL | Bearer token (archive) / None (catalog) | `ampel` |
| Babamul | API token (invitation only) | `babamul` |

### CI

- Added GitHub Actions workflow with `cargo check`, `cargo test`, `cargo fmt --check`
- Added URL verification job that checks all broker API and documentation URLs resolve

### Other

- Facade crate bumped to 0.3.0 with new `rubin` feature flag (enabled by default)

## 0.1.0 — Initial Release

Extracted all astronomical data source clients from the [starfield](https://github.com/OrbitalCommons/starfield) monolith into independent crates.

### Crates

| Crate | Description | Tests |
|---|---|---|
| `starfield-horizons` | NASA JPL HORIZONS API client (vectors, observer, elements, approach, SPK generation) | 50 |
| `starfield-sbdb` | NASA JPL Small-Body Database API client (11 endpoints: lookup, CAD, fireball, sentry, scout, mission design, radar, identification, observability, NHATS) | 52 |
| `starfield-gaia` | ESA Gaia star catalog loader and downloader (DR1 CSV/gzip) | 2 |
| `starfield-hipparcos` | Hipparcos star catalog loader and downloader | 9 |
| `starfield-datasources` | Facade crate re-exporting all of the above behind feature flags | — |

### Migration from `starfield`

The data source code was extracted from these locations in `starfield`:

| Old import (starfield) | New import (this repo) |
|---|---|
| `starfield::horizons::HorizonsClient` | `starfield_horizons::HorizonsClient` |
| `starfield::horizons::EphemerisRequest` | `starfield_horizons::EphemerisRequest` |
| `starfield::horizons::parser::*` | `starfield_horizons::parser::*` |
| `starfield::sbdb::SbdbClient` | `starfield_sbdb::SbdbClient` |
| `starfield::sbdb::types::*` | `starfield_sbdb::types::*` |
| `starfield::sbdb::query::SbdbQueryParams` | `starfield_sbdb::query::SbdbQueryParams` |
| `starfield::catalogs::gaia::GaiaCatalog` | `starfield_gaia::GaiaCatalog` |
| `starfield::data::gaia_downloader::*` | `starfield_gaia::downloader::*` |
| `starfield::catalogs::hipparcos::HipparcosCatalog` | `starfield_hipparcos::HipparcosCatalog` |
| `starfield::data::download_hipparcos` | `starfield_hipparcos::download_hipparcos` |

### Dependency Direction

- `starfield-horizons` depends on `starfield` (for `StarfieldError`, `Result`, `SpiceKernel`)
- `starfield-sbdb` depends on `starfield` (for `StarfieldError`, `Result`)
- `starfield-gaia` depends on `starfield` (for `StarfieldError`, `Result`, `StarCatalog`, `StarData`)
- `starfield-hipparcos` depends on `starfield` (for `StarfieldError`, `Result`, `StarCatalog`, `StarData`)

### Using the Facade Crate

Add `starfield-datasources` to get everything, or pick individual crates:

```toml
# Everything
[dependencies]
starfield-datasources = "0.1"

# Or pick what you need
[dependencies]
starfield-horizons = "0.1"
```

Feature flags on the facade crate: `horizons`, `sbdb`, `gaia`, `hipparcos` (all enabled by default).
