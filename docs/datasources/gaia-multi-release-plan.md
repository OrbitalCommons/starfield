# Gaia Multi-Release Loader — Implementation Plan

Status: implemented 2026-04-16 (all phases complete)
Owner: matt@exclosure.io

## 1. Goals

- First-class loaders for Gaia **DR1**, **DR2**, and **DR3** in the `starfield-gaia` crate.
- One `Entry` struct per release that exposes **every field the release publishes** (no "lowest common denominator" flattening).
- Shared code wherever the releases genuinely overlap (core astrometry, HTTP + MD5 plumbing, parquet reader harness) — never at the cost of hiding release-specific fields.
- Parquet as the primary on-disk format. No custom CSV parsing on the hot path.
- Keep DR1 working; it has real users.

## 2. Non-goals

- Gaia's non-`gaia_source` tables (`gaia_universe_model`, `rvs_mean_spectrum`, `xp_continuous_mean_spectrum`, etc.). Out of scope for this plan.
- TAP/ADQL queries against the Gaia archive. File-based loading only.
- Backwards-compat shims for the old top-level `GaiaCatalog` / `GaiaEntry`. Break them; the crate is pre-1.0.

## 3. Type names

| Concept | DR1 | DR2 | DR3 |
|---|---|---|---|
| Module | `dr1` | `dr2` | `dr3` |
| Entry struct | `Dr1Entry` | `Dr2Entry` | `Dr3Entry` |
| Catalog struct | `Dr1Catalog` | `Dr2Catalog` | `Dr3Catalog` |
| Release marker (zero-sized) | `Dr1` | `Dr2` | `Dr3` |
| `Release` enum variant | `Release::Dr1` | `Release::Dr2` | `Release::Dr3` |
| Cache subdir | `gaia/dr1/` | `gaia/dr2/` | `gaia/dr3/` |
| Cargo feature | `dr1` | `dr2` | `dr3` |
| Reference epoch | J2015.0 | J2015.5 | J2016.0 |

### Sub-structs embedded in each `Entry`

| Sub-struct | DR1 | DR2 | DR3 |
|---|---|---|---|
| `GaiaCore` (shared) | ✓ | ✓ | ✓ |
| `TgasBlock` (DR1 TGAS-only astrometry + PM) | ✓ | — | — |
| `BpRpPhotometry` | — | ✓ | ✓ |
| `RadialVelocity` (DR2 form) | — | ✓ | — |
| `RadialVelocityDr3` (adds `rv_method_used`, template params) | — | — | ✓ |
| `AstroParams` (DR2 Apsis priam/flame) | — | ✓ | — |
| `AstrometricExtra` (chi2, n_good_obs, priors_used, ...) | — | ✓ | ✓ |
| `IpdQuality` (RUWE, `ipd_gof_*`, `ipd_frac_*`) | — | — | ✓ |
| `ScanDirection` (`scan_direction_strength/mean_k1..4`) | — | — | ✓ |
| `GspPhot` (distance, teff, logg, mh, ebpminrp) | — | — | ✓ |
| `DataLinks` (`has_xp_*`, `has_rvs`, `has_epoch_*`) | — | — | ✓ |
| `Classifications` (`in_qso_candidates`, `non_single_star`, ...) | — | — | ✓ |

All-or-nothing groups are `Option<Sub>`. Partial-presence fields are `Option<T>` inside the sub-struct.

## 4. Module layout

```
crates/gaia/src/
├── lib.rs
├── common/
│   ├── mod.rs
│   ├── core.rs         # GaiaCore
│   ├── traits.rs       # GaiaSource, GaiaRelease, Release enum, VarFlag
│   └── reader.rs       # ParquetSourceReader<R>: projection + streaming
├── download/
│   ├── mod.rs
│   ├── client.rs       # HTTP + MD5 (lifted from current downloader.rs)
│   ├── release.rs      # per-DR URL / filename / MD5 conventions
│   └── transcode.rs    # CSV.gz → parquet, only if a DR lacks native parquet
├── dr1/
│   ├── mod.rs
│   ├── entry.rs        # Dr1Entry + TgasBlock + DR1-only helpers
│   ├── catalog.rs      # Dr1Catalog (thin wrapper)
│   └── schema.rs       # column names, build_entry(&RecordBatch, row)
├── dr2/
│   ├── mod.rs
│   ├── entry.rs
│   ├── catalog.rs
│   └── schema.rs
└── dr3/
    ├── mod.rs
    ├── entry.rs
    ├── catalog.rs
    └── schema.rs
```

## 5. Shared traits

```rust
// common/traits.rs
pub enum Release { Dr1, Dr2, Dr3 }

pub trait GaiaSource {
    fn core(&self) -> &GaiaCore;
    fn release(&self) -> Release;
}

pub trait GaiaRelease: 'static {
    const RELEASE: Release;
    const BASE_URL: &'static str;
    const MD5_FILENAME: &'static str;
    const FILE_REGEX: &'static str;   // scrape pattern for index page
    const CACHE_SUBDIR: &'static str; // "gaia/dr3"

    type Entry: GaiaSource + Send + 'static;

    fn projection(schema: &arrow::datatypes::Schema) -> Result<parquet::arrow::ProjectionMask>;
    fn build_entry(batch: &arrow::record_batch::RecordBatch, row: usize) -> Result<Self::Entry>;
}
```

`Dr1Catalog`, `Dr2Catalog`, `Dr3Catalog` are each a thin newtype over a shared generic `GaiaCatalogBase<R: GaiaRelease>` that owns `HashMap<u64, R::Entry>`, `mag_limit`, and implements `StarCatalog`, `merge`, `brighter_than`. The per-release newtype exists so users can write `Dr3Catalog::from_parquet(path, 18.0)` without turbofish.

## 6. Reader strategy — Arrow CSV

Phase 0 probe result (2026-04-16): **ESA's CDN hosts CSV.gz only** for gaia_source across DR1/DR2/DR3. No native parquet; no `parquet/` sibling directories. Mirrors weren't pursued — they add trust surface we don't need.

**Decision:** skip parquet entirely. Read CSV.gz through the `arrow` CSV reader with a typed `Schema` per release. This keeps the schema-driven design we wanted (typed columns, null handling, projection, `RecordBatch`-based `build_entry`) without any transcode step.

### What we keep from the original parquet plan

- Null handling in the schema — `Option<f64>` maps to `Array::is_null(i)`.
- Column projection — declared per release; only projected columns are parsed.
- Streaming batches — `arrow::csv::Reader` yields `RecordBatch`es; DR3 files stay memory-bounded.
- Schema guard — missing/renamed columns surface as arrow errors at build time, not silent drift.

### What we give up vs. parquet

- No random access, no filter pushdown into file storage. Mag-limit filtering happens in our per-batch loop, not at the I/O layer. Acceptable — Gaia usage is load-once, work-in-RAM.

### Reader contract

```rust
pub struct CsvSourceReader<R: GaiaRelease> { /* ... */ }

impl<R: GaiaRelease> CsvSourceReader<R> {
    /// Open a .csv or .csv.gz file; gzip is detected by extension.
    pub fn open(path: &Path, mag_limit: f64) -> Result<Self>;
}

impl<R: GaiaRelease> Iterator for CsvSourceReader<R> {
    type Item = Result<R::Entry>;
}
```

Each `GaiaRelease` impl declares:
- `fn arrow_schema() -> SchemaRef` — the columns we parse.
- `fn build_entry(batch: &RecordBatch, row: usize) -> Result<Self::Entry>` — row-wise constructor.

## 7. Public surface

```rust
// lib.rs
pub mod common;
#[cfg(feature = "dr1")] pub mod dr1;
#[cfg(feature = "dr2")] pub mod dr2;
#[cfg(feature = "dr3")] pub mod dr3;

pub use common::{GaiaCore, GaiaSource, Release};
#[cfg(feature = "dr1")] pub use dr1::{Dr1Catalog, Dr1Entry};
#[cfg(feature = "dr2")] pub use dr2::{Dr2Catalog, Dr2Entry};
#[cfg(feature = "dr3")] pub use dr3::{Dr3Catalog, Dr3Entry};

pub mod prelude {
    pub use crate::{GaiaCore, GaiaSource, Release};
    #[cfg(feature = "dr1")] pub use crate::{Dr1Catalog, Dr1Entry};
    #[cfg(feature = "dr2")] pub use crate::{Dr2Catalog, Dr2Entry};
    #[cfg(feature = "dr3")] pub use crate::{Dr3Catalog, Dr3Entry};
}
```

Old top-level `GaiaCatalog` and `GaiaEntry` are removed.

## 8. Cargo features

```toml
# crates/gaia/Cargo.toml
[features]
default = ["dr3"]
dr1 = []
dr2 = []
dr3 = []
all-releases = ["dr1", "dr2", "dr3"]
```

Facade crate (`starfield-datasources`) exposes a new `gaia-all` convenience feature that forwards `all-releases`; its existing `gaia` feature stays default-on and maps to `dr3`.

## 9. Work breakdown (phases → todos)

### Phase 0 — Probes & scaffolding
- **0.1** Probe ESA CDN for DR1/DR2 parquet availability; pick mirror-vs-transcode path.
- **0.2** Add `arrow` + `parquet` to workspace dependencies.
- **0.3** Bump `starfield-gaia` to `0.2.0`; document in CHANGELOG.
- **0.4** Create the module skeleton (empty files at the paths in §4).

### Phase 1 — Shared foundation
- **1.1** `common/core.rs` — `GaiaCore` + `VarFlag` enum.
- **1.2** `common/traits.rs` — `Release`, `GaiaSource`, `GaiaRelease`.
- **1.3** `common/reader.rs` — `ParquetSourceReader<R>` with projection + mag-limit filter.
- **1.4** `download/client.rs` — lift HTTP + MD5 helpers from current `downloader.rs`, generalize over cache subdir.
- **1.5** `download/release.rs` — `GaiaRelease` impls for `Dr1`, `Dr2`, `Dr3` (URLs, regex, MD5 filename).
- **1.6** Shared `GaiaCatalogBase<R>` + `StarCatalog` impl in `common/`.

### Phase 2 — DR3 (reference implementation)
- **2.1** `dr3/entry.rs` — `Dr3Entry`, all sub-structs.
- **2.2** `dr3/schema.rs` — column list, `build_entry` from arrow `RecordBatch`.
- **2.3** `dr3/catalog.rs` — `Dr3Catalog` newtype.
- **2.4** Unit tests: parser round-trip on a small committed parquet snippet.
- **2.5** Ignored integration test: download one file, load, sanity-check count.

### Phase 3 — DR2
- **3.1** `dr2/entry.rs` — `Dr2Entry` + DR2 sub-structs.
- **3.2** `dr2/schema.rs`.
- **3.3** `dr2/catalog.rs`.
- **3.4** Tests (unit + ignored integration).
- **3.5** If Phase 0 decided transcode: wire `download/transcode.rs` for DR2.

### Phase 4 — DR1
- **4.1** `dr1/entry.rs` — `Dr1Entry` + `TgasBlock`.
- **4.2** `dr1/schema.rs`.
- **4.3** `dr1/catalog.rs`.
- **4.4** Tests.
- **4.5** Transcode wiring if needed.

### Phase 5 — Facade + polish
- **5.1** Update `starfield-datasources` facade features; bump to `0.5.0`.
- **5.2** Rewrite `docs/data-sources/gaia-catalog.md` for multi-release API.
- **5.3** Update any `examples/` that use the old `GaiaCatalog` API.
- **5.4** Run `cargo test --workspace --all-features`; fix fallout.

## 10. Versioning

- `starfield-gaia`: **0.1.0 → 0.2.0** (breaking API).
- `starfield-datasources` (facade): **0.4.0 → 0.5.0** (re-exports change, new `gaia-all` feature). Flagged for confirmation; CLAUDE.md's minor-bump rule is specifically for new crates, so patch would also be defensible.

## 11. Open questions

1. **Parquet for DR1/DR2** — resolved in Phase 0.
2. **Synthetic catalogs** — currently `GaiaCatalog::create_synthetic` exists and is used in downstream tests. Plan: port it to DR3 only; drop the DR1/DR2 synthetic paths unless a consumer asks.
3. **`StarCatalog` trait** — confirm the upstream `starfield` crate's trait can be implemented generically for `GaiaCatalogBase<R>`, or whether each DR needs its own impl.
