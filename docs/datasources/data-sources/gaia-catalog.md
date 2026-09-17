# Gaia Star Catalog

## 1. Overview

Gaia is a space observatory operated by the European Space Agency (ESA), launched on 19 December 2013. Its mission is to create the most precise three-dimensional map of the Milky Way by measuring positions, distances, and motions of stars with unprecedented accuracy.

### Data Releases

| Release | Date | Sources | Notes |
|---------|------|---------|-------|
| DR1 | September 2016 | ~1.1 billion | Positions and G-band magnitudes; limited parallaxes and proper motions (TGAS subset of ~2 million) |
| DR2 | April 2018 | ~1.7 billion | Five-parameter astrometry, radial velocities for ~7 million stars |
| EDR3 | December 2020 | ~1.8 billion | Improved astrometry and photometry; same source list as full DR3 |
| DR3 | June 2022 | ~1.8 billion | Full release: astrophysical parameters, variable star classifications, spectra, non-stellar objects |
| DR4 | Expected ~2026 | TBD | Full mission data, improved time-series, exoplanet orbits |

### Precision

Gaia achieves astrometric precision that varies with source brightness:

| G magnitude | Position precision | Parallax precision | Proper motion precision |
|-------------|-------------------|-------------------|------------------------|
| < 15 | ~20 microarcseconds | ~20 microarcseconds | ~20 microarcseconds/yr |
| 17 | ~100 microarcseconds | ~100 microarcseconds | ~70 microarcseconds/yr |
| 20 | ~700 microarcseconds | ~700 microarcseconds | ~500 microarcseconds/yr |
| 21 | ~2000 microarcseconds | ~2000 microarcseconds | ~1500 microarcseconds/yr |

### Reference Epoch

- **DR1**: J2015.0
- **DR2 and later**: J2016.0

All positions, parallaxes, and proper motions in DR2+ are given at epoch J2016.0 (JD 2457389.0 TDB, which corresponds to 2016-01-01T12:00:00 TDB). To obtain positions at a different epoch, proper motion must be applied.

### Coordinate System

Gaia uses the International Celestial Reference System (ICRS), materialized by the International Celestial Reference Frame (ICRF). Positions are reported as right ascension (ra) and declination (dec) in this frame.

---

## 2. Data Access

### Gaia Archive

The primary access point is the ESA Gaia Archive:

- **Main portal**: https://gea.esac.esa.int/archive/
- **TAP endpoint**: https://gea.esac.esa.int/tap-server/tap
- **Bulk download CDN**: https://cdn.gea.esac.esa.int/Gaia/

### Access Methods

#### TAP/ADQL Queries

The archive exposes a Table Access Protocol (TAP) service that accepts queries written in Astronomical Data Query Language (ADQL). This is the most flexible method for obtaining specific subsets of the catalog.

Example using `curl`:

```bash
curl -X POST "https://gea.esac.esa.int/tap-server/tap/sync" \
  --data "REQUEST=doQuery" \
  --data "LANG=ADQL" \
  --data "FORMAT=csv" \
  --data "QUERY=SELECT TOP 100 source_id, ra, dec, parallax, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < 6.0" \
  -o bright_stars.csv
```

#### Bulk Download via HTTP

Pre-partitioned CSV files are available for direct download from the CDN. Files are organized by data release and table:

```
https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/
https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/
https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/
```

Each release directory contains gzipped CSV files partitioned by HEALPix index or source ID range, along with an `MD5SUM.txt` file for integrity verification.

#### File Naming Conventions

- **DR1**: `GaiaSource_000-000-000.csv.gz` through `GaiaSource_000-020-XXX.csv.gz`
- **DR3**: `GaiaSource_XXXX.csv.gz` where `XXXX` is a zero-padded partition number

### How `starfield-gaia` Accesses Gaia Data

The crate ships **first-class loaders for DR1, DR2, and DR3** via the bulk download CDN. Each release lives in its own module with its own entry type exposing every field that release publishes — no lowest-common-denominator flattening.

| Release | Module | Entry | Catalog | Download base |
|---|---|---|---|---|
| DR1 | `starfield_gaia::dr1` | `Dr1Entry` | `Dr1Catalog` | `https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/` |
| DR2 | `starfield_gaia::dr2` | `Dr2Entry` | `Dr2Catalog` | `https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/` |
| DR3 | `starfield_gaia::dr3` | `Dr3Entry` | `Dr3Catalog` | `https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/` |

Pipeline (identical for every release, parameterized by the [`GaiaRelease`] trait):

1. **Index discovery**: Fetches the HTML directory listing at the release's base URL and extracts filenames via a per-release regex.

2. **Download with caching**: Each gzipped CSV file is downloaded to `~/.cache/starfield/gaia/{dr1,dr2,dr3}/`. `starfield-datasource-utils` writes to `.tmp` first and renames atomically.

3. **MD5 verification**: The release's checksum file (`MD5SUM.txt` for DR1/DR2, `_MD5SUM.txt` for DR3) is downloaded once and consulted for every file.

4. **Streaming Arrow CSV**: Files are kept gzipped on disk. `Dr*Catalog::from_csv_file` pipes through `flate2::GzDecoder` and the `arrow` CSV reader, which uses a typed per-release [`Schema`](arrow::datatypes::Schema) — nullability is declared up front, not inferred, so missing/renamed columns become typed errors. DR3's ECSV `#`-prefixed YAML header is stripped before parsing.

5. **Magnitude filtering at read time**: The caller-supplied `mag_limit` is applied per row as batches stream past, so fainter entries never materialize.

6. **Catalog merging**: `Dr*Catalog::merge(other)` combines two catalogs of the same release; existing `source_id` entries win.

### Cache Layout

```
~/.cache/starfield/gaia/
  dr1/
    MD5SUM.txt
    GaiaSource_000-000-000.csv.gz
    ...
  dr2/
    MD5SUM.txt
    GaiaSource_1000172165251650944_1000424567594791808.csv.gz
    ...
  dr3/
    _MD5SUM.txt
    GaiaSource_000000-003111.csv.gz
    ...
```

---

## 3. Data Fields

Each release's entry type organizes its fields into coherent sub-structs. Every entry embeds a shared [`GaiaCore`] with astrometry and G-band photometry; release-specific data (BP/RP, RV, GSP-Phot, TGAS cross-ids, etc.) lives in additional fields and is wrapped in `Option<_>` when absent.

### Shared `GaiaCore`

Present on every `Dr{1,2,3}Entry.core`:

| Field | Type | Description |
|-------|------|-------------|
| `source_id` | `u64` | Unique source identifier within the release |
| `solution_id` | `u64` | Identifier of the processing pipeline solution |
| `ref_epoch` | `f64` | Reference epoch (J2015.0 for DR1, J2015.5 for DR2, J2016.0 for DR3) |
| `random_index` | `Option<u64>` | Random index for uniform sub-sampling |
| `ra` / `dec` | `f64` | Right ascension / declination, ICRS, at `ref_epoch` |
| `ra_error` / `dec_error` | `f32` | Standard errors (mas) |
| `ra_dec_corr` | `Option<f32>` | Correlation between RA and Dec |
| `parallax` + `_error` | `Option<f64>` / `f32` | Parallax and error (mas) |
| `pmra` + `_error`, `pmdec` + `_error` | `Option<f64>` / `f32` | Proper motion components (mas/yr) |
| `l`, `b` | `f64` | Galactic longitude / latitude (deg) |
| `ecl_lon`, `ecl_lat` | `f64` | Ecliptic longitude / latitude (deg) |
| `phot_g_mean_mag` | `f64` | G-band mean magnitude — used for `mag_limit` filtering |
| `phot_g_mean_flux` + `_error`, `phot_g_n_obs` | `Option<_>` | G-band photometry extras |
| `phot_variable_flag` | `VarFlag` enum | `NotAvailable` / `NotVariable` / `Variable` |
| `astrometric_n_obs_al`, `astrometric_excess_noise` (+ `_sig`) | `Option<_>` | Astrometric quality |
| `astrometric_primary_flag`, `duplicated_source`, `matched_observations` | `Option<_>` | Source-level flags |

### Release-specific sub-structs

| Sub-struct | DR1 | DR2 | DR3 |
|---|:-:|:-:|:-:|
| `AstrometricExtra` (release-specific quality metrics) | ✓ | ✓ | ✓ |
| `ScanDirection` (k1..k4 strengths and means) | ✓ | — | — |
| `TgasBlock` (`hip`, `tycho2_id` — splice via [`attach_tgas`]) | ✓ | — | — |
| `BpRpPhotometry` (BP + RP + color indices) | — | ✓ | ✓ |
| `RadialVelocity` (DR2 form: 6 fields) | — | ✓ | — |
| `RadialVelocityDr3` (adds `rv_method_used`, template metadata) | — | — | ✓ |
| `AstroParams` (Apsis priam + flame: Teff, AG, EBPRP, radius, luminosity with percentile bounds) | — | ✓ | — |
| `IpdQuality` (`ruwe`, `ipd_gof_*`, `ipd_frac_*`) | — | — | ✓ |
| `GspPhot` (Teff, log g, [M/H], distance, extinction) | — | — | ✓ |
| `DataLinks` (`has_xp_continuous`, `has_rvs`, `has_epoch_*`, …) | — | — | ✓ |
| `Classifications` (`in_qso_candidates`, DSC combmod probabilities, `non_single_star`) | — | — | ✓ |

All sub-structs whose fields are published as a correlated block (BP/RP, RV, GSP-Phot, Apsis) are wrapped in `Option<_>` on the entry — `None` means Gaia didn't publish that block for that source.

### Extending

Columns beyond the curated set are easy to add — extend the per-release `COLUMNS` table in `src/dr{1,2,3}/schema.rs`, add fields to the corresponding sub-struct, and populate them in `build_entry`. The Arrow schema is the single source of truth; if a column is absent from `COLUMNS` it simply isn't read.

---

## 4. How `starfield-gaia` Uses It

### Parsing Pipeline

Per-release parsing is driven by three pieces, tied together by the [`GaiaRelease`] trait:

1. **Typed Arrow schema** (`src/dr{1,2,3}/schema.rs`): declares every column's name, Arrow `DataType`, and nullability. This is the single source of truth — missing, renamed, or re-typed columns surface as typed errors at read time, not as silently-wrong values.

2. **Projection**: at `CsvSourceReader::open`, the CSV's actual header row is read and each schema column is looked up by name. The resulting projection maps "schema column N = CSV column M"; real Gaia files carry ~100 columns we don't need, so this is also the place we drop them.

3. **Row constructor** (`build_entry(&RecordBatch, row)` per release): builds the nested `Dr{N}Entry` from the batch using typed accessors in `common/parse.rs`. All-or-nothing blocks like `BpRpPhotometry` / `RadialVelocity` / `GspPhot` are wrapped `None` when Gaia reports their key fields as null, so presence is explicit in the type.

The CSV reader itself is streaming: Arrow delivers `RecordBatch`es of 8192 rows. Mag-limit filtering runs per row in the iterator, so fainter entries never materialize into `Dr{N}Entry` structs.

DR3 files come as ECSV — a `# %ECSV 1.0` YAML preamble before the actual CSV header. `CsvSourceReader::open` auto-detects this from the release's `IS_ECSV` constant and consumes the preamble before handing bytes to Arrow.

### Loading code path

```rust
use starfield_gaia::{Dr3Catalog, prelude::*};
use starfield::catalogs::StarCatalog;

// Load a single file.
let catalog = Dr3Catalog::from_csv_file("GaiaSource_000000-003111.csv.gz", 18.0)?;
println!("{} stars brighter than G=18", catalog.len());

// Merge chunks.
let mut combined = Dr3Catalog::new();
for path in starfield_gaia::dr3::list_cached()? {
    combined.merge(Dr3Catalog::from_csv_file(&path, 18.0)?);
}

// Field-level access via the nested sub-structs.
if let Some(star) = combined.get_star(1234567890123456789) {
    if let Some(bp_rp) = &star.bp_rp {
        println!("BP-RP color: {:?}", bp_rp.bp_rp);
    }
    if let Some(rv) = &star.radial_velocity {
        println!("RV: {:?} km/s", rv.radial_velocity);
    }
}
# Ok::<(), starfield::StarfieldError>(())
```

### DR1 TGAS cross-ids

DR1 publishes astrometry (parallax, proper motion) only for the ~2M TGAS sources, and the Hipparcos / Tycho-2 cross-ids live in a separate `tgas_source` catalog. Load them and splice:

```rust
use starfield_gaia::dr1::{Dr1Catalog, load_tgas_block_map};

let mut catalog = Dr1Catalog::from_csv_file("GaiaSource_000-000-000.csv.gz", 20.0)?;
let tgas_map = load_tgas_block_map("TgasSource_000-000-000.csv.gz")?;
catalog.attach_tgas(&tgas_map);   // Entries not in the map are left untouched.
# Ok::<(), starfield::StarfieldError>(())
```

### Download strategy

Per-release entry points live in each module: `starfield_gaia::dr3::download_file`, `download_all(max_files)`, `list_cached()`. They delegate to the generic [`Downloader<R: GaiaRelease>`], which handles the regex-driven index scrape, atomic `.tmp` downloads, and MD5 verification against the release's checksum file. Failures on individual files log to stderr and don't abort the batch.

### `StarCatalog` trait implementation

`Dr{1,2,3}Catalog` all implement upstream `starfield::catalogs::StarCatalog` via the shared `GaiaCatalogBase<R>`. Methods:

- `get_star(id)`: lookup by `source_id` (cast to `u64`)
- `stars()`: iterator over all `Dr{N}Entry` references
- `star_data()`: converts to the generic `StarData` type. `b_v` is populated from `BpRpPhotometry::bp_rp` when available (DR2/DR3); `None` for DR1.
- `brighter_than(magnitude)`: filter to `Vec<StarData>`
- `stars_in_field(ra, dec, fov)`: cone search via great-circle distance

### Derived calculations

`GaiaCore` provides:
- **`unit_vector()`**: RA/Dec → ICRS Cartesian unit vector.
- **`cartesian_position()`**: 3D position in parsecs via `1000.0 / parallax_mas`. `None` when parallax is absent or non-positive.

---

## 5. Query Strategies

### ADQL Syntax

ADQL (Astronomical Data Query Language) is a SQL-like language for querying astronomical databases. It extends SQL with geometric functions for celestial coordinates.

#### Cone Search (Circle on the Sky)

Find all sources within a radius of a given position:

```sql
SELECT source_id, ra, dec, parallax, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE 1 = CONTAINS(
    POINT('ICRS', ra, dec),
    CIRCLE('ICRS', 83.633, -5.55, 1.0)
)
```

This queries a 1-degree radius circle centered on the Orion Nebula region.

#### Box Search (Rectangle on the Sky)

```sql
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE ra BETWEEN 80.0 AND 85.0
  AND dec BETWEEN -10.0 AND 0.0
```

For regions not crossing the RA=0/360 boundary this is simpler than a polygon query. For regions crossing RA=0, use `OR` logic or the `POLYGON` function.

#### Magnitude-Limited Query

```sql
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
       radial_velocity, teff_gspphot
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < 10.0
ORDER BY phot_g_mean_mag ASC
```

#### Nearby Stars (Parallax-Based Distance Filter)

```sql
SELECT source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag,
       1000.0/parallax AS distance_pc
FROM gaiadr3.gaia_source
WHERE parallax > 0
  AND parallax_over_error > 10
  AND parallax > 50.0  -- closer than 20 parsecs
ORDER BY parallax DESC
```

The `parallax_over_error > 10` filter ensures reliable distance estimates.

#### Cross-Match with Other Catalogs

Gaia DR3 provides pre-computed cross-matches with other major catalogs:

```sql
-- Cross-match with Hipparcos
SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag,
       h.hip, h.hpmag
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.hipparcos2_best_neighbour AS h
  ON g.source_id = h.source_id
WHERE g.phot_g_mean_mag < 6.0

-- Cross-match with Tycho-2
SELECT g.source_id, t.original_ext_source_id AS tycho_id
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.tycho2tdsc_merge_best_neighbour AS t
  ON g.source_id = t.source_id
```

Available cross-match tables include:
- `hipparcos2_best_neighbour`
- `tycho2tdsc_merge_best_neighbour`
- `panstarrs1_best_neighbour`
- `sdssdr13_best_neighbour`
- `allwise_best_neighbour`
- `tmass_psc_best_neighbour` (2MASS)

#### High Proper Motion Stars

```sql
SELECT source_id, ra, dec, pmra, pmdec,
       SQRT(pmra*pmra + pmdec*pmdec) AS total_pm,
       phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE pmra IS NOT NULL
  AND pmdec IS NOT NULL
  AND SQRT(pmra*pmra + pmdec*pmdec) > 500.0
ORDER BY total_pm DESC
```

### Pagination for Large Results

The Gaia TAP service limits synchronous queries to relatively small result sets (~3 million rows). For larger queries, use asynchronous mode:

1. Submit an asynchronous job via the TAP `/async` endpoint.
2. Poll the job status until completion.
3. Download the result file.

Alternatively, partition queries by sky region (HEALPix level) or source ID range:

```sql
-- Partition by HEALPix level 5 pixel
SELECT source_id, ra, dec, phot_g_mean_mag
FROM gaiadr3.gaia_source
WHERE source_id BETWEEN 0 AND 562949953421311  -- First HEALPix L12 range
  AND phot_g_mean_mag < 15.0
```

The `source_id` in Gaia encodes the HEALPix level-12 pixel index in its most significant bits, so range queries on `source_id` correspond to spatial regions on the sky.

---

## 6. Data Volume Considerations

### Full Catalog Sizes

| Release | Table | Compressed Size | Uncompressed Size | Sources |
|---------|-------|----------------|-------------------|---------|
| DR1 | `gaia_source` | ~12 GB (gzipped CSV) | ~55 GB | ~1.1 billion |
| DR3 | `gaia_source` | ~650 GB (gzipped CSV) | ~2.5 TB | ~1.8 billion |

### Practical Subsets by Magnitude Limit

Approximate source counts and download sizes for magnitude-limited subsets of DR3:

| G magnitude limit | Approx. sources | Approx. CSV size (uncompressed) | Approx. CSV size (gzipped) |
|-------------------|-----------------|-------------------------------|--------------------------|
| < 6.0 | ~9,000 | ~5 MB | ~1 MB |
| < 8.0 | ~80,000 | ~40 MB | ~8 MB |
| < 10.0 | ~650,000 | ~350 MB | ~70 MB |
| < 12.0 | ~5 million | ~2.5 GB | ~500 MB |
| < 14.0 | ~40 million | ~20 GB | ~4 GB |
| < 16.0 | ~250 million | ~120 GB | ~25 GB |
| < 18.0 | ~800 million | ~400 GB | ~80 GB |
| < 21.0 (full) | ~1.8 billion | ~2.5 TB | ~650 GB |

These numbers are approximate. For starfield's use cases (star field rendering, astrometry), magnitude limits of 12-16 are typically sufficient and keep data volumes manageable.

### Strategies for Reducing Data Volume

1. **Column selection**: Query only the columns you need. The full `gaia_source` table has 152 columns; starfield uses 17. Requesting fewer columns via ADQL dramatically reduces download size.

2. **Spatial partitioning**: Download only the sky regions relevant to your use case rather than the full sky.

3. **Quality filters**: Exclude low-quality measurements:
   ```sql
   WHERE parallax_over_error > 5
     AND ruwe < 1.4
     AND astrometric_excess_noise < 1.0
   ```

4. **Source type filtering**: Exclude non-stellar sources if only stars are needed:
   ```sql
   WHERE classprob_dsc_combmod_star > 0.9
   ```

---

## 7. Practical Notes

### Rate Limits and Etiquette

The Gaia Archive does not publish strict rate limits, but users should follow these guidelines:

- **Synchronous queries**: Limited to results under ~3 million rows (configurable by the archive). Queries exceeding this limit will fail and should be resubmitted as asynchronous jobs.
- **Concurrent connections**: Keep concurrent downloads to a reasonable number (2-4 simultaneous connections). The CDN can handle more, but aggressive parallelism may result in throttling or IP bans.
- **Query complexity**: Avoid extremely complex JOIN operations on the full `gaia_source` table. Use indexed columns (`source_id`, `ra`, `dec`, `phot_g_mean_mag`, `parallax`) in WHERE clauses.
- **Large result sets**: Use asynchronous TAP jobs for queries expected to return more than a few hundred thousand rows.

### Download Strategies for Large Datasets

1. **Incremental download**: The starfield downloader processes files sequentially, which is resilient to interruptions. Already-downloaded files are detected by the cache and skipped on retry. Use the `max_files` parameter to limit batch sizes.

2. **Checksum verification**: Always verify downloads against the `MD5SUM.txt` file provided in each data release directory. The starfield downloader does this automatically.

3. **Temporary file pattern**: Write downloads to a `.tmp` file first, then rename atomically. This prevents partially downloaded files from being treated as complete. The starfield downloader implements this pattern.

4. **Compression**: Keep gzipped files on disk and decompress during parsing. This roughly halves storage requirements with minimal performance cost due to the I/O-bound nature of CSV parsing.

### Cache Management

Starfield stores cached Gaia files at `~/.cache/starfield/gaia/`. To manage this cache:

- **List cached files**: Use `list_cached_gaia_files()` from `src/data/gaia_downloader.rs` to enumerate all cached `.csv` and `.csv.gz` files.
- **Clear cache**: Delete the `~/.cache/starfield/gaia/` directory to force re-download.
- **Disk usage**: Monitor cache size, especially if downloading large portions of the catalog. A full DR1 download is ~12 GB compressed.

### Known Limitations in Current Implementation

1. **No ADQL client**: `starfield-gaia` does not include a TAP/ADQL client. For custom subset queries, use external tools (TOPCAT, `astroquery`, `curl`) to obtain CSVs, then load via `Dr{1,2,3}Catalog::from_csv_file`.

2. **B-V only from BP-RP**: `StarData::b_v` is populated from `BpRpPhotometry::bp_rp` for DR2/DR3 (BP-RP is a useful but not identical proxy for Johnson B-V). DR1 entries return `None` since no comparable color is published.

3. **Sequential downloads**: `Downloader::download_all` processes files one at a time. For full-catalog retrieval, parallel downloads would improve throughput.

4. **Curated column set**: For DR3 we expose ~95 columns covering every documented sub-product (BP/RP, RV, RUWE, IPD, GSP-Phot, datalink flags, classifications). The long tail of per-correlation fields and obscure quality metrics is not wired up — extending is straightforward (add to `COLUMNS` + sub-struct + `build_entry`).

5. **No non-`gaia_source` tables**: Variability time series, XP spectra, epoch RV, and cross-match tables are out of scope.

### Useful External Tools

- **TOPCAT** (http://www.star.bris.ac.uk/~mbt/topcat/): Desktop application for interactive catalog exploration and TAP queries.
- **astroquery** (Python): `astroquery.gaia` module provides programmatic TAP access.
- **Aladin Lite** (https://aladin.u-strasbg.fr/AladinLite/): Browser-based sky viewer with Gaia overlay support.
- **VizieR** (https://vizier.u-strasbg.fr/): Alternative access point for Gaia tables with pre-built queries.
