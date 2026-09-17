# Hipparcos Star Catalog

## 1. Overview

The Hipparcos catalog is the product of the European Space Agency's (ESA) **Hipparcos** space astrometry mission, which operated from 1989 to 1993. The satellite performed high-precision measurements of stellar positions, proper motions, and parallaxes from a heliocentric orbit, free from the atmospheric distortions that limit ground-based astrometry.

Key facts:

- **Stars cataloged:** 118,218 entries in the main catalog (`hip_main.dat`)
- **Positional accuracy:** ~1 milliarcsecond (mas) for bright stars
- **Reference epoch:** J1991.25 (the mean observation epoch)
- **Coordinate system:** ICRS (International Celestial Reference System)
- **Original publication:** 1997 (ESA SP-1200)
- **Revised reduction:** 2007 by Floor van Leeuwen, improving parallaxes by up to a factor of 4 for bright stars
- **Superseded by:** Gaia DR3 for most applications, though Hipparcos remains the reference for bright-star proper motions over a ~25-year baseline when combined with Gaia

The Hipparcos catalog served as the primary positional reference for stellar astronomy from 1997 until the Gaia mission published its first data release in 2016.

## 2. Data Format

The main catalog file is `hip_main.dat`, a fixed-width ASCII text file with one record per star. Each line contains 78 fields separated by pipe (`|`) characters. Fields that have no measured value for a given star are left blank.

### Format characteristics

- **Encoding:** ASCII text
- **Record separator:** newline (`\n`)
- **Field separator:** pipe (`|`)
- **Total fields per record:** 78 (H0 through H77)
- **Typical line length:** ~450 characters
- **Total records:** 118,218
- **File size:** approximately 51 MB uncompressed

### Reference epoch

All positional data (RA, Dec, proper motions) are referred to epoch **J1991.25** in the **ICRS** frame. The ICRS is essentially aligned with the FK5/J2000 equatorial system to within the uncertainties of the FK5 catalog. To propagate positions to another epoch, apply proper motion corrections:

```
RA(t)  = RA(J1991.25)  + pmRA  * (t - 1991.25) / cos(Dec)
Dec(t) = Dec(J1991.25) + pmDE  * (t - 1991.25)
```

where `pmRA` is the proper motion in RA already multiplied by `cos(Dec)` (field H12), and time `t` is in Julian years.

## 3. Download Source

### Primary URL (used by starfield)

```
https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat
```

This is the CDS (Centre de Donnees astronomiques de Strasbourg) FTP mirror, accessed over HTTPS. The starfield downloader stores this URL as the constant `HIPPARCOS_URL` in `src/data/downloader.rs`.

### Mirror URLs

| Source | URL |
|--------|-----|
| CDS FTP (primary) | `https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat` |
| CDS VizieR catalog page | `https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=I/239` |
| ESA Hipparcos archive | `https://www.cosmos.esa.int/web/hipparcos/catalogues` |

### File sizes

| File | Size |
|------|------|
| `hip_main.dat` (uncompressed) | ~51 MB |
| `hip_main.dat.gz` (gzipped) | ~15 MB |

The starfield downloader fetches the uncompressed `.dat` file directly. The download function reports approximately 36 MB to the user (this varies depending on transfer encoding).

### Cache location

Starfield caches the file at:

```
~/.cache/starfield/hip_main.dat
```

The downloader also checks for a `hip_main.dat` in the project root (useful for CI environments) and copies it to the cache if found.

## 4. Field Reference

The following table documents all 78 fields in `hip_main.dat`. The "Pipe Index" column gives the zero-based index when the line is split on the `|` character, which is how the starfield parser accesses fields. The "Bytes" column gives the 1-based byte positions from the original CDS format specification.

| Pipe Index | Field ID | Bytes | Format | Units | Label | Description |
|:----------:|:--------:|:-----:|:------:|:-----:|:-----:|:------------|
| 0 | H0 | 1 | A1 | --- | Catalog | Catalogue identifier (always `H`) |
| 1 | H1 | 9-14 | I6 | --- | HIP | Hipparcos identifier number |
| 2 | H2 | 16 | A1 | --- | Proxy | Proximity flag (`H` = HIP, `T` = Tycho) |
| 3 | H3 | 18-28 | A11 | --- | RAhms | Right ascension in h m s (ICRS, J1991.25) |
| 4 | H4 | 30-40 | A11 | --- | DEdms | Declination in d ' " (ICRS, J1991.25) |
| 5 | H5 | 42-46 | F5.2 | mag | Vmag | Johnson V magnitude |
| 6 | H6 | 48 | I1 | --- | VarFlag | Coarse variability flag (1, 2, or 3) |
| 7 | H7 | 50 | A1 | --- | r_Vmag | Source of V magnitude (`G`=ground, `H`=HIP, `T`=Tycho) |
| 8 | H8 | 52-63 | F12.8 | deg | RAdeg | Right ascension in decimal degrees (ICRS, J1991.25) |
| 9 | H9 | 65-76 | F12.8 | deg | DEdeg | Declination in decimal degrees (ICRS, J1991.25) |
| 10 | H10 | 78 | A1 | --- | AstroRef | Reference flag for astrometry |
| 11 | H11 | 80-86 | F7.2 | mas | Plx | Trigonometric parallax |
| 12 | H12 | 88-95 | F8.2 | mas/yr | pmRA | Proper motion in RA (mu_alpha * cos(delta)) |
| 13 | H13 | 97-104 | F8.2 | mas/yr | pmDE | Proper motion in Dec (mu_delta) |
| 14 | H14 | 106-111 | F6.2 | mas | e_RAdeg | Standard error in RA*cos(Dec) |
| 15 | H15 | 113-118 | F6.2 | mas | e_DEdeg | Standard error in Dec |
| 16 | H16 | 120-125 | F6.2 | mas | e_Plx | Standard error in parallax |
| 17 | H17 | 127-132 | F6.2 | mas/yr | e_pmRA | Standard error in pmRA |
| 18 | H18 | 134-139 | F6.2 | mas/yr | e_pmDE | Standard error in pmDE |
| 19 | H19 | 141-145 | F5.2 | --- | DE:RA | Correlation coefficient, Dec/RA*cos(delta) |
| 20 | H20 | 147-151 | F5.2 | --- | Plx:RA | Correlation coefficient, Plx/RA*cos(delta) |
| 21 | H21 | 153-157 | F5.2 | --- | Plx:DE | Correlation coefficient, Plx/Dec |
| 22 | H22 | 159-163 | F5.2 | --- | pmRA:RA | Correlation coefficient, pmRA/RA*cos(delta) |
| 23 | H23 | 165-169 | F5.2 | --- | pmRA:DE | Correlation coefficient, pmRA/Dec |
| 24 | H24 | 171-175 | F5.2 | --- | pmRA:Plx | Correlation coefficient, pmRA/Plx |
| 25 | H25 | 177-181 | F5.2 | --- | pmDE:RA | Correlation coefficient, pmDE/RA*cos(delta) |
| 26 | H26 | 183-187 | F5.2 | --- | pmDE:DE | Correlation coefficient, pmDE/Dec |
| 27 | H27 | 189-193 | F5.2 | --- | pmDE:Plx | Correlation coefficient, pmDE/Plx |
| 28 | H28 | 195-199 | F5.2 | --- | pmDE:pmRA | Correlation coefficient, pmDE/pmRA |
| 29 | H29 | 201-203 | I3 | % | F1 | Percentage of rejected data |
| 30 | H30 | 205-209 | F5.2 | --- | F2 | Goodness-of-fit parameter |
| 31 | H31 | 211-216 | I6 | --- | HIP (rep) | HIP number (repeated for convenience) |
| 32 | H32 | 218-223 | F6.3 | mag | BTmag | Mean BT magnitude (Tycho photometry) |
| 33 | H33 | 225-229 | F5.3 | mag | e_BTmag | Standard error on BTmag |
| 34 | H34 | 231-236 | F6.3 | mag | VTmag | Mean VT magnitude (Tycho photometry) |
| 35 | H35 | 238-242 | F5.3 | mag | e_VTmag | Standard error on VTmag |
| 36 | H36 | 244 | A1 | --- | m_BTmag | Reference flag for BT and VT mag |
| 37 | H37 | 246-251 | F6.3 | mag | B-V | Johnson B-V color index |
| 38 | H38 | 253-257 | F5.3 | mag | e_B-V | Standard error on B-V |
| 39 | H39 | 259 | A1 | --- | r_B-V | Source of B-V (`G`=ground, `T`=Tycho) |
| 40 | H40 | 261-264 | F4.2 | mag | V-I | Cousins V-I color index |
| 41 | H41 | 266-269 | F4.2 | mag | e_V-I | Standard error on V-I |
| 42 | H42 | 271 | A1 | --- | r_V-I | Source of V-I |
| 43 | H43 | 273 | A1 | --- | CombMag | Flag for combined V mag, B-V, V-I |
| 44 | H44 | 275-281 | F7.4 | mag | Hpmag | Median magnitude in Hipparcos system |
| 45 | H45 | 283-288 | F6.4 | mag | e_Hpmag | Standard error on Hpmag |
| 46 | H46 | 290-294 | F5.3 | mag | Hpscat | Scatter of Hpmag |
| 47 | H47 | 296-298 | I3 | --- | o_Hpmag | Number of observations for Hpmag |
| 48 | H48 | 300 | A1 | --- | m_Hpmag | Reference flag for Hpmag |
| 49 | H49 | 302-306 | F5.2 | mag | Hpmax | Hpmag at maximum brightness (5th percentile) |
| 50 | H50 | 308-312 | F5.2 | mag | HPmin | Hpmag at minimum brightness (95th percentile) |
| 51 | H51 | 314-320 | F7.2 | d | Period | Variability period in days |
| 52 | H52 | 322 | A1 | --- | HvarType | Variability type (C/D/M/P/R/U) |
| 53 | H53 | 324 | A1 | --- | moreVar | Additional variability data flag (1/2) |
| 54 | H54 | 326 | A1 | --- | morePhoto | Light curve annex flag (A/B/C) |
| 55 | H55 | 328-337 | A10 | --- | CCDM | CCDM (Catalog of Components of Double and Multiple Stars) identifier |
| 56 | H56 | 339 | A1 | --- | n_CCDM | Historical status flag for CCDM |
| 57 | H57 | 341-342 | I2 | --- | Nsys | Number of entries with same CCDM |
| 58 | H58 | 344-345 | I2 | --- | Ncomp | Number of components in this entry |
| 59 | H59 | 347 | A1 | --- | MultFlag | Double/multiple systems flag (C/G/O/V/X) |
| 60 | H60 | 349 | A1 | --- | Source | Astrometric source flag (P/F/I/L/S) |
| 61 | H61 | 351 | A1 | --- | Qual | Solution quality flag (A/B/C/D/S) |
| 62 | H62 | 353-354 | A2 | --- | m_HIP | Component identifiers |
| 63 | H63 | 356-358 | I3 | deg | theta | Position angle between components |
| 64 | H64 | 360-366 | F7.3 | arcsec | rho | Angular separation between components |
| 65 | H65 | 368-372 | F5.3 | arcsec | e_rho | Standard error on angular separation |
| 66 | H66 | 374-378 | F5.2 | mag | dHp | Magnitude difference of components |
| 67 | H67 | 380-383 | F4.2 | mag | e_dHp | Standard error on magnitude difference |
| 68 | H68 | 385 | A1 | --- | Survey | Survey star flag (`S` if survey star) |
| 69 | H69 | 387 | A1 | --- | Chart | Identification chart flag (`D`/`G`) |
| 70 | H70 | 389 | A1 | --- | Notes | Notes existence flag |
| 71 | H71 | 391-396 | I6 | --- | HD | Henry Draper catalog number |
| 72 | H72 | 398-407 | A10 | --- | BD | Bonner Durchmusterung identifier |
| 73 | H73 | 409-418 | A10 | --- | CoD | Cordoba Durchmusterung identifier |
| 74 | H74 | 420-429 | A10 | --- | CPD | Cape Photographic Durchmusterung identifier |
| 75 | H75 | 431-434 | F4.2 | mag | (V-I)red | V-I color index used for reductions |
| 76 | H76 | 436-447 | A12 | --- | SpType | Spectral type string |
| 77 | H77 | 449 | A1 | --- | r_SpType | Source of spectral type |

### Variability type codes (H52)

| Code | Meaning |
|:----:|:--------|
| C | Constant (no detected variability) |
| D | Duplicity-induced variability |
| M | Micro-variable |
| P | Periodic variable |
| R | Revised color index |
| U | Unsolved variable |

### Double/multiple systems flag (H59)

| Code | Meaning |
|:----:|:--------|
| C | Component solution (individual positions) |
| G | Acceleration (orbital motion detected) |
| O | Orbital solution available |
| V | Variability-induced mover (VIM) |
| X | Stochastic solution |

### Solution quality (H61)

| Code | Meaning |
|:----:|:--------|
| A | Reliable (percentage of rejected data < 10%) |
| B | Moderately reliable |
| C | Unreliable |
| D | Very unreliable |
| S | Suspected non-single star |

## 5. How Starfield Uses It

### Parsing approach

The Hipparcos parser lives in `src/catalogs/hipparcos.rs`. It reads `hip_main.dat` line by line using a buffered reader, splitting each line on the `|` delimiter and extracting fields by their pipe-delimited index.

```rust
let fields: Vec<&str> = line.split('|').collect();
```

Lines are skipped if they:
- Are shorter than 110 characters (insufficient data)
- Have fewer than 10 pipe-delimited fields
- Have an unparseable HIP number (field index 1)
- Have an unparseable magnitude (field index 5)
- Have unparseable RA or Dec (field indices 8 and 9)

### Fields extracted

The parser extracts 8 fields into a `HipparcosEntry` struct:

| Struct Field | Pipe Index | Catalog Field | Type | Required |
|:------------|:----------:|:--------------|:----:|:--------:|
| `hip` | 1 | HIP | `usize` | Yes |
| `mag` | 5 | Vmag | `f64` | Yes |
| `ra` | 8 | RAdeg | `f64` | Yes |
| `dec` | 9 | DEdeg | `f64` | Yes |
| `parallax` | 11 | Plx | `Option<f64>` | No |
| `pm_ra` | 12 | pmRA | `Option<f64>` | No |
| `pm_dec` | 13 | pmDE | `Option<f64>` | No |
| `b_v` | 37 | B-V | `Option<f64>` | No |

Required fields cause the line to be skipped if they cannot be parsed. Optional fields default to `None` when absent or unparseable.

### Magnitude filtering

The `from_dat_file` method accepts a `mag_limit` parameter. Stars with `Vmag > mag_limit` are silently excluded (they are not counted as "skipped" in the diagnostic output). The default magnitude limit used by `CatalogSource::Hipparcos` is **8.0**, which retains approximately 40,000 stars visible to the naked eye and binoculars.

### StarData conversion

The `StarCatalog` trait implementation maps each `HipparcosEntry` to a `StarData` struct:

```rust
StarData::new(star.hip as u64, star.ra, star.dec, star.mag, star.b_v)
```

This provides a uniform interface across catalog types. The `StarData` struct stores the position as an `Equatorial` coordinate (internally in radians), plus magnitude and optional B-V color index.

### StarCatalog trait

`HipparcosCatalog` implements the full `StarCatalog` trait, providing:

- `get_star(id)` -- lookup by HIP number
- `stars()` -- iterate over all loaded entries
- `len()` -- count of loaded entries
- `filter(predicate)` -- filter entries by arbitrary predicate
- `star_data()` -- iterate entries as `StarData`
- `filter_star_data(predicate)` -- filter as `StarData`
- `brighter_than(magnitude)` -- convenience filter by magnitude
- `stars_in_field(ra, dec, fov)` -- find stars within a circular field of view

### Cartesian positions

`HipparcosEntry` provides `unit_vector()` and `cartesian_position()` methods. The `cartesian_position()` method converts parallax to distance in parsecs (`1000 / parallax_mas`) and scales the unit direction vector. Returns `None` if parallax is missing or non-positive.

### Storage

Stars are stored in a `HashMap<usize, HipparcosEntry>` keyed by HIP number, providing O(1) lookup by identifier.

## 6. Related Catalogs

### Tycho-2 Catalog

- **Stars:** 2,539,913
- **Positional accuracy:** ~25 mas at mean epoch, ~60 mas at J2000
- **Proper motion accuracy:** ~2.5 mas/yr
- **Origin:** Derived from Hipparcos satellite star mapper data combined with ground-based Astrographic Catalogue
- **Relation to Hipparcos:** Includes all Hipparcos stars plus additional fainter stars. Tycho-2 proper motions are derived from the ~100-year baseline between the Astrographic Catalogue and the Hipparcos epoch
- **CDS identifier:** I/259

### Gaia DR3

- **Stars:** ~1.8 billion sources
- **Positional accuracy:** ~0.01-0.5 mas (depending on magnitude)
- **Parallax accuracy:** ~0.01-0.5 mas
- **Proper motion accuracy:** ~0.01-0.5 mas/yr
- **Reference epoch:** J2016.0
- **Relation to Hipparcos:** Gaia supersedes Hipparcos for nearly all applications. However, combining Hipparcos (epoch ~1991.25) with Gaia (epoch ~2016.0) provides a ~25-year proper motion baseline that can reveal long-period astrometric binaries and improve acceleration solutions. The Hipparcos-Gaia Catalog of Accelerations (HGCA) exploits this baseline.
- **In starfield:** Gaia support exists in `src/catalogs/gaia.rs` with `GaiaCatalog` and `GaiaEntry` types

### Bright Star Catalogue (HR/Yale)

- **Stars:** 9,110 (all stars visible to the naked eye, V < 6.5)
- **Relation to Hipparcos:** A subset. Hipparcos provides much more precise positions and proper motions for these stars.

## 7. Practical Notes

### Auto-download and caching

Calling `starfield::data::downloader::download_hipparcos()` will:

1. Check `~/.cache/starfield/hip_main.dat` -- return immediately if present and non-empty
2. Check `./hip_main.dat` (project root) -- copy to cache if found
3. Download from the CDS URL and save to cache

This means the first run on a new machine will download ~51 MB, and subsequent runs use the cached copy.

### Parsing robustness

The parser is deliberately tolerant:

- Lines shorter than 110 characters are silently skipped
- Lines with fewer than 10 pipe-separated fields are skipped
- Missing optional fields (`parallax`, `pm_ra`, `pm_dec`, `b_v`) result in `None` rather than errors
- IO errors on individual lines are logged and skipped
- An empty file or a file where zero stars pass validation returns an explicit error

### Known data quirks

- **Missing positions:** A small number of catalog entries have blank RA/Dec fields. These are entries where the astrometric solution failed. The parser skips them.
- **Negative parallaxes:** The catalog contains negative parallax measurements (a statistical artifact for distant stars). The parser stores these as `Option<f64>`, and `cartesian_position()` returns `None` for non-positive parallaxes.
- **Duplicate HIP numbers:** Not present in the original catalog. The parser uses `HashMap::insert` which would overwrite any duplicates, but this is not expected to occur in practice.
- **Epoch J1991.25 vs J2000:** The catalog positions are at J1991.25, not J2000. Accurate work requires propagating positions forward using proper motions. The starfield parser currently stores positions as-is at J1991.25.
- **Proper motion convention:** Field H12 (`pmRA`) is `mu_alpha * cos(delta)`, the proper motion in RA already corrected for the cos(Dec) projection factor. This is the standard convention in modern astrometric catalogs.

### Performance considerations

- Parsing the full catalog (118,218 stars) takes on the order of 100ms on modern hardware
- The `HashMap` storage allows O(1) lookups by HIP number but does not support spatial indexing natively
- For field-of-view queries, `stars_in_field()` performs a linear scan with a dot-product angular distance check -- adequate for the catalog size but not optimal for repeated queries over many fields
- Applying a magnitude limit at load time (e.g., `mag_limit = 8.0`) reduces memory usage by excluding the faintest ~60% of catalog entries

### References

- ESA, 1997, *The Hipparcos and Tycho Catalogues*, ESA SP-1200
- van Leeuwen, F., 2007, *Hipparcos, the New Reduction of the Raw Data*, Astrophysics and Space Science Library, Vol. 350
- CDS catalog page: `https://cdsarc.cds.unistra.fr/viz-bin/cat/I/239`
