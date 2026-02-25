# Extracting Johnson-Cousins B-V from Gaia Photometry

## Background

The Johnson-Cousins B-V color index is a standard measure of stellar color, defined as the difference between blue (B) and visual (V) apparent magnitudes. It correlates strongly with effective temperature and spectral type. Many astronomical applications (star rendering, HR diagrams, spectral classification) require B-V, but Gaia uses its own photometric system (G, BP, RP) which does not map 1:1 to Johnson-Cousins bands.

## Current State in Starfield

The Gaia catalog implementation (`src/catalogs/gaia.rs`) currently:

- Uses **Gaia DR1** data, which provides only G-band photometry
- Returns `None` for B-V color in `star_data()` (line 613)
- Approximates V magnitude by returning G magnitude directly in `approx_v_magnitude()` (line 83)
- Has no BP or RP magnitude fields in `GaiaEntry`

The downloader (`src/data/gaia_downloader.rs`) is hardcoded to DR1:
```
https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/
```

**DR1 does not include BP/RP photometry at all.** Upgrading to DR2 or DR3 is a prerequisite for any B-V computation.

## Gaia Data Release Comparison

| Column | DR1 | DR2 | DR3 |
|---|---|---|---|
| `phot_g_mean_mag` | Yes | Yes | Yes |
| `phot_bp_mean_mag` | No | Yes (81.6% of sources) | Yes (85.1%) |
| `phot_rp_mean_mag` | No | Yes (81.7%) | Yes (85.8%) |
| `bp_rp` (pre-computed) | No | Yes (~81%) | Yes (~85%) |
| Effective temperature | No | `teff_val` (9.5%) | `teff_gspphot` (26%) |
| Synthetic Johnson B, V | No | No | `b_jkc_mag`, `v_jkc_mag` in GSPC table (~12%) |
| Total sources | 1.14B | 1.69B | 1.81B |
| Total columns in gaia_source | ~57 | 96 | 153 |

### Bulk Download URLs

| Release | URL |
|---|---|
| DR1 | `https://cdn.gea.esac.esa.int/Gaia/gdr1/gaia_source/csv/` |
| DR2 | `https://cdn.gea.esac.esa.int/Gaia/gdr2/gaia_source/csv/` |
| DR3 | `https://cdn.gea.esac.esa.int/Gaia/gdr3/gaia_source/` |

DR2 has ~3,386 gzipped CSV files (~550 GB compressed). DR3 has ~3,386 files (~1.3 TB compressed) and uses ECSV format (YAML header before CSV data). Files are partitioned by HEALPix level-8 index ranges derived from `source_id >> 43`.

### TAP/ADQL Filtered Downloads

For subsets (e.g. stars brighter than magnitude 12), the ESA TAP service avoids downloading the full catalog:

```
Endpoint: https://gea.esac.esa.int/tap-server/tap
```

Example query for DR3 bright stars with color data:
```sql
SELECT source_id, ra, dec, parallax, pmra, pmdec,
       phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
       bp_rp, teff_gspphot
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag < 12
```

- Synchronous queries: limited to 2,000 rows (testing only)
- Asynchronous queries: limited to ~3,000,000 rows per query
- Output formats: `votable`, `csv`, `json`, `fits`
- For results exceeding 3M rows, split on `random_index`

## Approaches to Compute B-V

### Approach 1: Two-Polynomial Route (Recommended)

Combine two polynomial transformations to get B-V from BP-RP:

**Step 1: G - V from BP-RP** (official ESA polynomial)

Gaia EDR3 (Riello et al. 2021, Table 5.7):
```
G - V = -0.02704 + 0.01424*(BP-RP) - 0.2156*(BP-RP)^2 + 0.01426*(BP-RP)^3
σ = 0.03017
Valid: -0.5 < BP-RP < 5.0
```

Gaia DR2 (Evans et al. 2018, Table 5.8):
```
G - V = -0.01760 - 0.006860*(BP-RP) - 0.1732*(BP-RP)^2
σ = 0.045858
Valid: -0.5 < BP-RP < 2.75
```

**Step 2: B - G from BP-RP** (third-party fit from qwim.ca)

```
B - G = -0.39137 + 1.6034*(BP-RP) - 0.15377*(BP-RP)^2
σ = 0.084
Valid: 0.49 <= BP-RP <= 3.9
```

**Combined:**
```
B - V = (B - G) + (G - V)
```

Using the EDR3 G-V polynomial and the qwim.ca B-G polynomial:
```
B - V = (-0.39137 + 1.6034*x - 0.15377*x^2) + (0.02704 - 0.01424*x + 0.2156*x^2 - 0.01426*x^3)
      = -0.36433 + 1.58916*x + 0.06183*x^2 - 0.01426*x^3
where x = BP-RP
```

Combined uncertainty: ~0.09 mag (dominated by the B-G polynomial).

Coverage: ~81% of DR2, ~85% of DR3 sources.

### Approach 2: Inverse Ballesteros from Effective Temperature

The Ballesteros (2012) formula relates B-V to effective temperature via blackbody approximation:

**Forward (B-V → T_eff):**
```
T_eff = 4600 * [1/(0.92*(B-V) + 1.7) + 1/(0.92*(B-V) + 0.62)]
```

**Inverse (T_eff → B-V), analytically derived:**

Let `k = T_eff / 4600`. The formula reduces to a quadratic in `u = 0.92*(B-V) + 1.7`:
```
k*u^2 - (1.08k + 2)*u + 1.08 = 0
```

Discriminant (always positive):
```
D = 1.1664*k^2 + 4
```

Solution:
```
u = [(1.08k + 2) - sqrt(1.1664*k^2 + 4)] / (2k)
B-V = (u - 1.7) / 0.92
```

In Rust:
```rust
fn bv_from_teff(teff: f64) -> f64 {
    let k = teff / 4600.0;
    let discriminant = 1.1664 * k * k + 4.0;
    let u = ((1.08 * k + 2.0) - discriminant.sqrt()) / (2.0 * k);
    (u - 1.7) / 0.92
}
```

Accuracy: ~0.05 mag for FGK stars (4000-7000 K). Degrades for very hot (O, B) and very cool (M) stars because real spectra deviate from Planck functions. Does not account for metallicity, surface gravity, or extinction.

Coverage: 9.5% of DR2 (`teff_val`), 26% of DR3 (`teff_gspphot`).

### Approach 3: Gaia DR3 Synthetic Photometry (GSPC)

The Gaia DR3 Synthetic Photometry Catalogue provides pre-computed Johnson-Cousins magnitudes from BP/RP spectra:

- Table: `gaiadr3.synthetic_photometry_gspc`
- Columns: `b_jkc_mag` (Johnson B), `v_jkc_mag` (Johnson V)
- Coverage: ~220 million sources (~12% of DR3)
- This is the most accurate method, as it uses the full BP/RP spectral energy distribution

ADQL query:
```sql
SELECT gs.source_id, gs.ra, gs.dec, gs.phot_g_mean_mag, gs.bp_rp,
       sp.b_jkc_mag, sp.v_jkc_mag
FROM gaiadr3.gaia_source AS gs
JOIN gaiadr3.synthetic_photometry_gspc AS sp USING (source_id)
WHERE gs.phot_g_mean_mag < 12
```

### Approach 4: Numerical Inversion of ESA Polynomial

ESA provides `BP-RP = f(B-V)` (EDR3):
```
BP-RP = 0.06483 + 1.575*(B-V) - 0.7815*(B-V)^2 + 0.5707*(B-V)^3 - 0.176*(B-V)^4
σ = 0.0659
Valid: -0.5 < B-V < 3.5
```

This can be numerically inverted (Newton-Raphson or bisection) to find `B-V` given `BP-RP`. The polynomial is monotonically increasing over the validity range, so the inverse is unique. This avoids relying on third-party fits but requires iterative computation.

## Additional Official ESA Polynomials

### DR2 G-V from B-V
```
G - V = -0.02907 - 0.02385*(B-V) - 0.2297*(B-V)^2 - 0.001768*(B-V)^3
σ = 0.06285
Valid: -0.3 < B-V < 2.4
```

### EDR3 G-V from B-V
```
G - V = -0.04749 - 0.0124*(B-V) - 0.2901*(B-V)^2 + 0.02008*(B-V)^3
σ = 0.04772
Valid: -0.4 < B-V < 3.3
```

### EDR3 G-R from BP-RP
```
G - R = -0.02275 + 0.3961*(BP-RP) - 0.1243*(BP-RP)^2 - 0.01396*(BP-RP)^3
σ = 0.03167
Valid: 0.0 < BP-RP < 4.0
```

### EDR3 G-Ic from BP-RP
```
G - Ic = 0.01753 + 0.76*(BP-RP) - 0.0991*(BP-RP)^2 + 0.03765*(BP-RP)^3
Valid: 0.5 < BP-RP < 2.0
```

## Sample Validation Points (Mamajek Dwarf Star Table)

| Spectral Type | B-V | BP-RP |
|---|---|---|
| A0V | 0.000 | -0.037 |
| G2V (Sun) | 0.650 | 0.823 |
| K0V | 0.816 | 0.983 |
| M0V | 1.420 | 1.840 |
| M5V | 1.830 | 3.350 |

These are for main-sequence dwarfs only.

## Recommendation for Starfield

1. **Upgrade from DR1 to DR2.** DR2 is simpler than DR3 (plain CSV vs ECSV) and provides the essential `bp_rp` column for 81% of sources. The bulk download URL changes to `gdr2/gaia_source/csv/`.

2. **Add `bp_rp` and `teff_val` to `GaiaEntry`** as `Option<f64>` fields. Update the CSV parser's column detection to handle these new columns.

3. **Implement B-V estimation** using a priority cascade:
   - If `bp_rp` is available: use the two-polynomial approach (B-G + G-V)
   - Else if `teff_val` is available: use the inverse Ballesteros formula
   - Else: return `None`

4. **Consider TAP/ADQL** for filtered downloads of bright stars instead of bulk download.

5. **Improve `approx_v_magnitude()`** to use `V = G - f(BP-RP)` when BP-RP is available.

## References

- Ballesteros, F.J. 2012, "New insights into black bodies", EPL 97, 34008. [arXiv:1201.1809](https://arxiv.org/abs/1201.1809)
- Evans, D.W. et al. 2018, "Gaia Data Release 2: Photometric content and validation", A&A 616, A4. [DOI](https://doi.org/10.1051/0004-6361/201832756)
- Riello, M. et al. 2021, "Gaia Early Data Release 3: Photometric content and validation", A&A 649, A3. [DOI](https://doi.org/10.1051/0004-6361/202039587)
- Pancino, E. et al. 2022, "Gaia EDR3 view on Galactic globular clusters", A&A 664, A109. [DOI](https://doi.org/10.1051/0004-6361/202243939)
- Montegriffo, P. et al. 2023, "Gaia Data Release 3: The Galaxy in your preferred photometric system", A&A 674, A3. [DOI](https://doi.org/10.1051/0004-6361/202243709)
- Jordi, C. et al. 2010, "Gaia broad band photometry", A&A 523, A48. [DOI](https://doi.org/10.1051/0004-6361/201015441)
- Mamajek, E.E., "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence", [Rochester table](https://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt)
- qwim.ca, "Photometric Transforms for K and M Dwarfs", [qwim.ca/Stars/photom.html](https://qwim.ca/Stars/photom.html)
- ESA Gaia DR2 Photometric Relations: [Documentation](https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu5pho/sec_cu5pho_calibr/ssec_cu5pho_PhotTransf.html)
- ESA Gaia EDR3 Photometric Relations: [Documentation](https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu5pho/cu5pho_sec_photSystem/cu5pho_ssec_photRelations.html)
- ESA Gaia DR3 GSPC Data Model: [Documentation](https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_performance_verification/ssec_dm_synthetic_photometry_gspc.html)
