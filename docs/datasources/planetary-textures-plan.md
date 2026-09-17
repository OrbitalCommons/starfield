# Resolved Planetary Appearance — Data Source Plan

One consumer: `focalplane` is adding resolved planets, moons and atmospheric
limbs to its focal-plane renderer. `starfield` supplies the geometry (body-fixed
frames, sub-observer points, apparent ellipses — see
`starfield/docs/solar-system-bodies-plan.md`). `focalplane` owns the radiometry.
This repository supplies the middle layer neither of them should own: **the
measured appearance of each body** — where the light comes from on the disk, and
what colour it is.

The split this document adds to the existing one:

> geometry in `starfield` · **measured appearance in `starfield-datasources`** · radiometry in `focalplane`

"Appearance" is deliberately three separable products, because they come from
different archives, at different sizes, with different update cadences:

| Product | What it is | Size | Delivery |
|---|---|---|---|
| **Spectral albedo** | geometric albedo vs wavelength, disk-integrated | ~100 KB/body | embedded |
| **Albedo map** | normal-albedo texture in body-fixed coordinates | 5 MB – 6 GB | tiered |
| **Photometric parameters** | Hapke / Minnaert / Lommel-Seeliger coefficients, phase curves | ~1 KB/body | embedded |

A renderer needs all three. Spectral albedo alone gives a flat disk of the right
colour; a map alone gives structure with no colour and the wrong limb darkening.

### Reconciliation with focalplane's plan

`focalplane`'s `docs/solar-system-plan.md` (branch `fp-planets`) lists Phase 1
items D1.1–D1.4 against this repo. They are the same effort under different
names, with one real disagreement resolved in their favour:

| focalplane | here | Status |
|---|---|---|
| D1.1 `starfield-solar-spectrum` (TSIS-1 HSRS v2, 1 nm) | *was absent* | **Adopted.** §2.3 originally pushed the solar spectrum to focalplane as radiometry. That was wrong: it is a tabulated archive product with provenance, which is what this repo is for. Next crate. |
| D1.2 `starfield-reflectance-library` | §3.4 terrain-unit endmembers | Same concept; their split into a standalone crate is better than folding it into the map work. |
| D1.3 `starfield-planet-maps` | PR 5 | Same crate. |
| D1.4 SHA-256 helper in `datasource-utils` | *was absent* | Adopted; lands with PR 5, which is what needs it. |
| *(not on their list)* | `starfield-planet-spectra` | Their D1.2 endmember list is entirely rocky/Earth surfaces, but their F3.7 renders the gas giants. Karkoschka fills that gap. |

### Settled interface contract

Agreed with focalplane and not to be re-litigated without telling them:

- **Map sampling input is east-positive planetocentric radians**, row 0 = north.
  Every archive convention is converted *inside* the map (§2.2). focalplane calls
  `sub_observer_point(..).to_planetocentric(radii).lon_rad` and applies no
  `LongitudeSense`.
- **`sample_area(lon_rad, lat_rad, sky_radius_rad, mu, epoch: Option<&Time>)`**
  returns a composited `EndmemberMix`. The caller passes the **sky** footprint
  and the emission cosine *separately*; the map does the projection.

  This was the one design disagreement. focalplane initially offered to pass the
  already-foreshortened surface footprint (`sky_radius / mu`). Rejected because
  1/μ is singular at the limb — so the clamp policy is inseparable from mip-level
  selection and must live in one place — and because foreshortening is
  *anisotropic*: dividing a scalar radius by μ inflates both axes, giving correct
  sharpness along the stretched direction and heavy over-blur across it, smearing
  along the limb exactly where a terminator test looks. Passing the full local
  Jacobian is strictly more information and stays available if the
  ellipsoid-vs-sphere difference turns out to matter at Jupiter's or Saturn's
  limb; not built until it does.
- **Sampling scale.** Disks are 30–400 px against mosaics of 100 m–1 km, i.e.
  10³–10⁶ texels per pixel. Point sampling would alias badly. Default embedded
  tier is ~0.1°/px (~4 km at the Mars equator); `sample_area` is backed by a mip
  pyramid whose level-selection rule is documented and whose chosen level is
  **exposed**, so the consumer can assert on it.
- **Always return `EndmemberMix`**, never a bare scalar; a uniform albedo is a
  one-endmember mix. Endmember ids are a stable enum shared between the map and
  reflectance crates.
- **Bodies are keyed on NAIF id** at the crate boundary, both sides.
- **Frames come from `frame_for`, on starfield 0.16.1 or later.** `frame_for(399)`
  gives `ItrsFrame` and `frame_for(301)` gives `PckFrame` when a Moon PA segment
  is loaded (DE440 preferred over DE421), matching what Horizons uses — ITRF93
  for Earth, MOON_ME for the Moon. Everything else falls back to the embedded
  IAU 2015 table, so the kernel-free path works offline. **0.16.0 does not have
  that fallback** and returns `DataError` for a body with no text kernel loaded.
- **`f_nu_cgs_at_nm` stays here; the `Spectrum` impl stays in focalplane.** The
  unit change is not radiometry, so the factor is written once in the crate that
  owns the units; the trait impl would invert the dependency. λ² is applied at the
  *requested* wavelength, not the bin centre — the box mean is a statement about
  F_λ and the λ² is exact.

## 1. What exists today

Checked against the working trees, not from memory:

| Area | State | Consequence |
|---|---|---|
| `starfield::planetlib::Body` | 11 bodies (8 planets, Sun, Moon, Pluto); `name()`, `naif_id()`, `radii_km()`, `flattening()`, `rotational_elements()` | **No moons besides Luna.** This plan needs its own body identifier keyed on NAIF id. |
| `starfield::planetarylib` | starfield 0.16.1: `text_pck`, `iau2015.csv`, `RotationalElements`, `IauFrame`, `PckFrame` | Body-fixed rotation is done. Map registration has something to register *to*. |
| `PlanetaryConstants::frame_for` | **0.16.1 or later required.** In 0.16.0 it returned `DataError` for any body with no text kernel loaded; since 0.16.1 it falls back to the embedded IAU 2015 table, with kernel values winning once read. | The convenience constructor the map path uses. On 0.16.0 the kernel-free call for Mars fails, and `IauFrame::from_naif_id(499)` was the workaround. Do not pin below 0.16.1. |
| `Position::angular_semi_diameter`, `Position::apparent_ellipse` | landed on starfield `main` at `93608df` (#182), in `src/positions/angular_size.rs` — **after** the 0.15.0 tag, so absent from a 0.15 checkout | Verified against `origin/main`. Grep the tag and you will wrongly conclude this is missing. |
| `planetarylib::occult` | landed on `main` (#182) | `Occultation::{None, Partial, Full}`. |
| `Position::sub_observer_point` / `sub_solar_point` | landing on `main` today (#187), `src/planetarylib/subpoint.rs` | The map-sampling entry point. Was the hard blocker for PR 5; no longer is. |
| `focalplane::photometry::Spectrum` | trait: `spectral_irradiance(Wavelength) -> f64` (erg s⁻¹ cm⁻² Hz⁻¹), `irradiance(&Band)` | An albedo is *not* a spectrum — it is dimensionless. See §2.3. |
| `focalplane` deps | pulls `starfield-gaia`, `starfield-nsa`, `starfield-datasource-utils` by git rev | Dependency runs focalplane → datasources. **Nothing here may depend on focalplane.** |
| `starfield-datasource-utils` | `cache_dir`, `ensure_cache_subdir`, `download_to_file`, `build_http_client` | Reuse for every downloading crate here. |
| `crates/bright-galaxies` | `include_str!`ed CSV + typed accessors, no network | The template for embedded-data crates. |
| datasources issue #8 | "Add planetary physical constants (GM, J2, radii)" | Superseded — radii now live in `starfield::planetlib::Body`. Close it, referencing this plan. |

## 2. Design decisions

### 2.1 Bodies are NAIF ids, not a re-exported enum

`starfield::planetlib::Body` covers 11 bodies. The first wave covers ~25,
including Io (501), Europa (502), Titan (606) and Charon (901). Re-exporting
`Body` would force an upstream change to `starfield` for every moon added here.

Each crate keys its tables on `i32` NAIF id and exposes its own
`SpectralBody` / `MappedBody` enum for the subset it actually has data for.
`From<starfield::planetlib::Body>` is provided for the overlap. This keeps the
"which bodies do we have data for" question answerable per crate, which is the
question a renderer actually asks.

### 2.2 Longitude convention is a stored property, not an assumption

This is the single highest-risk item in the plan, and the reason the user asked
for the starfield conventions up front.

`starfield`'s IAU rotational elements produce a body-fixed frame via
`Rz(W) · Rx(90° − δ) · Rz(90° + α)`. The `W` angle is measured from the node in
the direction of rotation, so the *native* frame longitude is **east-positive
planetocentric** for every body.

The archives disagree with that and with each other:

| Product | Longitude in the file | Latitude in the file |
|---|---|---|
| USGS Mars mosaics (modern) | east-positive | planetocentric |
| Mars, IAU/IAG cartographic convention | **west-positive** | planetographic |
| USGS Moon mosaics | east-positive | planetocentric |
| USGS Mercury MDIS mosaics | east-positive | planetocentric |
| Older Viking-era Mars products | west-positive | planetographic |
| Jupiter System III | west-positive | planetographic |

A map registered with the wrong handedness is *mirrored*, and mirrored Mars
looks entirely plausible until compared against an ephemeris. So:

```rust
pub enum Longitude { EastPositive, WestPositive }
pub enum Latitude  { Planetocentric, Planetographic }

pub struct MapGrid {
    pub longitude: Longitude,
    pub latitude: Latitude,
    pub lon0_deg: f64,   // longitude of the left edge
    // ...
}
```

Every map carries its grid. `AlbedoMap::sample_body_fixed(lon_rad, lat_rad)`
takes **east-positive planetocentric** coordinates — i.e. exactly what
`starfield`'s frame yields — and does the conversion internally, using
`Body::flattening()` for the ographic↔ocentric latitude conversion
(`tan φ_c = (1 − f)² tan φ_g`). Callers never do the conversion themselves;
that is how the mirror bug gets in.

A required regression test per map: sample a landmark of known body-fixed
coordinates and assert it lands where the IAU frame says it should. Olympus Mons
at 18.65°N, 226.2°E is the Mars case; Tycho at 43.31°S, 348.68°E is the Moon.

### 2.3 Albedo is dimensionless; this repo does not implement `Spectrum`

`focalplane`'s `Spectrum` returns spectral irradiance in erg s⁻¹ cm⁻² Hz⁻¹.
A geometric albedo is a dimensionless ratio. Converting one to the other needs
the solar spectrum, the heliocentric distance and the observer distance — none
of which a data crate has, and two of which are `starfield` geometry.

So these crates expose:

```rust
pub struct SpectralAlbedo { /* wavelengths_nm: Vec<f64>, albedo: Vec<f64> */ }
impl SpectralAlbedo {
    pub fn at_nm(&self, nm: f64) -> Option<f64>;      // linear interpolation
    pub fn mean_over(&self, lo_nm: f64, hi_nm: f64) -> Option<f64>;
    pub fn range_nm(&self) -> (f64, f64);
}
```

`focalplane` builds the `Spectrum` impl by multiplying this against its own
solar spectrum and the inverse-square factors. That keeps the radiometry
boundary exactly where the starfield plan put it, and keeps this repo free of a
focalplane dependency (§1).

### 2.4 Full-disk albedo is not geometric albedo

The Karkoschka tables (§3.1) label the Jupiter and Saturn columns *full-disk
albedo at phase angle 6.8° and 5.7°*, and the Uranus, Neptune and Titan columns
*geometric albedo* (phase ≈ 0). Saturn's is further qualified "at zero ring
tilt" — the globe only, rings excluded.

Silently mixing the two introduces a ~5–10% error on the two brightest targets.
The type therefore records what it holds:

```rust
pub enum AlbedoKind {
    Geometric,
    FullDisk { phase_angle_deg: f64 },
}
```

`AlbedoKind::Geometric` is *not* synthesised from the full-disk columns in the
first wave. Doing so needs a phase function; that arrives with the photometric
parameters crate (PR 4), which is where the correction belongs.

Saturn additionally needs a `rings_included: bool` flag, because every
disk-integrated Saturn measurement in the literature is one or the other and
the difference is up to a magnitude.

### 2.5 Tiered map delivery

Full-resolution mosaics are impossible to vendor — the LRO WAC global mosaic is
**5,959,263,751 bytes** (verified). The Mars and Mercury products are comparable.

Each mapped body therefore ships:

- an **embedded** equirectangular map, downsampled to 2048×1024 or smaller,
  stored as PNG (~0.5–2 MB/body) — enough for a disk a few hundred pixels
  across, works with no network, and is what the tests run against;
- a **`download_full()`** path fetching the archive product into
  `~/.cache/starfield/` via `starfield-datasource-utils`, `#[ignore]`d in tests.

The embedded maps are generated by a checked-in script from the archive
products, never hand-edited, with the source URL and a checksum recorded
alongside. That script is a deliverable, not a convenience.

## 3. Data sources, per body

Only sources whose download path has been verified are listed as *confirmed*.

### 3.1 Giant planets and Titan — confirmed, best-in-class

**Karkoschka (1994, 1998), PDS Atmospheres volume `gbat_0001`**, dataset
`ESO-J/S/N/U-SPECTROPHOTOMETER-4-V2.0`, DOI `10.17189/2bp8-k793`.

| File | Coverage | Bodies |
|---|---|---|
| `1995low.tab` | 300–1050 nm, 1 nm resolution, 0.4 nm sampling, 1875 rows | Jupiter, Saturn, Uranus, Neptune, Titan |
| `1995high.tab` | 520–995 nm, 0.4 nm resolution, 0.1 nm sampling, 4750 rows | Jupiter, Saturn, Uranus |
| `1993.tab` | 300–1000 nm, 1 nm resolution, 0.4 nm sampling, 1750 rows | Jupiter, Saturn, Uranus, Neptune, Titan |

Base URL `https://pds-atmospheres.nmsu.edu/PDS/data/gbat_0001/data/`.
PDS3, fixed-width ASCII, 54-byte records (`1995low`), detached `.lbl`.
Columns: vacuum λ, air λ, CH₄ absorption coefficient (km-amagat)⁻¹, then one
albedo column per body.

`1995low.tab` is 101,250 bytes and covers 300–1050 nm at 0.4 nm sampling — it
spans the entire silicon response with room to spare. It is embedded verbatim,
label included, and parsed at load. This is the highest-fidelity visible-band
planetary albedo dataset that exists in a machine-readable archive, and it is
small enough to vendor. It is the anchor of the whole effort.

### 3.2 Rocky planets — literature tables

No equivalent single archive exists. The plan is a curated CSV with per-row
provenance, assembled from:

- **Mallama, Krobusek & Pavlov (2017)**, *Icarus* 282, 19–33 (arXiv:1609.05048)
  — reference magnitudes and albedos for all 8 planets in 7 Johnson-Cousins
  and 5 Sloan bands. Band-integrated rather than spectral, but it is the
  standard reference and the **primary validation target**: any spectral
  albedo we ship must reproduce these band integrals.
- **Mallama & Hilton (2018)** — the phase functions used by *The Astronomical
  Almanac*. Feeds the photometric-parameters crate.
- **Madden & Kaltenegger (2018)**, *Astrobiology* 18(12), 1559 (arXiv:1807.11442)
  — spectra and geometric albedos for 19 solar-system bodies, 0.45–2.5 µm,
  including moons. Wider body coverage than anything else; blue cutoff at
  450 nm leaves the u/B end uncovered, so it supplements rather than replaces
  per-body work.
- Mercury: MESSENGER MASCS/VIRS (Izenberg et al. 2014, Domingue et al. 2010).
- Venus: cloud-top albedo, Irvine (1968) and successors.
- Earth: full-disk earthshine spectra.
- Mars: HST/STIS full-disk, plus the bright/dark terrain split (§3.4).

**Open item:** none of these is archived as a downloadable table the way
Karkoschka is; several exist only as published figures. Digitising a figure is
not a data source. Where no tabulated source is found, the crate ships the
band-integrated Mallama values and reports `range_nm()` honestly rather than
fabricating a spectrum. This is called out in §6.

### 3.3 Maps — USGS Astrogeology, confirmed

Product pages under `https://astrogeology.usgs.gov/search/map/<slug>` (require a
browser `User-Agent`; they 403 otherwise). Each page carries a direct GeoTIFF
link on `https://planetarymaps.usgs.gov/mosaic/<Name>.tif`, which 302-redirects
to `https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/<Name>.tif`.

The CKAN endpoint at `/ckan/api/3/action/package_search` returns HTML, not JSON —
**it is not a usable discovery API.** The crate therefore carries a curated,
pinned table of (body, product, URL, checksum) rather than querying at runtime.
Pinning is the right call for a data source anyway: reproducibility beats
freshness.

Confirmed products — URL and size verified by fetch:

| Body | Product | Size |
|---|---|---|
| Moon | `Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif` | 5.96 GB, monochrome 643 nm |
| Mars | `Mars_Viking_ClrMosaic_global_925m.tif` | 0.74 GB, colour |
| Mars | `Mars_Viking_MDIM21_ClrMosaic_global_232m.tif` | 11.87 GB, colour |
| Ganymede | `Ganymede_Voyager_GalileoSSI_Global_ClrMosaic_1435m.tif` | colour |
| Callisto | `Callisto_Voyager_GalileoSSI_global_mosaic_1km.tif` | monochrome |
| Mercury | MESSENGER MDIS basemap, 665 m; MD3 colour (1000/750/430 nm → RGB) | MD3 is the one with real spectral meaning |
| Earth | Blue Marble / MODIS | |

Product slugs cannot be guessed — `astrogeology.usgs.gov` returns HTTP 200 with a
generic catalog page for an unknown slug, so a wrong slug looks like a hit. Treat
"the page has a `planetarymaps.usgs.gov` link" as the existence test.

**Jupiter has no controlled global mosaic** in this catalog, and will not: there is
no solid surface to control a photogrammetric network to. See §3.5.

Moons are otherwise the sparse case: the Galileans are covered from
Voyager/Galileo, the mid-size Saturnians from Cassini ISS, Triton from Voyager 2
(one hemisphere only). Coverage is body-by-body and is the subject of PR 5's
survey.

### 3.5 Time-variable features — a static map is not enough

A map is a fixed property of the body only where the surface is fixed. For the
Moon, Mercury, Callisto and Ganymede that holds. For the two features most likely
to be asked for by name, it does not, and baking them into a texture produces
something confidently wrong rather than merely coarse.

**Jupiter's Great Red Spot drifts.** Its longitude in System III moves at roughly
0.36°/day — 13–16° per year, and the rate has been accelerating since the 1980s.
A texture with the GRS painted in is wrong by a visible fraction of the disk
within weeks and by a hemisphere within a decade. Jupiter's belts and zones also
reorganise on multi-year timescales.

**Mars' polar caps are seasonal, by a very large amount.** The southern seasonal
CO₂ cap reaches roughly −40° latitude at maximum extent (Ls ≈ 80–90°) and has
essentially sublimated away by Ls ≈ 320–330°. That is a swing of order 50° of
latitude over one Mars year. Any static mosaic bakes in whatever cap extent
happened to exist when its source frames were taken, so the caps are wrong for
most of the year — including, usually, being present when they should be absent.

Both therefore become **parametric overlays on top of the base map**, not part of
it:

```rust
/// Evaluated at observation time and composited over the base albedo map.
pub trait TimeVariableFeature {
    fn contribution(&self, lon_rad: f64, lat_rad: f64, t: &Time) -> Option<f64>;
}
```

- Mars seasonal caps: cap edge latitude as a function of solar longitude Ls, per
  hemisphere, from the MARCI/MOC recession literature. Ls is a `starfield`
  ephemeris quantity, so this needs no new geometry.
- Jupiter GRS: centre longitude from a System III drift model with an epoch, plus
  a size that has itself been shrinking. Needs the drift refit periodically, so
  the model carries its fit epoch and validity window and refuses to extrapolate
  far outside it — the same discipline `SpectralAlbedo` applies to wavelength.

This is new scope relative to the original PR list. It lands as **PR 5b (Mars
seasonal caps)** and **PR 7b (Jupiter GRS and belt model)**, each depending on
the map framework in PR 5. Neither blocks the static-surface bodies, which is
the argument for keeping the base-map path free of any time dependence.

### 3.4 Making colour vary across the disk

The maps are monochrome or false-colour; the spectra are disk-integrated. Naïvely
multiplying gives a body whose colour is uniform and only its brightness varies —
which is visibly wrong for Mars and the Moon, the two bodies most likely to be
rendered large.

The fix is a **terrain-unit decomposition**: 2–3 endmember spectra per body plus
a per-pixel mixing fraction derived from the colour mosaic.

- Moon: mare vs highlands.
- Mars: bright (dust-covered) vs dark (basaltic) terrain, plus polar ice.
- Mercury: high-reflectance plains vs low-reflectance material.

This is what separates "right average colour" from "looks right". It is deferred
to PR 6 because it depends on both the map crate and the spectral crate, but the
`SpectralAlbedo` type is designed for it now: endmembers are just several
`SpectralAlbedo`s and a mixing map.

## 4. Pull request sequence

Sizes: S ≈ half a day, M ≈ one to two days.

**PR 1 — `starfield-planet-spectra`, Karkoschka core (S–M).** *This pass.*
New crate. Embedded `1995low.tab` + `.lbl`, PDS3 fixed-width parser,
`SpectralAlbedo`, `AlbedoKind`, `SpectralBody`, interpolation and band-mean
accessors. Downloader for the `1995high` / `1993` variants. Facade feature
`planet-spectra`; facade minor bump. Unit tests against real embedded rows.

**PR 2 — rocky-body spectral albedos (M).**
Extends the same crate with the curated table of §3.2 and per-row provenance.
Validation against Mallama+2017 band integrals.

**PR 3 — moon spectral albedos (M).**
Galileans, mid-size Saturnians, Triton, Charon. Madden & Kaltenegger as the
backbone, per-body literature where it beats that.

**PR 4 — `starfield-planet-photometry` (M).**
Phase functions (Mallama & Hilton 2018), Hapke/Minnaert/Lommel-Seeliger
coefficients, limb-darkening parameters. Unblocks the full-disk → geometric
conversion deferred in §2.4.

**PR 5 — `starfield-planet-maps`, survey + framework (M).**
`MapGrid`, `AlbedoMap`, `sample_body_fixed`, the pinned product table, the
downsampling script, `download_full()`. Lands with the Moon and Mars embedded
maps and the landmark regression tests of §2.2. Includes the moon-coverage
survey.

**PR 6 — terrain-unit decomposition (M).**
§3.4, for the Moon and Mars first.

**PR 7 — moon maps (M).** Galileans and Saturnians, coverage permitting.

Dependency graph:

```
PR1 Karkoschka ──┬─ PR2 rocky spectra ──┐
                 └─ PR3 moon spectra    ├─ PR6 terrain units
PR4 photometry                          │
PR5 map framework ──┬───────────────────┘
                    └─ PR7 moon maps
```

Critical path: **PR1 → PR5 → PR6**.

## 5. Validation strategy

| Product | Reference | Path |
|---|---|---|
| Karkoschka parse | embedded real rows; row count and λ range from the PDS label | CI |
| Spectral albedo, planets | Mallama+2017 band-integrated albedos | CI, tolerance per band |
| Colour | B−V from Mallama+2017 Table 10 vs synthesised from our spectra | CI |
| Map registration | named landmarks at known body-fixed coordinates (§2.2) | CI, embedded maps |
| Map download | checksum against pinned table | `#[ignore]`d |
| Disk-integrated brightness | JPL Horizons apparent magnitude at three epochs | `#[ignore]`d fetch, fixtures in CI |

The Horizons row is the end-to-end check and the one that matters: render the
body with the full stack and compare integrated flux against the ephemeris.
`starfield-horizons` already parses observer tables, so the fixtures are cheap.

## 6. Open questions

1. **The rocky-planet spectral gap (§3.2).** Karkoschka gives the giant planets a
   0.4 nm-sampled archive; Mercury, Venus, Earth and Mars have nothing
   equivalent that is both tabulated and public. Options: ship band-integrated
   Mallama values only, honestly reporting coarse sampling; or contact the
   Madden & Kaltenegger authors for the machine-readable catalog. Recommend the
   former for PR 2, the latter in parallel.
2. **Saturn's rings.** They dominate Saturn's appearance and are not a surface
   map — they need their own opacity/scattering model, which is arguably
   focalplane radiometry. Out of scope here beyond flagging `rings_included` on
   the albedo, but somebody has to own it.
3. **Earth.** The one body whose appearance is time-variable at the hour scale
   (clouds). A static map is wrong in a way the others are not. For the
   Earth-limb-from-Mars-orbit study the limb profile may matter more than the
   disk; confirm with focalplane what fidelity is actually needed.
4. **Pluto/Charon** are in `Body` but their maps (New Horizons) cover one
   hemisphere well and the other poorly. Ship the good hemisphere and mark the
   rest as no-data, or omit?
