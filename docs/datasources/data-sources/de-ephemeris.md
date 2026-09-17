# JPL Development Ephemeris (DE-Series) Reference

## 1. Overview

The JPL Development Ephemeris (DE) series are the definitive high-precision
planetary and lunar ephemerides produced by the Jet Propulsion Laboratory's
Solar System Dynamics group. Each DE release provides the positions and
velocities of the Sun, Moon, planets, and selected other bodies over a span of
time, computed by numerically integrating the equations of motion for the solar
system.

DE ephemerides are the standard reference for planetary positions used
worldwide by spacecraft navigators, observatory software, almanac offices, and
open-source astronomy libraries alike. They are published as SPK (Spacecraft
and Planet Kernel) binary files in the NAIF Double Array File (DAF) container
format, conventionally distributed with the `.bsp` extension.

### Historical Progression

| Ephemeris | Year | Notes |
|-----------|------|-------|
| DE102 | 1981 | First widely distributed DE; Voyager-era |
| DE200 | 1982 | Standard for The Astronomical Almanac 1984--2003 |
| DE405 | 1997 | ICRF-aligned; dominant ephemeris for a decade |
| DE421 | 2008 | Fit to Mars and lunar laser ranging data |
| DE430 | 2013 | Improved spacecraft tracking fits (MESSENGER, Cassini) |
| DE440 | 2021 | Current recommended general-purpose ephemeris |
| DE441 | 2021 | Extended-range companion to DE440 (13200 BC -- AD 17191) |

Each successive release incorporates additional observational data (spacecraft
tracking, radar ranging, lunar laser ranging, VLBI) and improved dynamical
models (asteroid perturbation ring, solar oblateness, general relativity
parameters).

---

## 2. Available Ephemeris Files

### DE421 (2008)

| Property | Value |
|----------|-------|
| Coverage | 1899 Dec 14 -- 2053 Oct 8 |
| File size | ~17 MB |
| Bodies | Mercury--Pluto barycenters, Sun, Moon, Earth |
| Data type | SPK Type 2 (Chebyshev position) |
| Fit data | Mars ranging, LLR, planetary radar |

DE421 is the best choice for quick calculations, prototyping, and CI
environments. It covers the period most users care about with modern accuracy,
and its small size makes it fast to download and embed.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp`
- NAIF Generic Kernels: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de421.bsp`

### DE430 / DE430t (2013)

| Property | Value |
|----------|-------|
| Coverage (DE430t) | 1549 Dec 20 -- 2650 Jan 25 |
| File size | ~115 MB |
| Bodies | Mercury--Pluto barycenters, Sun, Moon, Earth |
| Data type | SPK Type 2 |
| Fit data | DE421 data + MESSENGER, Cassini, Mars rovers, LLR |

DE430 improved the fits to Mercury (MESSENGER tracking) and Saturn (Cassini
tracking). The "t" variant (de430t.bsp) is the full-range truncated file
commonly distributed by NAIF.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de430t.bsp`
- NAIF Generic Kernels: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430t.bsp`

### DE440 (2021) -- Recommended

| Property | Value |
|----------|-------|
| Coverage | 1549 Dec 20 -- 2650 Jan 25 |
| File size | ~115 MB |
| Bodies | Mercury--Pluto barycenters, Sun, Moon, Earth |
| Data type | SPK Type 2 |
| Fit data | All DE430 data + Juno, Cassini Grand Finale, MESSENGER extended mission |

DE440 is the current recommended ephemeris for most applications. It
incorporates the most extensive set of modern spacecraft tracking data and is
the standard used by JPL for current mission navigation.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440.bsp`
- NAIF Generic Kernels: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp`

### DE440s (2021) -- Short Version

| Property | Value |
|----------|-------|
| Coverage | 1849 Dec 26 -- 2150 Jan 22 |
| File size | ~32 MB |
| Bodies | Mercury--Pluto barycenters, Sun, Moon, Earth |
| Data type | SPK Type 2 |

DE440s contains the same polynomial coefficients as DE440, trimmed to a 300-year
window. It is the best balance between accuracy and file size for applications
that only need near-modern dates.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de440s.bsp`
- NAIF Generic Kernels: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp`

### DE441 (2021) -- Extended Range

| Property | Value |
|----------|-------|
| Coverage | ~13200 BC -- AD 17191 |
| File size | ~3.1 GB |
| Bodies | Mercury--Pluto barycenters, Sun, Moon, Earth |
| Data type | SPK Type 2 |

DE441 is the extended-range companion to DE440. Within the DE440 time span
it is identical to DE440. It is intended for paleoclimate research, historical
eclipse verification, archaeoastronomy, and far-future mission planning. Its
large size makes it impractical for most applications; prefer DE440 or DE440s
unless you genuinely need dates outside their range.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de441.bsp`
- NAIF Generic Kernels: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441.bsp`

### DE405 (1997) -- Legacy

| Property | Value |
|----------|-------|
| Coverage | 1599 Dec 9 -- 2201 Feb 20 |
| File size | ~55 MB |
| Data type | SPK Type 2 |

DE405 remains available for backward compatibility and reproducing published
results that were computed against it. For new work, use DE440.

**Download URLs:**

- JPL SSD: `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de405.bsp`

### Quick Selection Guide

| Use case | Recommended file | Size |
|----------|-----------------|------|
| Getting started / CI / tests | `de421.bsp` | ~17 MB |
| Production (modern dates) | `de440s.bsp` | ~32 MB |
| Production (wide range) | `de440.bsp` | ~115 MB |
| Ancient/far-future dates | `de441.bsp` | ~3.1 GB |
| Legacy compatibility | `de405.bsp` | ~55 MB |

---

## 3. Bodies Included

All DE ephemeris files contain the following 12 segments:

| NAIF ID | Body | Center ID | Center Body | Description |
|---------|------|-----------|-------------|-------------|
| 1 | Mercury Barycenter | 0 | Solar System Barycenter (SSB) | Mercury system center of mass |
| 2 | Venus Barycenter | 0 | SSB | Venus system center of mass (= Venus, no moons) |
| 3 | Earth-Moon Barycenter (EMB) | 0 | SSB | Earth-Moon system center of mass |
| 4 | Mars Barycenter | 0 | SSB | Mars system center of mass |
| 5 | Jupiter Barycenter | 0 | SSB | Jupiter system center of mass |
| 6 | Saturn Barycenter | 0 | SSB | Saturn system center of mass |
| 7 | Uranus Barycenter | 0 | SSB | Uranus system center of mass |
| 8 | Neptune Barycenter | 0 | SSB | Neptune system center of mass |
| 9 | Pluto Barycenter | 0 | SSB | Pluto system center of mass |
| 10 | Sun | 0 | SSB | Sun center |
| 301 | Moon | 3 | EMB | Moon relative to Earth-Moon Barycenter |
| 399 | Earth | 3 | EMB | Earth relative to Earth-Moon Barycenter |

### Important Note on Barycenters

For planets without substantial moons (Mercury and Venus), the barycenter is
effectively the planet body itself. For Mars, the mass of Phobos and Deimos is
negligible, so the Mars Barycenter is also essentially Mars itself. For the gas
giants, the barycenter and the planet body can be separated by thousands of
kilometers; to get the position of Jupiter itself (NAIF ID 599), you need a
separate satellite SPK file (e.g., `jup365.bsp`).

### Chain Resolution

DE files do not store every body relative to the SSB. The Moon (301) and Earth
(399) are stored relative to the Earth-Moon Barycenter (3), not relative to the
SSB. To compute a vector between two arbitrary bodies, you must walk the chain
of segments from each body back to their common root (the SSB).

**Example: Earth-to-Mars vector**

To find where Mars is as seen from Earth:

```
  Mars Barycenter (4) relative to SSB (0)        [segment 0 -> 4]
- Earth-Moon Barycenter (3) relative to SSB (0)   [segment 0 -> 3]
- Earth (399) relative to EMB (3)                  [segment 3 -> 399]
```

This gives: `position(Mars) = pos(4, SSB) - pos(3, SSB) - pos(399, EMB)`

Or equivalently, the chain to Mars from the SSB is just `[(0, 4)]`, and the
chain to Earth from the SSB is `[(0, 3), (3, 399)]`. Subtracting Earth's chain
from Mars's chain yields the Earth-to-Mars vector.

**Example: Earth-to-Moon vector**

```
  Moon (301) relative to EMB (3)                   [segment 3 -> 301]
- Earth (399) relative to EMB (3)                   [segment 3 -> 399]
```

Since both are relative to the same center (EMB), the EMB-to-SSB chain cancels
out, and you can simply subtract the two EMB-relative positions.

---

## 4. SPK Format Details

### DAF Container Format

All DE files use the NAIF Double Array File (DAF) format as their binary
container. The DAF format is defined by three structural layers:

**File Record (Record 1, bytes 0--1023):**

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 8 | LOCIDW | File ID word, e.g. `"DAF/SPK "` |
| 8 | 4 | ND | Number of double-precision summary components (2 for SPK) |
| 12 | 4 | NI | Number of integer summary components (6 for SPK) |
| 16 | 60 | LOCIFN | Internal filename |
| 76 | 4 | FWARD | Record number of first summary record |
| 80 | 4 | BWARD | Record number of last summary record |
| 84 | 4 | FREE | First free double-precision address |

Each record is 1024 bytes. Records are 1-indexed. Byte order is detected
automatically by checking whether ND and NI decode to sensible small integers
in little-endian vs. big-endian. In practice, all modern DE files are
little-endian.

**Comment Records (Records 2 through FWARD-1):**

Optional ASCII text describing the file contents, provenance, and creation date.

**Summary Records (Records FWARD, FWARD+2, ...):**

Each summary record contains a linked-list pointer (NEXT, PREV, NSUM) followed
by packed segment summaries. Each summary contains ND doubles and NI integers:

For SPK files (ND=2, NI=6), each segment summary contains:

| Index | Type | Field | Description |
|-------|------|-------|-------------|
| 0 | f64 | start_second | Segment start epoch (TDB seconds past J2000.0) |
| 1 | f64 | end_second | Segment end epoch (TDB seconds past J2000.0) |
| 2 | i32 | target | Target body NAIF ID |
| 3 | i32 | center | Center body NAIF ID |
| 4 | i32 | frame | Reference frame code (1 = J2000/ICRF) |
| 5 | i32 | data_type | SPK segment type (2 for DE files) |
| 6 | i32 | start_i | Start address in file (1-indexed double-word) |
| 7 | i32 | end_i | End address in file (1-indexed double-word) |

Each summary record is paired with a name record (the next record) that
provides an ASCII label for each segment.

### SPK Type 2 Segments

All DE ephemeris files use SPK Type 2, which stores position as Chebyshev
polynomial coefficients. Velocity is not stored explicitly -- it is obtained by
analytically differentiating the position polynomials.

**Segment Layout:**

A Type 2 segment (addressed by `start_i` through `end_i`) consists of:

1. **Coefficient records** -- A sequence of fixed-size records, each covering one
   time interval
2. **Metadata** -- Four trailing doubles at the end of the segment

**Metadata (last 4 doubles):**

| Field | Description |
|-------|-------------|
| `init` | Initial epoch of first record (TDB seconds past J2000.0) |
| `intlen` | Length of each time interval (seconds) |
| `rsize` | Size of each record (number of doubles) |
| `n` | Number of records |

**Each Coefficient Record (rsize doubles):**

| Offset | Count | Field |
|--------|-------|-------|
| 0 | 1 | `MID` -- Midpoint of the time interval (TDB seconds) |
| 1 | 1 | `RADIUS` -- Half-length of the time interval (seconds) |
| 2 | n_coeffs | X Chebyshev coefficients |
| 2 + n_coeffs | n_coeffs | Y Chebyshev coefficients |
| 2 + 2*n_coeffs | n_coeffs | Z Chebyshev coefficients |

Where `n_coeffs = (rsize - 2) / 3`.

The number of coefficients per component varies by body and ephemeris. Inner
planets (Mercury, Venus) and the Moon typically use more coefficients and
shorter time intervals (8--16 days) due to their faster orbital motion. Outer
planets use fewer coefficients and longer intervals (32 days or more).

### Reference Frame and Units

| Property | Value |
|----------|-------|
| Reference frame | J2000 (ICRF-aligned, frame code 1) |
| Position units | kilometers (km) |
| Velocity units | kilometers per second (km/s), derived from differentiation |
| Time system | Barycentric Dynamical Time (TDB) |
| Time representation | Seconds past J2000.0 TDB epoch (JD 2451545.0) |

---

## 5. How Starfield Uses DE Files

Starfield implements a complete DE ephemeris reader in the `jplephem` module.
The relevant source files are:

| File | Purpose |
|------|---------|
| `src/jplephem/daf.rs` | DAF binary container reader |
| `src/jplephem/spk.rs` | SPK segment parser and Type 2/3 evaluator |
| `src/jplephem/kernel.rs` | High-level API with name resolution and chain building |
| `src/jplephem/chebyshev.rs` | Chebyshev polynomial evaluation and differentiation |
| `src/jplephem/names.rs` | NAIF body name/ID mappings |
| `src/data/downloader.rs` | Auto-downloading and caching of BSP files |

### DAF Reader (`daf.rs`)

The `DAF` struct handles:

- **Automatic endianness detection:** Reads ND/NI in both byte orders and picks
  whichever gives small positive values.
- **Memory mapping:** Uses `memmap2` for efficient access to large files. Falls
  back to standard file I/O if memory mapping is unavailable.
- **In-memory buffers:** Supports `DAF::from_bytes()` for use with
  `include_bytes!()`, enabling compile-time embedding of ephemeris data.
- **Summary iteration:** Walks the linked list of summary records and unpacks
  segment metadata.
- **Array reading:** Reads arbitrary ranges of f64 values from the file using
  1-indexed double-word addresses.

### SPK Segment Parsing (`spk.rs`)

The `SPK` struct opens a DAF file and parses all segment summaries into
`Segment` objects. Each `Segment` holds:

- Target and center body IDs
- Time bounds (TDB seconds and Julian dates)
- SPK data type (2 or 3)
- File address range for the segment data

Segment data is loaded lazily on first access and cached for subsequent
evaluations. The loading process reads the trailing metadata (init, intlen,
rsize, n), validates consistency, and stores the coefficient array.

### Chebyshev Evaluation (`chebyshev.rs`)

Position evaluation at a given time involves:

1. **Record selection:** Compute `record_index = floor((et - init) / intlen)`
   to find which coefficient record covers the requested time.

2. **Time normalization:** Map the physical time to the interval `[-1, 1]`:
   ```
   t_normalized = (et - MID) / RADIUS
   ```

3. **Clenshaw recurrence:** Evaluate the Chebyshev series for each component
   (X, Y, Z) using the Clenshaw algorithm, which is numerically stable and
   efficient:
   ```
   b[n+1] = 0, b[n] = 0
   for k = n-1 down to 1:
       b[k] = 2*x*b[k+1] - b[k+2] + c[k]
   result = c[0] + x*b[1] - b[2]
   ```

4. **Velocity by differentiation:** The derivative of a Chebyshev series uses
   the identity `dT_n(x)/dx = n * U_{n-1}(x)` where `U` is the Chebyshev
   polynomial of the second kind. The derivative in normalized time is then
   rescaled to physical units by dividing by `RADIUS`:
   ```
   velocity_km_s = d(position)/d(t_normalized) / RADIUS
   ```

### BFS Chain Resolution (`kernel.rs`)

The `SpiceKernel` struct wraps an `SPK` and precomputes paths from the Solar
System Barycenter (SSB, NAIF ID 0) to every reachable body using breadth-first
search over the segment graph.

For each body, the chain is a list of `(center, target)` pairs. To compute a
body's SSB-relative position, starfield walks the chain and sums the position
vectors:

```rust
let mut total_pos = Vector3::zeros();
for &(center, target) in chain {
    let seg = spk.get_segment_mut(center, target)?;
    let (pos, vel) = seg.compute_and_differentiate(tdb_seconds, 0.0)?;
    total_pos += pos;
}
```

For example, the chain for Earth (399) is `[(0, 3), (3, 399)]`:
- Segment `(0, 3)`: EMB position relative to SSB
- Segment `(3, 399)`: Earth position relative to EMB

Summing these gives Earth's position relative to SSB.

### Unit Conversion

SPK files store positions in kilometers and velocities in km/s. The
`SpiceKernel::compute_at()` method converts to astronomical units (AU) and
AU/day using:

```
AU_KM = 149,597,870.700  (IAU 2012 exact definition)
position_au = position_km / AU_KM
velocity_au_day = velocity_km_s * 86400.0 / AU_KM
```

### Auto-Download and Caching

The `Loader` struct provides zero-configuration access to DE files:

```rust
let loader = starfield::Loader::new();
let mut kernel = loader.open("de421.bsp")?;
```

Behind the scenes:
1. `download_or_cache()` checks `~/.cache/starfield/` for the file.
2. If not found, `resolve_url()` maps the filename to a known URL. Files
   matching `*.bsp` are resolved against `https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/`.
3. The file is downloaded with a progress bar to a temporary file, then
   atomically renamed into the cache directory.

You can also use `SpiceKernel::open()` or `SpiceKernel::from_bytes()` directly
if you manage file paths yourself.

---

## 6. Accuracy

### Position Accuracy by Body

| Body | Approximate accuracy | Notes |
|------|---------------------|-------|
| Moon | ~1 meter | Constrained by lunar laser ranging |
| Inner planets (Mercury--Mars) | Sub-kilometer | Constrained by radar and spacecraft tracking |
| Jupiter, Saturn | ~1--10 km | Constrained by spacecraft tracking (Juno, Cassini) |
| Uranus, Neptune | ~10--100 km | Limited spacecraft data (Voyager flybys only) |
| Pluto | ~100 km -- few thousand km | New Horizons flyby improved this significantly |

These accuracy figures apply to DE440/441 within the well-observed portion of
their time span (roughly 1900--2050). Accuracy degrades for dates far from the
modern observational era.

### Primary Error Sources

- **Asteroid perturbations:** The main belt contains ~340 individually modeled
  asteroids plus a ring representing the remainder. Imprecise asteroid masses
  are the dominant error source for inner planet positions.
- **Solar oblateness (J2):** Affects Mercury's orbit most strongly.
- **Trans-Neptunian object masses:** Affects Pluto and Neptune predictions.
- **General relativity parameters:** Post-Newtonian effects are included but
  parameterized; small uncertainties propagate over centuries.

### Degradation Over Time

DE ephemerides are most accurate near the epoch of the observations that
constrain them. Extrapolating to the year 3000 or back to 1000 BC incurs
progressively larger errors, though the positions remain useful for most
astronomical purposes across the full coverage span.

---

## 7. Comparison with Other Ephemerides

Three major planetary ephemeris series exist worldwide:

| Ephemeris | Institution | Country | SPK Type | Notes |
|-----------|------------|---------|----------|-------|
| DE (Development Ephemeris) | JPL/NASA | USA | Type 2 | The series documented here |
| INPOP (Int. de Num. Plan. de l'Observatoire de Paris) | IMCCE | France | Type 2 | Comparable accuracy, different asteroid model |
| EPM (Ephemerides of Planets and the Moon) | IAA RAS | Russia | Type 20 | Uses a different Chebyshev representation |

All three series:
- Agree at the sub-kilometer level for inner planets within their common
  well-constrained time spans.
- Use overlapping but not identical sets of observational data.
- Are independently coded numerical integrations, providing valuable cross-checks.

INPOP ephemerides are distributed in SPK format and can be read by the same DAF/SPK
code used for DE files. EPM uses SPK Type 20 (Chebyshev with velocity), which
starfield does not currently support.

---

## 8. Practical Notes

### Choosing an Ephemeris

- **`de421.bsp` (~17 MB)** -- Start here. Fast to download, covers 1900--2053,
  perfectly adequate accuracy for most applications. Starfield's test suite uses
  this file.
- **`de440s.bsp` (~32 MB)** -- Best balance for production. Same accuracy as
  DE440, covers 1850--2150.
- **`de440.bsp` (~115 MB)** -- Full DE440 for applications needing dates back
  to 1550 or forward to 2650.
- **`de441.bsp` (~3.1 GB)** -- Only use this if you genuinely need positions
  before 1550 or after 2650.

### Performance Considerations

- **Memory mapping:** Starfield uses `memmap2` to memory-map BSP files. This
  means the OS manages paging -- only the portions of the file actually read
  are loaded into physical memory. A 115 MB DE440 file does not consume 115 MB
  of RAM.
- **Lazy loading:** Segment coefficient data is loaded and cached on first
  access. Subsequent evaluations for the same segment reuse the cached data.
- **Compile-time embedding:** For constrained environments (WebAssembly,
  embedded), use `SpiceKernel::from_bytes(include_bytes!("de421.bsp"))` to embed
  the ephemeris directly in your binary. DE421's 17 MB size makes this feasible.

### Cache Location

Starfield caches downloaded files in `~/.cache/starfield/`. This follows the
XDG Base Directory convention on Linux. The cache location is determined by:

```rust
let home = std::env::var("HOME").unwrap_or(".".to_string());
PathBuf::from(home).join(".cache").join("starfield")
```

You can override this by passing a custom directory to `Loader::with_data_dir()`.

### Mercury Barycenter vs Mercury

For Mercury (and Venus), the "barycenter" and the planet body are the same
point because these planets have no moons. The DE files contain segments for
Mercury Barycenter (NAIF ID 1), not Mercury body (NAIF ID 199). When starfield
resolves a request for "mercury", it returns the position from the barycenter
segment, which is the planet itself.

Note: Some DE files include a segment for Mercury body (1 -> 199), but this
segment may have a more limited time range than the barycenter segment (0 -> 1).
For Mercury, the barycenter segment is sufficient and preferred.

### Satellite SPK Files

DE ephemerides only contain data for planet barycenters, not for individual
moons (other than Earth's Moon) or the planet bodies of the gas giants. To get
positions of Jupiter (599), its Galilean moons (501--504), Saturn's rings, etc.,
you need additional satellite SPK files from NAIF:

- `jup365.bsp` -- Jupiter system satellites
- `sat441.bsp` -- Saturn system satellites
- etc.

These are resolved by starfield from
`https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/`.

---

## 9. Code Examples

### Basic: Load and Query

```rust
use starfield::jplephem::SpiceKernel;
use starfield::Timescale;

let ts = Timescale::default();
let mut kernel = SpiceKernel::open("de421.bsp")?;

let t = ts.tdb_jd(2451545.0); // J2000.0
let state = kernel.compute_at("earth", &t)?;

println!("Earth position: ({}, {}, {}) AU",
    state.position.x, state.position.y, state.position.z);
println!("Earth velocity: ({}, {}, {}) AU/day",
    state.velocity.x, state.velocity.y, state.velocity.z);
```

### Auto-Download

```rust
use starfield::Loader;

let loader = Loader::new();
let mut kernel = loader.open("de421.bsp")?; // downloads if needed

let ts = loader.timescale();
let t = ts.tdb_jd(2460000.0);
let mars = kernel.compute_at("mars", &t)?;
```

### Raw km/s Access

```rust
use starfield::jplephem::SpiceKernel;
use starfield::Timescale;

let ts = Timescale::default();
let mut kernel = SpiceKernel::open("de421.bsp")?;

let t = ts.tdb_jd(2451545.0);
let (pos_km, vel_km_s) = kernel.compute_km("moon", &t)?;

println!("Moon position: {} km from SSB", pos_km.norm());
println!("Moon velocity: {} km/s", vel_km_s.norm());
```

### Compile-Time Embedded

```rust
use starfield::jplephem::SpiceKernel;

static BSP: &[u8] = include_bytes!("path/to/de421.bsp");
let mut kernel = SpiceKernel::from_bytes(BSP)?;
// Use as normal -- no filesystem access needed
```

---

## 10. References

- Park, R. S., et al. (2021). "The JPL Planetary and Lunar Ephemerides DE440
  and DE441." _The Astronomical Journal_, 161(3), 105.
  https://doi.org/10.3847/1538-3881/abd414

- Folkner, W. M., et al. (2014). "The Planetary and Lunar Ephemerides DE430
  and DE431." _IPN Progress Report_, 42-196.

- Folkner, W. M., Williams, J. G., & Boggs, D. H. (2009). "The Planetary and
  Lunar Ephemeris DE421." _IPN Progress Report_, 42-178.

- NAIF SPK Required Reading:
  https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html

- NAIF DAF Required Reading:
  https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html

- JPL SSD Planetary Ephemeris page:
  https://ssd.jpl.nasa.gov/planets/eph_export.html

- NAIF Generic Kernels:
  https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
