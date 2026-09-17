# JPL SPK (Spacecraft and Planet Kernel) Binary Files

Implementation notes for building Rust SPK readers. This document covers the DAF
container format, all SPK segment types, available kernel files, and the
algorithms needed to evaluate ephemeris data.

---

## 1. Overview

SPK files are NAIF SPICE binary ephemeris files that store position and velocity
data for solar system bodies. They are the primary mechanism by which JPL
distributes planetary, satellite, asteroid, and spacecraft trajectory data.

- **Format**: Binary, stored inside a DAF (Double precision Array File) container
- **Maintainer**: NAIF (Navigation and Ancillary Information Facility) at NASA's
  Jet Propulsion Laboratory
- **Users**: NASA, ESA, JAXA, and virtually all spaceflight organizations worldwide
  use SPK files for mission planning, navigation, and scientific analysis
- **File extension**: `.bsp` (Binary SPK)
- **Reference toolkit**: NAIF SPICE toolkit (available in Fortran, C, MATLAB, Python via SpiceyPy)
- **Python reference**: `skyfield` and `jplephem` packages read SPK files natively

SPK files contain one or more **segments**, each providing ephemeris data for a
specific body relative to a specific center body over a specific time range. A
single file can contain segments for many bodies (e.g., `de440.bsp` contains
segments for all planet barycenters, the Sun, the Moon, and the Earth).

---

## 2. DAF Container Format

DAF is the general-purpose binary container used by NAIF for SPK, PCK (Planet
Constants Kernel), and CK (C-matrix/pointing Kernel) files. Understanding DAF is
a prerequisite for reading any SPK file.

### 2.1 Overall File Structure

A DAF file is organized as a sequence of **records**, each exactly **1024 bytes**
long. Records are numbered starting from 1.

```
Record 1:             File Record (header)
Records 2..FWARD-1:   Comment Area (ASCII text, optional)
Record FWARD:         First Summary Record
Record FWARD+1:       First Name Record
...                   Additional Summary/Name record pairs
Remaining records:    Element Records (coefficient data)
```

### 2.2 File Record (Record 1) - First 1024 Bytes

The file record contains all metadata needed to parse the rest of the file.

| Byte Offset | Size (bytes) | Field  | Description |
|-------------|-------------|--------|-------------|
| 0-7         | 8           | LOCIDW | File ID word, ASCII. `"DAF/SPK "` for SPK files, `"DAF/PCK "` for PCK files. Used to identify the file type. |
| 8-11        | 4           | ND     | Number of double-precision components per summary. For SPK files, always **2** (start epoch, end epoch). |
| 12-15       | 4           | NI     | Number of integer components per summary. For SPK files, always **6** (target, center, frame, data_type, start_addr, end_addr). |
| 16-75       | 60          | LOCIFN | Internal filename, ASCII, right-padded with spaces. |
| 76-79       | 4           | FWARD  | Record number of the first summary record (1-indexed). |
| 80-83       | 4           | BWARD  | Record number of the last summary record (1-indexed). |
| 84-87       | 4           | FREE   | First free double-precision address in the file (1-indexed). |
| 88-95       | 8           | (unused) | Numeric format identifier string (e.g., `"LTL-IEEE"` or `"BIG-IEEE"`). |
| 96-1023     | 928         | (unused) | Reserved/padding, typically zero-filled. |

**ND and NI** determine the structure of every summary in the file. For SPK:
- ND = 2 (two doubles: start epoch, end epoch)
- NI = 6 (six integers: target, center, frame, data_type, start_i, end_i)

**Endianness detection**: Read ND and NI as both little-endian and big-endian
`u32`. Valid values are small (1-10). Whichever interpretation yields small
positive values is the file's byte order. All subsequent multi-byte reads must
use this byte order.

```rust
let nd_le = LittleEndian::read_u32(&header[8..12]);
let ni_le = LittleEndian::read_u32(&header[12..16]);
let nd_be = BigEndian::read_u32(&header[8..12]);
let ni_be = BigEndian::read_u32(&header[12..16]);

let endian = if nd_le > 0 && nd_le < 10 && ni_le > 0 && ni_le < 10 {
    Endian::Little
} else if nd_be > 0 && nd_be < 10 && ni_be > 0 && ni_be < 10 {
    Endian::Big
} else {
    // Invalid file
};
```

### 2.3 Comment Records (Records 2 through FWARD-1)

If FWARD > 2, records 2 through FWARD-1 contain ASCII comment text. Comments are
stored as raw bytes with NUL (`0x00`) as end-of-comment and `0x04` (EOT) as
line separators (converted to `\n` when reading). Comments typically describe the
file's provenance, creation parameters, and the bodies contained within.

### 2.4 Summary Records

Summary records form a **doubly-linked list**, starting at record number FWARD
and ending at BWARD. Each summary record is paired with the immediately following
**name record** (at record_number + 1).

#### Summary Record Layout (1024 bytes)

| Byte Offset | Size | Type | Field | Description |
|-------------|------|------|-------|-------------|
| 0-7         | 8    | f64  | NEXT  | Record number of next summary record (0 if last) |
| 8-15        | 8    | f64  | PREV  | Record number of previous summary record (0 if first) |
| 16-23       | 8    | f64  | NSUM  | Number of summaries in this record |
| 24-...      | variable | mixed | Summaries | Packed summary entries |

The maximum number of summaries per record is:

```
max_summaries = floor((1024 - 24) / summary_step)
```

where:

```
summary_length = ND + ceil(NI / 2)   // in double-words
summary_step   = summary_length * 8  // in bytes
```

For SPK (ND=2, NI=6): `summary_length = 2 + 3 = 5`, `summary_step = 40 bytes`,
`max_summaries = floor(1000 / 40) = 25`.

#### Individual Summary Layout (40 bytes for SPK)

Each summary contains ND doubles followed by NI integers. The integers are packed
as pairs of `i32` values into 8-byte (double-word) slots.

| Offset within summary | Size | Type | Field | Description |
|-----------------------|------|------|-------|-------------|
| 0-7                   | 8    | f64  | START_EPOCH | Start epoch in TDB seconds past J2000 |
| 8-15                  | 8    | f64  | END_EPOCH   | End epoch in TDB seconds past J2000 |
| 16-19                 | 4    | i32  | TARGET      | NAIF target body ID |
| 20-23                 | 4    | i32  | CENTER      | NAIF center body ID |
| 24-27                 | 4    | i32  | FRAME       | Reference frame ID |
| 28-31                 | 4    | i32  | DATA_TYPE   | SPK segment type (2, 3, 5, 21, etc.) |
| 32-35                 | 4    | i32  | START_ADDR  | Start address in element records (1-indexed, in double-words) |
| 36-39                 | 4    | i32  | END_ADDR    | End address in element records (1-indexed, in double-words) |

**Address convention**: START_ADDR and END_ADDR are 1-indexed addresses in units
of double-precision words (8 bytes). To convert to a byte offset:
`byte_offset = (address - 1) * 8`.

#### Name Record Layout

The name record immediately follows its summary record (at record_number + 1).
It contains segment source labels packed at the same offsets as their
corresponding summaries (each label occupies `summary_step` bytes, right-padded
with spaces or NULs).

### 2.5 Element Records

All remaining records after the summary/name pairs contain the actual ephemeris
data (Chebyshev coefficients, discrete states, difference arrays, etc.). Segments
reference ranges within this data using the START_ADDR and END_ADDR fields from
their summaries.

To read a segment's data:

```rust
// addresses are 1-indexed double-word positions
let data: Vec<f64> = daf.read_array(start_addr, end_addr)?;
// data.len() == end_addr - start_addr + 1
```

The byte range in the file is:
```
start_byte = (start_addr - 1) * 8
end_byte   = end_addr * 8    // exclusive
```

---

## 3. SPK Segment Types - Complete Reference

SPK defines over 20 segment types, each using a different mathematical
representation. The DATA_TYPE field in each summary identifies the type.

### Implementation Priority Summary

| Priority | Types | Rationale |
|----------|-------|-----------|
| **Implemented** | 2, 3 | Planetary ephemerides, satellite kernels |
| **Highest** | 21 | All modern Horizons small-body files |
| **High** | 1 | Legacy small-body files (Type 21 predecessor) |
| **Medium** | 5 | Osculating element propagation |
| **Low** | 8, 9, 12, 13, 14 | Interpolation types, uncommon |
| **Not needed** | 10, 15, 17, 18, 19, 20 | Domain-specific or very rare |

---

### Type 1: Modified Difference Arrays (MDA)

**Status**: DEPRECATED in favor of Type 21, but still present in archived files.
**Priority**: Should implement for backward compatibility with legacy data.

**Used by**: Older JPL spacecraft trajectories, pre-2018 Horizons small-body
outputs, archived mission SPK files.

#### Mathematical Background

Modified Difference Arrays use the Shampine-Gordon method for polynomial
interpolation via divided differences. The method stores a reference state and a
table of modified divided differences that can reconstruct a polynomial
interpolation of the trajectory at any point within a record's time span.

#### Segment Layout

The segment data consists of a sequence of fixed-size records followed by an
epoch directory.

**Segment structure** (double-precision words):

```
[ Record_0 | Record_1 | ... | Record_{N-1} | Epoch_Dir | N ]
```

- The last word is `N`, the number of records (as f64, cast to integer).
- Before `N` is the **epoch directory**: contains `floor((N-1) / 100)` epoch
  values for fast searching when N > 100.

**Each record** contains exactly **71 double-precision values** (MAXDIM = 15 for Type 1):

| Word Index | Count | Field | Description |
|------------|-------|-------|-------------|
| 0          | 1     | TL    | Reference epoch (TDB seconds past J2000) |
| 1-15       | 15    | G     | Stepsize function coefficients |
| 16-18      | 3     | REFPOS | Reference position (X, Y, Z) in km |
| 19-21      | 3     | REFVEL | Reference velocity (VX, VY, VZ) in km/s |
| 22-36      | 15    | DT    | Modified difference array for X |
| 37-51      | 15    | DT    | Modified difference array for Y |
| 52-66      | 15    | DT    | Modified difference array for Z |
| 67         | 1     | KQMAX1 | Maximum integration order + 1 |
| 68-70      | 3     | KQ    | Integration order for each component |

Total: **71 doubles per record** = 568 bytes.

#### Evaluation Algorithm

To compute position and velocity at epoch `t`:

1. **Find the record**: Search epoch directory, then linear scan to find the
   record whose reference epoch `TL` is nearest to `t`.

2. **Compute normalized time**: `dt = t - TL`

3. **Reconstruct position via difference table**: For each component (X, Y, Z):
   ```
   Initialize: tp = dt
   pos = refpos[comp] + refvel[comp] * dt + dt * dt * kq_sum(...)
   ```

   The full reconstruction uses the Shampine-Gordon recurrence:
   ```
   fc[0] = 1.0
   for i in 0..kq[comp]:
       fc[i+1] = dt / g[i] * fc[i]

   w[kq[comp]-1] = diff_table[comp][kq[comp]-1]
   for j in (0..kq[comp]-1).rev():
       w[j] = diff_table[comp][j] + fc[j+1] * w[j+1]

   position[comp] = refpos[comp] + dt * (refvel[comp] + dt * w[0])
   ```

4. **Velocity via derivative**: Similar recurrence with derivative terms:
   ```
   velocity[comp] = refvel[comp] + dt * (2 * w[0] + dt * w'[0])
   ```
   where `w'` uses modified recurrence coefficients.

**Reference implementation**: `spktype01` Python package by Shushi Uetsuki.

---

### Type 2: Chebyshev Polynomials (Position Only)

**Status**: ALREADY IMPLEMENTED in starfield (`src/jplephem/spk.rs`).
**Priority**: This is the single most important type.

**Used by**: ALL JPL planetary ephemerides (DE421, DE430, DE440, DE441), bulk
asteroid perturber kernels (sb441-n16.bsp, sb441-n373.bsp), numerically
integrated satellite orbits, and many other high-precision ephemerides.

#### Segment Layout

Type 2 segments contain a sequence of fixed-length records, each covering a
uniform time interval. Metadata is stored as the **last 4 words** of the segment.

**Segment structure** (double-precision words):

```
[ Record_0 | Record_1 | ... | Record_{N-1} | INIT | INTLEN | RSIZE | N ]
```

**Segment metadata** (last 4 double-precision values):

| Word (from end) | Field  | Description |
|-----------------|--------|-------------|
| N-4             | INIT   | Initial epoch of first record (TDB seconds past J2000) |
| N-3             | INTLEN | Time length of each record interval (seconds) |
| N-2             | RSIZE  | Record size in double-precision words |
| N-1             | N      | Number of records |

**Each record** (RSIZE double-precision values):

| Word Offset | Count | Field | Description |
|-------------|-------|-------|-------------|
| 0           | 1     | MID   | Midpoint epoch of this interval (TDB seconds past J2000) |
| 1           | 1     | RADIUS | Half-length of this interval (seconds). RADIUS = INTLEN / 2 |
| 2           | D     | C_X   | Chebyshev coefficients for X position (degree 0 to D-1) |
| 2+D         | D     | C_Y   | Chebyshev coefficients for Y position |
| 2+2D        | D     | C_Z   | Chebyshev coefficients for Z position |

where `D = (RSIZE - 2) / 3` is the number of coefficients per component
(polynomial degree + 1).

**Validation**: `N * RSIZE + 4 == total_segment_length`

#### Evaluation Algorithm

To compute position and velocity at epoch `t` (TDB seconds past J2000):

**Step 1: Find the record index**
```
index = floor((t - INIT) / INTLEN)
index = clamp(index, 0, N-1)
```

**Step 2: Read record data**
```
record_offset = index * RSIZE
MID    = data[record_offset]
RADIUS = data[record_offset + 1]
C_X    = data[record_offset + 2 .. record_offset + 2 + D]
C_Y    = data[record_offset + 2 + D .. record_offset + 2 + 2D]
C_Z    = data[record_offset + 2 + 2D .. record_offset + 2 + 3D]
```

**Step 3: Normalize time to [-1, 1]**
```
tau = (t - MID) / RADIUS
```
The value `tau` must be in [-1, 1]. Clamp for floating-point edge cases.

**Step 4: Evaluate position via Clenshaw recurrence**

For each component (using coefficients `c[0..D]`):

```
b[D]   = 0
b[D-1] = 0
for k = D-1 down to 1:
    b[k] = 2 * tau * b[k+1] - b[k+2] + c[k]
position = c[0] + tau * b[1] - b[2]
```

This is numerically stable and costs O(D) operations.

**Step 5: Compute velocity via Chebyshev derivative**

The derivative of a Chebyshev expansion with respect to `tau` uses:

```
dT_n(tau)/dtau = n * U_{n-1}(tau)
```

where `U_n` is the Chebyshev polynomial of the second kind. Evaluate `U_n`
via the recurrence:

```
U_0(x) = 1
U_1(x) = 2x
U_n(x) = 2x * U_{n-1}(x) - U_{n-2}(x)
```

Then the derivative of the expansion is:

```
df/dtau = sum_{n=1}^{D-1} c[n] * n * U_{n-1}(tau)
```

**Step 6: Rescale velocity to physical units**

The derivative is with respect to normalized time `tau`. To get km/s:

```
velocity = (df/dtau) / RADIUS
```

Since RADIUS is in seconds and the coefficients are in km, this gives km/s.

#### Rust Implementation Reference

From `src/jplephem/spk.rs`:

```rust
let t = normalize_time(et, record_mid, record_radius)?;
let poly_x = ChebyshevPolynomial::new(coeffs_x);
let position_x = poly_x.evaluate(t);
let velocity_x = rescale_derivative(poly_x.derivative(t), record_radius)?;
```

---

### Type 3: Chebyshev Polynomials (Position + Velocity)

**Status**: ALREADY IMPLEMENTED in starfield (`src/jplephem/spk.rs`).

**Used by**: Satellite kernels with analytical orbit theories (jup365.bsp,
sat441l.bsp, mar097.bsp, etc.). When velocity is independently computed (not
derived from position polynomials), Type 3 provides both sets of coefficients.

#### Segment Layout

Identical structure to Type 2, but each record is twice as long because it
contains separate Chebyshev coefficients for both position and velocity.

**Segment metadata**: Same as Type 2 (INIT, INTLEN, RSIZE, N at end of segment).

**Each record** (RSIZE double-precision values):

| Word Offset | Count | Field | Description |
|-------------|-------|-------|-------------|
| 0           | 1     | MID   | Midpoint epoch |
| 1           | 1     | RADIUS | Half-interval |
| 2           | D     | C_X   | Position X coefficients |
| 2+D         | D     | C_Y   | Position Y coefficients |
| 2+2D        | D     | C_Z   | Position Z coefficients |
| 2+3D        | D     | C_VX  | Velocity X coefficients |
| 2+4D        | D     | C_VY  | Velocity Y coefficients |
| 2+5D        | D     | C_VZ  | Velocity Z coefficients |

where `D = (RSIZE - 2) / 6`.

#### Evaluation Algorithm

**Position**: Identical to Type 2 Clenshaw evaluation using the position
coefficients `C_X`, `C_Y`, `C_Z`.

**Velocity**: Apply Clenshaw evaluation **directly** to the velocity
coefficients `C_VX`, `C_VY`, `C_VZ`, then rescale:

```
velocity = evaluate(C_V*, tau) / RADIUS
```

The velocity coefficients are Chebyshev expansions of `velocity * RADIUS`
(i.e., velocity scaled by the half-interval), so dividing by RADIUS recovers
the physical velocity in km/s.

**Key difference from Type 2**: For Type 2, velocity is obtained by
differentiating the position polynomials. For Type 3, velocity has its own
independent polynomial expansion, which can be more accurate when the velocity
is known independently (e.g., from an analytical theory).

---

### Type 5: Discrete States + Two-Body Propagation

**Status**: Not yet implemented.
**Priority**: Medium. Starfield already has Kepler propagation in `keplerlib`.

**Used by**: Traditional asteroid/comet ephemerides from osculating elements,
some older spacecraft trajectory files.

#### Segment Layout

Type 5 stores a sequence of discrete state vectors at specific epochs, plus the
gravitational parameter (GM) of the central body.

**Segment structure**:

```
[ State_0 | State_1 | ... | State_{N-1} | GM ]
```

Each state record is **6 double-precision values** (48 bytes):

| Word Offset | Field | Description |
|-------------|-------|-------------|
| 0           | EPOCH | State epoch (TDB seconds past J2000) |
| 1           | X     | Position X (km) |
| 2           | Y     | Position Y (km) |
| 3           | Z     | Position Z (km) |
| 4           | VX    | Velocity X (km/s) |
| 5           | VY    | Velocity Y (km/s) |
| 6           | VZ    | Velocity Z (km/s) |

Wait -- Type 5 actually stores **two-body discrete states**. The exact layout:

**Segment structure**:

```
[ (epoch_0, state_0) | (epoch_1, state_1) | ... | Epochs | N | GM ]
```

- The last value is **GM** (gravitational parameter of central body, km^3/s^2)
- The second-to-last value is **N** (number of states)
- Before N: **epoch directory** (for fast lookup when N > 100)
- Each state: 6 doubles (X, Y, Z, VX, VY, VZ) at a specific epoch

#### Evaluation Algorithm

To compute state at epoch `t`:

1. **Find bracketing states**: Binary search the epoch list to find states at
   times `t_i <= t <= t_{i+1}`.

2. **Select nearest state**: Choose the state at `t_i` or `t_{i+1}` (whichever
   is closer to `t`).

3. **Two-body propagation**: From the selected state, propagate forward or
   backward to `t` using Keplerian two-body mechanics:
   - Convert state to orbital elements
   - Propagate mean anomaly: `M = M_0 + n * dt` where `n = sqrt(GM / a^3)`
   - Solve Kepler's equation: `M = E - e*sin(E)` (iteratively)
   - Convert back to Cartesian state

This provides a reasonable approximation for bodies on near-Keplerian orbits
(asteroids, comets, distant moons).

---

### Type 8: Lagrange Interpolation (Equally Spaced)

**Status**: Not implemented.
**Priority**: Low -- uncommon in practice.

**Used by**: Some mission-specific trajectory files where uniform sampling is
natural.

#### Segment Layout

```
[ State_0 | State_1 | ... | State_{N-1} | INIT | STEP | DEGREE+1 | N ]
```

Each state: 6 doubles (X, Y, Z, VX, VY, VZ) in km and km/s.

**Metadata** (last 4 words):
- INIT: epoch of first state
- STEP: uniform time step between states (seconds)
- DEGREE+1: window size (polynomial degree + 1); must be even
- N: number of states

#### Evaluation Algorithm

1. Find the window of DEGREE+1 states centered around `t`
2. Evaluate Lagrange interpolating polynomial through those states
3. Evaluate derivative of Lagrange polynomial for velocity

Lagrange polynomial at equally spaced nodes:
```
L_j(t) = prod_{k != j} (t - t_k) / (t_j - t_k)
position(t) = sum_j state_j * L_j(t)
```

---

### Type 9: Lagrange Interpolation (Unequally Spaced)

**Status**: Not implemented.
**Priority**: Low -- uncommon.

**Used by**: Trajectory files with non-uniform time sampling.

#### Segment Layout

```
[ State_0 | State_1 | ... | State_{N-1} | Epoch_0 | ... | Epoch_{N-1} | Epoch_Dir | DEGREE+1 | N ]
```

Each state: 6 doubles. Epochs are explicitly stored (non-uniform spacing).

**Metadata** (last 2 words):
- DEGREE+1: window size
- N: number of states

#### Evaluation

Same Lagrange interpolation as Type 8, but at non-uniform knot times.

---

### Type 10: Space Command Two-Line Elements

**Status**: Not implemented.
**Priority**: Not needed -- completely different domain from planetary ephemerides.

**Used by**: Earth-orbiting satellites using NORAD TLE sets.

#### Structure

Contains NORAD Two-Line Element sets:
- Inclination, RAAN, eccentricity, argument of perigee, mean anomaly, mean motion
- Plus associated drag terms (B* coefficient)

#### Evaluation

Requires the SGP4/SDP4 propagator, which is a fundamentally different algorithm
from anything else in the SPK system. SGP4 models atmospheric drag, solar/lunar
perturbations, and Earth oblateness effects specific to Earth-orbiting objects.

---

### Type 12: Hermite Interpolation (Equally Spaced)

**Status**: Not implemented.
**Priority**: Low.

**Used by**: Some mission-specific trajectory files.

#### Segment Layout

```
[ State_0 | State_1 | ... | State_{N-1} | INIT | STEP | DEGREE+1 | N ]
```

Each state: 6 doubles (position + velocity). Same metadata as Type 8.

#### Evaluation

Hermite interpolation uses both position **and** velocity values at each knot,
producing a polynomial that matches the function value and first derivative at
every knot point. This gives C^1 continuity (continuous first derivative) at
record boundaries, unlike Lagrange (Type 8/9) which only guarantees C^0.

For a window of DEGREE+1 states, Hermite interpolation constructs a polynomial
of degree `2*(DEGREE+1) - 1` (twice the Lagrange degree) using divided
differences on the doubled set of knots.

---

### Type 13: Hermite Interpolation (Unequally Spaced)

**Status**: Not implemented.
**Priority**: Low-medium.

**Used by**: Some mission-specific trajectory files, ESA reconstructed orbits.

#### Segment Layout

```
[ State_0 | ... | State_{N-1} | Epoch_0 | ... | Epoch_{N-1} | Epoch_Dir | DEGREE+1 | N ]
```

Same as Type 9 but with Hermite interpolation instead of Lagrange.

#### Evaluation

Same Hermite algorithm as Type 12, but with non-uniform knot spacing. The
non-uniform spacing makes the divided difference computation slightly more
involved but the algorithm is otherwise identical.

---

### Type 14: Chebyshev Polynomials (Unequally Spaced)

**Status**: Not implemented.
**Priority**: Low -- uncommon.

**Used by**: Rare cases where different time intervals require different
polynomial degrees or interval lengths.

#### Segment Layout

Unlike Types 2/3 which use fixed-length records at uniform intervals, Type 14
allows each record to have a different length and cover a different time span.

```
[ Record_0 | Record_1 | ... | Record_{N-1} | Epochs | N ]
```

Each record has its own MID, RADIUS, and coefficient count. The epoch directory
lists the start/end times of each record.

#### Evaluation

Same Clenshaw recurrence as Types 2/3, but with a binary search to find the
correct record and variable polynomial degrees per record.

---

### Type 15: Precessing Conic Propagation

**Status**: Not implemented.
**Priority**: Low -- rarely used in practice.

**Used by**: Very approximate satellite ephemerides where only the mean orbital
elements and secular J2 perturbations matter.

#### Segment Layout

The entire segment is a single record containing:

| Word | Field | Description |
|------|-------|-------------|
| 0    | EPOCH | Reference epoch (TDB seconds past J2000) |
| 1-3  | TP    | Trajectory pole unit vector (perpendicular to orbital plane) |
| 4-6  | PA    | Periapsis unit vector at epoch |
| 7    | P     | Semi-latus rectum (km) |
| 8    | ECC   | Eccentricity |
| 9    | J2FLG | J2 processing flag (1 = apply J2 corrections) |
| 10-12| PV    | Central body pole unit vector |
| 13   | GM    | Central body gravitational parameter (km^3/s^2) |
| 14   | J2    | Central body J2 coefficient |
| 15   | RE    | Central body equatorial radius (km) |

#### Evaluation

1. Propagate mean anomaly from epoch: `M = M_0 + n * dt`
2. Apply J2 secular perturbations if J2FLG = 1:
   - Nodal regression: `dOmega/dt = -1.5 * n * J2 * (RE/a)^2 * cos(i) / (1-e^2)^2`
   - Apsidal precession: `domega/dt = 0.75 * n * J2 * (RE/a)^2 * (5*cos^2(i) - 1) / (1-e^2)^2`
3. Solve Kepler's equation for eccentric anomaly
4. Convert to Cartesian position and velocity

---

### Type 17: Equinoctial Elements

**Status**: Not implemented.
**Priority**: Low.

**Used by**: Some mean-element satellite ephemerides.

#### Structure

Stores equinoctial orbital elements and their time rates:
- Semi-major axis `a`, mean longitude rate `dn/dt`
- `h = e * sin(omega + Omega)`, `k = e * cos(omega + Omega)`
- `p = tan(i/2) * sin(Omega)`, `q = tan(i/2) * cos(Omega)`
- Mean longitude at epoch, plus rates for all elements
- Precession rate of the node

#### Evaluation

Propagate elements linearly in time, convert equinoctial elements to Cartesian
state. Equinoctial elements avoid singularities at zero eccentricity and zero
inclination that affect classical Keplerian elements.

---

### Type 18: ESOC/DDID Hermite/Lagrange Interpolation

**Status**: Not implemented.
**Priority**: Not needed unless supporting ESA mission data.

**Used by**: Mars Express, Rosetta, SMART-1, Venus Express, and other ESA missions.
Designed by the European Space Operations Centre (ESOC).

#### Structure

Has two subtypes:
- **Subtype S**: Hermite interpolation (uses position + velocity)
- **Subtype T**: Lagrange interpolation (uses position only)

A subtype flag in the segment distinguishes the two.

#### Evaluation

Similar to Types 12/13 (Hermite) or Types 8/9 (Lagrange) but with ESOC-specific
record layout conventions.

---

### Type 19: ESOC/DDID Piecewise Interpolation

**Status**: Not implemented.
**Priority**: Not needed.

An enhanced version of Type 18 that requires fewer segments by allowing
variable-size "mini-segments" within a single SPK segment. Each mini-segment
can use either Hermite or Lagrange interpolation independently.

---

### Type 20: Chebyshev Polynomials (Velocity Only)

**Status**: Not implemented.
**Priority**: Not needed unless supporting Russian ephemerides.

**Used by**: Russian EPM (Ephemerides of Planets and the Moon) series only.
This is the velocity-first counterpart to Type 2.

#### Structure

Same layout as Type 2, but coefficients represent **velocity** rather than
position.

#### Evaluation

1. Evaluate Chebyshev expansion directly to get velocity
2. **Integrate** the Chebyshev expansion analytically to get position:
   ```
   integral of T_n(x) = (T_{n+1}(x)/(2(n+1)) - T_{n-1}(x)/(2(n-1)))
   ```
   with appropriate treatment of the T_0 and T_1 terms.
3. Add an integration constant (stored as the first velocity coefficient's
   contribution to position)

---

### Type 21: Extended Modified Difference Arrays

**Status**: Not implemented.
**Priority**: HIGHEST for new implementation -- unlocks the entire Horizons
small-body catalog.

**Used by**: ALL modern Horizons-generated small-body SPK files (post-October
2018), modern spacecraft trajectory reconstructions.

This is the modern successor to Type 1. Since October 2018, JPL's Horizons
system generates Type 21 exclusively for small-body SPK files, making this the
essential type for anyone wanting to work with asteroid or comet ephemerides
from Horizons.

#### Key Difference from Type 1

Type 1 uses a fixed MAXDIM of 15 (maximum number of difference table entries per
component). Type 21 uses a **variable MAXDIM** that is stored per-segment,
typically larger than 15. This allows higher-precision interpolation.

#### Segment Layout

**Segment structure**:

```
[ Record_0 | Record_1 | ... | Record_{N-1} | Epoch_Dir | N | DLSIZE ]
```

- The last word is **DLSIZE** (Difference Line Size = 4 * MAXDIM + 11),
  giving the size of each record in double-precision words.
- The second-to-last word is **N**, the number of records.
- Before N: **epoch directory** with `floor((N-1) / 100)` values for fast
  searching (only present when N > 100).

To determine MAXDIM from DLSIZE:
```
MAXDIM = (DLSIZE - 11) / 4
```

**Each record** (DLSIZE double-precision values):

| Word Index | Count | Field | Description |
|------------|-------|-------|-------------|
| 0          | 1     | TL    | Reference epoch (TDB seconds past J2000) |
| 1..MAXDIM  | MAXDIM | G   | Stepsize function coefficients |
| MAXDIM+1   | 1     | REFPOS_X | Reference X position (km) |
| MAXDIM+2   | 1     | REFPOS_Y | Reference Y position (km) |
| MAXDIM+3   | 1     | REFPOS_Z | Reference Z position (km) |
| MAXDIM+4   | 1     | REFVEL_X | Reference X velocity (km/s) |
| MAXDIM+5   | 1     | REFVEL_Y | Reference Y velocity (km/s) |
| MAXDIM+6   | 1     | REFVEL_Z | Reference Z velocity (km/s) |
| MAXDIM+7   | MAXDIM | DT_X | Modified difference array for X |
| 2*MAXDIM+7 | MAXDIM | DT_Y | Modified difference array for Y |
| 3*MAXDIM+7 | MAXDIM | DT_Z | Modified difference array for Z |
| 4*MAXDIM+7 | 1     | KQMAX1 | Maximum integration order + 1 |
| 4*MAXDIM+8 | 3     | KQ    | Integration orders for X, Y, Z |

Total: **4*MAXDIM + 11 = DLSIZE** doubles per record.

**Verification**: `record_count * DLSIZE + epoch_dir_count + 2 == segment_length`
where `epoch_dir_count = max(0, floor((N-1) / 100))`.

#### Evaluation Algorithm

The evaluation algorithm is identical to Type 1 (Shampine-Gordon modified
divided differences), just with variable MAXDIM. Here is the complete algorithm:

**Step 1: Find the correct record**

If N <= 100, do a linear scan. Otherwise, use the epoch directory for an
initial bracket, then linear scan within that bracket.

```
// Epoch directory has floor((N-1)/100) entries
// Entry i corresponds to record i*100
// Binary search directory, then linear scan
```

Search for the record whose reference epoch `TL` is closest to `t` without
exceeding it. Handle boundary conditions for the first and last records.

**Step 2: Extract record fields**

```rust
let tl = record[0];
let g = &record[1..1+maxdim];
let refpos = [record[maxdim+1], record[maxdim+2], record[maxdim+3]];
let refvel = [record[maxdim+4], record[maxdim+5], record[maxdim+6]];
let dt = [
    &record[maxdim+7 .. 2*maxdim+7],     // X differences
    &record[2*maxdim+7 .. 3*maxdim+7],    // Y differences
    &record[3*maxdim+7 .. 4*maxdim+7],    // Z differences
];
let kqmax1 = record[4*maxdim+7] as usize;
let kq = [
    record[4*maxdim+8] as usize,
    record[4*maxdim+9] as usize,
    record[4*maxdim+10] as usize,
];
```

**Step 3: Compute position**

```
delta = t - tl

// Build factorial-like coefficients from the stepsize function
tp = delta
fc[0] = 1.0
for i in 0..kqmax1-1:
    fc[i+1] = tp / g[i]
    tp = delta + g[i]

// For each component c = 0, 1, 2:
//   Accumulate from highest order down
w[kq[c]-1] = dt[c][kq[c]-1]
for j = kq[c]-2 down to 0:
    w[j] = dt[c][j] + fc[j+1] * w[j+1]

position[c] = refpos[c] + delta * (refvel[c] + delta * w[0])
```

**Step 4: Compute velocity**

```
// Build derivative coefficients
wc[0] = 0.0
for i in 0..kqmax1-1:
    wc[i+1] = fc[i+1] * (i+1) / (delta + g[i])  // derivative of fc

// For each component c:
//   Similar recurrence with wc mixed in
dw[kq[c]-1] = dt[c][kq[c]-1] * wc[kq[c]-1]
for j = kq[c]-2 down to 0:
    dw[j] = dt[c][j] * wc[j] + fc[j+1] * dw[j+1] + wc[j+1] * w[j+1]
    // (where w[j+1] values come from the position computation)

velocity[c] = refvel[c] + delta * (2.0 * w[0] + delta * dw[0])
```

**Reference implementations**:
- `spktype21` Python package (Shushi Uetsuki / MITSuMo group)
- `jplephem` Python package (Brandon Rhodes) -- handles Types 1 and 21

---

## 4. Available Kernel Files from JPL

### 4.1 Planetary Ephemerides

These contain positions for the Sun, Moon, planet barycenters, and sometimes
the Earth itself. All use Type 2 (Chebyshev position-only) segments.

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `de421.bsp` | ~17 MB | 2 | 1899-2053 | 10 Sun, Moon, planet barycenters 1-9, 199, 299, 399 | Good for modern applications, lightweight |
| `de430.bsp` | ~115 MB | 2 | 1549-2650 | Same as de421 | Updated from de421, includes Pluto |
| `de440.bsp` | ~115 MB | 2 | 1549-2650 | Same | Latest standard ephemeris (2021), improved inner planets |
| `de440s.bsp` | ~32 MB | 2 | 1849-2150 | Same | Short-span version of de440 |
| `de441.bsp` | ~3.1 GB | 2 | 13200 BC - AD 17191 | Same | Extended range, identical accuracy to de440 over modern era |

**Recommended default**: `de440s.bsp` (32 MB, covers 1849-2150, sufficient for
most applications). Use `de440.bsp` for historical work, `de441.bsp` only if
you need deep-time coverage.

**Download URL**: `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/`

### 4.2 Asteroid Perturber Kernels

These are large Type 2 kernels containing Chebyshev-fit ephemerides for asteroids
massive enough to gravitationally perturb other bodies. They are used alongside
planetary ephemerides for high-precision work.

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `sb441-n16.bsp` | 616 MB | 2 | ~1800-2200 | 16 most massive asteroids | Ceres, Pallas, Vesta, Hygiea + 12 others |
| `sb441-n373.bsp` | 14.1 GB | 2 | ~1800-2200 | 373 perturber asteroids | Full set used in DE441 integration |
| `sb441-n373s.bsp` | 937 MB | 2 | Reduced span | 373 perturber asteroids | Shorter time coverage, same bodies |
| `codes_300ast_20100725.bsp` | 59 MB | 2 | 1800-2200 | 300 asteroids | CODES/Baer mass model asteroids |

**Download URL**: `https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/`

### 4.3 Satellite Kernels

Natural satellite ephemerides, typically using Type 2 or Type 3 segments. These
are large files because they contain many moons over long time spans.

#### Jupiter System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `jup365.bsp` | 1.03 GB | 2/3 | 1600-2200 | 13 moons | Galilean moons (Io, Europa, Ganymede, Callisto) + 9 inner/outer moons |
| `jup347.bsp` | ~500 MB | 2/3 | 1600-2200 | 12 moons | Previous generation |
| `jup387xl.bsp` | ~2.5 GB | 2/3 | Extended | 13+ moons | Extended version with additional small moons |

#### Saturn System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `sat441l.bsp` | 609 MB | 2/3 | 1749-2250 | 19 moons | Major moons: Mimas through Phoebe, plus smaller bodies |
| `sat441xl.bsp` | ~1.5 GB | 2/3 | Extended | 19+ moons | Extended coverage version |
| `sat456.bsp` | ~700 MB | 2/3 | 1749-2250 | 19+ moons | Updated satellite ephemeris |

#### Mars System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `mar099.bsp` | 1.10 GB | 2/3 | 1600-2600 | Phobos + Deimos | Long-span, high-precision satellite ephemeris |

#### Uranus System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `ura184.bsp` | 4.14 GB | 2/3 | 1600-2399 | Major moons | Miranda, Ariel, Umbriel, Titania, Oberon + others |
| `ura111.30kyr.bsp` | 6.78 GB | 2/3 | 30,000 yr span | Major moons | Ultra-long coverage for dynamical studies |

#### Neptune System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `nep097.bsp` | 3.01 GB | 2/3 | 1600-2399 | Triton + others | Standard Neptune satellite ephemeris |
| `nep104.bsp` | ~3 GB | 2/3 | 1600-2399 | Updated set | Updated from nep097 |
| `Triton.nep097.30kyr.bsp` | 2.40 GB | 2/3 | 30,000 yr span | Triton only | Ultra-long Triton coverage |

#### Pluto System

| File | Size | Type | Coverage | Bodies | Notes |
|------|------|------|----------|--------|-------|
| `plu060.bsp` | 111 MB | 2/3 | ~1900-2100 | 5 moons | Charon, Nix, Hydra, Kerberos, Styx |

**Download URL**: `https://ssd.jpl.nasa.gov/ftp/eph/satellites/bsp/`

### 4.4 Notable Individual Small-Body Kernels

These are typically Type 21 (modern) or Type 1 (legacy) segments generated by
JPL's Horizons system for specific asteroids or comets.

| Body | File | Size | Notes |
|------|------|------|-------|
| Apophis (99942) | Solutions 216-220 | 69-176 MB | Multiple trajectory solutions, extensively studied PHA |
| Comet 67P/Churyumov-Gerasimenko | `sb-67p-k151-6.bsp` | 117 KB | Rosetta target |
| Comet Siding Spring (C/2013 A1) | 8 kernel files | 1-37 MB | Mars close approach Oct 2014 |
| Didymos/Dimorphos (65803) | `sb-65803-205.bsp` | 231 KB | DART mission target asteroid |
| Dimorphos | `dimorphos_s542.bsp` | 48.8 MB | DART impact target moon |

**Download URL**: `https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/`

### 4.5 Download URLs Summary

| Category | URL |
|----------|-----|
| Planetary ephemerides | `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/` |
| Satellites | `https://ssd.jpl.nasa.gov/ftp/eph/satellites/bsp/` |
| Asteroid perturbers | `https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/asteroids_de441/` |
| Individual small bodies | `https://ssd.jpl.nasa.gov/ftp/eph/small_bodies/` |
| NAIF generic kernels | `https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/` |

---

## 5. Segment Addressing and Chain Resolution

### 5.1 NAIF Body ID Convention

Bodies are identified by integer IDs following NAIF conventions:

| ID Range | Category | Examples |
|----------|----------|---------|
| 0 | Solar System Barycenter (SSB) | Origin of the ICRF |
| 1-9 | Planet barycenters | 1=Mercury BC, 3=Earth-Moon BC, 5=Jupiter BC |
| 10 | Sun | |
| 1xx | Planet (xx=99) or satellites (xx=01,02,...) | 199=Mercury, 399=Earth, 301=Moon, 401=Phobos |
| 2xxxxxx | Asteroids | 2000001=Ceres, 2000004=Vesta |
| 1xxxxxxx | Comets | |

### 5.2 Segment Graph

Each SPK segment defines a directed edge in a body graph:
```
center_body ---> target_body
```

For example, `de440.bsp` contains these segments:

```
SSB (0) ---> Mercury Barycenter (1)
SSB (0) ---> Venus Barycenter (2)
SSB (0) ---> Earth-Moon Barycenter (3)
SSB (0) ---> Mars Barycenter (4)
SSB (0) ---> Jupiter Barycenter (5)
SSB (0) ---> Saturn Barycenter (6)
SSB (0) ---> Uranus Barycenter (7)
SSB (0) ---> Neptune Barycenter (8)
SSB (0) ---> Pluto Barycenter (9)
SSB (0) ---> Sun (10)
Earth-Moon Barycenter (3) ---> Moon (301)
Earth-Moon Barycenter (3) ---> Earth (399)
Mercury Barycenter (1) ---> Mercury (199)
Venus Barycenter (2) ---> Venus (299)
Mars Barycenter (4) ---> Mars (499)
```

### 5.3 Chain Resolution

To compute the position of body A relative to the SSB, you must find a **chain**
of segments from SSB to body A and sum their contributions.

**Example**: Position of Mars (499) relative to SSB:

```
Chain: SSB(0) -> Mars Barycenter(4) -> Mars(499)
Position = segment(0,4).compute(t) + segment(4,499).compute(t)
```

**Example**: Position of the Moon (301) relative to Earth (399):

```
Moon relative to SSB:  chain(0->3) + chain(3->301)
Earth relative to SSB: chain(0->3) + chain(3->399)
Moon relative to Earth: (chain_moon - chain_earth)
                      = chain(3->301) - chain(3->399)
```

### 5.4 BFS Algorithm

Starfield uses Breadth-First Search from SSB (ID=0) to build chains:

```rust
// Build adjacency list from segments
let mut adj: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
for seg in &spk.segments {
    adj.entry(seg.center).or_default().push((seg.center, seg.target));
}

// BFS from SSB
let mut queue = VecDeque::new();
let mut parent: HashMap<i32, (i32, i32)> = HashMap::new();
queue.push_back(0);
parent.insert(0, (0, 0)); // sentinel

while let Some(node) = queue.pop_front() {
    if let Some(edges) = adj.get(&node) {
        for &(center, target) in edges {
            if !parent.contains_key(&target) {
                parent.insert(target, (center, target));
                queue.push_back(target);
            }
        }
    }
}

// Reconstruct chain for target by walking parent pointers back to SSB
```

This precomputation runs once when opening a kernel. Subsequent lookups are O(1)
hash map access plus O(chain_length) segment evaluations.

---

## 6. Time System

### 6.1 TDB (Barycentric Dynamical Time)

All epochs in SPK files are in **TDB seconds past J2000**.

- **J2000 epoch**: 2000 January 1, 12:00:00.000 TDB = JD 2451545.0 TDB
- **TDB**: The independent time argument for solar system dynamics, closely
  tracks TT (Terrestrial Time) with periodic variations < 2 ms

### 6.2 Conversions

```
TDB_seconds = (JD_TDB - 2451545.0) * 86400.0
JD_TDB = 2451545.0 + TDB_seconds / 86400.0
```

For sub-microsecond precision, split the computation:
```rust
pub fn jd_to_seconds(jd: f64) -> f64 {
    (jd - 2451545.0) * 86400.0
}
```

The `compute_and_differentiate(tdb, tdb2)` interface in starfield accepts a
split time where `tdb + tdb2` is the total TDB seconds. This preserves
precision when working with large epoch values.

### 6.3 TDB vs TT

TDB differs from TT by periodic terms:

```
TDB - TT = 0.001657 * sin(628.3076 * T + 6.2401)
          + 0.000022 * sin(575.3385 * T + 4.2970)
          + ... (smaller terms)
```

where T is Julian centuries of TDB past J2000. The maximum difference is
approximately 1.7 ms. For most applications (position accuracy > 1 km), TDB
and TT can be used interchangeably. For sub-km precision, the correction
matters.

---

## 7. Reference Frames

### 7.1 Frame IDs in SPK Files

The FRAME integer in each summary identifies the reference frame:

| Frame ID | Name | Description |
|----------|------|-------------|
| 1 | J2000 | ICRF-aligned Earth mean equator and equinox of J2000. This is the standard frame for virtually all SPK files. |
| 2 | B1950 | FK4 mean equator and equinox of B1950.0. Used by some legacy files. |
| 17 | ECLIPJ2000 | Mean ecliptic and equinox of J2000. Rotated from J2000 by the obliquity of the ecliptic (~23.4 degrees). |

**In practice**: Nearly all modern SPK files use frame 1 (J2000/ICRF). The J2000
frame is aligned with the International Celestial Reference Frame to within
the accuracy of the frame tie (~0.01 arcseconds).

### 7.2 J2000 to ECLIPJ2000 Rotation

The obliquity of the ecliptic at J2000:
```
epsilon = 23.439291111 degrees = 0.40909280422 radians
```

Rotation matrix (J2000 equatorial -> ECLIPJ2000 ecliptic):
```
R = [[1,          0,           0         ],
     [0,  cos(eps),   sin(eps) ],
     [0, -sin(eps),   cos(eps) ]]
```

---

## 8. Units

### 8.1 Internal SPK Units

All SPK files store:
- **Positions**: kilometers (km)
- **Velocities**: kilometers per second (km/s) for most types
  - Exception: some Type 2 velocity is derived as km/s from position differentiation
  - Exception: Type 20 stores velocity coefficients in km/s directly

### 8.2 Starfield API Units

Starfield's high-level `SpiceKernel` API converts to astronomical units:

```rust
const AU_KM: f64 = 149_597_870.700;  // IAU 2012 exact definition
const S_PER_DAY: f64 = 86400.0;

// Position: km -> AU
position_au = position_km / AU_KM;

// Velocity: km/s -> AU/day
velocity_au_day = velocity_km_s * S_PER_DAY / AU_KM;
```

The raw `Segment::compute_and_differentiate()` returns km and km/s.

---

## 9. Implementation Notes for Rust

### 9.1 Binary Parsing

- Use the `byteorder` crate for endian-aware reading (already used in starfield)
- Use `memmap2` for memory-mapped I/O on large kernels (already used in starfield)
- Records are always 1024 bytes; addresses are 1-indexed double-word positions
- Byte offset from address: `(address - 1) * 8`

### 9.2 Memory-Mapped I/O

For large kernels (de441.bsp = 3.1 GB, sb441-n373.bsp = 14.1 GB), memory-mapped
I/O is essential. The OS will page in data on demand. Starfield already implements
this via `memmap2::Mmap`.

```rust
// Current starfield approach: mmap with file fallback
if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
    self.map = Some(mmap);
}
```

### 9.3 Type 2/3 Implementation (Already Done)

Fixed-size records with uniform time intervals make these straightforward:

1. Read metadata (last 4 words of segment)
2. Compute record index from time: `floor((t - INIT) / INTLEN)`
3. Extract coefficients from `data[index * RSIZE .. (index+1) * RSIZE]`
4. Clenshaw evaluation
5. Derivative via Chebyshev second-kind polynomials

See `src/jplephem/spk.rs` for the complete implementation.

### 9.4 Type 21 Implementation (Next Priority)

Key considerations:

- **Variable record size**: Read DLSIZE from last word of segment, derive MAXDIM
- **Epoch search**: For N > 100, use epoch directory for O(log N) initial bracket
- **Stepsize function**: The `G` array encodes non-uniform integration steps
- **Divided difference recurrence**: Careful with indexing -- the algorithm builds
  coefficients from highest order down
- **Velocity**: Requires tracking both `fc` and `wc` (derivative of `fc`)
  coefficients simultaneously

```rust
struct Type21Record {
    tl: f64,                    // Reference epoch
    g: Vec<f64>,                // Stepsize function [MAXDIM]
    refpos: [f64; 3],           // Reference position
    refvel: [f64; 3],           // Reference velocity
    dt: [Vec<f64>; 3],          // Difference tables [3][MAXDIM]
    kqmax1: usize,              // Max integration order + 1
    kq: [usize; 3],            // Integration orders per component
}
```

### 9.5 Type 1 Implementation

Identical to Type 21 but with fixed MAXDIM = 15, so record size is always
`4 * 15 + 11 = 71` doubles. Can share the evaluation code with Type 21.

### 9.6 Type 5 Implementation

Requires:
1. Binary search of epoch list
2. Kepler equation solver (already exists in `keplerlib`)
3. State vector <-> orbital elements conversion
4. Two-body propagation from nearest state to target epoch

### 9.7 Segment Caching

Starfield caches decoded segment data in the `Segment::data` field (lazy
initialization). This avoids re-parsing coefficient arrays on repeated queries
to the same segment. For Type 21, consider caching the parsed records and
epoch list similarly.

### 9.8 Error Handling

Follow the existing pattern using `JplephemError`:
- `OutOfRangeError` when epoch is outside segment coverage
- `UnsupportedDataType` for unimplemented segment types
- `InvalidFormat` for corrupt or unexpected data layouts
- `BodyNotFound` when no segment exists for a requested body pair

### 9.9 Testing Strategy

- **Unit tests**: Verify coefficient extraction and Clenshaw evaluation against
  known values
- **Python comparison tests**: Use the `pybridge` infrastructure to compare
  Rust results against `skyfield`/`jplephem` for the same inputs
- **Round-trip tests**: Evaluate at segment boundary times and verify continuity
- **Edge cases**: First/last record, exact epoch boundaries, very short segments

### 9.10 Reference Python Packages

| Package | Types Supported | Notes |
|---------|----------------|-------|
| `jplephem` | 1, 2, 3, 21 | Brandon Rhodes, used by Skyfield |
| `spktype01` | 1 | Shushi Uetsuki, standalone Type 1 reader |
| `spktype21` | 21 | Shushi Uetsuki, standalone Type 21 reader |
| `spiceypy` | All | Python wrapper around NAIF C toolkit |

For algorithm details, `spktype21` and `jplephem` are the most readable
reference implementations. The NAIF C toolkit (`cspice`) is authoritative but
substantially harder to read.
