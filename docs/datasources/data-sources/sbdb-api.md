# JPL Small-Body Database (SBDB) API Ecosystem

Implementation reference for building a Rust client crate targeting the JPL SSD/CNEOS API
service. All information derived from the official JPL documentation at
<https://ssd-api.jpl.nasa.gov/>.

---

## 1. Overview

The JPL Small-Body Database (SBDB) is the authoritative database of orbital elements,
physical parameters, and discovery circumstances for all known asteroids and comets
(approximately 1.5 million objects as of 2026). It is maintained by NASA's Jet Propulsion
Laboratory Solar System Dynamics (SSD) group and the Center for Near-Earth Object Studies
(CNEOS).

The API ecosystem exposes 17 individual endpoints. All share these properties:

- **Base URL:** `https://ssd-api.jpl.nasa.gov/`
- **Method:** HTTP GET (all endpoints)
- **Auth:** None required. No API keys.
- **Format:** All responses are JSON
- **Rate limits:** No published rate limits. Reasonable use expected (avoid aggressive polling)
- **CORS:** Responses include appropriate CORS headers for browser use

### Common Response Envelope

Every API response includes a `signature` object:

```json
{
  "signature": {
    "version": "string",
    "source": "string"
  }
}
```

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200  | OK (may include empty results or soft errors) |
| 300  | Multiple Choices (ambiguous object query matched multiple objects) |
| 400  | Bad Request (invalid parameters) |
| 405  | Method Not Allowed (must use GET) |
| 500  | Internal Server Error (database unavailable) |
| 503  | Service Unavailable (temporary overload/maintenance) |

### API Catalog

| API | Endpoint | Description |
|-----|----------|-------------|
| SBDB | `sbdb.api` | Single object lookup with full detail |
| SBDB Query | `sbdb_query.api` | Bulk filtered queries across all objects |
| Close Approach (CAD) | `cad.api` | Asteroid/comet close approaches to planets |
| Fireball | `fireball.api` | Atmospheric impact events from US Government sensors |
| Sentry | `sentry.api` | Earth impact risk monitoring |
| Scout | `scout.api` | NEOCP unconfirmed object analysis |
| NHATS | `nhats.api` | Human-accessible NEA mission data |
| Mission Design | `mdesign.api` | Small-body trajectory design |
| SB Identification | `sb_ident.api` | Identify objects in a field of view |
| SB Observability | `sbwobs.api` | Observable small bodies for a given night |
| SB Radar | `sb_radar.api` | Radar astrometry measurements |
| Horizons | `horizons.api` | Ephemeris generation (query-parameter interface) |
| Horizons File | `horizons_file.api` | Ephemeris generation (file-based interface) |
| Horizons Lookup | `horizons_lookup.api` | Horizons body lookup |
| JD Converter | `jdconv.api` | Julian Day / calendar date converter |
| Periodic Orbits | `periodic_orbits.api` | Periodic orbit database |
| SB Satellites | (via `sbdb.api?sat=true`) | Small-body satellite data |

---

## 2. SBDB API (Single Object Lookup)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sbdb.api`

Retrieves comprehensive data for a single small body: orbital elements, physical parameters,
close-approach data, discovery circumstances, virtual impactor data, radar observations,
satellite data, and alternate designations.

### 2.1 Request Parameters

#### Object Selection (exactly one required)

| Parameter | Type | Description |
|-----------|------|-------------|
| `sstr` | string | Search string: designation, MPC packed form, case-insensitive name, SPK-ID. Wildcard `*` allowed |
| `spk` | integer | SPK-ID (e.g., `2000433` for Eros) |
| `des` | string | Designation or IAU number (e.g., `2015 AB`, `141P`, `433`) |

#### Optional Parameters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `neo` | integer | `0` | `0`, `1`, `2` | NEO filter: 0=all, 1=NEO only, 2=non-NEO match returns minimal data |
| `alt-des` | boolean | `false` | | Include alternate designations |
| `alt-spk` | boolean | `false` | | Include alternate SPK-IDs |
| `full-prec` | boolean | `false` | | Full precision output (16 significant digits) |
| `soln-epoch` | boolean | `false` | | Orbit at JPL solution epoch instead of MPC epoch |
| `cd-epoch` | boolean | `false` | | Add calendar date/time of orbit epoch |
| `cd-tp` | boolean | `false` | | Add calendar date/time of perihelion passage |
| `cov` | string | (none) | `mat`, `vec`, `src` | Orbital covariance: matrix, vector, or square-root form |
| `nv-fmt` | string | (none) | `jd`, `cd` | Format for `not_valid_before`/`not_valid_after` |
| `anc-data` | boolean | `false` | | Output ancillary data availability summary |
| `no-orbit` | boolean | `false` | | Suppress default orbit output |
| `alt-orbits` | boolean | `false` | | Output alternate non-default orbit solutions |
| `orbit-defs` | boolean | `false` | | Output orbit parameter definitions |
| `sat` | boolean | `false` | | Output satellite data |
| `phys-par` | boolean | `false` | | Output physical parameters |
| `ca-data` | boolean | `false` | | Output close-approach data |
| `ca-body` | string | (all) | `Merc`, `Venus`, `Earth`, `Mars`, `Juptr`, `Satrn`, `Urnus`, `Neptn`, `Pluto`, `Moon` | Limit close approaches to specified body |
| `ca-time` | string | `cd` | `cd`, `jd`, `both` | Close approach time format |
| `ca-tunc` | string | `num` | `num`, `fmt`, `both` | Close approach time uncertainty format |
| `ca-unc` | boolean | `false` | | Include close approach distance uncertainty ellipse |
| `radar-obs` | boolean | `false` | | Output radar astrometry data |
| `r-name` | boolean | `false` | | Include radar station names |
| `r-observer` | boolean | `false` | | Include observer field in radar data |
| `r-notes` | boolean | `false` | | Include notes field in radar data |
| `vi-data` | boolean | `false` | | Output virtual impactor (Sentry) data |
| `discovery` | boolean | `false` | | Output discovery circumstances |
| `raw-citation` | boolean | `false` | | Output IAU citation in raw LaTeX format |

### 2.2 Response Structure

#### `object` (always present)

```json
{
  "object": {
    "des": "string",
    "spkid": "string",
    "fullname": "string",
    "shortname": "string",
    "prefix": "string | null",
    "kind": "string",
    "neo": true,
    "pha": true,
    "orbit_class": {
      "name": "string",
      "code": "string"
    },
    "orbit_id": "string",
    "des_alt": [
      { "des": "string" }
    ],
    "spkid_alt": ["string"],
    "anc_data": {
      "ca_data": { "count": 0, "ref_url": "string" },
      "ra_data": { "count": 0, "ref_url": "string" },
      "vi_data": { "count": 0, "ref_url": "string" },
      "nhats_data": { "count": 0, "ref_url": "string" }
    }
  }
}
```

**`kind` values:**

| Code | Meaning |
|------|---------|
| `an` | Numbered asteroid |
| `au` | Unnumbered asteroid |
| `cn` | Numbered comet |
| `cu` | Unnumbered comet |

#### `orbit` (default, suppressed with `no-orbit=true`)

```json
{
  "orbit": {
    "orbit_id": "string",
    "epoch": "string (JD TDB)",
    "cd_epoch": "string (YYYY-MMM-DD.D, if cd-epoch=true)",
    "equinox": "string",
    "elements": [
      {
        "name": "string",
        "label": "string",
        "title": "string",
        "value": "string",
        "sigma": "string",
        "units": "string | null"
      }
    ],
    "model_pars": [
      {
        "n": 0,
        "kind": "SET | EST | CON",
        "name": "string",
        "value": "string",
        "title": "string",
        "desc": "string",
        "sigma": "string",
        "units": "string | null"
      }
    ],
    "covariance": {
      "epoch": "string (JD)",
      "data": [0.0],
      "labels": ["string"],
      "elements": []
    },
    "cov_epoch": "string (JD)",
    "moid": "string (AU)",
    "moid_jup": "string (AU)",
    "t_jup": "string",
    "condition_code": "string (0-9 or D)",
    "rms": "string",
    "first_obs": "string (YYYY-MM-DD)",
    "last_obs": "string (YYYY-MM-DD)",
    "data_arc": "string (days)",
    "n_obs_used": "string",
    "n_del_obs_used": "string",
    "n_dop_obs_used": "string",
    "pe_used": "string",
    "sb_used": "string",
    "two_body": "boolean | null",
    "soln_date": "string (datetime)",
    "source": "string (JPL | MPC | SAO)",
    "producer": "string",
    "not_valid_before": "string | null",
    "not_valid_after": "string | null",
    "comment": "string | null"
  }
}
```

**Orbital elements (`elements` array, by `name`):**

| Name | Label | Units | Description |
|------|-------|-------|-------------|
| `e` | Eccentricity | (none) | Orbital eccentricity |
| `a` | Semi-major axis | AU | Semi-major axis (not defined for parabolic orbits) |
| `q` | Perihelion distance | AU | Perihelion distance |
| `i` | Inclination | deg | Orbital inclination w.r.t. ecliptic |
| `om` | Longitude of ascending node | deg | Longitude of the ascending node |
| `w` | Argument of perihelion | deg | Argument of perihelion |
| `tp` | Time of perihelion | JD (TDB) | Time of perihelion passage |
| `cd_tp` | Time of perihelion (cal.) | (date) | Calendar date of perihelion (if `cd-tp=true`) |
| `ma` | Mean anomaly | deg | Mean anomaly at epoch |
| `n` | Mean motion | deg/d | Mean motion |
| `per` | Orbital period | d | Sidereal orbital period |
| `ad` | Aphelion distance | AU | Aphelion distance |

#### `phys_par` (if `phys-par=true`)

Array of physical parameter objects:

```json
{
  "phys_par": [
    {
      "name": "string",
      "value": "string",
      "sigma": "string | null",
      "units": "string | null",
      "ref": "string | null",
      "notes": "string | null",
      "title": "string | null",
      "desc": "string | null"
    }
  ]
}
```

**Common physical parameter names:**

| Name | Units | Description |
|------|-------|-------------|
| `H` | mag | Absolute magnitude (V-band) |
| `G` | (none) | Magnitude slope parameter |
| `diameter` | km | Effective diameter |
| `extent` | km | Tri-axial body extents |
| `albedo` | (none) | Geometric albedo |
| `rot_per` | h | Rotation period |
| `BV` | mag | Color index B-V |
| `UB` | mag | Color index U-B |
| `IR` | (none) | IR albedo |
| `spec_T` | (none) | Tholen spectral type |
| `spec_B` | (none) | SMASSII spectral type |
| `GM` | km^3/s^2 | Standard gravitational parameter |
| `density` | g/cm^3 | Bulk density |
| `pole` | (none) | Spin pole direction |

#### `ca_data` (if `ca-data=true`)

Array of close-approach records:

```json
{
  "ca_data": [
    {
      "body": "string",
      "jd": "string (JD TDB)",
      "cd": "string (YYYY-MMM-DD hh:mm)",
      "sigma_t": "string (minutes)",
      "sigma_tf": "string (d_hh:mm format)",
      "dist": "string (AU)",
      "dist_min": "string (AU)",
      "dist_max": "string (AU)",
      "v_rel": "string (km/s)",
      "v_inf": "string (km/s)",
      "unc_major": "string (km)",
      "unc_minor": "string (km)",
      "unc_angle": "string (deg)",
      "orbit_ref": "string"
    }
  ]
}
```

#### `vi_data` (if `vi-data=true`)

Array of virtual impactor records:

```json
{
  "vi_data": [
    {
      "date": "string (YYYY-MM-DD.DD UTC)",
      "dt": 0.0,
      "ps": "string (Palermo scale)",
      "ts": "string (Torino scale)",
      "ip": "string (impact probability)",
      "width": "string (Earth radii)",
      "energy": "string (Mt TNT)",
      "stretch": "string (Earth radii/sigma)",
      "dist": "string (Earth radii)",
      "sigma_vi": "string",
      "sigma_lov": "string",
      "sigma_imp": "string",
      "v_inf": "string (km/s)",
      "v_imp": "string (km/s)",
      "h": "string",
      "diam": "string (km)",
      "mass": "string (kg)",
      "method": "IOBS | LOV | MC"
    }
  ]
}
```

#### `radar_obs` (if `radar-obs=true`)

```json
{
  "radar_obs": [
    {
      "epoch": "string (YYYY-MM-DD hh:mm:ss UT)",
      "value": "string",
      "sigma": "string",
      "units": "us | Hz",
      "freq": "string (MHz)",
      "rcvr": "string (station code)",
      "xmit": "string (station code)",
      "rcvr_name": "string (if r-name=true)",
      "xmit_name": "string (if r-name=true)",
      "bp": "C | P",
      "observer": "string | null (if r-observer=true)",
      "notes": "string | null (if r-notes=true)"
    }
  ]
}
```

#### `discovery` (if `discovery=true`)

```json
{
  "discovery": {
    "date": "string (YYYY-MMM-DD)",
    "location": "string | null",
    "site": "string | null",
    "who": "string",
    "ref": "string",
    "name": "string | null",
    "discovery": "string | null",
    "citation": "string | null",
    "cref": "string | null"
  }
}
```

#### `sat` (if `sat=true`)

Array of satellite objects:

```json
{
  "sat": [
    {
      "fullname": "string",
      "prov_des": "string",
      "year": 0,
      "iau_num": 0,
      "iau_name": "string | null",
      "oid": "string",
      "orbit": {},
      "notes": "string | null",
      "ref": "string | null"
    }
  ]
}
```

Satellite orbital elements use meters (not AU) for `a` and `q`.

#### Multiple Object Match (HTTP 300)

```json
{
  "code": 300,
  "message": "specified query matched more than one object",
  "list": [
    { "pdes": "string", "name": "string" }
  ]
}
```

---

## 3. SBDB Query API (Bulk Filtered Queries)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sbdb_query.api`

Powerful bulk query interface supporting SQL-like filtering across the entire small-body
database. Returns tabular data with user-selected columns.

### 3.1 Request Parameters

#### Information Mode

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `info` | string | `count`, `field`, `all` | Returns dataset statistics (`count`), field metadata (`field`), or both (`all`) |

#### Query Mode

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fields` | string | (none) | Comma-separated field names to return (case-sensitive) |
| `sort` | string | (none) | Up to 3 fields comma-separated; prefix `-` for descending |
| `limit` | integer | (none) | Max records to return |
| `limit-from` | integer | (none) | Pagination offset (requires `limit`) |
| `full-prec` | boolean | `false` | Full precision output |

#### SBDB Filter Parameters

| Parameter | Type | Values | Description |
|-----------|------|--------|-------------|
| `sb-ns` | string | `n`, `u` | Numbered (`n`) or unnumbered (`u`) only |
| `sb-kind` | string | `a`, `c` | Asteroids (`a`) or comets (`c`) only |
| `sb-group` | string | see below | Predefined orbital group |
| `sb-class` | string | see below | Orbit class code(s), comma-separated |
| `sb-sat` | boolean | | Only objects with known satellites |
| `sb-xfrag` | boolean | | Exclude comet fragments |
| `sb-cdata` | string | JSON | Custom field constraints (max 2048 chars) |

**`sb-group` values:**

| Value | Description |
|-------|-------------|
| `neo` | Near-Earth Objects |
| `pha` | Potentially Hazardous Asteroids |
| `atira` | Atira-class (IEO) |
| `aten` | Aten-class |
| `apollo` | Apollo-class |
| `amor` | Amor-class |
| `mars-crosser` | Mars-crossing asteroids |
| `MBA` | Main Belt asteroids (all sub-classes) |
| `trojan` | Jupiter Trojans |
| `centaur` | Centaurs |
| `TNO` | Trans-Neptunian Objects |

### 3.2 Filter Constraint Syntax (`sb-cdata`)

JSON-formatted constraint expressions. Structure:

```json
{"AND": ["field|operator|value", "field|operator|value"]}
```

or

```json
{"OR": ["field|operator|value", "field|operator|value"]}
```

Individual constraint format: `"FIELD|OPERATOR|VALUE"` or `"FIELD|OPERATOR|VALUE1|VALUE2"`
for range operators.

**Operators:**

| Code | Meaning | Values Required |
|------|---------|-----------------|
| `EQ` | Equal to | 1 |
| `NE` | Not equal to | 1 |
| `LT` | Less than | 1 |
| `GT` | Greater than | 1 |
| `LE` | Less than or equal to | 1 |
| `GE` | Greater than or equal to | 1 |
| `RG` | Inclusive range | 2 (min and max) |
| `RE` | Regular expression match | 1 (regex pattern) |
| `DF` | Value is defined (not NULL) | 0 |
| `ND` | Value is undefined (NULL) | 0 |

**Example:** All PHAs with diameter > 1 km:

```
sb-group=pha&sb-cdata={"AND":["diameter|GT|1"]}
```

### 3.3 Queryable Fields

#### Object Identity

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `spkid` | string | | SPK-ID |
| `full_name` | string | | Full name/designation |
| `pdes` | string | | Primary designation |
| `name` | string | | IAU name (if assigned) |
| `prefix` | string | | Name prefix (e.g., comet type) |
| `kind` | string | | Object kind: `an`, `au`, `cn`, `cu` |
| `class` | string | | Orbit classification code |
| `neo` | string | | `Y` if NEO |
| `pha` | string | | `Y` if PHA |
| `orbit_id` | string | | Orbit solution identifier |

#### Orbital Elements

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `epoch` | float | JD (TDB) | Osculation epoch |
| `equinox` | string | | Reference equinox |
| `e` | float | (none) | Eccentricity |
| `a` | float | AU | Semi-major axis |
| `q` | float | AU | Perihelion distance |
| `i` | float | deg | Inclination |
| `om` | float | deg | Longitude of ascending node |
| `w` | float | deg | Argument of perihelion |
| `ma` | float | deg | Mean anomaly |
| `tp` | float | JD (TDB) | Time of perihelion passage |
| `per` | float | d | Orbital period |
| `n` | float | deg/d | Mean motion |
| `ad` | float | AU | Aphelion distance |

#### Orbital Element Uncertainties

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `sigma_e` | float | | 1-sigma uncertainty in eccentricity |
| `sigma_a` | float | AU | 1-sigma uncertainty in semi-major axis |
| `sigma_q` | float | AU | 1-sigma uncertainty in perihelion distance |
| `sigma_i` | float | deg | 1-sigma uncertainty in inclination |
| `sigma_om` | float | deg | 1-sigma uncertainty in node |
| `sigma_w` | float | deg | 1-sigma uncertainty in arg. perihelion |
| `sigma_tp` | float | d | 1-sigma uncertainty in perihelion time |
| `sigma_ma` | float | deg | 1-sigma uncertainty in mean anomaly |
| `sigma_per` | float | d | 1-sigma uncertainty in period |
| `sigma_n` | float | deg/d | 1-sigma uncertainty in mean motion |
| `sigma_ad` | float | AU | 1-sigma uncertainty in aphelion distance |

#### Derived Orbital Properties

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `t_jup` | float | | Tisserand parameter w.r.t. Jupiter |
| `moid` | float | AU | Minimum orbit intersection distance (Earth) |
| `moid_jup` | float | AU | MOID w.r.t. Jupiter |
| `moid_ld` | float | LD | MOID in lunar distances |

#### Orbit Quality

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `source` | string | | Orbit source (`JPL`, `MPC`, `SAO`) |
| `soln_date` | string | | Solution date |
| `producer` | string | | Orbit producer |
| `data_arc` | float | d | Observational data arc span |
| `first_obs` | string | | First observation date |
| `last_obs` | string | | Last observation date |
| `n_obs_used` | integer | | Number of optical observations used |
| `n_del_obs_used` | integer | | Number of radar delay observations used |
| `n_dop_obs_used` | integer | | Number of radar Doppler observations used |
| `condition_code` | string | | Orbit condition code (0-9, D) |
| `rms` | float | | Weighted RMS residual of fit |
| `two_body` | string | | `Y` if 2-body mechanics used |

#### Non-gravitational Parameters

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `A1` | float | AU/d^2 | Radial non-gravitational parameter |
| `A2` | float | AU/d^2 | Transverse non-gravitational parameter |
| `A3` | float | AU/d^2 | Normal non-gravitational parameter |
| `DT` | float | d | Non-gravitational delay parameter |

#### Physical Parameters

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `H` | float | mag | Absolute magnitude |
| `G` | float | | Magnitude slope parameter |
| `M1` | float | mag | Total comet magnitude parameter |
| `M2` | float | mag | Nuclear comet magnitude parameter |
| `K1` | float | | Total comet magnitude slope |
| `K2` | float | | Nuclear comet magnitude slope |
| `PC` | float | | Comet phase coefficient |
| `diameter` | float | km | Effective body diameter |
| `extent` | string | km | Tri-axial extents |
| `GM` | float | km^3/s^2 | Standard gravitational parameter |
| `density` | float | g/cm^3 | Bulk density |
| `rot_per` | float | h | Rotation period |
| `pole` | string | | Spin-pole direction |
| `albedo` | float | | Geometric albedo |
| `BV` | float | mag | Color index B-V |
| `UB` | float | mag | Color index U-B |
| `IR` | float | | IR albedo |
| `spec_T` | string | | Tholen spectral type |
| `spec_B` | string | | SMASSII spectral type |

### 3.4 Orbit Classifications

#### Asteroid Classes

| Code | Name | Definition |
|------|------|------------|
| `IEO` | Atira | Interior Earth Object: a < 1.0 AU, Q (aphelion) < 0.983 AU |
| `ATE` | Aten | a < 1.0 AU, Q > 0.983 AU (Earth-crossing from inside) |
| `APO` | Apollo | a > 1.0 AU, q < 1.017 AU (Earth-crossing from outside) |
| `AMO` | Amor | 1.017 AU < q < 1.3 AU (Earth-approaching, non-crossing) |
| `MCA` | Mars-crosser | 1.3 AU < q < 1.666 AU |
| `IMB` | Inner Main Belt | 2.0 AU < a < 2.5 AU, q > 1.666 AU |
| `MBA` | Main Belt | 2.5 AU < a < 2.82 AU, q > 1.666 AU |
| `OMB` | Outer Main Belt | 2.82 AU < a < 3.27 AU, q > 1.666 AU |
| `TJN` | Jupiter Trojan | Near Jupiter L4/L5 Lagrange points |
| `CEN` | Centaur | 5.5 AU < a < 30.1 AU |
| `TNO` | Trans-Neptunian Object | a > 30.1 AU |
| `HUN` | Hungaria | 1.78 AU < a < 2.0 AU, e < 0.18, i: 16-34 deg |
| `HIL` | Hilda | 3.7 AU < a < 4.1 AU (3:2 resonance with Jupiter) |
| `AST` | Asteroid | Generic asteroid (not otherwise classified) |
| `PAA` | Parabolic Asteroid | e approximately 1.0 |
| `HYA` | Hyperbolic Asteroid | e > 1.0 |

#### Comet Classes

| Code | Name | Definition |
|------|------|------------|
| `COM` | Comet | Generic long-period comet (P > 200 yr) |
| `HYP` | Hyperbolic Comet | e > 1.0 (unbound orbit) |
| `PAR` | Parabolic Comet | e approximately 1.0 |
| `JFc` | Jupiter-family Comet | 2 < T_Jupiter < 3 |
| `JFC` | Jupiter-family Comet (alt) | Alternative JFC classification |
| `HTC` | Halley-type Comet | T_Jupiter < 2, P < 200 yr |
| `ETc` | Encke-type Comet | a < 2.0 AU, T_Jupiter > 3 (only 2P/Encke) |
| `CTc` | Chiron-type Comet | T_Jupiter > 3, a > 5.5 AU |

**NEO Definition:** An object with q < 1.3 AU.

**PHA Definition:** An asteroid with MOID <= 0.05 AU and H <= 22.0.

### 3.5 Response Structure

#### Info Mode (`info=count`)

```json
{
  "info": {
    "count": {
      "au": "string (count of unnumbered asteroids)",
      "cu": "string (count of unnumbered comets)",
      "cn": "string (count of numbered comets)",
      "an": "string (count of numbered asteroids)"
    }
  }
}
```

#### Query Mode

```json
{
  "signature": { "version": "string", "source": "string" },
  "count": "string",
  "fields": ["string"],
  "data": [
    ["value", "value", ...]
  ]
}
```

Values in the `data` array are strings. The `fields` array provides the column names
matching the requested `fields` parameter.

---

## 4. Close Approach Data API (CAD)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/cad.api`

Returns close-approach data for asteroids and comets approaching the major planets, the Moon,
and Pluto.

### 4.1 Request Parameters

#### Date Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date-min` | string | `now` | Start date: `YYYY-MM-DD`, `YYYY-MM-DDThh:mm:ss`, or `now` |
| `date-max` | string | `+60` | End date: same formats, or `+D` for D days from now (max 36525). URL-encode `+` as `%2B` |

#### Distance Filters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `dist-min` | string | (none) | AU or LD | Minimum approach distance (append `LD` for lunar distances) |
| `dist-max` | string | `0.05` | AU or LD | Maximum approach distance |
| `min-dist-min` | string | (none) | AU or LD | Lower bound on 3-sigma minimum possible distance |
| `min-dist-max` | string | (none) | AU or LD | Upper bound on 3-sigma minimum possible distance |

#### Magnitude/Velocity Filters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `h-min` | float | (none) | Minimum absolute magnitude H |
| `h-max` | float | (none) | Maximum absolute magnitude H (e.g., 17.75 for larger objects) |
| `v-inf-min` | float | (none) | Minimum velocity at infinity (km/s) |
| `v-inf-max` | float | (none) | Maximum velocity at infinity (km/s) |
| `v-rel-min` | float | (none) | Minimum relative velocity (km/s) |
| `v-rel-max` | float | (none) | Maximum relative velocity (km/s) |

#### Object Type Filters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `class` | string | (none) | Orbit class code | Filter by orbit classification |
| `pha` | boolean | `false` | | PHAs only |
| `nea` | boolean | `false` | | NEAs only |
| `comet` | boolean | `false` | | Comets only |
| `nea-comet` | boolean | `false` | | NEAs and comets |
| `neo` | boolean | `true` | | NEOs only (default ON) |
| `kind` | string | (none) | `a`, `an`, `au`, `c`, `cn`, `cu`, `n`, `u` | Object kind filter |

#### Target Body

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `body` | string | `Earth` | `Merc`, `Venus`, `Earth`, `Mars`, `Juptr`, `Satrn`, `Urnus`, `Neptn`, `Pluto`, `Moon`, `ALL` | Close-approach body |

#### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sort` | string | `date` | Sort field: `date`, `dist`, `dist-min`, `v-inf`, `v-rel`, `h`, `object`. Prefix `-` for descending |
| `limit` | integer | (none) | Max results |
| `limit-from` | integer | (none) | Pagination offset (requires `limit`) |
| `total-only` | boolean | `false` | Return only count, no data |
| `diameter` | boolean | `false` | Include diameter and diameter_sigma |
| `fullname` | boolean | `false` | Include full object name/designation |

#### Object Selection (optional, for single-object queries)

| Parameter | Type | Description |
|-----------|------|-------------|
| `spk` | integer | SPK-ID |
| `des` | string | Designation (use `%20` for spaces) |

### 4.2 Response Structure

```json
{
  "signature": { "version": "1.5", "source": "NASA/JPL SBDB Close Approach Data API" },
  "count": 0,
  "total": 0,
  "fields": ["des", "orbit_id", "jd", "cd", "dist", "dist_min", "dist_max", "v_rel", "v_inf", "t_sigma_f", "h"],
  "data": [
    ["string", "string", "string", "string", "string", "string", "string", "string", "string", "string", "string"]
  ]
}
```

**Data fields (positional in arrays):**

| Index | Field | Type | Units | Description |
|-------|-------|------|-------|-------------|
| 0 | `des` | string | | Primary designation |
| 1 | `orbit_id` | string | | Orbit solution ID |
| 2 | `jd` | string | JD (TDB) | Julian date of close approach |
| 3 | `cd` | string | | Calendar date/time (TDB) |
| 4 | `dist` | string | AU | Nominal approach distance |
| 5 | `dist_min` | string | AU | 3-sigma minimum distance |
| 6 | `dist_max` | string | AU | 3-sigma maximum distance |
| 7 | `v_rel` | string | km/s | Relative velocity at close approach |
| 8 | `v_inf` | string | km/s | Velocity at infinity (v_rel for massless body) |
| 9 | `t_sigma_f` | string | | Time uncertainty formatted (e.g., `13:02`, `2_09:08`) |
| 10 | `h` | string | mag | Absolute magnitude |
| 11 | `diameter` | string/null | km | Diameter (if `diameter=true`) |
| 12 | `diameter_sigma` | string/null | km | Diameter uncertainty (if `diameter=true`) |
| 13 | `fullname` | string | | Full name (if `fullname=true`) |
| 14 | `body` | string | | Approach body name (if `body=ALL`) |

The `fields` array in the response reflects which columns are present based on optional
output parameters.

---

## 5. Fireball API

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/fireball.api`

Returns atmospheric fireball (bolide) events detected by US Government sensors.

### 5.1 Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `date-min` | string | (none) | Start date: `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ss` |
| `date-max` | string | (none) | End date (same format) |
| `energy-min` | float | (none) | Minimum total radiated energy (joules x 10^10) |
| `energy-max` | float | (none) | Maximum total radiated energy (joules x 10^10) |
| `impact-e-min` | float | (none) | Minimum estimated total impact energy (kt TNT) |
| `impact-e-max` | float | (none) | Maximum estimated total impact energy (kt TNT) |
| `vel-min` | float | (none) | Minimum velocity (km/s) |
| `vel-max` | float | (none) | Maximum velocity (km/s) |
| `alt-min` | float | (none) | Minimum altitude (km) |
| `alt-max` | float | (none) | Maximum altitude (km) |
| `req-loc` | boolean | `false` | Require lat/lon data |
| `req-alt` | boolean | `false` | Require altitude data |
| `req-vel-comp` | boolean | `false` | Require velocity component data |
| `vel-comp` | boolean | `false` | Include velocity components in output |
| `sort` | string | `-date` | Sort field: `date`, `energy`, `impact-e`, `vel`, `alt`. Prefix `-` for descending |
| `limit` | integer | (none) | Max records |

### 5.2 Response Structure

```json
{
  "signature": { "version": "1.2", "source": "NASA/JPL Fireball Data API" },
  "count": 0,
  "fields": ["date", "lat", "lat-dir", "lon", "lon-dir", "alt", "energy", "impact-e"],
  "data": [
    ["string", "string", "string", "string", "string", "string", "string", "string"]
  ]
}
```

**Data fields:**

| Field | Type | Units | Nullable | Description |
|-------|------|-------|----------|-------------|
| `date` | string | | No | Peak brightness date/time `YYYY-MM-DD hh:mm:ss` (GMT) |
| `lat` | string | deg | Yes | Latitude at peak brightness (decimal degrees) |
| `lat-dir` | string | | Yes | `N` or `S` |
| `lon` | string | deg | Yes | Longitude at peak brightness (decimal degrees) |
| `lon-dir` | string | | Yes | `E` or `W` |
| `alt` | string | km | Yes | Altitude above geoid |
| `energy` | string | J x 10^10 | No | Total radiated energy |
| `impact-e` | string | kt TNT | No | Estimated total impact energy |

**Velocity component fields (if `vel-comp=true`):**

| Field | Type | Units | Nullable | Description |
|-------|------|-------|----------|-------------|
| `vx` | string | km/s | Yes | Earth-centered velocity X component |
| `vy` | string | km/s | Yes | Earth-centered velocity Y component |
| `vz` | string | km/s | Yes | Earth-centered velocity Z component |

Only `date`, `energy`, and `impact-e` are guaranteed non-null. All other fields may be null.

---

## 6. Sentry API (Impact Risk Monitoring)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sentry.api`

Provides access to the Sentry impact monitoring system, which continuously scans the catalog
of known asteroids for potential future Earth impacts. Monitors approximately 1,500 objects
at any given time.

### 6.1 Request Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `spk` | integer | (none) | | Select specific object by SPK-ID |
| `des` | string | (none) | | Select specific object by designation |
| `h-max` | float | (none) | [-10, 100] | Maximum absolute magnitude H |
| `ps-min` | integer | (none) | [-20, 20] | Minimum cumulative Palermo Scale |
| `ip-min` | float | (none) | [1e-10, 1] | Minimum impact probability |
| `days` | integer | (none) | abs > 6 | Observation recency filter (negative inverts) |
| `all` | boolean | `false` | | Request complete virtual impactor dataset |
| `removed` | boolean | `false` | | Request list of removed objects |

### 6.2 Query Modes

The mode is determined by parameter combinations:

- **Mode O** (Object): `des` or `spk` specified. Returns detailed data for one object.
- **Mode S** (Summary): No object selector, no `all` flag. Returns summary table.
- **Mode V** (Virtual Impactors): `all=true`. Returns full VI table.
- **Mode R** (Removed): `removed=true`. Returns previously-monitored objects.

### 6.3 Response Structure

#### Mode O: Object Detail

```json
{
  "signature": { "version": "2.0", "source": "NASA/JPL Sentry Data API" },
  "summary": {
    "des": "string",
    "fullname": "string",
    "method": "IOBS | LOV | MC",
    "ps_cum": "string",
    "ps_max": "string",
    "ts_max": "string",
    "ip": "string",
    "n_imp": 0,
    "energy": "string (Mt TNT)",
    "h": "string",
    "diameter": "string (km)",
    "mass": "string (kg)",
    "v_inf": "string (km/s)",
    "v_imp": "string (km/s)",
    "pdate": "string (UTC)",
    "cdate": "string (Pacific)",
    "first_obs": "string (UTC)",
    "last_obs": "string (UTC)",
    "darc": "string (days)",
    "nobs": 0,
    "ndel": 0,
    "ndop": 0,
    "nsat": 0
  },
  "data": [
    {
      "date": "string (YYYY-MM-DD.DD)",
      "ip": "string",
      "ps": "string",
      "ts": "string",
      "energy": "string (Mt)",
      "dist": "string (Earth radii, LOV only)",
      "width": "string (Earth radii, LOV only)",
      "sigma_imp": "string (LOV only)",
      "sigma_lov": "string (LOV only)",
      "stretch": "string (Earth radii/sigma, LOV only)",
      "sigma_mc": "string (MC only)",
      "sigma_vi": "string (IOBS only)"
    }
  ]
}
```

#### Mode S: Summary Table

```json
{
  "signature": { "version": "2.0", "source": "NASA/JPL Sentry Data API" },
  "count": 0,
  "data": [
    {
      "des": "string",
      "fullname": "string",
      "id": "string",
      "h": "string",
      "diameter": "string (km)",
      "ip": "string",
      "ps_cum": "string",
      "ps_max": "string",
      "ts_max": "string",
      "n_imp": 0,
      "range": "string (year range)",
      "last_obs": "string",
      "last_obs_jd": "string",
      "v_inf": "string (km/s)"
    }
  ]
}
```

#### Mode R: Removed Objects

```json
{
  "signature": { "version": "2.0", "source": "NASA/JPL Sentry Data API" },
  "count": 0,
  "data": [
    {
      "des": "string",
      "removed": "string (YYYY-MM-DD HH:MM:SS UTC)"
    }
  ]
}
```

### 6.4 Palermo Technical Impact Hazard Scale

A logarithmic scale comparing the probability of a detected potential impact to the
"background" annual probability of an impact of equal or greater energy:

```
PS = log10(Pi / fB)
```

Where:
- `Pi` = impact probability for the detected event
- `fB` = annual background frequency of impacts at the same energy or greater

| Range | Interpretation |
|-------|----------------|
| PS < -2 | No likely consequences; no concern |
| -2 < PS < 0 | Situation merits careful monitoring |
| PS > 0 | Impact probability exceeds background; warrants concern |

### 6.5 Torino Impact Hazard Scale

An integer 0-10 scale combining impact probability and kinetic energy, intended for
public communication:

| Range | Category |
|-------|----------|
| 0 | No hazard: negligible collision chance, or object too small to penetrate atmosphere |
| 1 | Normal: meriting careful monitoring; chance of collision extremely unlikely |
| 2-4 | Meriting attention by astronomers: uncommon close encounter |
| 5-7 | Threatening: close encounter posing serious, but uncertain, threat |
| 8-10 | Certain collision: ranging from local damage (8) to global catastrophe (10) |

---

## 7. Scout API (NEOCP Analysis)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/scout.api`

Analyzes unconfirmed objects on the Minor Planet Center's Near-Earth Object Confirmation
Page (NEOCP). Provides orbit analysis, impact risk assessment, and ephemerides for recently
discovered objects awaiting confirmation.

### 7.1 Request Parameters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `tdes` | string | (none) | | NEOCP temporary designation (e.g., `P10uUSw`) |
| `plot` | string | (none) | `el`, `ca`, `sr`, colon-delimited combos | Plot type(s) for specified object |
| `file` | string | (none) | `list`, `mpc` | Get list of data files or MPC-format data file |
| `orbits` | boolean | `false` | | Include sampled orbit data |
| `n-orbits` | integer | 1000 | [1, 1000] | Max sampled orbits to return |
| `eph-start` | string | `now` | | Ephemeris start time (YYYY-MM-DD or ISO 8601 or `now`) |
| `eph-stop` | string | (none) | | Ephemeris stop time |
| `eph-step` | string | (auto) | `#d`, `#h`, `#m` | Ephemeris step size |
| `obs-code` | string | `500` | MPC code | Observatory code for ephemeris |
| `fov-diam` | float | (none) | (0, 1800] | Field-of-view diameter (arcmin) |
| `fov-ra` | string | median RA | | FOV center RA |
| `fov-dec` | string | median Dec | | FOV center Dec |
| `fov-vmag` | float | (none) | [0, 40] | V-magnitude limit for ephemeris |
| `ranges` | boolean | `false` | | Include topocentric/heliocentric ranges |

### 7.2 Query Modes

- **Mode S** (Summary): No `tdes` specified. Returns summary of all NEOCP objects.
- **Mode O** (Object): `tdes` specified. Returns detailed data for one object.
- **Mode E** (Ephemeris): `tdes` + `eph-start` specified. Returns ephemeris data.

### 7.3 Response Structure

#### Mode S: Summary

```json
{
  "signature": { "version": "string", "source": "string" },
  "count": "string",
  "data": [
    {
      "objectName": "string",
      "nObs": 0,
      "arc": 0.0,
      "rmsN": 0.0,
      "H": 0.0,
      "rating": 0,
      "moid": 0.0,
      "caDist": 0.0,
      "vInf": 0.0,
      "phaScore": 0,
      "neoScore": 0,
      "geocentricScore": 0,
      "ieoScore": 0,
      "tisserandScore": 0,
      "lastRun": "string (YYYY-MM-DD HH:mm:ss)",
      "ra": "string",
      "dec": "string",
      "elong": "string",
      "rate": 0.0,
      "Vmag": 0.0,
      "unc": 0.0,
      "uncP1": 0.0
    }
  ]
}
```

**Key summary fields:**

| Field | Type | Description |
|-------|------|-------------|
| `objectName` | string | NEOCP temporary designation |
| `nObs` | integer | Number of observations |
| `arc` | float | Observation arc (days) |
| `rmsN` | float | Normalized RMS residual |
| `H` | float | Estimated absolute magnitude |
| `rating` | integer | Interest rating (0-100). Higher = more interesting |
| `moid` | float | Minimum orbit intersection distance (AU) |
| `caDist` | float | Close approach distance (AU) |
| `vInf` | float | Velocity at infinity (km/s) |
| `phaScore` | integer | PHA likelihood score |
| `neoScore` | integer | NEO likelihood score |
| `geocentricScore` | integer | Geocentric orbit likelihood |
| `ieoScore` | integer | Interior Earth orbit likelihood |
| `tisserandScore` | integer | Tisserand parameter score (comet vs asteroid) |
| `Vmag` | float | Estimated visual magnitude |
| `unc` | float | Positional uncertainty (arcsec) |
| `uncP1` | float | Positional uncertainty at +1 day (arcsec) |

#### Mode O: Object Detail

Same fields as summary plus:

```json
{
  "neo1kmScore": "string",
  "tEphem": "string",
  "file": {
    "size": "string | null",
    "mpc": { "name": "string" }
  },
  "orbits": {
    "count": "string",
    "fields": ["string"],
    "data": [["mixed"]]
  }
}
```

**Orbit data fields:** `idx`, `epoch`, `ec`, `qr`, `tp`, `om`, `w`, `inc`, `H`, `dca`,
`tca`, `moid`, `vinf`, `geoEcc`, `impFlag`

#### Plot Types (base64-encoded PNG)

| Code | Plots Generated |
|------|----------------|
| `el` | qr_e_fig, qr_in_fig, qr_H_fig, H_hist_fig |
| `ca` | moid_hist_fig, ca_dist_cdf_fig, tca_hist_fig, ca_dist_hist_fig |
| `sr` | rms_sysran_fig, H_sysran_fig |

---

## 8. NHATS API (Near-Earth Asteroid Accessibility)

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/nhats.api`

Provides data on near-Earth asteroids that are potentially accessible for future human
exploration missions, based on round-trip trajectory analysis.

### 8.1 Request Parameters

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `dv` | integer | 12 | 4-12 | Maximum total mission delta-v (km/s) |
| `dur` | integer | 450 | 60-450 in steps of 30 | Maximum total mission duration (days) |
| `stay` | integer | 8 | 8, 16, 24, 32 | Minimum stay time at asteroid (days) |
| `launch` | string | `2020-2045` | `2020-2025`, `2025-2030`, `2030-2035`, `2035-2040`, `2040-2045`, `2020-2045` | Launch window |
| `h` | integer | (none) | 16-30 | Maximum absolute magnitude H (Mode S only) |
| `occ` | integer | (none) | 0-8 | Maximum orbit condition code (Mode S only) |
| `spk` | integer | (none) | | Select by SPK-ID (triggers Mode O) |
| `des` | string | (none) | | Select by designation (triggers Mode O) |
| `plot` | boolean | `false` | | Include base64-encoded plot image (Mode O only) |

### 8.2 Query Modes

- **Mode S** (Summary): No `des`/`spk`. Returns list of accessible asteroids.
- **Mode O** (Object): `des` or `spk` specified. Returns detailed trajectory data.

### 8.3 Response Structure

#### Mode S: Summary

```json
{
  "signature": { "version": "string", "source": "string" },
  "count": 0,
  "data": [
    {
      "des": "string",
      "fullname": "string",
      "orbit_id": "string",
      "h": "string",
      "min_size": "string (meters)",
      "max_size": "string (meters)",
      "size": "string (meters) | null",
      "size_sigma": "string | null",
      "occ": "string (orbit condition code)",
      "min_dv": { "dv": "string (km/s)", "dur": "string (days)" },
      "min_dur": { "dv": "string (km/s)", "dur": "string (days)" },
      "n_via_traj": 0,
      "obs_start": "string (YYYY-MM-DD)",
      "obs_end": "string (YYYY-MM-DD)",
      "obs_mag": "string",
      "obs_flag": "string",
      "radar_obs_a": "string (YYYY-MM-DD) | null",
      "radar_snr_a": "string | null",
      "radar_obs_g": "string (YYYY-MM-DD) | null",
      "radar_snr_g": "string | null"
    }
  ]
}
```

#### Mode O: Object Detail

Includes all summary fields at top level plus detailed trajectory information:

```json
{
  "computed": "string (YYYY-MM-DD)",
  "min_dv_traj": {
    "tid": "string",
    "dv_total": "string (km/s)",
    "dur_total": "string (days)",
    "dur_out": "string (days)",
    "dur_at": "string (days)",
    "dur_ret": "string (days)",
    "launch": "string (YYYY-MM-DD)",
    "c3": "string (km^2/s^2)",
    "v_dep_earth": "string (km/s)",
    "dv_dep_park": "string (km/s)",
    "vrel_arr_neo": "string (km/s)",
    "vrel_dep_neo": "string (km/s)",
    "vrel_arr_earth": "string (km/s)",
    "v_arr_earth": "string (km/s)",
    "dec_dep": "string (deg)",
    "dec_arr": "string (deg)"
  },
  "min_dur_traj": { }
}
```

`min_dur_traj` has the same structure as `min_dv_traj`.

---

## 9. Mission Design API

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/mdesign.api`

Provides trajectory parameters for planning missions to small bodies. Supports four modes:
accessible target search, pre-computed mission lookup, porkchop plot generation, and
flyby/extension target search.

### 9.1 Mode A: Accessible Targets

Find the most accessible small bodies for a given launch window.

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `crit` | integer | (required) | 1-6 | Optimality criterion |
| `year` | integer | current+4 | | Launch year(s), comma-separated |
| `lim` | integer | 200 | >0 | Max records |

**Optimality criteria (`crit`):**

| Value | Criterion |
|-------|-----------|
| 1 | Minimum departure V-infinity |
| 2 | Minimum arrival V-infinity |
| 3 | Minimum total delta-v |
| 4 | Minimum TOF + minimum departure V-infinity |
| 5 | Minimum TOF + minimum arrival V-infinity |
| 6 | Minimum TOF + minimum total delta-v |

**Response fields:** `name`, `date0`, `MJD0`, `datef`, `MJDF`, `c3_dep` (km^2/s^2),
`vinf_dep` (km/s), `vinf_arr` (km/s), `dv_tot` (km/s), `tof` (days), `class`, `H`,
`condition_code`, `neo` (Y/N), `pha` (Y/N), `bin` (Y/N), `pdes`

### 9.2 Mode Q: Pre-Computed Missions

Look up pre-computed mission parameters for a specific object.

| Parameter | Type | Description |
|-----------|------|-------------|
| `des` | string | Designation (one of `des`/`spk`/`sstr` required) |
| `spk` | integer | SPK-ID |
| `sstr` | string | Search string |
| `class` | boolean | If true, return full orbit class name instead of 3-letter code |

**Response:**

```json
{
  "object": {
    "des": "string",
    "fullname": "string",
    "spkid": "string",
    "orbit_class": "string",
    "condition_code": "string",
    "data_arc": "string (days)",
    "orbit_id": "string",
    "md_orbit_id": "string",
    "computed_on": "string"
  },
  "fields": ["MJD0", "MJDf", "vinf_dep", "vinf_arr", "phase_ang", "earth_dist", "elong_arr", "decl_dep", "approach_ang"],
  "selectedMissions": [[]]
}
```

**Field units:** MJD0/MJDf (Modified Julian Date), vinf_dep/vinf_arr (km/s), phase_ang (deg),
earth_dist (AU), elong_arr (deg), decl_dep (deg), approach_ang (deg)

### 9.3 Mode M: Porkchop Plot Maps

Generate 2D trajectory parameter grids for visualization.

Additional parameters beyond Mode Q:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `mjd0` | integer | 33282-73459 | First launch date (MJD) |
| `span` | integer | 10-9200 | Launch date period duration (days) |
| `tof-min` | integer | 10-9200 | Minimum time of flight (days) |
| `tof-max` | integer | 10-9200 | Maximum time of flight (days) |
| `step` | integer | 1,2,5,10,15,20,30 | Time step (days) |

**Response includes 2D arrays:** `vinf_dep`, `vinf_arr`, `phase_ang`, `earth_dist`,
`elong_arr`, `decl_dep`, `approach_ang` -- each n x m where n = TOF steps, m = departure date steps.

### 9.4 Mode T: Flyby/Extension Target Search

Find small bodies that pass near a given trajectory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `ec` | float | Eccentricity of reference orbit |
| `qr` | float | Perihelion distance (AU) |
| `tp` | float | Time of perihelion passage (JD) |
| `in` | float | Inclination (deg) |
| `om` | float | Longitude of ascending node (deg) |
| `w` | float | Argument of periapsis (deg) |
| `jd0` | float | Start of time span (JD) |
| `jdf` | float | End of time span (JD); max span 1 year |
| `maxout` | integer | Max output records |
| `maxdist` | float | Max close-approach distance (AU) |

**Response data fields:** `full_name`, `date`, `jd`, `min_dist_au`, `min_dist_km`,
`rel_vel` (km/s), `class`, `H`, `condition_code`, `neo`, `pha`, `sats`, `spkid`, `pdes`

---

## 10. Small-Body Identification API

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sb_ident.api`

Identifies known small bodies within a specified field of view at a given observation time.
Useful for image analysis and observation planning.

### 10.1 Request Parameters

#### Observer Location (one method required)

**Method 1: MPC Observatory Code**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mpc-code` | string | MPC observer location code |

**Method 2: Geodetic Coordinates**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lat` | float | | Latitude (deg, north-positive) [-90, 90] |
| `lon` | float | | Longitude (deg, east-positive) [-180, 180] |
| `alt` | float | 0 | Altitude above WGS-84 ellipsoid (km) |

**Method 3: Parallax Constants**

| Parameter | Type | Description |
|-----------|------|-------------|
| `lon` | float | Longitude (deg, east-positive) |
| `dxy` | float | Parallax constant perpendicular to spin axis (AU x 10^7) |
| `dz` | float | Parallax constant along spin axis (AU x 10^7) |

**Method 4: Geocentric State Vector**

| Parameter | Type | Description |
|-----------|------|-------------|
| `xobs` | string | Position (km) and optional velocity (km/s), J2000 equatorial, comma-separated |

**Method 5: Heliocentric State Vector**

| Parameter | Type | Description |
|-----------|------|-------------|
| `xobs-hel` | string | Position (AU) and optional velocity (AU/d), J2000 equatorial, comma-separated |

#### Observation Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs-time` | string | (required) | `YYYY-MM-DD[_hh:mm:ss]` or Julian Date |
| `vmag-lim` | float | (none) | Visual magnitude threshold |

#### Field of View (one method required)

**Method 1: FOV Edges**

| Parameter | Type | Description |
|-----------|------|-------------|
| `fov-ra-lim` | string | RA edges: `hh-mm-ss[.ss]` comma-separated. Use `M` prefix for negative |
| `fov-dec-lim` | string | Dec edges: `dd-mm-ss[.ss]` comma-separated. Use `M` prefix for negative |

**Method 2: FOV Center + Half-Width**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fov-ra-center` | string | | Center RA: `hh-mm-ss[.ss]` |
| `fov-dec-center` | string | | Center Dec: `dd-mm-ss[.ss]` |
| `fov-ra-hwidth` | float | 0.5 | Half-width in RA (deg) |
| `fov-dec-hwidth` | float | 0.5 | Half-width in Dec (deg) |

#### Filtering

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `two-pass` | boolean | `false` | Second pass with high-fidelity numerical integration |
| `mag-required` | boolean | `true` | Skip objects lacking magnitude parameters |
| `suppress-first-pass` | boolean | `true` | Suppress first-pass output when two-pass is true |
| `sb-kind` | string | (none) | Object type: `a` (asteroids) |
| `sb-group` | string | (none) | Group: `neo`, etc. |
| `req-elem` | boolean | `false` | Include osculating orbital elements |

### 10.2 Response Structure

```json
{
  "signature": { "version": "1.1", "source": "NASA/JPL Small-Body Identification API" },
  "summary": {},
  "observer": {
    "obs_date": "string",
    "location": "string",
    "fov_center": "string",
    "fov_offset": "string",
    "frame": "J2000"
  },
  "n_first_pass": 0,
  "n_second_pass": 0,
  "fields_first": ["string"],
  "fields_second": ["string"],
  "data_first_pass": [[]],
  "data_second_pass": [[]],
  "elem_first_pass": [[]],
  "elem_second_pass": [[]],
  "sb_constraints": {}
}
```

**Data columns (when `req-elem=false`):** Object name, Astrometric RA, Astrometric Dec,
RA offset (arcsec), Dec offset (arcsec), total offset (arcsec), visual magnitude V,
RA rate (deg/s), Dec rate (deg/s), RA error estimate (arcsec, first-pass only),
Dec error estimate (arcsec, first-pass only).

**Element columns (when `req-elem=true`):** Object name, H, G, e, q (AU), tp (JD),
om (deg), w (deg), i (deg), epoch (JD).

---

## 11. SB Radar API

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sb_radar.api`

Returns radar astrometry measurement data for small bodies.

### 11.1 Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `spk` | integer | (none) | Select by SPK-ID |
| `des` | string | (none) | Select by designation |
| `kind` | string | (none) | Object type: `a`, `an`, `au`, `c`, `cn`, `cu`, `n`, `u` |
| `bp` | string | (none) | Reference point: `P` (peak/center), `C` (center of mass) |
| `type` | string | (none) | Measurement type: `R` (range/delay), `P` (Doppler) |
| `observer` | boolean | `false` | Include observer field |
| `notes` | boolean | `false` | Include notes field |
| `ref` | boolean | `false` | Include reference field |
| `fullname` | boolean | `false` | Include full object name |
| `modified` | boolean | `false` | Include modification date |
| `coords` | boolean | `false` | Include station geodetic coordinates |
| `c-coords` | boolean | `false` | Use cylindrical station coordinates instead |

### 11.2 Response Structure

```json
{
  "signature": { "version": "string", "source": "string" },
  "count": "string",
  "fields": ["string"],
  "data": [[]]
}
```

**Standard fields:** `des`, `epoch`, `value`, `sigma`, `units` (us or Hz), `freq` (MHz),
`rcvr`, `xmit`, `bp`

**Optional fields:** `observer`, `notes`, `ref`, `fullname`, `modified`

**Geodetic coordinates:** `longitude` (deg), `latitude` (deg), `altitude`, `alt_units`

**Cylindrical coordinates:** `longitude` (deg), `d_xy` (10^-10 AU), `d_z` (10^-10 AU)

---

## 12. SB Observability API

**Endpoint:** `GET https://ssd-api.jpl.nasa.gov/sbwobs.api`

Determines which small bodies are optically observable from a specified observatory on a
given night.

### 12.1 Request Parameters

#### Observer Location (one method required)

Same location methods as SB Identification API (MPC code, geodetic, parallax).

#### Observation Constraints

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs-time` | string | (required) | Observation date: `YYYY-MM-DD[_hh:mm:ss]` |
| `obs-end` | string | (none) | End observation time |
| `optical` | boolean | `true` | Require sun below horizon |
| `elong-min` | float | (none) | Minimum solar elongation (deg, required if `optical=false`) |
| `elong-max` | float | (none) | Maximum solar elongation (deg) |
| `glat-min` | float | (none) | Minimum galactic latitude (deg) |
| `glat-max` | float | (none) | Maximum galactic latitude (deg) |
| `elev-min` | float | 30 | Minimum elevation above horizon (deg) |
| `time-min` | integer | 0 | Minimum observable time (minutes) |
| `vmag-min` | float | (none) | Minimum visual magnitude |
| `vmag-max` | float | (none) | Maximum visual magnitude |
| `mag-required` | boolean | `false` | Require magnitude data |
| `helio-min` | float | (none) | Minimum heliocentric distance (AU) |
| `helio-max` | float | (none) | Maximum heliocentric distance (AU) |
| `dist-min` | float | (none) | Minimum topocentric distance (AU) |
| `dist-max` | float | (none) | Maximum topocentric distance (AU) |

#### Output Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fmt-ra-dec` | boolean | `true` | Sexagesimal format (false = decimal degrees) |
| `maxoutput` | integer | (none) | Max records |
| `output-sort` | string | (none) | Sort field: `name`, `rise`, `trans`, `set`, `maxt`, `ra`, `dec`, `vmag`, `helio`, `topo`, `oes`, `oem`, `glat` |
| `output-sort-r` | boolean | `false` | Descending sort |

SBDB filter parameters (`sb-kind`, `sb-group`, `sb-class`, etc.) are also supported.

### 12.2 Response Structure

Contains night information (sunrise/sunset, twilight times, moon data) and observable
object array. Each object record includes: designation, full name, rise/transit/set times (UT),
max observable time, RA, Dec, visual magnitude, heliocentric range (AU), topocentric range (AU),
sun angle (deg), moon angle (deg), galactic latitude (deg).

---

## 13. Data Fields Reference

### Comprehensive Field Table

| Field | Category | Type | Units | APIs |
|-------|----------|------|-------|------|
| `spkid` | Identity | string | | SBDB, Query |
| `full_name` / `fullname` | Identity | string | | All |
| `des` / `pdes` | Identity | string | | All |
| `name` | Identity | string | | Query |
| `prefix` | Identity | string | | SBDB, Query |
| `kind` | Identity | string | | SBDB, Query, CAD |
| `neo` | Identity | boolean/string | | SBDB, Query |
| `pha` | Identity | boolean/string | | SBDB, Query |
| `class` | Identity | string | | Query, CAD, MDesign |
| `orbit_id` | Identity | string | | SBDB, Query, NHATS |
| `e` | Orbital | float | (none) | SBDB, Query |
| `a` | Orbital | float | AU | SBDB, Query |
| `q` | Orbital | float | AU | SBDB, Query |
| `i` | Orbital | float | deg | SBDB, Query |
| `om` | Orbital | float | deg | SBDB, Query |
| `w` | Orbital | float | deg | SBDB, Query |
| `ma` | Orbital | float | deg | SBDB, Query |
| `tp` | Orbital | float | JD (TDB) | SBDB, Query |
| `n` | Orbital | float | deg/d | SBDB, Query |
| `per` | Orbital | float | d | SBDB, Query |
| `ad` | Orbital | float | AU | SBDB, Query |
| `epoch` | Orbital | float | JD (TDB) | SBDB, Query |
| `t_jup` | Derived | float | (none) | SBDB, Query |
| `moid` | Derived | float | AU | SBDB, Query, Scout |
| `moid_jup` | Derived | float | AU | SBDB, Query |
| `moid_ld` | Derived | float | LD | Query |
| `condition_code` | Quality | string | | SBDB, Query, NHATS, MDesign |
| `data_arc` | Quality | float | d | SBDB, Query, Sentry |
| `n_obs_used` | Quality | integer | | SBDB, Query |
| `n_del_obs_used` | Quality | integer | | SBDB, Query |
| `n_dop_obs_used` | Quality | integer | | SBDB, Query |
| `rms` | Quality | float | | SBDB, Query |
| `source` | Quality | string | | SBDB, Query |
| `producer` | Quality | string | | SBDB, Query |
| `soln_date` | Quality | string | | SBDB, Query |
| `first_obs` | Quality | string | | SBDB, Query, Sentry |
| `last_obs` | Quality | string | | SBDB, Query, Sentry |
| `two_body` | Quality | string | | SBDB, Query |
| `H` | Physical | float | mag | SBDB, Query, CAD, Sentry, Scout, NHATS, MDesign |
| `G` | Physical | float | (none) | SBDB, Query |
| `M1` | Physical | float | mag | Query (comets) |
| `M2` | Physical | float | mag | Query (comets) |
| `K1` | Physical | float | (none) | Query (comets) |
| `K2` | Physical | float | (none) | Query (comets) |
| `PC` | Physical | float | (none) | Query (comets) |
| `diameter` | Physical | float | km | SBDB, Query, Sentry, NHATS |
| `extent` | Physical | string | km | SBDB, Query |
| `GM` | Physical | float | km^3/s^2 | SBDB, Query |
| `density` | Physical | float | g/cm^3 | SBDB, Query |
| `rot_per` | Physical | float | h | SBDB, Query |
| `pole` | Physical | string | | SBDB, Query |
| `albedo` | Physical | float | (none) | SBDB, Query |
| `BV` | Physical | float | mag | SBDB, Query |
| `UB` | Physical | float | mag | SBDB, Query |
| `IR` | Physical | float | (none) | SBDB, Query |
| `spec_T` | Physical | string | | SBDB, Query |
| `spec_B` | Physical | string | | SBDB, Query |
| `A1` | Non-grav | float | AU/d^2 | Query |
| `A2` | Non-grav | float | AU/d^2 | Query |
| `A3` | Non-grav | float | AU/d^2 | Query |
| `DT` | Non-grav | float | d | Query |
| `jd` | Close-approach | string | JD (TDB) | CAD |
| `cd` | Close-approach | string | | CAD |
| `dist` | Close-approach | string | AU | CAD, Sentry |
| `dist_min` | Close-approach | string | AU | CAD |
| `dist_max` | Close-approach | string | AU | CAD |
| `v_rel` | Close-approach | string | km/s | CAD, MDesign |
| `v_inf` | Close-approach | string | km/s | CAD, Sentry, Scout |
| `t_sigma_f` | Close-approach | string | | CAD |
| `body` | Close-approach | string | | CAD |
| `ip` | Impact | string | (probability) | Sentry |
| `ps` / `ps_cum` / `ps_max` | Impact | string | (log scale) | Sentry |
| `ts` / `ts_max` | Impact | string | (0-10) | Sentry |
| `n_imp` | Impact | integer | | Sentry |
| `energy` | Impact | string | Mt TNT | Sentry, Fireball |
| `sigma_lov` | Impact | string | | Sentry |
| `sigma_vi` | Impact | string | | Sentry |
| `width` | Impact | string | Earth radii | Sentry |
| `stretch` | Impact | string | Earth radii/sigma | Sentry |

### Important Type Notes for Rust Implementation

1. **Nearly all numeric values are returned as JSON strings**, not JSON numbers. The Rust
   deserializer must parse these: `"0.0123"` -> `f64`. Use a custom deserializer or
   intermediate string type.

2. **Boolean fields vary by API:** SBDB returns JSON booleans (`true`/`false`), Query returns
   strings (`"Y"`/`"N"`), CAD uses boolean query parameters.

3. **Nullable fields:** Many physical parameters and some close-approach fields may be `null`
   or absent. Use `Option<T>` extensively.

4. **The `data` arrays are positional:** Fields are identified by their position in the
   array, matched against the `fields` array in the response. Build a field-index map at
   parse time.

5. **Date formats vary by API:**
   - `YYYY-MM-DD` (common)
   - `YYYY-MMM-DD` (3-letter month abbreviation)
   - `YYYY-MM-DD hh:mm:ss` (fireball)
   - `YYYY-MM-DD.DD` (Sentry VI dates, fractional day)
   - Julian Date as string or number

6. **Distance units:** AU by default. Some parameters accept `LD` suffix for lunar distances.
   1 LD = 0.00257 AU approximately.

---

## 14. Example Requests

### SBDB: Look up Apophis (99942)

```
GET https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=Apophis&full-prec=true&phys-par=true&ca-data=true&vi-data=true
```

### SBDB Query: All PHAs with diameter > 1 km

```
GET https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=spkid,full_name,e,a,i,H,diameter&sb-group=pha&sb-cdata={"AND":["diameter|GT|1"]}&sort=-diameter
```

### SBDB Query: All numbered Amor-class asteroids

```
GET https://ssd-api.jpl.nasa.gov/sbdb_query.api?fields=spkid,full_name,a,e,q,i,H&sb-class=AMO&sb-ns=n&limit=50
```

### CAD: Earth close approaches within 0.05 AU in next year

```
GET https://ssd-api.jpl.nasa.gov/cad.api?date-min=now&date-max=%2B365&dist-max=0.05&body=Earth&sort=dist&fullname=true
```

### CAD: Apophis close approaches to all bodies

```
GET https://ssd-api.jpl.nasa.gov/cad.api?des=99942&body=ALL&date-min=2025-01-01&date-max=2040-01-01
```

### Fireball: All events in the last month

```
GET https://ssd-api.jpl.nasa.gov/fireball.api?date-min=2026-01-24&date-max=2026-02-24&req-loc=true&vel-comp=true
```

### Sentry: Objects with Palermo scale > -3

```
GET https://ssd-api.jpl.nasa.gov/sentry.api?ps-min=-3
```

### Sentry: Detailed data for Bennu

```
GET https://ssd-api.jpl.nasa.gov/sentry.api?des=101955
```

### Scout: All current NEOCP objects

```
GET https://ssd-api.jpl.nasa.gov/scout.api
```

### NHATS: Accessible asteroids with delta-v < 6 km/s

```
GET https://ssd-api.jpl.nasa.gov/nhats.api?dv=6&dur=360&stay=8
```

### Mission Design: Accessible targets for 2028 launch

```
GET https://ssd-api.jpl.nasa.gov/mdesign.api?crit=3&year=2028&lim=20
```

### Mission Design: Pre-computed missions to Bennu

```
GET https://ssd-api.jpl.nasa.gov/mdesign.api?des=101955
```

### SB Identification: Objects in a 1-degree FOV

```
GET https://ssd-api.jpl.nasa.gov/sb_ident.api?mpc-code=F51&obs-time=2026-03-01_10:00:00&fov-ra-center=12-30-00&fov-dec-center=25-00-00&fov-ra-hwidth=0.5&fov-dec-hwidth=0.5&two-pass=true
```

### SB Radar: All radar observations of Apophis

```
GET https://ssd-api.jpl.nasa.gov/sb_radar.api?des=99942&observer=true&fullname=true
```

---

## 15. Rate Limits and Practical Concerns

### No Authentication

All APIs are free and require no API key, token, or registration.

### Rate Limiting

No published rate limits exist. JPL expects "reasonable use." In practice:

- Avoid more than a few requests per second for sustained periods
- Use pagination (`limit`/`limit-from`) for large result sets rather than requesting
  everything at once
- Cache responses where possible; orbital data changes infrequently (days to weeks)
- The SBDB Query API can return very large JSON payloads for unfiltered queries

### Pagination

APIs supporting pagination use `limit` and `limit-from`:

```
# First 100 results
?limit=100

# Next 100 results
?limit=100&limit-from=100

# Results 200-299
?limit=100&limit-from=200
```

The response includes a `total` or `count` field to know when to stop.

### Null and Missing Data

Many fields may be null or absent, especially for poorly-characterized objects:

- Physical parameters (diameter, albedo, rotation period) are unknown for most objects
- Spectral types are only available for a small fraction
- Radar data exists for only ~1,000 objects
- Comet-specific fields (M1, M2, K1, K2) are null for asteroids and vice versa

### Data Freshness

- Orbital elements are updated as new observations are processed (typically within days)
- Sentry impact monitoring is updated continuously
- Scout data is updated as new NEOCP submissions arrive
- Close-approach data is recomputed with each orbit update
- Database counts grow by approximately 2,000-3,000 objects per month

### Response Size Considerations

- SBDB Query with no `limit` on all objects: can be 100+ MB
- CAD with broad date/distance ranges: can return 100,000+ records
- Always use `limit` for initial exploration, then paginate
- Consider streaming JSON parsing (e.g., `serde_json::StreamDeserializer`) for bulk queries

### URL Encoding

- Spaces in designations: use `%20` (e.g., `des=2015%20AB`)
- Plus sign in date offsets: use `%2B` (e.g., `date-max=%2B60`)
- JSON in `sb-cdata`: URL-encode the entire JSON string
- Commas in multi-value parameters: typically not encoded

### Error Handling

All APIs return error information in the JSON response body, even for HTTP 200:

```json
{
  "signature": { "version": "string", "source": "string" },
  "message": "string describing the error",
  "code": 400
}
```

Or for object-not-found in some APIs:

```json
{
  "error": "specified object was not found"
}
```

Check for `error`, `message`, and `code` fields in every response.

### Rust Implementation Notes

1. **Use `reqwest` with query parameter builders** rather than string formatting URLs.
   This handles URL encoding correctly.

2. **Define enums for closed sets:** orbit classes, object kinds, body names, sort fields.

3. **Use `#[serde(deserialize_with = ...)]`** for the string-to-number conversion pattern
   that appears everywhere in these APIs.

4. **Model the `fields`/`data` pattern** (CAD, Query, Fireball, Radar) with a generic
   tabular response type that maps field names to array positions at parse time.

5. **Use `Option<T>` aggressively.** Nearly every field can be null for some object.

6. **Consider builder patterns** for the complex parameter sets (especially SBDB Query
   filters and SB Identification FOV parameters).

7. **Implement `From<&str>` / `FromStr`** for orbit class and object kind enums to
   simplify parsing.

8. **Time handling:** Use a JD/calendar date abstraction. Multiple date formats exist
   across APIs; a unified internal representation (e.g., JD f64) with format-specific
   parsing is cleanest.
