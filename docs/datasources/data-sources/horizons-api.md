# NASA JPL HORIZONS System -- Implementation Reference

This document serves as a comprehensive technical reference for implementing a Rust client
for the NASA JPL HORIZONS on-line solar system ephemeris computation service.

---

## 1. System Overview

**HORIZONS** is an on-line solar system data and ephemeris computation service provided and
maintained by the **Solar System Dynamics Group (SSD)** at NASA's **Jet Propulsion Laboratory
(JPL)**. It provides access to high-precision ephemerides for solar system objects via
multiple interfaces.

The system computes positions, velocities, and observational quantities for solar system
bodies as functions of time, from the perspective of any specified observer location.

### Object Coverage

| Category | Count | Notes |
|----------|-------|-------|
| Asteroids | 1,479,000+ | All except single-opposition objects with <30-day arcs (NEOs/PHAs excepted) |
| Comets | 4,043+ | All apparitions with solutions |
| Natural Satellites (Moons) | 424+ | All known moons with trajectory models |
| Spacecraft | 239+ | Active and historical missions |
| Planets | 8 | Mercury through Neptune (plus Pluto) |
| Dwarf Planets | 5+ | Ceres, Pluto, Eris, Makemake, Haumea |
| Lagrange Points | 8 | Earth-Sun L1/L2/L4/L5, Earth-Moon L1/L2/L4/L5 |
| Barycenters | 10 | Solar System Barycenter + 9 planetary system barycenters |
| Sun | 1 | |

### Update Frequency

- PHAs/NEOs: typically within 2 hours of new observations
- Other small bodies: few days to 2 weeks
- Database refresh: hourly with new orbital solutions

### Key Properties

- **No API key required** -- free public service
- **No authentication** -- all endpoints are open
- Planetary ephemeris: JPL DE441 (13200 BC to AD 17191 for barycenters)
- Coordinate frame: ICRF (aligned with J2000 to ~0.0002 arcsec)

---

## 2. API Endpoints

### 2.1 URL API (GET)

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?{params}
```

All parameters are passed as URL query string key-value pairs. This is the primary
programmatic interface.

**Practical URL length limit:** ~2000 characters (browser/proxy dependent). Use the File
API for larger requests.

### 2.2 File API (POST)

```
POST https://ssd.jpl.nasa.gov/api/horizons_file.api
```

For requests that exceed URL length limits (large TLIST arrays, TLE data, etc.).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Horizons input file content (key-value pairs between `!$$SOF` and `!$$EOF`) |
| `format` | string | No | `json` (default) or `text` |

**Input file format:**
```
!$$SOF
COMMAND='499'
MAKE_EPHEM='YES'
EPHEM_TYPE='VECTORS'
CENTER='500@0'
START_TIME='2024-01-01'
STOP_TIME='2024-02-01'
STEP_SIZE='1 d'
!$$EOF
```

### 2.3 Lookup API (GET)

```
GET https://ssd.jpl.nasa.gov/api/horizons_lookup.api?{params}
```

Resolves names, designations, SPK-IDs, IAU numbers, and MPC packed designations to
standardized HORIZONS identifiers.

**Parameters:**

| Parameter | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `sstr` | string | *required* | alphanumeric, space, `/`, `'`, `-`, `!`, `=`, `\|`, `.` | Search term |
| `group` | string | (none) | `ast`, `com`, `pln`, `sat`, `sct`, `mb`, `sb` | Filter by object type |
| `format` | string | `json` | `json`, `text` | Output format |

**Group codes:**

| Code | Meaning |
|------|---------|
| `ast` | Asteroids only |
| `com` | Comets only |
| `pln` | Planets only |
| `sat` | Natural satellites only |
| `sct` | Spacecraft only |
| `mb` | All major bodies (planets + satellites + spacecraft) |
| `sb` | All small bodies (asteroids + comets) |

**Response fields (JSON):**

| Field | Description |
|-------|-------------|
| `count` | Number of matches |
| `name` | Object identifier |
| `type` | Object category |
| `pdes` | Primary provisional designation |
| `spkid` | Primary SPK-ID |
| `alias` | List of alternate designations |
| `signature` | API version metadata |

### 2.4 Response Format

All endpoints support `format=json` (default) and `format=text`.

**JSON response structure (ephemeris):**
```json
{
  "signature": {
    "source": "NASA/JPL Horizons API",
    "version": "1.3"
  },
  "result": "...full text output..."
}
```

**JSON response structure (SPK file):**
```json
{
  "signature": {
    "source": "NASA/JPL Horizons API",
    "version": "1.3"
  },
  "spk": "base64-encoded-binary-data...",
  "spk_file_id": "suggested_filename.bsp"
}
```

**JSON response structure (error):**
```json
{
  "signature": {
    "source": "NASA/JPL Horizons API",
    "version": "1.3"
  },
  "result": "...error text from Horizons..."
}
```

Note: HORIZONS-level errors (invalid object, bad time range, etc.) are returned as HTTP 200
with the error text inside the `result` field. The client must parse `result` to detect
these.

### 2.5 HTTP Status Codes

| Code | Meaning | Notes |
|------|---------|-------|
| 200 | OK | Check `result` payload for Horizons-level errors |
| 400 | Bad Request | Invalid keywords, content, or incorrect HTTP method |
| 405 | Method Not Allowed | Wrong HTTP method (e.g., POST to URL API) |
| 500 | Internal Server Error | Database unavailable |
| 503 | Service Unavailable | Server overloaded or maintenance |

---

## 3. Object Identification (`COMMAND` Parameter)

The `COMMAND` parameter selects the target body. Different syntax applies to major bodies
vs. small bodies.

### 3.1 Major Bodies

Major bodies have pre-computed trajectories interpolated to millimeter-level accuracy from
stored ephemeris files (DE441). They can serve as both targets and coordinate centers.

| Syntax | Example | Object |
|--------|---------|--------|
| Numeric ID | `'499'` | Mars center |
| Numeric ID | `'301'` | Moon |
| Numeric ID | `'10'` | Sun |
| Numeric ID | `'0'` | Solar System Barycenter |
| Name | `'Mars'` | Mars (case-insensitive) |
| Name | `'Io'` | Io (501) |
| List command | `'MB'` | List all major bodies |

### 3.2 Small Bodies (Asteroids and Comets)

Small bodies have statistically estimated orbital elements that are numerically integrated
on-demand. They **cannot** be used as coordinate centers.

**The semicolon (`;`) suffix is required** to distinguish small bodies from major bodies
with the same numeric ID. In URLs, encode as `%3B`.

| Syntax | Example (raw) | Example (URL-encoded) | Object |
|--------|---------------|----------------------|--------|
| IAU number + `;` | `'1;'` | `COMMAND='1%3B'` | 1 Ceres |
| IAU number + `;` | `'433;'` | `COMMAND='433%3B'` | 433 Eros |
| Designation | `'DES=1999 AN10;'` | `COMMAND='DES%3D1999+AN10%3B'` | 1999 AN10 |
| Designation | `'DES=2020 F3;'` | `COMMAND='DES%3D2020+F3%3B'` | C/2020 F3 (NEOWISE) |
| Name | `'Apophis;'` | `COMMAND='Apophis%3B'` | 99942 Apophis |
| SPK-ID | `'DES=2099942;'` | `COMMAND='DES%3D2099942%3B'` | 99942 Apophis (by SPK-ID) |
| List command | `'SB'` | `COMMAND='SB'` | List small-body search fields |

### 3.3 Search Flags

Append these to the `COMMAND` value for comets:

| Flag | Description |
|------|-------------|
| `NOFRAG` | Exclude comet fragments from results |
| `CAP` | Select closest apparition to current date |
| `CAP < JD#` | Closest apparition before specified Julian Day |
| `CAP < YEAR` | Closest apparition before specified year |

Example: `COMMAND='DES=73P; NOFRAG; CAP'`

### 3.4 User-Defined Objects

Set `COMMAND=';'` (URL: `COMMAND='%3B'`) and provide orbital elements via additional
parameters. See Section 9 for full details.

### 3.5 TLE Objects

Set `COMMAND='TLE'` and provide Two-Line Element data. Up to 600 TLE pairs (1200 lines).
Newlines encoded as `%0A` in URLs.

### 3.6 Body ID Numbering Scheme

#### Solar System Barycenter and Sun

| ID | Object |
|----|--------|
| 0 | Solar System Barycenter (SSB) |
| 10 | Sun |

#### Planetary System Barycenters

| ID | System |
|----|--------|
| 1 | Mercury Barycenter |
| 2 | Venus Barycenter |
| 3 | Earth-Moon Barycenter (EMB) |
| 4 | Mars Barycenter |
| 5 | Jupiter Barycenter |
| 6 | Saturn Barycenter |
| 7 | Uranus Barycenter |
| 8 | Neptune Barycenter |
| 9 | Pluto Barycenter |

#### Planet Centers

| ID | Planet |
|----|--------|
| 199 | Mercury |
| 299 | Venus |
| 399 | Earth |
| 499 | Mars |
| 599 | Jupiter |
| 699 | Saturn |
| 799 | Uranus |
| 899 | Neptune |
| 999 | Pluto |

The pattern is `X99` where `X` is the planetary system number (1-9).

#### Natural Satellites (Moons)

Moons are numbered `X01` through `X99` within each planetary system:

| ID Range | System | Examples |
|----------|--------|----------|
| 301 | Earth | 301 = Moon |
| 401-402 | Mars | 401 = Phobos, 402 = Deimos |
| 501-599 | Jupiter | 501 = Io, 502 = Europa, 503 = Ganymede, 504 = Callisto |
| 601-699 | Saturn | 601 = Mimas, 602 = Enceladus, 603 = Tethys, 606 = Titan |
| 701-799 | Uranus | 701 = Ariel, 702 = Umbriel, 703 = Titania, 704 = Oberon |
| 801-899 | Neptune | 801 = Triton |
| 901-999 | Pluto | 901 = Charon |

#### Lagrange Points

| ID | Point |
|----|-------|
| 391 | Earth-Sun L1 |
| 392 | Earth-Sun L2 |
| 393 | Earth-Sun L4 (leading) |
| 394 | Earth-Sun L5 (trailing) |
| 395 | Earth-Moon L1 |
| 396 | Earth-Moon L2 |
| 397 | Earth-Moon L4 |
| 398 | Earth-Moon L5 |

#### Spacecraft (Negative IDs)

| ID | Spacecraft |
|----|------------|
| -31 | Voyager 1 |
| -32 | Voyager 2 |
| -48 | Hubble Space Telescope (HST) |
| -64 | OSIRIS-REx (OSIRIS-APEX) |
| -82 | Cassini |
| -98 | New Horizons |
| -143 | Mars Odyssey |
| -170 | James Webb Space Telescope (JWST) |
| -227 | Kepler |
| -234 | STEREO-A |
| -236 | STEREO-B |

Use `COMMAND='MB'` to get a full listing.

#### Asteroids and Comets (SPK-IDs)

| SPK-ID Range | Type |
|-------------|------|
| 2000001+ | Numbered asteroids (SPK-ID = 2000000 + IAU number) |
| 3000001+ | Unnumbered asteroids |
| 1000001+ | Comets |

Note: When using the API, reference asteroids by IAU number with semicolon suffix (e.g.,
`'1;'` for Ceres) rather than by SPK-ID.

---

## 4. Five Ephemeris Types (`EPHEM_TYPE`)

### 4.1 OBSERVER -- Plane-of-Sky Observables

Generates sky-plane observational quantities from a specified observer location.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `QUANTITIES` | `'A'` | Comma-separated list of quantity codes (1-48) or preset group letter |
| `CENTER` | geocentric | Observer location |
| `ANG_FORMAT` | `HMS` | `HMS` or `DEG` for RA/Dec format |
| `APPARENT` | `AIRLESS` | `AIRLESS` or `REFRACTED` |
| `EXTRA_PREC` | `NO` | Extra decimal digits in RA/Dec |
| `CSV_FORMAT` | `NO` | Comma-separated output |
| `R_T_S_ONLY` | `NO` | Only rise/transit/set events |

#### All 48 Observer Quantity Codes

| Code | Name | Description |
|------|------|-------------|
| 1 | Astrometric RA & Dec | ICRF right ascension and declination, corrected for light-time |
| 2 | Apparent RA & Dec | Apparent RA/Dec in ICRF or of-date frame, with aberration and light deflection |
| 3 | Rates: RA & Dec | Angular rates of change: `dRA*cos(Dec)` and `d(Dec)/dt` in arcsec/hour |
| 4 | Apparent Az & El | Apparent azimuth (N=0, E=90) and elevation, topocentric |
| 5 | Rates: Az & El | Angular rates: `dAz*cos(El)` and `d(El)/dt` in arcsec/minute |
| 6 | Satellite X & Y | Differential RA/Dec of satellite relative to primary body center, and position angle |
| 7 | Local Apparent Sidereal Time | Local apparent sidereal time at observer site (HH:MM:SS.ff) |
| 8 | Airmass & Extinction | Optical airmass and visual magnitude extinction |
| 9 | Visual Mag & Surface Brightness | Apparent visual magnitude (V-band) and surface brightness (mag/arcsec^2) |
| 10 | Illuminated Fraction | Fraction of target disk illuminated by Sun (0.0 to 1.0) |
| 11 | Defect of Illumination | Maximum angular extent of dark limb, in arcsec |
| 12 | Satellite Angular Separation/Visibility | Satellite-to-primary angular separation with visibility code |
| 13 | Target Angular Diameter | Apparent angular diameter of target body, in arcsec |
| 14 | Observer Sub-Longitude & Sub-Latitude | Planetographic/planetocentric longitude and latitude of sub-observer point |
| 15 | Sun Sub-Longitude & Sub-Latitude | Planetographic/planetocentric longitude and latitude of sub-solar point |
| 16 | Sub-Sun Position Angle & Distance | Position angle and angular distance of sub-solar point from disk center |
| 17 | North Pole Position Angle & Distance | Position angle and angular distance of target north pole from disk center |
| 18 | Heliocentric Ecliptic Lon & Lat | Target heliocentric ecliptic longitude and latitude (J2000 ecliptic) |
| 19 | Heliocentric Range & Range-Rate | Distance from Sun (AU) and radial velocity (km/s) |
| 20 | Observer Range & Range-Rate | Distance from observer (AU or km) and radial velocity (km/s) |
| 21 | One-Way Light-Time | Down-leg light travel time from target to observer (minutes) |
| 22 | Speed wrt Sun & Observer | Target speed relative to Sun and relative to observer (km/s) |
| 23 | Sun-Observer-Target Elongation | Solar elongation angle (degrees) with `/r` leading/trailing indicator |
| 24 | Sun-Target-Observer Phase Angle | Phase angle: Sun-Target-Observer (degrees) |
| 25 | Target-Observer-Moon / Moon Illumination | Lunar elongation from target (degrees) and Moon illuminated fraction (%) |
| 26 | Observer-Primary-Target Angle | For satellites: angle between observer, primary body center, and satellite |
| 27 | Sun-Target Radial & Velocity Position Angle | Position angle of Sun-to-target radius vector and velocity vector |
| 28 | Orbit Plane Angle | Angle between observer line of sight and target orbital plane |
| 29 | Constellation ID | 3-letter IAU constellation abbreviation containing the target |
| 30 | Delta-T (TDB - UT) | Difference TDB minus UT in seconds |
| 31 | Observer Ecliptic Lon & Lat | Target ecliptic longitude and latitude as seen from observer |
| 32 | North Pole RA & Dec | Target body north pole ICRF right ascension and declination |
| 33 | Galactic Longitude & Latitude | Target galactic coordinates (degrees) |
| 34 | Local Apparent Solar Time | Solar time at observer site (HH:MM:SS.ff) |
| 35 | Earth-to-Observer-Site Light-Time | Light travel time from Earth center to observer site (seconds); zero for geocentric |
| 36 | RA & Dec Uncertainty | 1-sigma position uncertainties in RA and Dec (arcsec); small bodies only |
| 37 | Plane-of-Sky Error Ellipse | 1-sigma POS error ellipse: semi-major axis, semi-minor axis, position angle |
| 38 | POS Uncertainty (RSS) | Root-sum-square of POS error ellipse axes (arcsec) |
| 39 | Range & Range-Rate 3-Sigma | 3-sigma uncertainties in range (km) and range-rate (km/s) |
| 40 | Doppler & Delay 3-Sigma | 3-sigma uncertainties in Doppler shift (Hz) and round-trip delay (seconds) |
| 41 | True Anomaly Angle | Instantaneous true anomaly angle (degrees) |
| 42 | Local Apparent Hour Angle | Hour angle of target at observer site |
| 43 | Phase Angle & Bisector | Phase angle and phase angle bisector longitude/latitude |
| 44 | Apparent Longitude of Sun (L_s) | Apparent sub-solar longitude on target body; season indicator for Mars |
| 45 | Inertial Apparent RA & Dec | Apparent RA/Dec in inertial ICRF frame (for spacecraft observers) |
| 46 | Rate: Inertial RA & Dec | Angular rates of inertial apparent RA/Dec |
| 47 | Sky Motion: Rate & Angles | Total sky-plane angular rate, position angle of motion, and path angle |
| 48 | Lunar Sky Brightness & SNR | Estimated V-band sky brightness from scattered moonlight and resulting SNR |

#### Preset Quantity Groups

| Group | Description |
|-------|-------------|
| `A` | All available quantities (default) |
| `B` | Geocentric: custom set for geocentric observer |
| `C` | Small-body geocentric |
| `D` | Small-body topocentric |
| `E` | Spacecraft geocentric |
| `F` | Spacecraft topocentric |

#### Filtering Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ELEV_CUT` | `-90` | Minimum elevation angle in degrees; range [-90, 90] |
| `SKIP_DAYLT` | `NO` | Skip output when Sun above horizon (`YES`/`NO`) |
| `SOLAR_ELONG` | `0,180` | Solar elongation bounds in degrees (min,max) |
| `AIRMASS` | `38.0` | Maximum airmass cutoff value |
| `LHA_CUTOFF` | `0.0` | Local hour angle limit (0.0 to 12.0 hours) |
| `ANG_RATE_CUTOFF` | `0.0` | Minimum plane-of-sky angular rate cutoff (arcsec/hour) |
| `R_T_S_ONLY` | `NO` | Only output rise/transit/set events |

### 4.2 VECTORS -- Cartesian State Vectors

Generates Cartesian position and velocity vectors in a specified reference frame.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VEC_TABLE` | `3` | Table content type (1-6, with optional uncertainty modifiers) |
| `VEC_CORR` | `NONE` | Aberration correction |
| `OUT_UNITS` | `KM-S` | Output distance and time units |
| `REF_PLANE` | `ECLIPTIC` | Reference plane for coordinates |
| `VEC_LABELS` | `YES` | Include component labels (X, Y, Z, etc.) |
| `VEC_DELTA_T` | `NO` | Include TDB-UT difference |
| `CSV_FORMAT` | `NO` | Comma-separated output |

#### VEC_TABLE Types

| Value | Output Columns |
|-------|---------------|
| `1` | Position only: X, Y, Z |
| `2` | State vector: X, Y, Z, VX, VY, VZ |
| `3` | State + extras: X, Y, Z, VX, VY, VZ, LT, RG, RR (default) |
| `4` | Position + extras: X, Y, Z, LT, RG, RR |
| `5` | Velocity only: VX, VY, VZ |
| `6` | Extras only: LT, RG, RR |

Where the "extras" are:
- **LT** -- One-way light-time (seconds)
- **RG** -- Range; distance from coordinate center (km or AU)
- **RR** -- Range-rate; radial velocity (km/s or AU/day)

#### Uncertainty Modifiers (Small Bodies Only)

Append to VEC_TABLE value (e.g., `'2xa'`):

| Modifier | Uncertainty Coordinate System |
|----------|-------------------------------|
| `x` | Cartesian XYZ (sigma_X, sigma_Y, sigma_Z, ...) |
| `a` | Along-track / Cross-track / Normal (ACN) |
| `r` | Radial / Transverse / Normal (RTN) |
| `p` | Plane-of-sky (POS_RA, POS_DEC, POS_range) |

Multiple modifiers can be combined: `'2xarp'` outputs state vector plus all four
uncertainty systems.

#### VEC_CORR Aberration Corrections

| Value | Description |
|-------|-------------|
| `NONE` | Geometric (no correction) |
| `LT` | Light-time corrected (astrometric) |
| `LT+S` | Light-time + stellar aberration corrected (apparent) |

#### OUT_UNITS

| Value | Distance | Time |
|-------|----------|------|
| `KM-S` | Kilometers | Seconds |
| `AU-D` | Astronomical Units | Days |
| `KM-D` | Kilometers | Days |

### 4.3 ELEMENTS -- Keplerian Orbital Elements

Generates osculating orbital elements at each timestep, referenced to a major body center.

**Constraint:** The `CENTER` parameter must be a major body (not a topocentric site).

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TP_TYPE` | `ABSOLUTE` | Periapsis time format |
| `OUT_UNITS` | `KM-S` | Output units |
| `REF_PLANE` | `ECLIPTIC` | Reference plane |
| `ELM_LABELS` | `YES` | Include element labels |
| `CSV_FORMAT` | `NO` | Comma-separated output |

#### Output Elements (13 per timestep)

| Symbol | Name | Unit | Description |
|--------|------|------|-------------|
| JDTDB | Epoch | Julian Day (TDB) | Julian Date of osculating elements |
| EC | Eccentricity | dimensionless | 0=circle, <1=ellipse, 1=parabola, >1=hyperbola |
| QR | Periapsis Distance | AU or km | Closest approach distance to center body |
| IN | Inclination | degrees | Angle of orbit plane to reference plane |
| OM | Longitude of Ascending Node | degrees | Direction of orbit plane intersection |
| W | Argument of Perihelion | degrees | Direction of periapsis within orbit plane |
| Tp | Periapsis Time | Julian Day (TDB) | Time of periapsis passage |
| N | Mean Motion | degrees/unit-time | Angular rate: `360/period` |
| MA | Mean Anomaly | degrees | Position along orbit at epoch |
| TA | True Anomaly | degrees | True angular position at epoch |
| A | Semi-Major Axis | AU or km | Mean orbital radius |
| AD | Apoapsis Distance | AU or km | Farthest distance from center body |
| PR | Orbital Period | unit-time | Time for one complete orbit |

**TP_TYPE values:**

| Value | Description |
|-------|-------------|
| `ABSOLUTE` | Tp as absolute Julian Day number (default) |
| `RELATIVE` | Tp as days relative to epoch |

### 4.4 SPK -- Binary Ephemeris File Generation

Generates SPICE SPK binary trajectory files for **small bodies only** (asteroids and
comets).

**Response:**

| JSON Field | Description |
|------------|-------------|
| `spk` | Base64-encoded binary SPK file content |
| `spk_file_id` | Suggested output filename (e.g., `2099942.bsp`) |

**Decoding procedure:**
1. Parse JSON response
2. Extract the `spk` field (base64 string)
3. Base64-decode to raw bytes
4. Write bytes to a `.bsp` file

**Key details:**
- Generates **Type 21** SPK segments (Extended Modified Difference Arrays)
- Provides time-continuous trajectory (not discrete steps)
- `START_TIME` and `STOP_TIME` are interpreted as **TDB** (not UT)
- The resulting file can be loaded by the SPICE Toolkit (NAIF)
- Only `COMMAND`, `START_TIME`, and `STOP_TIME` are meaningful parameters

### 4.5 APPROACH -- Close-Approach Tables

Generates tables of close approaches between a small body and major solar system bodies.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CA_TABLE_TYPE` | `STANDARD` | `STANDARD` or `EXTENDED` |
| `TCA3SG_LIMIT` | `14400` | Max 3-sigma time uncertainty (minutes) |
| `CALIM_SB` | `0.05` | Small-body close-approach distance limit (AU) |
| `CALIM_PL` | (see below) | Per-planet distance limits (AU) |

**Default CALIM_PL values** (Mercury through Pluto, comma-separated):
```
.1,.1,.1,.1,1.0,1.0,1.0,1.0,.1,.003
```

| Planet | Default Limit (AU) |
|--------|-------------------|
| Mercury | 0.1 |
| Venus | 0.1 |
| Earth | 0.1 |
| Mars | 0.1 |
| Jupiter | 1.0 |
| Saturn | 1.0 |
| Uranus | 1.0 |
| Neptune | 1.0 |
| Pluto | 0.1 |
| Moon | 0.003 |

#### STANDARD Table Fields

| Field | Description |
|-------|-------------|
| Date (TDB) | Time of closest approach |
| Body | Name of close-approach body |
| CA Dist (AU) | Nominal minimum distance |
| MinDist (AU) | Minimum possible distance (3-sigma) |
| MaxDist (AU) | Maximum possible distance (3-sigma) |
| Vrel (km/s) | Relative velocity at closest approach |

#### EXTENDED Table Fields (15 fields)

| Field | Description |
|-------|-------------|
| Date (TDB) | Time of closest approach |
| Body | Name of close-approach body |
| CA Dist (AU) | Nominal minimum distance |
| MinDist (AU) | Minimum possible distance (3-sigma) |
| MaxDist (AU) | Maximum possible distance (3-sigma) |
| Vrel (km/s) | Relative velocity at closest approach |
| TCA3Sg (min) | 3-sigma time-of-CA uncertainty |
| SMaA-1Sg (km) | B-plane semi-major axis (1-sigma) |
| SMiA-1Sg (km) | B-plane semi-minor axis (1-sigma) |
| B.T (km) | B-plane B dot T component |
| B.R (km) | B-plane B dot R component |
| Theta (deg) | B-plane orientation angle |
| Nsig | Number of sigma to LOV intersection |
| P_i/p | Linearized impact probability |
| GeoFlag | Flag for geocentric vs. heliocentric |

---

## 5. Reference Systems

### 5.1 REF_SYSTEM

| Value | Description |
|-------|-------------|
| `ICRF` | International Celestial Reference Frame (default). Aligned with J2000 to ~0.0002 arcsec. Based on extragalactic radio sources (quasars). |
| `B1950` | FK4/B1950 dynamical frame. For historical data and older catalogs. |

### 5.2 REF_PLANE (VECTORS and ELEMENTS only)

| Value | Description |
|-------|-------------|
| `ECLIPTIC` | Ecliptic and mean equinox of reference epoch (default). For ICRF: IAU76 obliquity at J2000 (84381.448 arcsec). |
| `FRAME` | Earth mean equator and equinox of reference epoch (equatorial). |
| `BODY EQUATOR` | IAU body-fixed equatorial frame of the coordinate center body. |

### 5.3 Full Frame List

| Frame | Description | Use Case |
|-------|-------------|----------|
| ICRF | Fixed inertial frame; VLBI quasar-based | Default for all computations |
| Earth True Equator of Date | Precession + nutation (IAU76/80 model) | Apparent RA/Dec for Earth observers |
| Earth Ecliptic of Date | Dynamic ecliptic plane | Some observer quantities |
| Ecliptic at J2000 | Fixed ecliptic (IAU76 obliquity) | Asteroid/comet orbital elements |
| IAU Body-Frame | Body-fixed rotation model | Cartographic coordinates |
| Lunar Mean Earth Frame | Moon-centric (DE421), 1550-2650 AD | Lunar surface coordinates |
| FK4/B1950 | Older dynamical frame | Historical data |
| ITRF93 | Earth body frame + polar motion | Earth station coordinates |

### 5.4 Obliquity Constants

| Frame | Obliquity (arcsec) |
|-------|-------------------|
| J2000 ecliptic (with ICRF) | 84381.448 |
| B1950 ecliptic (with FK4) | 84404.836 |

---

## 6. Coordinate Centers (`CENTER`)

The `CENTER` parameter specifies the observer or coordinate origin location.

### 6.1 Syntax

| Syntax | Example | Meaning |
|--------|---------|---------|
| Body ID | `'500'` | Geocenter (Earth center) |
| Body ID @ Body | `'500@0'` | Solar System Barycenter |
| Body ID @ Body | `'500@10'` | Heliocenter (Sun center) |
| Site code | `'675'` | Palomar Mountain Observatory (Earth) |
| Site @ Body | `'675@399'` | Palomar Mountain on Earth (explicit) |
| Custom coords | `'coord@399'` | Custom Earth coordinates |
| Custom coords | `'coord@499'` | Custom Mars coordinates |
| Geocenter shorthand | `'geo'` or `'g@399'` | Earth geocenter |

### 6.2 Predefined Sites

- **2,300+** predefined Earth observation sites (matches Minor Planet Center list)
- Includes radar and radio telescope sites (some with negative ID numbers)
- Sites on other bodies: Apollo landing sites on Moon, Viking on Mars, etc.

**Listing sites:**

| Command | Result |
|---------|--------|
| `CENTER='*@399'` | List all Earth sites |
| `CENTER='*@301'` | List all Moon sites |
| `CENTER='*@499'` | List all Mars sites |

### 6.3 Custom Coordinates

When `CENTER='coord@BODYID'`, specify the location via:

| Parameter | Description |
|-----------|-------------|
| `COORD_TYPE` | `GEODETIC` or `CYLINDRICAL` |
| `SITE_COORD` | Comma-separated coordinate triplet |

**GEODETIC** (for bodies with defined ellipsoids):
```
SITE_COORD = 'East-Longitude, Latitude, Altitude(km)'
```
- East-Longitude: degrees east of prime meridian
- Latitude: geodetic latitude in degrees
- Altitude: height above reference ellipsoid in km

**CYLINDRICAL** (universal, works for any body):
```
SITE_COORD = 'East-Longitude, DXY(km), DZ(km)'
```
- East-Longitude: degrees east of prime meridian
- DXY: distance from spin-axis in km
- DZ: distance above equator plane in km

**Important longitude conventions:**
- Earth, Sun, Moon: positive = east longitude (despite prograde rotation)
- Retrograde rotators (Venus, Uranus moons): positive = east longitude
- Most prograde bodies: use negative east-longitude (= west-longitude - 360)

### 6.4 Topocentric Sites on Other Bodies

Any body with an IAU rotation model can serve as a topocentric site. The system computes
local horizon, airmass, rise/set, etc. relative to that body's surface.

---

## 7. Time Specification

### 7.1 TIME_TYPE

| Value | Description |
|-------|-------------|
| `UT` | Universal Time. Before 1962: UT1 (Earth rotation). After 1962: UTC with leap seconds. Default for OBSERVER. |
| `TT` | Terrestrial Time. Proper atomic time on Earth geoid. Differs from TDB by <= 0.002 sec. |
| `TDB` | Barycentric Dynamical Time. Independent variable of planetary equations. Default for VECTORS, ELEMENTS, APPROACH. |

### 7.2 Input Formats

| Format | Example | Notes |
|--------|---------|-------|
| Calendar (recommended) | `'2027-May-5 12:30:23.3348'` | Unambiguous month name |
| Calendar (numeric) | `'2028-05-04 18:00'` | ISO-like |
| Calendar (reversed) | `'04-05-2028 18:00'` | DD-MM-YYYY |
| Calendar (informal) | `'2 jan 1991 3:00:12.2'` | Flexible parsing |
| Julian Day | `'JD 2451545.0'` | Prefix with "JD" |
| Julian Day (compact) | `'JD2451545'` | No space |
| Day-of-Year | `'2016-365//12:00'` | Double-slash separates time |
| Ancient (BC) | `'278bc-jan-12 12:34'` | "bc" suffix required |
| Ancient (AD) | `'99ad-Aug-30'` | "ad" suffix for early dates |

### 7.3 STEP_SIZE Modes

| Mode | Syntax | Description |
|------|--------|-------------|
| Fixed days | `'1d'`, `'30 days'` | Uniform time step in days |
| Fixed hours | `'3 h'`, `'12h'` | Uniform time step in hours |
| Fixed minutes | `'10m'`, `'10 min'` | Uniform time step in minutes |
| Calendar year | `'1 year'`, `'1 y'` | Calendar-aware stepping |
| Calendar month | `'1 mo'`, `'3 mo'` | Calendar-aware stepping |
| N-step | `'100'` | Divide time range into 100 equal intervals |
| Rise/transit/set | `'1m TVH'` | True visual horizon with refraction |
| Rise/transit/set | `'5m GEO'` | Geometric horizon with refraction |
| Rise/transit/set | `'3m RAD'` | Radar horizon (no refraction) |
| Variable angular | `'VAR 600'` | Output after 600 arcsec sky-plane motion |

**Variable angular range:** 60 to 3600 arcseconds.

**Minimum step size:** 0.5 seconds (achieved via N-step mode for small time ranges).

### 7.4 Discrete Time Lists (TLIST)

Instead of START_TIME/STOP_TIME/STEP_SIZE, provide up to **10,000** discrete times:

| Parameter | Description |
|-----------|-------------|
| `TLIST` | Comma-separated list of times |
| `TLIST_TYPE` | `JD` (Julian Day), `MJD` (Modified Julian Day), or `CAL` (calendar) |

Example:
```
TLIST='2451545.0,2451546.0,2451547.5'
TLIST_TYPE='JD'
```

When using TLIST, `START_TIME`, `STOP_TIME`, and `STEP_SIZE` are ignored.

### 7.5 Output Format Parameters

| Parameter | Default | Values | Description |
|-----------|---------|--------|-------------|
| `TIME_DIGITS` | `MINUTES` | `MINUTES`, `SECONDS`, `FRACSEC` | Timestamp precision |
| `CAL_FORMAT` | `CAL` | `CAL`, `JD`, `BOTH` | Date column format |
| `CAL_TYPE` | `MIXED` | `MIXED`, `GREGORIAN` | Calendar system |
| `TIME_ZONE` | `+00:00` | UTC offset string | Time zone for output |

`CAL_TYPE` values:
- `MIXED`: Julian calendar before Oct 15 1582, Gregorian after
- `GREGORIAN`: Gregorian calendar throughout (default for modern work)

### 7.6 Ephemeris Coverage

| Object Class | Coverage |
|-------------|----------|
| Planetary barycenters (DE441) | 13200 BC to AD 17191 |
| Planet centers (e.g., 199, 299) | ~1600 to ~2500 (varies) |
| Moon (301) | ~13200 BC to AD 17191 (reduced accuracy outside 1550-2650) |
| Asteroids | Depends on orbit quality; typically decades from epoch |
| Comets | Depends on apparition; may be single-apparition only |
| Spacecraft | Mission-specific only; check individual time spans |
| Lagrange points | Same as parent system |

---

## 8. Physical Constants (`OBJ_DATA=YES`)

When `OBJ_DATA='YES'` (default), the response includes a physical data block before the
ephemeris table. The available fields depend on the object type.

### Returned Fields (when available)

| Field | Unit | Description |
|-------|------|-------------|
| Radius | km | Mean volumetric radius (equatorial, polar, or triaxial) |
| Density | g/cm^3 | Mean bulk density |
| Mass | kg | Total mass |
| GM | km^3/s^2 | Gravitational parameter (mass x G) |
| Flattening | dimensionless | Oblateness: (R_eq - R_pol) / R_eq |
| Rotation Period | hours | Sidereal rotation period |
| Rotation Rate | rad/s | Angular rotation rate |
| Solar Day | hours | Synodic day length |
| Surface Gravity | m/s^2 | Equatorial surface gravity |
| Geometric Albedo | dimensionless | Geometric visual albedo |
| V(1,0) | mag | Absolute visual magnitude at 1 AU |
| Obliquity | degrees | Axial tilt to orbital plane |
| Orbital Period | years or days | Sidereal orbital period |
| Orbital Speed | km/s | Mean orbital velocity |
| Escape Speed | km/s | Surface escape velocity |
| Hill Sphere Radius | AU or km | Gravitational sphere of influence |
| Angular Diameter | arcsec | As seen from 1 AU |
| Mean Temperature | K | Surface or cloud-top temperature |
| Atmospheric Pressure | bar | Surface atmospheric pressure |
| H (asteroids) | mag | Absolute magnitude parameter |
| G (asteroids) | dimensionless | Magnitude slope parameter |
| B-V | mag | Color index |
| Spectral Type | string | Tholen/SMASSII taxonomic class |
| Rotation Period | hours | For asteroids |
| M1, M2 (comets) | mag | Total and nuclear absolute magnitudes |
| K1, K2 (comets) | dimensionless | Magnitude scaling factors |

### Reference Constants

| Constant | Value | Unit |
|----------|-------|------|
| Sun GM | 1.32712440018 x 10^11 | km^3/s^2 |
| Earth GM | 3.986004418 x 10^5 | km^3/s^2 |
| Moon GM | 4.9028005821 x 10^3 | km^3/s^2 |
| AU | 149,597,870.700 | km |

---

## 9. User-Defined Objects

User-defined objects allow ephemeris generation for hypothetical bodies or objects not in
the HORIZONS database.

### 9.1 Heliocentric Ecliptic Orbital Elements

Set `COMMAND=';'` and provide the following parameters:

#### Required

| Parameter | Unit | Description |
|-----------|------|-------------|
| `OBJECT` | string | User-chosen name for the object |
| `EPOCH` | JD (TDB) | Julian Day of osculating elements (TDB timescale) |
| `ECLIP` | string | `J2000` or `B1950` -- ecliptic reference |
| `EC` | dimensionless | Eccentricity |

Plus **one** of these element sets:

**Set A: Periapsis time + distance**

| Parameter | Unit |
|-----------|------|
| `TP` | JD (TDB) |
| `QR` | AU |

**Set B: Mean anomaly + semi-major axis**

| Parameter | Unit |
|-----------|------|
| `MA` | degrees |
| `A` | AU |

**Set C: Mean anomaly + mean motion**

| Parameter | Unit |
|-----------|------|
| `MA` | degrees |
| `N` | degrees/day |

#### Optional Orientation

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `OM` | degrees | 0.0 | Longitude of ascending node |
| `W` | degrees | 0.0 | Argument of perihelion |
| `IN` | degrees | 0.0 | Inclination |
| `RAD` | km | 0.0 | Object radius |

#### Optional Magnitude Parameters (Asteroids)

| Parameter | Description |
|-----------|-------------|
| `H` | Absolute magnitude parameter |
| `G` | Magnitude slope parameter (default 0.15) |

#### Optional Magnitude Parameters (Comets)

| Parameter | Description |
|-----------|-------------|
| `M1` | Total absolute magnitude |
| `M2` | Nuclear absolute magnitude |
| `K1` | Total magnitude scaling factor |
| `K2` | Nuclear magnitude scaling factor |
| `PHCOF` | Phase coefficient for K2=5 |

#### Optional Non-Gravitational Parameters

| Parameter | Unit | Default | Description |
|-----------|------|---------|-------------|
| `A1` | AU/d^2 | 0 | Radial non-gravitational acceleration |
| `A2` | AU/d^2 | 0 | Transverse non-gravitational acceleration |
| `A3` | AU/d^2 | 0 | Normal non-gravitational acceleration |
| `R0` | AU | 2.808 | Normalizing heliocentric distance |
| `ALN` | dimensionless | 0.1112620426 | Normalizing factor |
| `NM` | dimensionless | 2.15 | Radial exponent |
| `NN` | dimensionless | 5.093 | Fall-off exponent |
| `NK` | dimensionless | 4.6142 | Normalization exponent |
| `DT` | days | 0 | Non-grav lag/delay (comets) |
| `AMRAT` | m^2/kg | 0 | Solar radiation pressure area-to-mass ratio |

#### Optional Covariance

| Parameter | Description |
|-----------|-------------|
| `SRC` | JPL square-root covariance matrix (upper-triangular, space-separated) |
| `EST` | Estimated non-gravitational parameter names |

### 9.2 Two-Line Elements (TLE)

Set `COMMAND='TLE'` and provide SGP4/SDP4 geocentric elements:

```
TLE='ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993
2 25544  51.6420  30.2134 0002345 210.5678 149.4246 15.49456789123456'
```

- **Maximum:** 600 TLE pairs (1200 lines + optional name lines)
- **URL encoding:** Newlines as `%0A`, spaces as `%20`
- **Propagation:** Standard SGP4/SDP4 model; limited to ~few days forecast accuracy

---

## 10. Small-Body Search/Filter

The `COMMAND` parameter accepts search expressions to find small bodies matching orbital and
physical criteria. Results are then selectable for ephemeris generation.

### 10.1 Search Syntax

```
COMMAND='search_expression_1; search_expression_2; ...'
```

Each expression is `KEYWORD operator VALUE`. Operators: `<`, `>`, `<>` (not equal), `=`.

### 10.2 All Search Keywords

#### Common (Asteroids and Comets)

| Type | Keyword | Description |
|------|---------|-------------|
| C | `NAME` | Name fragment (wildcard with `*`) |
| C | `DES` | Designation (e.g., `1990 MU`, `1993*`) |
| R | `EPOCH` | Julian Date of osculating elements |
| R | `A` | Semi-major axis (AU) |
| R | `EC` | Eccentricity |
| R | `IN` | Inclination (degrees) |
| R | `OM` | Longitude of ascending node (degrees) |
| R | `W` | Argument of perihelion (degrees) |
| R | `TP` | Perihelion Julian Date |
| R | `MA` | Mean anomaly (degrees) |
| R | `PER` | Orbital period (years) |
| R | `RAD` | Object radius (km) |
| R | `GM` | Gravitational parameter (km^3/s^2) |
| R | `QR` | Perihelion distance (AU) |
| R | `ADIST` | Aphelion distance (AU) |
| R | `ANGMOM` | Specific angular momentum (AU^2/day) |
| R | `N` | Mean motion (degrees/day) |
| R | `DAN` | Distance at ascending node (AU) |
| R | `DDN` | Distance at descending node (AU) |
| R | `L` | Ecliptic longitude (degrees) |
| R | `B` | Ecliptic latitude (degrees) |
| I | `NOBS` | Number of astrometric observations |
| C | `SOLN` | Solution ID |

Type key: **C** = character/string, **R** = real/numeric, **I** = integer

#### Asteroid-Specific

| Type | Keyword | Description |
|------|---------|-------------|
| C | `ASTNAM` | Asteroid name fragment |
| R | `B-V` | B-V color index |
| R | `H` | Absolute magnitude parameter |
| R | `G` | Magnitude slope parameter |
| R | `ROTPER` | Rotation period (hours) |
| R | `ALBEDO` | Geometric albedo |
| C | `STYP` | Spectral type (Tholen taxonomy) |

#### Comet-Specific

| Type | Keyword | Description |
|------|---------|-------------|
| C | `COMNAM` | Comet name fragment |
| I | `COMNUM` | Comet number |
| R | `M1` | Total absolute magnitude |
| R | `M2` | Nuclear absolute magnitude |
| R | `K1` | Total magnitude scaling factor |
| R | `K2` | Nuclear magnitude scaling factor |
| R | `PHCOF` | Phase coefficient |
| R | `A1` | Radial non-grav acceleration (AU/day^2) |
| R | `A2` | Transverse non-grav acceleration (AU/day^2) |
| R | `A3` | Normal non-grav acceleration (AU/day^2) |
| R | `DT` | Non-grav lag/delay (days) |

### 10.3 Search Directives

| Directive | Description |
|-----------|-------------|
| `COM` | Limit search to comets only |
| `AST` | Limit search to asteroids only |
| `LIST` | Display parameter values for matching objects |
| `NOFRAG` | Exclude comet fragments from results |
| `CAP` | Select closest apparition to current date |
| `CAP < JD#` | Closest apparition before specified Julian Day |
| `CAP < YEAR` | Closest apparition before specified integer year |

### 10.4 Search Examples

```
# S-type asteroids with a < 2.5 AU and i > 7.8 degrees, with known GM
COMMAND='A < 2.5; IN > 7.8; STYP = S; GM <> 0;'

# Comet 73P, closest apparition, no fragments
COMMAND='DES = 73P; NOFRAG; CAP'

# All objects with names starting with "mua"
COMMAND='NAME = mua*;'

# All comets with names containing "her"
COMMAND='COMNAM = HER*;'

# Asteroids with rotation period < 3 hours
COMMAND='AST; ROTPER < 3;'
```

---

## 11. Response Format

### 11.1 JSON Structure

Standard ephemeris responses:
```json
{
  "signature": {
    "source": "NASA/JPL Horizons API",
    "version": "1.3"
  },
  "result": "full text output including headers, ephemeris data, and footers"
}
```

SPK file responses:
```json
{
  "signature": {
    "source": "NASA/JPL Horizons API",
    "version": "1.3"
  },
  "spk": "base64-encoded-binary-content",
  "spk_file_id": "suggested_filename.bsp"
}
```

### 11.2 Parsing the `result` Field

The `result` field contains the full Horizons text output as a single string. Key
delimiters for parsing:

| Delimiter | Meaning |
|-----------|---------|
| `$$SOE` | **Start Of Ephemeris** -- data rows begin on the next line |
| `$$EOE` | **End Of Ephemeris** -- data rows end on the previous line |

**Parsing algorithm:**
1. Split `result` on newlines
2. Find the line containing `$$SOE`
3. Find the line containing `$$EOE`
4. Extract all lines between these delimiters as data rows
5. Parse each row according to the column format

### 11.3 CSV Output

When `CSV_FORMAT='YES'`:
- Column headers are included as the first row after `$$SOE`
- Data values are comma-separated
- Easier to parse programmatically than fixed-width format

### 11.4 Text (Non-JSON) Format

When `format=text`:
- First two lines: `API VERSION: X.X` and `API SOURCE: NASA/JPL Horizons API`
- Followed by blank lines
- Then the standard Horizons output (same content as `result` field in JSON)

---

## 12. Rate Limits and Practical Concerns

### Rate Limits

- **No API key required**
- **No published rate limits**
- Free public service; NASA expects "reasonable use"
- The service is shared infrastructure; excessive requests may trigger HTTP 503

### Request Size Limits

| Limit | Value |
|-------|-------|
| URL length (GET) | ~2000 characters practical limit |
| TLIST entries | 10,000 maximum |
| TLE pairs | 600 maximum (1200 lines) |
| Ephemeris time range | No hard limit, but large requests take time |

### Implementation Recommendations

- Use the File API (POST) for requests that would exceed ~1500 characters in the URL
- Implement exponential backoff for HTTP 503 responses
- Cache results where appropriate (ephemeris data is deterministic for given inputs)
- Use `CSV_FORMAT='YES'` for easier parsing
- Always check the `result` field for error messages even on HTTP 200 responses
- Use `OBJ_DATA='NO'` when physical constants are not needed (faster parsing)
- Set `QUANTITIES` to only the codes actually needed (reduces response size)
- Prefer `format=json` for programmatic access

### Error Detection

HORIZONS errors in the result text can be identified by patterns such as:
- `"No ephemeris for target"` -- object not found or time out of range
- `"Cannot find central body"` -- invalid CENTER
- `"Ambiguous target name"` -- COMMAND matches multiple objects
- `"No matches found"` -- search returned no results

For ambiguous matches, the response includes a numbered list of candidates that the client
can present for disambiguation.

---

## 13. Complete Parameter Reference Table

### Core Parameters (All Ephemeris Types)

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `format` | `json` | `json`, `text` | API output format |
| `COMMAND` | *(required)* | See Section 3 | Target body selection |
| `OBJ_DATA` | `YES` | `YES`, `NO` | Include physical/orbital data summary |
| `MAKE_EPHEM` | `YES` | `YES`, `NO` | Generate ephemeris table |
| `EPHEM_TYPE` | `OBSERVER` | `OBSERVER`, `VECTORS`, `ELEMENTS`, `SPK`, `APPROACH` | Ephemeris type |

### Time Parameters (OBSERVER, VECTORS, ELEMENTS, SPK)

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `CENTER` | geocentric | Site code, body ID, `coord@body` | Observer/coordinate center |
| `START_TIME` | *(required)* | Date string or `JD #` | Ephemeris start time |
| `STOP_TIME` | *(required)* | Date string or `JD #` | Ephemeris stop time |
| `STEP_SIZE` | `60 min` | Time step, N-steps, `VAR #` | Output interval |
| `TLIST` | *(none)* | Comma-separated times | Discrete time list (up to 10,000) |
| `TLIST_TYPE` | *(none)* | `JD`, `MJD`, `CAL` | Time list format |
| `TIME_TYPE` | varies | `UT`, `TT`, `TDB` | Input/output timescale |
| `TIME_DIGITS` | `MINUTES` | `MINUTES`, `SECONDS`, `FRACSEC` | Timestamp precision |
| `CAL_FORMAT` | `CAL` | `CAL`, `JD`, `BOTH` | Date output format |
| `CAL_TYPE` | `MIXED` | `MIXED`, `GREGORIAN` | Calendar system |
| `REF_SYSTEM` | `ICRF` | `ICRF`, `B1950` | Reference frame |
| `COORD_TYPE` | `GEODETIC` | `GEODETIC`, `CYLINDRICAL` | Custom site coordinate type |
| `SITE_COORD` | `0,0,0` | Coordinate triplet | Custom site coordinates |
| `CSV_FORMAT` | `NO` | `YES`, `NO` | Comma-separated output |

### OBSERVER-Specific Parameters

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `QUANTITIES` | `A` | `1`-`48` (comma-sep), or `A`-`F` | Observable quantity selection |
| `ANG_FORMAT` | `HMS` | `HMS`, `DEG` | RA/Dec angle format |
| `APPARENT` | `AIRLESS` | `AIRLESS`, `REFRACTED` | Refraction correction |
| `TIME_ZONE` | `+00:00` | UTC offset string | Time zone offset |
| `RANGE_UNITS` | `AU` | `AU`, `KM` | Range output units |
| `SUPPRESS_RANGE_RATE` | `NO` | `YES`, `NO` | Suppress range-rate column |
| `ELEV_CUT` | `-90` | -90 to 90 (integer) | Minimum elevation filter (degrees) |
| `SKIP_DAYLT` | `NO` | `YES`, `NO` | Skip daylight hours |
| `SOLAR_ELONG` | `0,180` | Min,Max (degrees) | Solar elongation filter |
| `AIRMASS` | `38.0` | Decimal value | Maximum airmass cutoff |
| `LHA_CUTOFF` | `0.0` | 0.0 to 12.0 | Local hour angle limit (hours) |
| `ANG_RATE_CUTOFF` | `0.0` | Decimal (arcsec/hr) | Minimum angular rate filter |
| `EXTRA_PREC` | `NO` | `YES`, `NO` | Extra RA/Dec decimal digits |
| `R_T_S_ONLY` | `NO` | `YES`, `NO` | Only rise/transit/set events |

### VECTORS-Specific Parameters

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `VEC_TABLE` | `3` | `1`-`6`, with `x`/`a`/`r`/`p` modifiers | Vector table format |
| `VEC_CORR` | `NONE` | `NONE`, `LT`, `LT+S` | Aberration correction |
| `OUT_UNITS` | `KM-S` | `KM-S`, `AU-D`, `KM-D` | Distance/time units |
| `REF_PLANE` | `ECLIPTIC` | `ECLIPTIC`, `FRAME`, `BODY EQUATOR` | Reference plane |
| `VEC_LABELS` | `YES` | `YES`, `NO` | Column labels in output |
| `VEC_DELTA_T` | `NO` | `YES`, `NO` | Include TDB-UT delta |

### ELEMENTS-Specific Parameters

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `TP_TYPE` | `ABSOLUTE` | `ABSOLUTE`, `RELATIVE` | Periapsis time format |
| `OUT_UNITS` | `KM-S` | `KM-S`, `AU-D`, `KM-D` | Output units |
| `REF_PLANE` | `ECLIPTIC` | `ECLIPTIC`, `FRAME`, `BODY EQUATOR` | Reference plane |
| `ELM_LABELS` | `YES` | `YES`, `NO` | Element labels in output |

### APPROACH-Specific Parameters

| Parameter | Default | Valid Values | Description |
|-----------|---------|--------------|-------------|
| `CA_TABLE_TYPE` | `STANDARD` | `STANDARD`, `EXTENDED` | Table detail level |
| `TCA3SG_LIMIT` | `14400` | Integer (minutes) | Max 3-sigma time uncertainty |
| `CALIM_SB` | `0.05` | Decimal (AU) | Small-body distance limit |
| `CALIM_PL` | `.1,.1,.1,.1,1,1,1,1,.1,.003` | 10 comma-separated AU values | Per-planet distance limits |

### SPK-Specific Parameters

SPK requests only use `COMMAND`, `START_TIME`, and `STOP_TIME`. All other ephemeris
parameters are ignored. The output is always a Type 21 SPK binary file.

---

## 14. Example Requests

### 14.1 Vector Ephemeris for Mars (from SSB, in AU-D)

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='499'&OBJ_DATA='NO'&MAKE_EPHEM='YES'&EPHEM_TYPE='VECTORS'&CENTER='500@0'&START_TIME='2024-01-01'&STOP_TIME='2024-02-01'&STEP_SIZE='1%20d'&OUT_UNITS='AU-D'&REF_PLANE='ECLIPTIC'&VEC_TABLE='2'&VEC_CORR='LT'&CSV_FORMAT='YES'
```

**Parameters explained:**
- `COMMAND='499'` -- Mars center
- `CENTER='500@0'` -- Solar System Barycenter
- `EPHEM_TYPE='VECTORS'` -- Cartesian state vectors
- `OUT_UNITS='AU-D'` -- AU and days
- `VEC_TABLE='2'` -- Position + velocity (X,Y,Z,VX,VY,VZ)
- `VEC_CORR='LT'` -- Light-time corrected
- `CSV_FORMAT='YES'` -- CSV output for easy parsing

### 14.2 Orbital Elements for Ceres (heliocentric)

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='1%3B'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='ELEMENTS'&CENTER='500@10'&START_TIME='2024-01-01'&STOP_TIME='2025-01-01'&STEP_SIZE='30%20d'&REF_PLANE='ECLIPTIC'&TP_TYPE='ABSOLUTE'&CSV_FORMAT='YES'
```

**Parameters explained:**
- `COMMAND='1%3B'` -- Ceres (1;) -- semicolon marks small body
- `CENTER='500@10'` -- Heliocenter (Sun center)
- `EPHEM_TYPE='ELEMENTS'` -- Keplerian orbital elements
- `TP_TYPE='ABSOLUTE'` -- Periapsis time as absolute JD

### 14.3 Observer Table for Apophis from Palomar

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='Apophis%3B'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER='675@399'&START_TIME='2029-04-01'&STOP_TIME='2029-04-30'&STEP_SIZE='1%20h'&QUANTITIES='1,2,9,20,23,24,36,37,47'&ANG_FORMAT='DEG'&EXTRA_PREC='YES'&CSV_FORMAT='YES'
```

**Parameters explained:**
- `COMMAND='Apophis%3B'` -- Asteroid Apophis (name search, small-body)
- `CENTER='675@399'` -- Palomar Mountain Observatory
- `QUANTITIES='1,2,9,20,23,24,36,37,47'` -- Astrometric RA/Dec, apparent RA/Dec, magnitude, range, elongation, phase angle, uncertainty, error ellipse, sky motion
- `EXTRA_PREC='YES'` -- Extra decimal digits for precise astrometry

### 14.4 Close-Approach Data for a Near-Earth Object

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='99942%3B'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='APPROACH'&START_TIME='2024-01-01'&STOP_TIME='2050-01-01'&CA_TABLE_TYPE='EXTENDED'&CALIM_SB='0.2'&TCA3SG_LIMIT='14400'
```

**Parameters explained:**
- `COMMAND='99942%3B'` -- Apophis by IAU number
- `EPHEM_TYPE='APPROACH'` -- Close-approach table
- `CA_TABLE_TYPE='EXTENDED'` -- Full B-plane details
- `CALIM_SB='0.2'` -- Show approaches within 0.2 AU

### 14.5 SPK File Generation for an Asteroid

```
GET https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='433%3B'&MAKE_EPHEM='YES'&EPHEM_TYPE='SPK'&START_TIME='2024-01-01'&STOP_TIME='2025-01-01'
```

**Parameters explained:**
- `COMMAND='433%3B'` -- 433 Eros
- `EPHEM_TYPE='SPK'` -- Generate binary SPK file
- Response will contain `spk` (base64) and `spk_file_id` fields

**Decoding the response (conceptual Rust pseudocode):**
```rust
let response: serde_json::Value = client.get(url).send()?.json()?;
let spk_b64 = response["spk"].as_str().unwrap();
let spk_bytes = base64::decode(spk_b64)?;
let filename = response["spk_file_id"].as_str().unwrap();
std::fs::write(filename, &spk_bytes)?;
```

### 14.6 Lookup API Query

```
GET https://ssd.jpl.nasa.gov/api/horizons_lookup.api?sstr=Apophis&group=ast
```

**Response example:**
```json
{
  "signature": {"source": "NASA/JPL ...", "version": "1.1"},
  "count": "1",
  "result": [
    {
      "pdes": "99942",
      "name": "Apophis",
      "spkid": "2099942",
      "alias": ["2004 MN4"]
    }
  ]
}
```

---

## 15. URL Encoding Quick Reference

Characters that must be encoded in URL query parameters:

| Character | Encoding | Common Context |
|-----------|----------|----------------|
| `;` (semicolon) | `%3B` | Small-body suffix |
| ` ` (space) | `%20` or `+` | Designations, step sizes |
| `=` (equals) | `%3D` | DES= prefix |
| `@` (at) | `%40` | CENTER site specs |
| `#` (hash) | `%23` | Fragment identifiers |
| `&` (ampersand) | `%26` | Would conflict with param separator |
| `+` (plus) | `%2B` | Would be interpreted as space |
| `,` (comma) | `%2C` | Usually OK unencoded in values |
| `/` (slash) | `%2F` | Path separator |
| `\n` (newline) | `%0A` | TLE line breaks |

---

## 16. Implementation Notes for a Rust Client

### Recommended Crate Dependencies

- `reqwest` -- HTTP client (async or blocking)
- `serde` / `serde_json` -- JSON deserialization
- `base64` -- Decoding SPK file responses
- `url` -- URL construction and encoding
- `chrono` -- Date/time handling for time parameters
- `thiserror` -- Error type definitions

### Suggested Type Structure

```
HorizonsClient
  |- query(params: EphemerisRequest) -> Result<HorizonsResponse>
  |- lookup(name: &str, group: Option<ObjectGroup>) -> Result<LookupResponse>

EphemerisRequest
  |- command: Command (enum: MajorBody(i32), SmallBody(String), TLE(String), UserDefined(...))
  |- ephem_type: EphemType (enum: Observer, Vectors, Elements, Spk, Approach)
  |- center: Center (enum: BodyCenter(i32), Site(String), Coordinates{...})
  |- time_spec: TimeSpec (enum: Range{start, stop, step}, DiscreteList{times, format})
  |- ... type-specific options ...

HorizonsResponse
  |- signature: Signature { source: String, version: String }
  |- result: Option<String>     // text output
  |- spk: Option<Vec<u8>>       // decoded binary SPK
  |- spk_file_id: Option<String>

Command
  |- MajorBody(i32)              // e.g., 499 for Mars
  |- Asteroid { number: u32 }    // appends semicolon
  |- Comet { designation: String, flags: Vec<SearchFlag> }
  |- Designation(String)         // DES=...
  |- Name(String)                // name search
  |- TLE(String)                 // two-line elements
  |- UserDefined(OrbitalElements)
  |- Search(Vec<SearchCriterion>)
```

### Key Implementation Considerations

1. **Semicolon handling:** Always append `;` for small bodies; URL-encode as `%3B`
2. **Error detection:** HTTP 200 does not mean success; always parse `result` for error patterns
3. **Ambiguity handling:** When HORIZONS returns a disambiguation list, parse it and either auto-select or propagate to the user
4. **Large requests:** Switch from GET to POST (File API) when the encoded URL would exceed ~1500 characters
5. **Response parsing:** The `$$SOE` / `$$EOE` delimiters are the primary mechanism for extracting data rows
6. **SPK decoding:** The `spk` field is standard base64; decode and write as binary
7. **Time formats:** Support at least ISO calendar format and Julian Day input; handle TDB/TT/UT correctly
8. **CSV mode:** Strongly prefer `CSV_FORMAT='YES'` for all programmatic parsing; fixed-width format is fragile
