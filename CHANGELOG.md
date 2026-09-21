# Changelog

## 0.17.1

- Combine local SPK files with `SpiceKernel::open_many` or `merge`, and named
  datastore kernels with `Loader::open_many`, so observers and targets can
  span planetary and satellite kernels.
- Prefer the latest covering segment for overlapping center/target pairs;
  retain earlier coverage outside later segments, including during light-time
  iteration. Preserve unsupported-type diagnostics across merged files.
- Route Mars, Saturn, Uranus, Neptune and Pluto satellite SPKs through the
  NAIF satellite artifact namespace, alongside Jupiter.
- Document the existing text-PCK loader for satellite radii and frames, and
  add a Mars-to-Io observation example.

## 0.17.0

- Join starfield and starfield-datasources into one history-preserving workspace,
  with one version, changelog, CI, and automated publication workflow.
- Keep `starfield` as the public facade; move shared calculations and traits to
  `starfield-core`. All internal dependencies use exact, local version pins.
- Add `catalogs`, `jpl`, and `surfaces` feature groups and individual datasource
  features, preserving Gaia release and NSA radial-profile controls.
- Consolidate HORIZONS and SBDB implementations, retaining HORIZONS calendar
  formatting and SBDB orbital-element snapshot support.
- Breaking: HORIZONS and SBDB now require their matching feature (or `jpl`).
  Replace `Loader::horizons_client()` / `Loader::sbdb_client()` with the clients'
  `new()` constructors. Core types and existing catalog paths remain available.
- Include the final datasource cutover: SFEMv4 planet-map headers identify
  albedo conventions and wavelength bands; Moon and Mars are normalized at V.
- The datasource Hipparcos loader is available at
  `starfield::catalogs::hipparcos::catalog::HipparcosCatalog`; the pre-existing
  `starfield::catalogs::hipparcos::HipparcosCatalog` retains its original type.
- Keep `starfield-datastore` independently versioned. Existing embedded data
  remains available offline; no asset download is added by this migration.

## 0.16.4

- Bump indicatif 0.17 → 0.18, dropping the unmaintained `number_prefix` (RUSTSEC-2025-0119) from the dependency tree; with starfield-datastore 0.1.2 this was the last path by which it reached downstream crates (#195)
- `Time::utc_iso` rounds to the last printed place instead of truncating, carrying through seconds, minutes, hours and the date, as Skyfield does; `2007-10-03T05:30:00Z` no longer prints as `05:29:59` (#194, closes #178)

## 0.16.3

- `keplerlib` no longer panics on degenerate catalogue elements (#192). `eccentric_anomaly` handles `e = 0` exactly — MPCORB's assumed-circular rows, `e` printed as `0.0000000`, used to produce a NaN state from an ∞·0 starting guess — and `propagate` returns a non-finite state instead of hitting `f64::clamp` with a NaN bound for elements that describe no bound orbit (`e ≥ 1` with `a > 0`, `a ≤ 0`, NaN). New `KeplerOrbit::try_at` returns an error for such orbits and `KeplerOrbit::is_finite` screens them; `at` keeps its signature. Regression tests cover the 1994 TG row, the near-parabolic damocloids, and the degenerate cases

## 0.16.2

- Data files resolve through `starfield-datastore` (new default-on `datastore` feature, rollout step 5): `Loader::open`, `open_text_pck`, `open_binary_pck`, `ensure_file`, `download_hipparcos` and the Gaia shard downloader go `local disk → STARFIELD_MIRROR → upstream (only with STARFIELD_ALLOW_UPSTREAM=1)`, with every fetch validated by starfield's own kernel magic numbers before it is cached. A flat `~/.cache/starfield/<file>` (or `hip_main.dat`, or a Gaia shard) left by the old downloader is adopted on first use, so nothing is re-downloaded. See `docs/datastore.md`
- New `data::artifacts`: public `kernel_artifact`, `hipparcos_artifact`, `gaia_artifact`, `gaia_md5sums_artifact`, `url_artifact`, `artifact_for` and the hermetic `download_or_cache_with(&Datastore, name)`, so other consumers and the server's manifest tooling build byte-identical keys (`naif/spk/<file>`, `naif/spk/satellites/<file>`, `naif/pck/<file>`, `naif/fk/<file>`, `naif/lsk/<file>`, `cds/I/239/hip_main.dat`, `gaia/<dr>/gaia_source/<file>`, `url/<sha256 of any other http(s) URL>`); `GaiaRelease` names the three releases' shard and MD5 locations. A resolved Gaia shard is exposed under its archive name in `~/.cache/starfield/gaia/` so suffix-based gzip detection keeps working; a shard failing its archive MD5 is dropped from the cache
- Gaia shard discovery reads the release's MD5 manifest instead of scraping the archive's directory page, which is now a JavaScript shell; an empty list is an error rather than a fabricated enumeration
- New `StarfieldError::Datastore(#[from] DatastoreError)`; `resolve_url` knows `.tls` (`NAIF_LSK_URL`); `HIPPARCOS_URL`, `JPL_BSP_URL`, `NAIF_SATELLITES_URL` are public
- Without the feature the direct downloader is unchanged

## 0.16.1

- `PlanetaryConstants::frame_for` now falls back to the embedded IAU 2015 table when no text kernel read so far defines the body, as the plan promised: `PlanetaryConstants::new().frame_for(499)` works offline, and kernel values still win when loaded. Earth (`ItrsFrame`) and the Moon with a principal-axes kernel (`PckFrame`) are unchanged
- `planetarylib::TEXT_MAGIC_NUMBERS` is public so content validators can reuse starfield's own list of text-kernel magic numbers; `text_pck::{parse, load}` are documented as the unchecked tokenizer beneath the validated `read_text`
- `data::downloader::tests::test_known_endpoints_reachable` is `#[ignore]`d: it made live HEAD requests to JPL and NAIF from the default suite and failed CI three times in one day on upstream hiccups

## 0.16.0

Completes the resolved-solar-system-bodies milestone from `docs/solar-system-bodies-plan.md`: disk geometry, illumination and non-Earth observers for the focalplane renderer.

- Add `Position::{phase_angle, illuminated_fraction, solar_elongation}` using the true Sun from the kernel (`magnitudelib` keeps its SSB approximation); new `StarfieldError::MissingObserver` instead of NaN when the observer is unknown; `Time::shift_days` (#180)
- Add `Position::observe_star` — apparent places of catalog stars for any barycentric observer, via `starlib::Star::observe_from`; matches Skyfield to < 1e-5 mas for 20 Hipparcos stars from Earth and Mars; shared `STAR_TARGET_ID` sentinel (#181)
- Add `Position::{angular_semi_diameter, apparent_ellipse}` — the projected outline of a triaxial ellipsoid via the Schur complement of the rotated shape matrix — and `planetarylib::occult::{Occultation, occultation}` for None/Partial/Full occultation of one body by another, searchable with `searchlib::find_discrete` (#182)
- Add `KeplerOrbit::barycentric_at` for spacecraft on a Kepler orbit about any body and `Position::from_spk_target` for arbitrary (including negative) SPK ids, reporting `UnsupportedDataType` for SPK types the reader lacks; DE440 planetary GM constants; `jplephem::spk::SUPPORTED_DATA_TYPES` (#183)
- `PlanetaryConstants::frame_for(301)` returns the loaded lunar principal-axes `PckFrame` (DE440 preferred over DE421) when present, else `IauFrame` (#185)
- Add `Position::{sub_observer_point, sub_solar_point}` returning planetographic `SubPoint`s with per-body east/west `LongitudeSense`, validated against JPL Horizons to < 0.01°; `horizons::EphemerisRequest::cal_format` so observer tables parse (#187)
- Add `Position::{north_pole_position_angle, bright_limb_position_angle}` in the observer's sky frame, validated against Horizons NP.ang / SN.ang (#186)

## 0.15.0

Foundation for resolved solar-system bodies (planets, Moon, limbs), built for the focalplane renderer; the plan and PR sequence live in `docs/solar-system-bodies-plan.md` (#171).

- Add `planetarylib` — a port of Skyfield's `planetarylib`: text PCK/FK kernel parser (`text_pck`), `PlanetaryConstants`, `RotationalElements`, and an embedded, network-free IAU WGCCRE 2015 table (`iau2015.csv`) of radii, flattening and rotational elements exposed through `planetlib::Body::{radii_km, mean_radius_km, flattening, rotational_elements}`; `Loader::open_text_pck` (#175)
- Add `jplephem::pck` binary PCK reader (DAF type-2 Euler-angle segments) replacing the stub; `PlanetaryConstants::{read_binary, build_frame, build_frame_named}` with TK-frame offsets; `planetarylib::PckFrame` (e.g. `MOON_PA_DE421`, matches Skyfield to 1e-9 rad); SPK types 2 and 3 now share a `chebyshev::ChebyshevRecords` loader; `Loader::open_binary_pck` (#176)
- Add `planetarylib::IauFrame` — ICRF → IAU body-fixed rotation from the WGCCRE elements including nut-prec series and the Mars-system quadratic phase term; `RotationalElements::{evaluate, evaluate_with_rates}`; `PlanetaryConstants::frame_for`. Validated against SpiceyPy `pxform` to < 1e-5 arcsec for Mars, Earth, Moon and Jupiter (#177)
- Add `framelib::ItrsFrame` — ICRF → ITRS through the existing precession/nutation/GAST/polar-motion chain (`Time::c_matrix`) (#172)
- Add `magnitudelib::moon_magnitude` (Allen V phase curve); `planetary_magnitude` now supports body 301 (#174)
- `data::resolve_url` resolves `.tpc`/`.bpc` to the NAIF generic PCK directory and `.tf` to `fk/satellites/` (#173, #176)
- The Python comparison environment gains `spiceypy`, pinned in `.spiceypy-version` (#177)

## 0.14.0

Five new modules extracted from the OrbitalCommons/planet9 research workspace, where they were built and hardened against published Planet Nine results. Zero new dependencies across all five.

- Add `nbodylib` — symplectic N-body integration, starfield's first perturbed-propagation capability: Wisdom-Holman in democratic-heliocentric coordinates (Duncan, Levison & Lee 1998), hardened universal-variable Kepler drift, Bulirsch-Stoer with recursive step halving, Chambers (1999) hybrid encounter switching, and a composable `ExtraForce` hook (J2-averaged giant-planet quadrupole, galactic tide via `framelib::GALACTIC`, custom closures). Ephemeris-seeded initial conditions via `planetlib`; `examples/nbody_giants.rs` (#153)
- Add `secularlib` — secular & resonance dynamics: quadrupole/octupole Hamiltonians with documented validity regimes, numerical Gauss-ring double averaging for the non-hierarchical regime, convergence-controlled Hansen coefficients, and perturber-generic Chirikov overlap / critical perihelion / libration detection with Neptune convenience wrappers (#152)
- Add `statslib` — circular statistics (circular mean/std, mean resultant length, Rayleigh test with the Mardia & Jupp small-n correction, Kuiper test, seeded Monte Carlo joint-significance helpers) and time-series primitives (multi-origin MSD diffusion estimator, median absolute deviation) (#149)
- Add `surveylib` — survey detection/completeness simulation: footprints with exact solid angles, logistic magnitude efficiency, k-of-n linking via exact Poisson-binomial tails, typed apparent-position geometry (with a regression test against declination/ecliptic-latitude conflation), deterministic expected-completeness accumulation, and multi-survey OR combination (#154)
- Add `magnitudelib::small_body` — physical photometry for hypothetical/small bodies (Neptune-anchored mass-radius, H from radius+albedo, reflected-light apparent magnitude, IAU two-term H-G phase law), cross-validated against the Mallama-Hilton Neptune model in-crate (#150)
- Add `catalogs::synthetic::orbits` (seeded synthetic orbital populations with deterministic-N resampling) and `sbdb::snapshot` (offline element diffing with wrap-aware angle deltas, plus `diff_against_live`, documenting the frozen-snapshot/drift-allowlist pattern) (#151)

## 0.13.0

- Upgrade `ndarray` 0.16 → 0.17 (#148). Semver-incompatible for consumers of starfield's ndarray-typed catalog/`StarData` APIs, hence a minor bump.
- Drop the unused `numpy` Rust crate dependency so the whole package — including the `python-tests` feature — resolves to a single ndarray 0.17 (it was the last pin holding 0.16). Python-side numpy is unaffected; it is reached via the pybridge.

## 0.12.6

- Add serde derives for `SersicProfile` and `StarData` so downstream consumers can serialize catalog primitives directly (#143)
- Add `AGENTS.md` as a symlink to `CLAUDE.md` so agent instructions are available under both expected names (#144)

## 0.12.5

- Add `ProperMotion { pmra, pmdec }` struct (mas/yr, Gaia DR3 convention: `pmra` carries cos(dec), `pmdec` is a plain Dec rate). Lives in `framelib::inertial` next to `Equatorial`; re-exported through `coordinates` and the crate root so callers write `starfield::ProperMotion`. Includes `ZERO` const, `new()`, `magnitude()`, serde derives (#136)

## 0.12.4

- `Time` is now `Send + Sync`. The lazy UT1 / TDB / delta-T caches were `Cell<Option<f64>>` (interior-mutable, single-threaded only); they're now `OnceLock<f64>`, which provides identical set-once cache semantics while making `Time` safe to share across threads. Unblocks embedding `Time` in `Arc<…>`-shared row structs (catalog records, indexes). Adds a compile-time `Send + Sync` assertion and a concurrent `tdb()` / `ut1()` / `delta_t()` consistency test as regression guards (#138).

## 0.12.3

- Security cleanup. `cargo audit` count drops from **4 vulnerabilities + 5 warnings** to **0 vulnerabilities + 3 warnings**:
  - Update `rustls-webpki` 0.103.10 → 0.103.13 — clears three vulnerabilities (CRL panic, wildcard / URI name-constraint bypasses) via transitive bump in the `reqwest` chain.
  - Disable default features on `image`, enable only `png` + `jpeg` — prunes the `rav1e → {core2, rand 0.9, paste}` AVIF subtree that triggered three unmaintained-crate warnings. AVIF decode/encode is no longer available through `starfield`; consumers can enable it on their own `image` dep.
  - Bump `pyo3` 0.19 → 0.24 and `numpy` 0.19 → 0.24 — clears `RUSTSEC-2025-0020` (`PyString::from_object` buffer overflow), affected only the `python-tests` dev feature. Bridge migrated to `Bound<'py, T>` / `&CStr` / `bind()` shapes.

## 0.12.2

- `Timescale` now wraps `Arc<TimescaleInner>` internally so cloning a `Timescale` (or a `Time` that holds one) is a refcount bump rather than a deep copy of the delta-T / leap-second / polar-motion tables. Unblocks embedding `Time` as a field in row-like structures (catalog records, indexes). `set_polar_motion_table` uses `Arc::make_mut` for copy-on-write so pre-existing `Time`s see the pre-mutation table (#134).

## 0.12.1

- Add `SersicProfile::total_flux_per_ie` helper for the `I_e` ↔ `F_total` conversion, with a built-in Lanczos g=7 `Γ` approximation (#128)
- Document the `I_e`-from-total-flux derivation on `SersicProfile::surface_brightness_at`, calling out the easy-to-drop `exp(b_n)` factor (Graham & Driver 2005, Eq. 4–6) (#126)
- Fix the position-angle convention translation in `SersicProfile::surface_brightness_at`'s docstring: `theta_AstroPy = 90° − position_angle_deg`, not `+ 90°`. The implementation was correct; only the docstring was wrong. Adds a regression test (#124)
- Wire AstroPy into the `python-tests` bridge alongside Skyfield; cross-checks `surface_brightness_at` against `astropy.modeling.Sersic2D` live (#123)

## 0.12.0

- Add `photometry` Cargo feature, off by default (#115, #116, #117)
- Add `Photometry` trait + `Band` enum for per-band fluxes / extinction / k-correction (#115)
- Add `RadialProfile` trait for measured azimuthally-averaged surface brightness (#116)
- Add `IsophoteSeries` trait + `IsophoteSample` for radius-resolved axis ratio + position angle (#117)
- Add `SersicProfile::b_n` and `SersicProfile::surface_brightness_at` Sérsic evaluator, cross-validated against `astropy.modeling.Sersic2D` (#122)

## 0.11.1

- Add `MinimalCatalog::load_with_progress` progress-callback hook for large catalog loads (#110)

## 0.11.0

- Reintegrate jplephem as an internal module (removes external crate dependency)

## 0.10.0

## 0.9.1

- Rename BinaryCatalog to MinimalCatalog

## 0.9.0

- Exclude test data from crates.io package
- Add lunar and solar eclipse detection
