# Changelog

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
