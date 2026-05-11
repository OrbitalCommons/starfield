# Changelog

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
