# Changelog

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
