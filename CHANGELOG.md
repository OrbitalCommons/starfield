# Changelog

## 0.10.0

### Breaking: jplephem extracted to standalone crate

The internal `src/jplephem/` module has been replaced by a dependency on
`starfield-jplephem` from the `starfield-datasources` workspace.

**For most users, nothing changes** -- `starfield::jplephem::SpiceKernel` and all
other paths still work via re-export.

**If you used `starfield::planetlib::PlanetState`**: It is now re-exported from
`starfield_jplephem::PlanetState`. The struct layout is identical.

**If you called `.compute_at()`, `.compute_km()`, or `.at()` on SpiceKernel**:
These methods now come from an extension trait. Add:
```rust
use starfield::jplephem_ext::SpiceKernelExt;
```

### Modernizing to use datasource crates directly

For new projects, consider depending on the individual datasource crates
from the `starfield-datasources` workspace instead of going through `starfield`:

| Crate | Replaces |
|---|---|
| `starfield-jplephem` | `starfield::jplephem` |
| `starfield-horizons` | `starfield::horizons` |
| `starfield-sbdb` | `starfield::sbdb` |
| `starfield-gaia` | `starfield::catalogs::gaia` + `starfield::data::gaia_downloader` |
| `starfield-hipparcos` | `starfield::catalogs::hipparcos` + `starfield::data::downloader` |

Or use the facade: `starfield-datasources` (feature flags for each source).

Repo: https://github.com/OrbitalCommons/starfield-datasources

## 0.9.1

- Rename BinaryCatalog to MinimalCatalog

## 0.9.0

- Exclude test data from crates.io package
- Add lunar and solar eclipse detection
