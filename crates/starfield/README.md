# Starfield

Astronomical calculations, catalogs, JPL clients, and planetary surface data.

```toml
starfield = { version = "0.17", features = ["catalogs", "jpl"] }
```

Core APIs retain their paths (`starfield::planetlib`, `starfield::time`, etc.).
Enable `catalogs`, `jpl`, or `surfaces` for groups, or individual sources such as
`hipparcos`, `gaia`, `horizons`, or `solar-spectrum`. `gaia-all` enables all Gaia
releases; `radial-profiles` opts into NSA's larger per-entry arrays.

See the [migration guide](https://github.com/OrbitalCommons/starfield/blob/main/docs/workspace-migration.md).
