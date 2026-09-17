# One Starfield workspace

The 0.17 family joins Starfield and the active `meawoppl/initial-workspace`
branch of starfield-datasources. The original datasource history is preserved verbatim under the
`datasources-history/12f24d1` tag in this repository. The migration PR has a single
parent so it follows Starfield's squash-only merge policy; use
`git log datasources-history/12f24d1` to inspect the original source history. `starfield-datastore` remains an independently versioned dependency.

## Consumer API

```toml
starfield = { version = "0.17", features = ["catalogs", "jpl"] }
```

The facade re-exports core types rather than wrapping them. Existing
`starfield::planetlib`, `time`, `positions`, `catalogs::StarCatalog`, and other
calculation paths retain their identity within the new release.

| Feature | Exports |
| --- | --- |
| `catalogs` | Gaia, extended Gaia, bright galaxies, Hipparcos, MAST, NSA under `starfield::catalogs` |
| `jpl` | HORIZONS, SBDB, MPC, Rubin under `starfield::jpl` |
| `surfaces` | Planet maps, planet spectra, reflectance library, solar spectrum under `starfield::surfaces` |
| `gaia-all` | Gaia DR1, DR2, DR3 |
| `radial-profiles` | NSA's optional radial profile arrays |
| `photometry` | Core photometry traits |
| `python-tests` | Embedded Python reference tests; requires Python |

Every datasource also has an individual feature with its hyphenated name.
`all-data` enables all sources and data options without enabling Python.
Default features enable datastore support, not every datasource. Datasource
artifact clients themselves require datastore support.

HORIZONS and SBDB remain available at `starfield::horizons` / `starfield::sbdb`
and `starfield::data::{horizons,sbdb}` when their features are enabled. Replace
`loader.horizons_client()` and `loader.sbdb_client()` with
`HorizonsClient::new()` and `SbdbClient::new()`. This removes a dependency cycle.

The existing core Hipparcos type remains at
`starfield::catalogs::hipparcos::HipparcosCatalog`. The datasource implementation
is at `starfield::catalogs::hipparcos::catalog::HipparcosCatalog`, alongside
`hipparcos::downloader` and `hipparcos::download_hipparcos`. Both implement the
same core traits. Existing `catalogs::GaiaCatalog` remains available; the new
release-specific Gaia loaders are under `catalogs::gaia`.

`starfield-datasources` is retained as a compatibility facade, now on the same
version as every other workspace crate. New consumers should use `starfield`.
The implementation crates remain separate in this first migration. They can
later fold into catalogs/JPL/surfaces groups behind these facade paths.

## Development and publication

- `cargo test --workspace --features starfield/all-data` runs core and datasource tests.
- `cargo test -p starfield-datasources --all-features` includes pull-through tests.
- `cargo test -p starfield-core --features python-tests -- --test-threads=1`
  runs reference comparisons without making Python a normal consumer dependency.
- `python3 devops/check_workspace.py` verifies the single version and exact local edges.
- `cargo publish --dry-run --workspace` verifies publication in dependency order.
- `python3 devops/check_workspace.py --packages` enforces a 9 MB archive budget
  (9.5 MB for the existing offline planet-map tiers), below the registry's 10 MB cap.

The existing Publish workflow publishes missing packages in dependency order
and tags the release only after publication succeeds. Retry that workflow on
partial failure; do not publish manually. A retry also repairs missing GitHub
release metadata after all crates have uploaded. Version bumps and changelog
entries remain explicit reviewable PR changes.

Exact pins align one release family; they cannot stop a consumer from also
resolving an older incompatible Starfield. Remove old direct dependencies and
git overrides together, then inspect `cargo tree -d` for duplicate core types.
The independently versioned datastore is intentionally outside this rule.

## Repository transition

The imported head is `12f24d109b936e81476bbe0fad127661d5546eea`. It includes
the Moon tier, NSA exception documentation, and indicatif update. The remaining
open datasource PRs at import are #54 (extended bright-galaxy cones) and #59
(agent-instruction rename). Port #54 to this workspace before retiring its
branch; #59 is superseded by the unified root AGENTS.md.

The final cutover is waiting for datasource PR #82 (SFEMv4 albedo conventions
and wavelength-band metadata). Import its final merged SHA, preserve that
history under a corresponding tag, and revalidate the tier binaries before
marking the migration ready.

After this PR is validated and merged, verify the automatic 0.17 publication,
then migrate focalplane, cfl-foundations/shared, zodiacal, and planet9 to the
facade. Transfer still-open datasource issues and add a redirect README before
archiving the old repository. Do not archive while the release or PR migration
is outstanding.

Embedded assets remain embedded in this release, preserving offline behavior.
Moving large tiers to digest-pinned datastore artifacts is a separate behavior
change that needs explicit prefetch/offline semantics and small hermetic fixtures.
