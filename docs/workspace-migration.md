# One Starfield workspace

The 0.17 family joins Starfield and the active `meawoppl/initial-workspace`
branch of starfield-datasources. The original datasource history is preserved verbatim under the
`datasources-history/0711def` tag in this repository. The migration PR has a single
parent so it follows Starfield's squash-only merge policy; use
`git log datasources-history/0711def` to inspect the original source history. `starfield-datastore` remains an independently versioned dependency.

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

The imported head is `0711deff6760b8be96f7bfab35329d76ae6027fe`. It includes
the Moon tier, NSA exception documentation, indicatif update, and SFEMv4 albedo
convention and wavelength-band metadata with V-band normalization. The remaining
open datasource PRs at import are #54 (extended bright-galaxy cones) and #59
(agent-instruction rename). Port #54 to this workspace before retiring its
branch; #59 is superseded by the unified root AGENTS.md.

The datasource agent confirmed this is the final cutover after PR #82, with
no further work in flight. All 11 imported data files match that tree byte-for-byte.

After this PR is validated and merged, verify the automatic 0.17 publication,
then migrate focalplane, cfl-foundations/shared, zodiacal, and planet9 to the
facade. All 13 open datasource issues have been transferred (see below). Add a redirect README before
archiving the old repository. Do not archive while the release or PR migration
is outstanding.

Embedded assets remain embedded in this release, preserving offline behavior.
Moving large tiers to digest-pinned datastore artifacts is a separate behavior
change that needs explicit prefetch/offline semantics and small hermetic fixtures.


## Transferred issues

GitHub transfers preserve discussions and redirect the old URLs. The old
repository has no remaining open issues as of this cutover.

| Previous issue | Starfield issue | Work |
| --- | --- | --- |
| [#80](https://github.com/OrbitalCommons/starfield-datasources/issues/80) | [#200](https://github.com/OrbitalCommons/starfield/issues/200) | Enable workspace-wide clippy for all targets |
| [#73](https://github.com/OrbitalCommons/starfield-datasources/issues/73) | [#201](https://github.com/OrbitalCommons/starfield/issues/201) | reflectance-library: FreshSnow and DesertSoil from the ECOSTRESS/ASTER library |
| [#71](https://github.com/OrbitalCommons/starfield-datasources/issues/71) | [#202](https://github.com/OrbitalCommons/starfield/issues/202) | reflectance-library: seasonal green/dry vegetation fraction from MOD13C2 |
| [#67](https://github.com/OrbitalCommons/starfield-datasources/issues/67) | [#203](https://github.com/OrbitalCommons/starfield/issues/203) | reflectance-library: Venus cloud-top reflectance endmember |
| [#66](https://github.com/OrbitalCommons/starfield-datasources/issues/66) | [#204](https://github.com/OrbitalCommons/starfield/issues/204) | reflectance-library: Mars bright and dark terrain endmembers from CRISM/OMEGA |
| [#65](https://github.com/OrbitalCommons/starfield-datasources/issues/65) | [#205](https://github.com/OrbitalCommons/starfield/issues/205) | reflectance-library: lunar mare and highland endmembers from RELAB/LSCC |
| [#60](https://github.com/OrbitalCommons/starfield-datasources/issues/60) | [#206](https://github.com/OrbitalCommons/starfield/issues/206) | AMPEL broker endpoints unreachable: ampel.zeuthen.desy.de no longer resolves in DNS |
| [#35](https://github.com/OrbitalCommons/starfield-datasources/issues/35) | [#207](https://github.com/OrbitalCommons/starfield/issues/207) | starfield-nsa: NsaEntry surface area gaps for high-quality galaxy rendering |
| [#6](https://github.com/OrbitalCommons/starfield-datasources/issues/6) | [#208](https://github.com/OrbitalCommons/starfield/issues/208) | Add CRTS (Catalina Real-Time Transient Survey) footprint and depth data |
| [#5](https://github.com/OrbitalCommons/starfield-datasources/issues/5) | [#209](https://github.com/OrbitalCommons/starfield/issues/209) | Add OSSOS survey characterization data loader |
| [#4](https://github.com/OrbitalCommons/starfield-datasources/issues/4) | [#210](https://github.com/OrbitalCommons/starfield/issues/210) | Add DES (Dark Energy Survey) data access client |
| [#3](https://github.com/OrbitalCommons/starfield-datasources/issues/3) | [#211](https://github.com/OrbitalCommons/starfield/issues/211) | Add Pan-STARRS DR2 catalog queries to starfield-mast |
| [#2](https://github.com/OrbitalCommons/starfield-datasources/issues/2) | [#212](https://github.com/OrbitalCommons/starfield/issues/212) | Add IRSA client (ZTF, WISE, 2MASS archive access) |

The datasource maintainer flagged #201 (formerly #73) as a human acquisition
step: request the ECOSTRESS Water and Soil archive categories from
https://speclib.jpl.nasa.gov/download, then feed the emailed archives to the
build script through a local path. The optional Lunar category also helps
#205. This scientific-data work remains open; importing the existing tiers
must not be mistaken for validating Earth's snow/soil endmembers.
