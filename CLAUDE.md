# starfield-datasources

## Versioning

- When adding a new datasource crate, increment the minor version of the `starfield-datasources` facade crate by 1
- Each new datasource crate starts at version 0.1.0

## Project Structure

- Workspace with datasource crates under `crates/`
- Facade crate at `crates/starfield-datasources/` re-exports all datasource crates behind feature flags
- Default branch is `meawoppl/initial-workspace`
- Local `main` and `meawoppl/initial-workspace` refs go stale — always `git fetch` and branch off
  `origin/meawoppl/initial-workspace`, not the local ref, or you will rebuild work that already merged
- `Cargo.lock` is not tracked

## Patterns

- Clients use `reqwest::blocking::Client` with timeouts (30-60s)
- Error handling uses `starfield::{Result, StarfieldError}`
- All dependencies come from workspace
- Integration tests requiring network are marked `#[ignore]`, always with a reason
- **Two kinds of network test, and they must not be confused.** *Live API queries*
  (HORIZONS, SBDB, MPC, alert brokers) exercise a service's behaviour; they are
  query-shaped and are never cached or mirrored. *Upstream-rot canaries* fetch
  archive artifacts — pinned USGS slugs, PDS tables, MAST products, broker
  endpoint URLs — and exist to detect that an upstream URL has died or relocated.
  Those carry the ignore reason `"live upstream; bypasses the mirror..."` and
  **must keep pointing at the real archive** once `starfield-datastore` lands.
  Re-pointing one at the mirror silently defeats it: the mirror would go on
  serving a copy of a product whose upstream URL died years ago. See
  OrbitalCommons/starfield#189
- **Direct-download exceptions.** Once `starfield-datastore` lands, every
  downloader routes through it (step 5 of its rollout). Two categories stay
  direct, and both must say so at the call site rather than silently bypassing:
  - `starfield-nsa`: the NYU host serving `nsa_v0_1_2.fits` has an incomplete
    TLS chain (missing intermediate), so `download_nsa` builds a one-off client
    with `danger_accept_invalid_certs(true)`, scoped to that crate. The datastore
    deliberately ships no per-source danger flag, because an invalid-cert fetch
    that is invisible in a pinned manifest is worse than an explicit exception.
  - The upstream-rot canaries below.
- A canary that cannot reach upstream must **fail**, never skip. `#[ignore]`
  already gates these off by default; any second gate (an env var, a missing
  credential) that turns into a silent pass converts the canary into a
  permanent false green, which is worse than not having it
- Real data excerpts should be used for parser unit tests
