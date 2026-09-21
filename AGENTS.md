# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build project: `cargo build`
- Run with release optimizations: `cargo build --release`
- Run example: `cargo run --example hipparcos`
- Run tests: `cargo test --workspace --features starfield/all-data`
- Run Python comparison tests: `cargo test -p starfield-core --features python-tests`
- Run Skyfield comparison example: `cargo run --example skyfield_comparison --features python-tests`
- Run single test: `cargo test test_synthetic_hipparcos`
- Run benchmarks: `cargo bench`

## Lint/Format
- Format code: `cargo fmt --all`
- Run clippy lints: `cargo clippy --workspace --features starfield/all-data`

## Commit Guidelines
- Always run `cargo fmt --all` and `cargo clippy --workspace --features starfield/all-data` before committing any changes
- Fix any formatting or linting issues before finalizing the commit
- Do not include attribution to Claude in commit messages

## Releasing
- Releases are automated: merging a version change to `main` publishes it. `.github/workflows/publish.yml` publishes every crate whose `name@version` is not yet on crates.io, in dependency order, then tags `vX.Y.Z` and creates the GitHub release from that version's `CHANGELOG.md` section
- To release, bump `version` in `Cargo.toml` and add a `## X.Y.Z` section to `CHANGELOG.md` in the same PR. The `Release Check` workflow fails the PR if the changelog entry is missing or a crate doesn't package
- Versioning: patch for additive changes, minor for breaking ones; every crate in the family shares one version
- Never run `cargo publish` by hand. If a publish fails after merge, fix the cause and re-run the Publish workflow (`gh workflow run publish.yml`)
- Publishing uses the `CARGO_REGISTRY_TOKEN` repository secret: a crates.io token with the `publish-new` and `publish-update` scopes, restricted to `starfield` and `starfield-*`

## Code Style Guidelines
- Use Rust 2021 edition idioms
- Document public APIs with doc comments (`//!` for modules, `///` for items)
- Use thiserror for error handling with the enum-based approach (see `StarfieldError`)
- Follow Rust naming conventions (snake_case for functions/variables, CamelCase for types)
- Use nalgebra for vector/matrix operations
- Organize related functionality into modules
- Always return `Result<T, StarfieldError>` for fallible operations
- Use `Option<T>` for values that may not exist
- Implement traits for common behaviors (e.g., `StarCatalog`, `StarPosition`)
- Use proper type aliases to make complex types more readable
- Never special case tests in production code
- Follow the conventions of python-skyfield as this is intended to be a Rust port
- Use the existing tooling to compare outputs with the Python reference implementation whenever possible
- Always run `cargo fmt --all` first, then clean up any `cargo clippy --workspace --features starfield/all-data` errors introduced
- Create examples in the examples directory for new functionality
- Always document functions with public visibility
- Keep module documentation up to date with changes
- For Python interop, prefer pyo3 direct Python evaluation over executing standalone Python scripts

## Python Reference Testing Infrastructure

This project validates its astronomical calculations against the Python Skyfield library using an in-process Python bridge. The infrastructure lives in `crates/core/src/pybridge/` and is gated behind the `python-tests` Cargo feature flag.

### Architecture

The bridge has three components:

1. **`crates/core/src/pybridge/bridge.rs`** — `PyRustBridge` struct that embeds a Python interpreter via PyO3. Calls `pyo3::prepare_freethreaded_python()` and uses `Python::with_gil()` for all Python interaction. The key method `run_py_to_json(code)` executes arbitrary Python code and retrieves results as JSON.

2. **`crates/core/src/pybridge/helpers.rs`** — Defines `PythonResult` enum with three variants: `Bytes`, `String`, and `Array` (with dtype/shape/data). Handles JSON deserialization and base64 decoding of binary data from Python.

3. **`crates/core/src/pybridge/helper.py`** — Loaded into every Python execution. Provides a `ResultCollector` class (instantiated as global `rust` object) with methods `collect_bytes()`, `collect_string()`, and `collect_array()`. Serializes results to JSON with base64 encoding for binary data.

### Data Flow

```
Rust test code
  → PyRustBridge::run_py_to_json(python_code_string)
    → Python executes code, calls rust.collect_bytes/string/array()
    → ResultCollector serializes to JSON (base64 for binary)
  → JSON string returned to Rust
  → PythonResult::try_from(json) deserializes
  → Rust test compares values against native Rust calculations
```

### Writing a Python Comparison Test

```rust
#[cfg(feature = "python-tests")]
#[test]
fn test_my_calculation() {
    let bridge = PyRustBridge::new().unwrap();
    let result = bridge.run_py_to_json(r#"
        from skyfield.api import load
        ts = load.timescale()
        t = ts.utc(2024, 1, 1)
        rust.collect_string(str(t.tt))
    "#).unwrap();
    let parsed = PythonResult::try_from(result.as_str()).unwrap();
    // Compare parsed value against Rust implementation
}
```

For NumPy arrays, use `rust.collect_array(np_array)` — the bridge preserves dtype, shape, and raw byte data.

### Environment Setup

- Python 3.10.8 managed via pyenv (see `.python-version`)
- Virtual environment "starfield" with `skyfield` (see `.skyfield-version`), `astropy` (see `.astropy-version`) and `spiceypy` (see `.spiceypy-version`). The bridge is library-agnostic — any package installed in the venv is reachable from `bridge.run_py_to_json(...)`.
- `spiceypy` is the reference for the IAU body-fixed frames: nothing else in the environment evaluates the WGCCRE polynomial elements, so `planetarylib::IauFrame` is checked against `spiceypy.pxform` and `spiceypy.sxform`. Those tests need generic kernels off the network (`pck00011.tpc` and the leapsecond kernel `naif0012.tls`, downloaded into `~/.cache/starfield/`) and are `#[ignore]`d; the matrices they produce are checked in as golden constants so CI runs the same comparison offline. Run them with `cargo test -p starfield-core --features python-tests -- --ignored --test-threads=1` — the embedded interpreter is not safe to drive from several test threads at once.
- `devops/setup_pyenv.sh` — installs pyenv, creates venv, installs dependencies, generates `.env.python`
- `devops/verify_pyenv.sh` — validates the Python environment is correctly configured
- `.env` and `.env.python` — set `PYO3_PYTHON`, `PYTHONPATH`, `LD_LIBRARY_PATH` for PyO3

### CI Integration

GitHub Actions (`.github/workflows/ci.yml`) runs two separate jobs:
1. **`test`** — standard `cargo fmt --all`, `cargo clippy --workspace --features starfield/all-data`, `cargo test --workspace --features starfield/all-data`
2. **`python-comparison`** — sets up Python 3.10 with Skyfield, AstroPy and SpiceyPy, then runs `cargo test -p starfield-core --features python-tests`

### Reference Source

A clone of the Python Skyfield source lives at `python-skyfield/` for reference. The bridge calls the *installed* Skyfield package (via pip), not this local clone.

### Feature Flag

In `crates/core/Cargo.toml`: `python-tests = ["pyo3", "anyhow"]`. The `crates/core/src/pybridge/` module is only compiled when this feature is enabled. Standard `cargo test --workspace --features starfield/all-data` skips all Python comparison tests.

## Embedded Reference Data

Reference tables that must work without a network live next to the code that
reads them and are pulled in with `include_str!`. The pattern, established by
`crates/core/src/planetarylib/iau2015.csv`:

- Comment header (`#` lines) naming the upstream file, its retrieval URL, the
  published reference, and the meaning and units of every column.
- One header row of column names, then one row per entity. Fields are comma
  separated; multi-valued fields are space separated inside a single field, so
  the file needs no quoting and no CSV crate.
- Parsed once into a `std::sync::LazyLock<HashMap<..>>`, with the parse errors
  surfaced as `StarfieldError::DataError` and an `.expect()` at the `LazyLock`,
  since a malformed embedded table is a build-time mistake.
- Checked in alongside a verbatim excerpt of the upstream file (here
  `crates/core/src/planetarylib/pck00011_excerpt.tpc`) so a unit test can assert the table
  and the original agree. Do not check in whole kernels; excerpt them.

## Kernel Fixtures and Golden Vectors

`test_data/de421.bsp` is the only kernel checked into the repository. Do not
add more; a test that needs another kernel follows the pattern established by
`crates/core/src/planetarylib/pck_frame.rs`:

- Build the binary in memory instead of on disk when the test only exercises
  parsing or interpolation. `crates/core/src/jplephem/pck.rs` has a
  `#[cfg(test)] pub(crate) mod test_support` that assembles a DAF/PCK file
  around coefficients the test chooses, and other modules' tests import it.
- When the real kernel is unavoidable, fetch it through `Loader` into
  `~/.cache/starfield/` and mark the test `#[ignore]` with a reason naming the
  file. CI does not download kernels.
- Check the reference values in as `const` golden arrays next to the ignored
  test, with a doc comment giving the Python that produced them, so the
  agreement is still tested when the kernel is absent.

## Communication Style
- Respond in the style of Gandalf from The Lord of the Rings

## Workspace boundaries

- All packages inherit the version from `[workspace.package]`; internal edges
  use an exact version and local path. Run `python3 devops/check_workspace.py`.
- `starfield-core` owns shared traits and calculations. Datasources depend on
  core; core must not acquire a normal dependency on a datasource or facade.
- `starfield` is the consumer facade. Keep its public namespaces stable when
  consolidating the implementation crates. See `docs/workspace-migration.md`.
- Tests use the repository's single `test_data/` fixture directory through
  symlinks in core and facade; it is excluded from published packages.
- Live API queries and upstream-rot canaries remain ignored network tests.
  Canaries must contact the actual upstream and fail on unreachability.
- Artifact downloads use the datastore. Preserve documented direct-download
  exceptions (currently NSA's host-specific TLS workaround).
