# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Build project: `cargo build`
- Run with release optimizations: `cargo build --release`
- Run example: `cargo run --example hipparcos`
- Run tests: `cargo test`
- Run Python comparison tests: `cargo test --features python-tests`
- Run Skyfield comparison example: `cargo run --example skyfield_comparison --features python-tests`
- Run single test: `cargo test test_synthetic_hipparcos`
- Run benchmarks: `cargo bench`

## Lint/Format
- Format code: `cargo fmt`
- Run clippy lints: `cargo clippy`

## Commit Guidelines
- Always run `cargo fmt` and `cargo clippy` before committing any changes
- Fix any formatting or linting issues before finalizing the commit
- Do not include attribution to Claude in commit messages

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
- Always run `cargo fmt` first, then clean up any `cargo clippy` errors introduced
- Create examples in the examples directory for new functionality
- Always document functions with public visibility
- Keep module documentation up to date with changes
- For Python interop, prefer pyo3 direct Python evaluation over executing standalone Python scripts

## Python Reference Testing Infrastructure

This project validates its astronomical calculations against the Python Skyfield library using an in-process Python bridge. The infrastructure lives in `src/pybridge/` and is gated behind the `python-tests` Cargo feature flag.

### Architecture

The bridge has three components:

1. **`src/pybridge/bridge.rs`** — `PyRustBridge` struct that embeds a Python interpreter via PyO3. Calls `pyo3::prepare_freethreaded_python()` and uses `Python::with_gil()` for all Python interaction. The key method `run_py_to_json(code)` executes arbitrary Python code and retrieves results as JSON.

2. **`src/pybridge/helpers.rs`** — Defines `PythonResult` enum with three variants: `Bytes`, `String`, and `Array` (with dtype/shape/data). Handles JSON deserialization and base64 decoding of binary data from Python.

3. **`src/pybridge/helper.py`** — Loaded into every Python execution. Provides a `ResultCollector` class (instantiated as global `rust` object) with methods `collect_bytes()`, `collect_string()`, and `collect_array()`. Serializes results to JSON with base64 encoding for binary data.

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
- Virtual environment "starfield" with `skyfield==1.53` (see `.skyfield-version`)
- `devops/setup_pyenv.sh` — installs pyenv, creates venv, installs dependencies, generates `.env.python`
- `devops/verify_pyenv.sh` — validates the Python environment is correctly configured
- `.env` and `.env.python` — set `PYO3_PYTHON`, `PYTHONPATH`, `LD_LIBRARY_PATH` for PyO3

### CI Integration

GitHub Actions (`.github/workflows/ci.yml`) runs two separate jobs:
1. **`test`** — standard `cargo fmt`, `cargo clippy`, `cargo test`
2. **`python-comparison`** — sets up Python 3.10 + Skyfield, then runs `cargo test --features python-tests`

### Reference Source

A clone of the Python Skyfield source lives at `python-skyfield/` for reference. The bridge calls the *installed* Skyfield package (via pip), not this local clone.

### Feature Flag

In `Cargo.toml`: `python-tests = ["pyo3", "numpy", "anyhow"]`. The `src/pybridge/` module is only compiled when this feature is enabled. Standard `cargo test` skips all Python comparison tests.

## Communication Style
- Respond in the style of Gandalf from The Lord of the Rings