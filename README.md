# Starfield

Astronomical data reduction toolkit with star catalogs, coordinate systems, and star finding algorithms inspired by Skyfield.

## Features

- Celestial coordinate transformations
- Star catalog management (Hipparcos, GAIA)
- Precession, nutation, and earth rotation calculations
- Time and date handling for astronomical applications
- Synthetic catalog generation for testing
- Python interoperability for comparing results with Skyfield (optional)

## Installation

```bash
cargo add starfield
```

## Example

```rust
use starfield::time::Time;
use starfield::catalogs::hipparcos::HipparcosCatalog;
use starfield::catalogs::StarCatalog;

fn main() {
    // Create a synthetic Hipparcos catalog for testing
    let catalog = HipparcosCatalog::create_synthetic();
    
    // Get current time
    let time = Time::now();
    
    // Find bright stars
    let bright_stars = catalog.brighter_than(3.0);
    
    println!("Found {} bright stars at {}", bright_stars.len(), time);
    
    // Print the brightest star information
    if let Some(brightest) = catalog.stars().min_by(|a, b| a.mag.partial_cmp(&b.mag).unwrap()) {
        println!(
            "Brightest star: HIP {} (magnitude {:.2})",
            brightest.hip, brightest.mag
        );
    }
}
```

## Command Line Tool

The package includes a simple command-line tool for analyzing star catalogs:

```bash
# Basic catalog statistics
cargo stats --catalog hipparcos --operation stats

# Filter a catalog by magnitude and save it
cargo stats --catalog hipparcos --operation filter --magnitude 6.0 --output bright_stars.bin
```

## Python Interoperability

Starfield provides optional Python interoperability for comparing results with the Python Skyfield library:

```bash
# Enable Python comparison tests
cargo test --features python-tests

# Run example comparing Rust calculations with Skyfield
cargo run --example skyfield_comparison --features python-tests
```

Example code using the Python bridge:

```rust
// This requires the python-tests feature to be enabled
use starfield::pybridge::{PyRustBridge, PythonResult};
use std::convert::TryFrom;

fn compare_with_skyfield() -> anyhow::Result<()> {
    // Create a bridge to Python
    let bridge = PyRustBridge::new()?;
    
    // Run Python code and get the result
    let python_code = r#"
    from skyfield.api import load
    ts = load.timescale()
    t = ts.utc(2023, 8, 15, 12, 0, 0)
    rust(t.tt)  # Return TT Julian Date
    "#;
    
    let result_json = bridge.run_py_to_json(python_code)?;
    let result = PythonResult::try_from(result_json.as_str())?;
    
    match result {
        PythonResult::String(s) => {
            println!("Skyfield result: {}", s);
            // Compare with Rust calculation
        },
        _ => println!("Unexpected result type"),
    }
    
    Ok(())
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.