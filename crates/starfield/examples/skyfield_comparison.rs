//! Example showing how to compare Rust calculations with Python Skyfield
//!
//! This example demonstrates the use of the `pybridge` module to verify
//! that our time calculations match the reference implementation in Skyfield.
//!
//! Run this example with:
//! ```bash
//! cargo run --example skyfield_comparison --features python-tests
//! ```

#[cfg(feature = "python-tests")]
fn main() -> anyhow::Result<()> {
    use starfield::pybridge::{PyRustBridge, PythonResult};
    use starfield::time::{Time, Timescale};
    use std::convert::TryFrom;

    println!("Creating PyRustBridge...");
    let bridge = PyRustBridge::new()?;

    // Get current Julian Date from Skyfield
    let python_code = r#"
from skyfield.api import load, JulianDate
from datetime import datetime

# Current time
now = datetime.now()
ts = load.timescale()
t = ts.utc(now.year, now.month, now.day, now.hour, now.minute, now.second)
rust(t.tt)  # TT Julian Date
"#;

    println!("Running Python code to get Julian Date...");
    let result_json = bridge.run_py_to_json(python_code)?;
    let result = PythonResult::try_from(result_json.as_str())?;

    let py_jd = match result {
        PythonResult::String(s) => s.parse::<f64>().unwrap(),
        _ => panic!("Expected String result"),
    };

    println!("Python Skyfield Julian Date (TT): {}", py_jd);

    // Calculate the same in Rust
    let ts = Timescale::default();
    let now = ts.now();
    // Use the tt() method to get the Julian date directly
    let rust_jd = now.tt();

    println!("Rust Starfield Julian Date (TT): {}", rust_jd);

    // Compare the two values
    let diff = (rust_jd - py_jd).abs();
    println!("Difference: {} seconds", diff * 86400.0);

    if diff < 0.001 / 86400.0 {
        // Less than 1 millisecond difference
        println!("✅ Calculations match within 1 millisecond!");
    } else {
        println!("❌ Calculations differ significantly");
    }

    Ok(())
}

#[cfg(not(feature = "python-tests"))]
fn main() {
    println!("This example requires the 'python-tests' feature.");
    println!("Run with: cargo run --example skyfield_comparison --features python-tests");
}
