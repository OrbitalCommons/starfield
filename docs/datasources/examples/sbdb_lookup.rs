//! Example: Look up a small body in the JPL Small-Body Database
//!
//! Queries detailed orbital and physical data for asteroid 433 Eros.
//!
//! Usage: cargo run --example sbdb_lookup

use starfield::sbdb::SbdbClient;

fn main() {
    let client = SbdbClient::new().expect("Failed to create SBDB client");

    // Look up asteroid Eros
    println!("Looking up asteroid Eros...\n");
    let response = client.lookup("Eros").expect("Failed to look up Eros");

    // Object identification
    let obj = &response.object;
    println!(
        "Object: {}",
        obj.fullname.as_deref().unwrap_or(&obj.designation)
    );
    println!("  Designation: {}", obj.designation);
    println!("  SPK-ID: {}", obj.spkid.as_deref().unwrap_or("N/A"));
    println!("  NEO: {}", obj.neo);
    println!("  PHA: {}", obj.pha);
    if let Some(ref class) = obj.orbit_class {
        println!("  Orbit class: {}", class.as_code());
    }

    // Orbital elements
    if let Some(ref orbit) = response.orbit {
        println!("\nOrbital Elements:");
        if let Some(e) = orbit.eccentricity {
            println!("  Eccentricity: {:.6}", e);
        }
        if let Some(a) = orbit.semi_major_axis {
            println!("  Semi-major axis: {:.6} AU", a);
        }
        if let Some(i) = orbit.inclination {
            println!("  Inclination: {:.4}°", i);
        }
        if let Some(p) = orbit.period {
            println!("  Period: {:.2} days ({:.2} years)", p, p / 365.25);
        }
        if let Some(moid) = orbit.moid_au {
            println!("  MOID (Earth): {:.6} AU", moid);
        }
    }

    // Physical parameters
    if let Some(ref phys) = response.phys_par {
        println!("\nPhysical Parameters:");
        if let Some(h) = phys.abs_magnitude_h {
            println!("  Absolute magnitude H: {:.2}", h);
        }
        if let Some(d) = phys.diameter_km {
            println!("  Diameter: {:.2} km", d);
        }
        if let Some(a) = phys.albedo {
            println!("  Albedo: {:.3}", a);
        }
        if let Some(r) = phys.rotation_period_h {
            println!("  Rotation period: {:.3} hours", r);
        }
    }

    // Also demonstrate a basic lookup
    println!("\n---\nBasic lookup for Ceres:");
    let ceres = client
        .lookup_basic("Ceres")
        .expect("Failed to look up Ceres");
    println!(
        "  {}  NEO={}  PHA={}",
        ceres
            .object
            .fullname
            .as_deref()
            .unwrap_or(&ceres.object.designation),
        ceres.object.neo,
        ceres.object.pha,
    );
}
