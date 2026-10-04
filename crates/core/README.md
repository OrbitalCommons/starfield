# Starfield core

Shared astronomical calculations, coordinate systems, catalog traits, and
artifact loading. Most consumers should depend on the `starfield` facade.
Datasource implementations depend on this crate to share one set of Rust types.
This package is released in lockstep with the Starfield workspace.

## Browser WebAssembly

Use `starfield-core = { version = "0.18.2", default-features = false }` for
`wasm32-unknown-unknown`. Native downloads, filesystem caches, and the `datastore`
feature are not available in the browser. Supply host-fetched data to in-memory
parsers such as `jplephem::spk::SPK::from_bytes`; the JavaScript random backend is
selected automatically. CI checks both the core and the `starfield` facade on
this target without default features.
