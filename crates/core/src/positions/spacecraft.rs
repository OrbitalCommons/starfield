//! Barycentric observers from spacecraft SPK segments
//!
//! [`Position::from_spk_target`] resolves any SPK target id — including the
//! negative ids NAIF assigns to spacecraft — to a barycentric position and
//! velocity, so a spacecraft with its own kernel can be an observer just like
//! a planet from `SpiceKernelExt::at`:
//!
//! ```ignore
//! let mut kernel = SpiceKernel::open("mro_psp.bsp")?;
//! let mro = Position::from_spk_target(&mut kernel, -74, &t)?;
//! let earth = mro.observe("earth", &mut kernel, &t)?.apparent(&mut kernel, &t)?;
//! ```
//!
//! # Supported segment types
//!
//! This crate's SPK reader implements data types 2, 3 and 21 (Chebyshev
//! position, Chebyshev position and velocity, and extended modified difference
//! arrays). Reconstructed spacecraft trajectories are commonly type 1 or
//! type 13; a target whose only segments are of another type reports
//! [`JplephemError::UnsupportedDataType`] rather than looking like a missing
//! body. For a spacecraft whose kernel is unsupported — or one that has no
//! kernel at all, because the trajectory is a study rather than a flown
//! mission — model the orbit with
//! [`KeplerOrbit::barycentric_at`](crate::keplerlib::KeplerOrbit::barycentric_at)
//! instead.

use crate::constants::{AU_KM, DAY_S};
use crate::jplephem::errors::{JplephemError, Result};
use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem::spk::{jd_to_seconds, SUPPORTED_DATA_TYPES};
use crate::positions::Position;
use crate::time::Time;

impl Position {
    /// Barycentric state of an SPK target id at `t`.
    ///
    /// Resolves the chain of segments from the solar system barycenter to
    /// `target_id` exactly as `SpiceKernelExt::at` does for a named body, and
    /// returns a [`crate::positions::PositionKind::Barycentric`] position in
    /// AU and AU/day with `target` set to the requested id. Negative ids are
    /// spacecraft; positive ids are the usual bodies, so
    /// `from_spk_target(kernel, 399, t)` is `kernel.at("earth", t)`.
    ///
    /// # Errors
    /// * [`JplephemError::UnsupportedDataType`] if the kernel holds segments
    ///   for this target but of a data type the reader does not implement
    ///   (see the module documentation)
    /// * [`JplephemError::Other`] if no chain of segments reaches the target
    /// * an out-of-range error if the segments do not cover `t`
    pub fn from_spk_target(kernel: &mut SpiceKernel, target_id: i32, t: &Time) -> Result<Position> {
        let chain = match kernel.get(&target_id.to_string()) {
            Ok(vf) => vf.chain,
            // A target with only unsupported segments is dropped when the file
            // is parsed and so looks unreachable; say what actually happened.
            Err(err) => return Err(unsupported_data_type(kernel, target_id).unwrap_or(err)),
        };

        let (position_km, velocity_km_s) =
            kernel.compute_chain_pub(&chain, jd_to_seconds(t.tdb()))?;

        Ok(Position::barycentric(
            position_km / AU_KM,
            velocity_km_s * DAY_S / AU_KM,
            target_id,
        ))
    }
}

/// The unsupported data type of a segment for `target_id`, if that is why the
/// target could not be resolved.
fn unsupported_data_type(kernel: &SpiceKernel, target_id: i32) -> Option<JplephemError> {
    let daf = &kernel.spk().daf;
    let summaries = daf.summaries().ok()?;

    for (_, values) in summaries {
        if values.len() < (daf.nd + daf.ni) as usize {
            continue;
        }
        let target = values[2] as i32;
        let data_type = values[5] as i32;
        if target == target_id && !SUPPORTED_DATA_TYPES.contains(&data_type) {
            return Some(JplephemError::UnsupportedDataType(data_type));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jplephem_ext::SpiceKernelExt;
    use crate::time::Timescale;

    fn de421_kernel() -> SpiceKernel {
        SpiceKernel::open("test_data/de421.bsp").expect("Failed to open DE421")
    }

    /// A DAF/SPK file in memory holding one segment of the given data type.
    ///
    /// Only the header, summary and name records are written: a segment of an
    /// unsupported type is rejected before its data array is ever read.
    fn synthetic_spk(target: i32, center: i32, data_type: i32) -> Vec<u8> {
        const RECORD_SIZE: usize = 1024;
        let start_i = 3 * RECORD_SIZE / 8 + 1;
        let end_i = start_i + 7;

        let mut file = vec![0u8; 3 * RECORD_SIZE + 64];
        file[0..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&2u32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&6u32.to_le_bytes()); // NI
        file[16..76].copy_from_slice(format!("{:<60}", "SYNTHETIC").as_bytes());
        file[76..80].copy_from_slice(&2u32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2u32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&((end_i + 1) as u32).to_le_bytes()); // FREE

        let summary = RECORD_SIZE;
        let mut put = |at: usize, value: f64| {
            file[at..at + 8].copy_from_slice(&value.to_le_bytes());
        };
        put(summary, 0.0); // NEXT
        put(summary + 8, 0.0); // PREV
        put(summary + 16, 1.0); // NSUM
        put(summary + 24, -1.0e9); // segment start, TDB seconds
        put(summary + 32, 1.0e9); // segment end, TDB seconds

        // Six integers packed two to a double-word.
        let ints = [target, center, 1, data_type, start_i as i32, end_i as i32];
        for (i, value) in ints.iter().enumerate() {
            let at = summary + 40 + (i / 2) * 8 + (i % 2) * 4;
            file[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }

        file[2 * RECORD_SIZE..2 * RECORD_SIZE + 40]
            .copy_from_slice(format!("{:<40}", "SYNTHETIC SPK").as_bytes());

        file
    }

    #[test]
    fn test_from_spk_target_earth_matches_named_lookup() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let by_name = kernel.at("earth", &t).unwrap();
        let by_id = Position::from_spk_target(&mut kernel, 399, &t).unwrap();

        assert_eq!(by_id.target, 399);
        assert_eq!(by_id.center, 0);
        assert_eq!(by_id.kind, crate::positions::PositionKind::Barycentric);
        assert!((by_id.position - by_name.position).norm() < 1e-15);
        assert!((by_id.velocity - by_name.velocity).norm() < 1e-15);
    }

    #[test]
    fn test_from_spk_target_moon_matches_named_lookup() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let by_name = kernel.at("moon", &t).unwrap();
        let by_id = Position::from_spk_target(&mut kernel, 301, &t).unwrap();

        assert!((by_id.position - by_name.position).norm() < 1e-15);
        assert!((by_id.velocity - by_name.velocity).norm() < 1e-15);
    }

    #[test]
    fn test_from_spk_target_missing_body() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        // -74 is Mars Reconnaissance Orbiter; DE421 has no spacecraft.
        let err = Position::from_spk_target(&mut kernel, -74, &t).unwrap_err();
        assert!(
            matches!(err, JplephemError::Other(ref m) if m.contains("No path from SSB")),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_from_spk_target_unsupported_data_type() {
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        // Type 1 (modified difference arrays) is the usual reconstructed
        // spacecraft trajectory, and is not implemented.
        let bytes = synthetic_spk(-74, 4, 1);
        let mut kernel = SpiceKernel::from_bytes(&bytes).unwrap();
        let err = Position::from_spk_target(&mut kernel, -74, &t).unwrap_err();
        assert!(
            matches!(err, JplephemError::UnsupportedDataType(1)),
            "unexpected error: {err}"
        );

        // Type 13 (Hermite interpolation) likewise.
        let bytes = synthetic_spk(-74, 4, 13);
        let mut kernel = SpiceKernel::from_bytes(&bytes).unwrap();
        let err = Position::from_spk_target(&mut kernel, -74, &t).unwrap_err();
        assert!(
            matches!(err, JplephemError::UnsupportedDataType(13)),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_from_spk_target_observes_and_apparent() {
        let mut kernel = de421_kernel();
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0);

        let mars = Position::from_spk_target(&mut kernel, 499, &t).unwrap();
        let earth = mars
            .observe("earth", &mut kernel, &t)
            .unwrap()
            .apparent(&mut kernel, &t)
            .unwrap();

        assert_eq!(earth.target, 399);
        assert!(earth.observer_barycentric.is_some());
        assert!(earth.distance() > 0.3 && earth.distance() < 2.7);
    }
}
