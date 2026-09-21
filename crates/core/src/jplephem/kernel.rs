//! High-level SpiceKernel API for planetary position lookups
//!
//! Provides a Skyfield-style interface where you load a BSP file, look up
//! a body by name, and compute its position at a given time.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Point3, Vector3};

use super::errors::{JplephemError, Result};
use super::names::target_id;
use super::spk::{jd_to_seconds, SPK};

/// AU in kilometers (IAU 2012 exact definition)
pub const AU_KM: f64 = 149_597_870.700;
/// Seconds per day
pub const S_PER_DAY: f64 = 86400.0;

/// State of a body: position in AU and velocity in AU/day relative to SSB
#[derive(Debug, Clone)]
pub struct PlanetState {
    /// Position in AU relative to SSB
    pub position: Point3<f64>,
    /// Velocity in AU/day relative to SSB
    pub velocity: Vector3<f64>,
}

/// A loaded SPK kernel with named body access and segment chain resolution
pub struct SpiceKernel {
    spk: SPK,
    /// Precomputed chains: target_id -> list of (center, target) segment pairs
    chains: HashMap<i32, Vec<(i32, i32)>>,
}

impl SpiceKernel {
    /// Open a BSP/SPK file and precompute segment chains
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let spk = SPK::open(path)?;
        let chains = Self::build_chains(&spk);
        Ok(SpiceKernel { spk, chains })
    }

    /// Open several local BSP/SPK files as one kernel, in load order.
    ///
    /// Segment chains span the union of the files. For overlapping
    /// `(center, target)` pairs, the last covering segment in the last loaded
    /// file wins. Earlier segments remain available outside that coverage.
    /// This follows SPICE load order for competing segments of the same pair.
    /// Files must describe compatible center chains in the J2000 frame.
    /// An empty list or any unreadable/invalid file returns an error.
    pub fn open_many<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let (first, rest) = paths.split_first().ok_or_else(|| {
            JplephemError::Other("open_many requires at least one SPK file".into())
        })?;
        let mut kernel = Self::open(first)?;
        for path in rest {
            kernel.merge(Self::open(path)?);
        }
        Ok(kernel)
    }

    /// Merge another loaded kernel, giving its segments later-load priority.
    ///
    /// Rebuilds chains over both kernels. Segment buffers and memory mappings
    /// remain owned by the combined kernel; the source kernel is consumed.
    /// As with [`Self::open_many`], precedence is per `(center, target)` pair.
    /// Resolve vector functions again after merging to pick up new chains.
    pub fn merge(&mut self, other: Self) {
        self.spk.append(other.spk);
        self.chains = Self::build_chains(&self.spk);
    }

    /// Create a SpiceKernel from an in-memory byte buffer
    ///
    /// Parses the same binary SPK/BSP format as `open`, but from `&[u8]`.
    /// Useful with `include_bytes!()` for compile-time embedded ephemeris data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// static BSP_DATA: &[u8] = include_bytes!("de421.bsp");
    /// let mut kernel = SpiceKernel::from_bytes(BSP_DATA).unwrap();
    /// ```
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let spk = SPK::from_bytes(data)?;
        let chains = Self::build_chains(&spk);
        Ok(SpiceKernel { spk, chains })
    }

    /// Build BFS chains from SSB (0) to every reachable target body
    fn build_chains(spk: &SPK) -> HashMap<i32, Vec<(i32, i32)>> {
        let mut adj: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
        for seg in &spk.segments {
            adj.entry(seg.center)
                .or_default()
                .push((seg.center, seg.target));
        }

        let mut chains = HashMap::new();
        let mut queue = std::collections::VecDeque::new();
        let mut parent: HashMap<i32, (i32, i32)> = HashMap::new();

        queue.push_back(0i32);
        parent.insert(0, (0, 0)); // sentinel

        while let Some(node) = queue.pop_front() {
            if let Some(edges) = adj.get(&node) {
                for &(center, target) in edges {
                    if let std::collections::hash_map::Entry::Vacant(e) = parent.entry(target) {
                        e.insert((center, target));
                        queue.push_back(target);
                    }
                }
            }
        }

        for &target_id in parent.keys() {
            if target_id == 0 {
                continue;
            }
            let mut chain = Vec::new();
            let mut current = target_id;
            while current != 0 {
                if let Some(&(center, target)) = parent.get(&current) {
                    if center == 0 && target == 0 {
                        break;
                    }
                    chain.push((center, target));
                    current = center;
                } else {
                    break;
                }
            }
            chain.reverse();
            chains.insert(target_id, chain);
        }

        chains
    }

    /// Get a VectorFunction for a body by name or numeric ID string
    ///
    /// Supported names: "earth", "mars", "moon", "sun", "mercury", "venus",
    /// "jupiter", "saturn", "uranus", "neptune", "pluto", "earth barycenter", etc.
    pub fn get(&self, name: &str) -> Result<VectorFunction> {
        let id = self.resolve_name(name)?;

        let chain = self.chains.get(&id).ok_or_else(|| {
            JplephemError::Other(format!(
                "No path from SSB to body {id} ('{name}') in this kernel"
            ))
        })?;

        Ok(VectorFunction {
            target_id: id,
            target_name: name.to_string(),
            chain: chain.clone(),
        })
    }

    /// Resolve a name to a NAIF ID
    fn resolve_name(&self, name: &str) -> Result<i32> {
        // Try parsing as numeric ID first
        if let Ok(id) = name.parse::<i32>() {
            return Ok(id);
        }

        target_id(name).ok_or_else(|| JplephemError::Other(format!("Unknown body name: '{name}'")))
    }

    /// Compute position and velocity for a chain of segments at a given TDB seconds.
    ///
    /// Returns raw (km, km/s) values from the underlying SPK data.
    pub fn compute_chain_pub(
        &mut self,
        chain: &[(i32, i32)],
        tdb_seconds: f64,
    ) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let mut total_pos = Vector3::new(0.0, 0.0, 0.0);
        let mut total_vel = Vector3::new(0.0, 0.0, 0.0);

        for &(center, target) in chain {
            let seg = self.spk.segment_at_mut(center, target, tdb_seconds)?;
            let (pos, vel) = seg.compute_and_differentiate(tdb_seconds, 0.0)?;
            total_pos += pos;
            total_vel += vel;
        }

        Ok((total_pos, total_vel))
    }

    /// Compute a body's state at a given Julian Date (TDB)
    ///
    /// Returns PlanetState with position in AU and velocity in AU/day,
    /// relative to the Solar System Barycenter (SSB).
    pub fn compute_at_jd(&mut self, name: &str, jd_tdb: f64) -> Result<PlanetState> {
        let vf = self.get(name)?;
        let tdb_seconds = jd_to_seconds(jd_tdb);
        let (pos_km, vel_km_s) = self.compute_chain_pub(&vf.chain, tdb_seconds)?;

        Ok(PlanetState {
            position: Point3::new(pos_km.x / AU_KM, pos_km.y / AU_KM, pos_km.z / AU_KM),
            velocity: Vector3::new(
                vel_km_s.x * S_PER_DAY / AU_KM,
                vel_km_s.y * S_PER_DAY / AU_KM,
                vel_km_s.z * S_PER_DAY / AU_KM,
            ),
        })
    }

    /// Compute raw position and velocity in km and km/s at a given Julian Date (TDB)
    pub fn compute_km_jd(
        &mut self,
        name: &str,
        jd_tdb: f64,
    ) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let vf = self.get(name)?;
        let tdb_seconds = jd_to_seconds(jd_tdb);
        self.compute_chain_pub(&vf.chain, tdb_seconds)
    }

    /// Access the underlying SPK for direct segment access
    pub fn spk(&self) -> &SPK {
        &self.spk
    }

    /// Access the underlying SPK mutably for computation
    pub fn spk_mut(&mut self) -> &mut SPK {
        &mut self.spk
    }
}

/// Represents a resolved path from SSB to a target body
///
/// Holds the chain of (center, target) segment pairs that must be summed
/// to compute the body's position relative to the SSB.
#[derive(Debug, Clone)]
pub struct VectorFunction {
    /// NAIF ID of the target body
    pub target_id: i32,
    /// Human-readable name
    pub target_name: String,
    /// Chain of (center, target) pairs from SSB to this body
    pub chain: Vec<(i32, i32)>,
}

impl std::fmt::Display for VectorFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VectorFunction({} [{}], {} segments)",
            self.target_name,
            self.target_id,
            self.chain.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::positions::Position;
    use crate::time::Timescale;

    // One type-2 Chebyshev record per segment, generated entirely in memory.
    // Each tuple is (center, target, start, end, x at midpoint, x velocity).
    fn synthetic(segments: &[(i32, i32, f64, f64, f64, f64)]) -> Vec<u8> {
        let mut bytes = vec![0u8; 3072 + segments.len() * 96];
        bytes[..8].copy_from_slice(b"DAF/SPK ");
        for (offset, value) in [
            (8, 2u32),
            (12, 6),
            (76, 2),
            (80, 2),
            (84, (bytes.len() / 8 + 1) as u32),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[88..96].copy_from_slice(b"LTL-IEEE");
        bytes[1040..1048].copy_from_slice(&(segments.len() as f64).to_le_bytes());
        for (i, &(center, target, start, end, x, velocity)) in segments.iter().enumerate() {
            let summary = 1048 + 40 * i;
            bytes[summary..summary + 8].copy_from_slice(&start.to_le_bytes());
            bytes[summary + 8..summary + 16].copy_from_slice(&end.to_le_bytes());
            let address = 385 + 12 * i;
            for (j, value) in [target, center, 1, 2, address as i32, (address + 11) as i32]
                .iter()
                .enumerate()
            {
                bytes[summary + 16 + 4 * j..summary + 20 + 4 * j]
                    .copy_from_slice(&value.to_le_bytes());
            }
            let radius = (end - start) / 2.0;
            let values = [
                (start + end) / 2.0,
                radius,
                x,
                velocity * radius,
                0.0,
                0.0,
                0.0,
                0.0,
                start,
                end - start,
                8.0,
                1.0,
            ];
            for (j, value) in values.iter().enumerate() {
                let offset = 3072 + 96 * i + 8 * j;
                bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn merged_satellite_can_be_observed_from_another_planet() {
        let mut kernel = SpiceKernel::from_bytes(&synthetic(&[
            (0, 4, -1e6, 1e6, AU_KM, 0.0),
            (4, 499, -1e6, 1e6, 0.0, 0.0),
            (0, 5, -1e6, 1e6, 2.0 * AU_KM, 0.0),
        ]))
        .unwrap();
        let satellite =
            SpiceKernel::from_bytes(&synthetic(&[(5, 501, -1e6, 1e6, 1000.0, 1.0)])).unwrap();
        assert!(kernel.get("501").is_err());
        assert!(satellite.get("501").is_err());
        // The satellite only becomes SSB-reachable after the union is built.
        kernel.merge(satellite);
        // The newer segment covers reception but not emission: iteration must
        // fall back to the older file when the light left the satellite.
        kernel.merge(
            SpiceKernel::from_bytes(&synthetic(&[(5, 501, -1.0, 100.0, 1000.0, 0.0)])).unwrap(),
        );
        let t = Timescale::default().tdb_jd(2451545.0);
        let mars = Position::from_spk_target(&mut kernel, 499, &t).unwrap();
        let observed = mars.observe("501", &mut kernel, &t).unwrap();
        let c_km_s = crate::constants::C_AUDAY * AU_KM / S_PER_DAY;
        let expected_km = (AU_KM + 1000.0) / (1.0 + 1.0 / c_km_s);
        // observe() subtracts light time from a full JD (about 20 us resolution).
        assert!(
            (observed.position.x * AU_KM - expected_km).abs() < 5e-5,
            "actual={} expected={} epoch={}",
            observed.position.x * AU_KM,
            expected_km,
            t.tdb()
        );
        assert_eq!(observed.center, 499);
        assert_eq!(observed.target, 501);
        let apparent = observed.apparent(&mut kernel, &t).unwrap();
        assert!(apparent.position.norm().is_finite());
    }

    #[test]
    fn overlapping_segments_use_load_order_and_preserve_outer_coverage() {
        let earlier = synthetic(&[(0, 5, -100.0, 100.0, 10.0, 0.0)]);
        let later = synthetic(&[(0, 5, -10.0, 10.0, 20.0, 0.0), (0, 5, -5.0, 5.0, 30.0, 0.0)]);
        let mut kernel = SpiceKernel::from_bytes(&earlier).unwrap();
        kernel.merge(SpiceKernel::from_bytes(&later).unwrap());
        for (epoch, expected) in [
            (-100.0, 10.0),
            (-50.0, 10.0),
            (-10.0, 20.0),
            (-5.0, 30.0),
            (0.0, 30.0),
            (5.0, 30.0),
            (10.0, 20.0),
            (50.0, 10.0),
            (100.0, 10.0),
        ] {
            let (position, _) = kernel.compute_chain_pub(&[(0, 5)], epoch).unwrap();
            assert_eq!(position.x, expected);
        }
        assert!(matches!(
            kernel.compute_chain_pub(&[(0, 5)], 101.0),
            Err(JplephemError::OutOfRangeError { .. })
        ));
        let mut reversed = SpiceKernel::from_bytes(&later).unwrap();
        reversed.merge(SpiceKernel::from_bytes(&earlier).unwrap());
        assert_eq!(reversed.compute_km_jd("5", 2451545.0).unwrap().0.x, 10.0);
    }

    #[test]
    fn open_many_reads_all_files_and_rejects_empty_or_missing_input() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base.bsp");
        let satellite = dir.path().join("satellite.bsp");
        std::fs::write(&base, synthetic(&[(0, 5, -100.0, 100.0, 10.0, 0.0)])).unwrap();
        std::fs::write(&satellite, synthetic(&[(5, 501, -100.0, 100.0, 2.0, 0.0)])).unwrap();
        let mut kernel = SpiceKernel::open_many(&[&base, &satellite]).unwrap();
        assert_eq!(kernel.compute_km_jd("501", 2451545.0).unwrap().0.x, 12.0);
        let mut loaded = crate::Loader::new()
            .with_data_dir(dir.path())
            .open_many(&["base.bsp", "satellite.bsp"])
            .unwrap();
        assert_eq!(loaded.compute_km_jd("501", 2451545.0).unwrap().0.x, 12.0);
        assert!(SpiceKernel::open_many::<&Path>(&[]).is_err());
        assert!(SpiceKernel::open_many(&[base, dir.path().join("missing.bsp")]).is_err());
    }
    #[test]
    fn merged_unsupported_segments_keep_their_diagnostic() {
        let mut kernel =
            SpiceKernel::from_bytes(&synthetic(&[(0, 4, -100.0, 100.0, 0.0, 0.0)])).unwrap();
        let mut unsupported = synthetic(&[(4, -74, -100.0, 100.0, 0.0, 0.0)]);
        unsupported[1076..1080].copy_from_slice(&13i32.to_le_bytes());
        kernel.merge(SpiceKernel::from_bytes(&unsupported).unwrap());
        let time = Timescale::default().tdb_jd(2451545.0);
        assert!(matches!(
            Position::from_spk_target(&mut kernel, -74, &time),
            Err(JplephemError::UnsupportedDataType(13))
        ));
    }

    #[test]
    fn loader_text_pck_adds_satellite_constants_from_local_cache() {
        let dir = tempfile::tempdir().unwrap();
        // Synthetic constants exercise the loader/parser seam without a kernel download.
        std::fs::write(dir.path().join("pck00011.tpc"),
            "KPL/PCK\n\\begindata\nBODY401_RADII = ( 13 11 9 )\nBODY401_POLE_RA = ( 0 0 0 )\nBODY401_POLE_DEC = ( 90 0 0 )\nBODY401_PM = ( 0 1 0 )\n\\begintext\n").unwrap();
        let constants = crate::Loader::new()
            .with_data_dir(dir.path())
            .open_text_pck("pck00011.tpc")
            .unwrap();
        assert_eq!(constants.radii(401), Some([13.0, 11.0, 9.0]));
        assert!(constants.frame_for(401).is_ok());
    }
}
