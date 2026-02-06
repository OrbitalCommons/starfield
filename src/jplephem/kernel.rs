//! High-level SpiceKernel API for planetary position lookups
//!
//! Provides a Skyfield-style interface where you load a BSP file, look up
//! a body by name, and compute its position at a given time.

use std::collections::HashMap;
use std::path::Path;

use nalgebra::{Point3, Vector3};

use crate::jplephem::errors::{JplephemError, Result};
use crate::jplephem::names::target_id;
use crate::jplephem::spk::{jd_to_seconds, SPK};
use crate::planetlib::PlanetState;
use crate::time::Time;

/// AU in kilometers (IAU 2012 exact definition)
const AU_KM: f64 = 149_597_870.700;
/// Seconds per day
const S_PER_DAY: f64 = 86400.0;

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
            let seg = self.spk.get_segment_mut(center, target)?;
            let (pos, vel) = seg.compute_and_differentiate(tdb_seconds, 0.0)?;
            total_pos += pos;
            total_vel += vel;
        }

        Ok((total_pos, total_vel))
    }

    /// Compute a body's state at a given time
    ///
    /// Returns PlanetState with position in AU and velocity in AU/day,
    /// relative to the Solar System Barycenter (SSB).
    pub fn compute_at(&mut self, name: &str, time: &Time) -> Result<PlanetState> {
        let vf = self.get(name)?;
        let tdb_seconds = jd_to_seconds(time.tdb());
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

    /// Compute raw position and velocity in km and km/s
    pub fn compute_km(&mut self, name: &str, time: &Time) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let vf = self.get(name)?;
        let tdb_seconds = jd_to_seconds(time.tdb());
        self.compute_chain_pub(&vf.chain, tdb_seconds)
    }

    /// Compute a body's barycentric position at a given time.
    ///
    /// Returns a [`Position`](crate::positions::Position) with PositionKind::Barycentric,
    /// position in AU and velocity in AU/day, relative to the SSB.
    pub fn at(&mut self, name: &str, time: &Time) -> Result<crate::positions::Position> {
        let state = self.compute_at(name, time)?;
        let vf = self.get(name)?;
        Ok(crate::positions::Position::barycentric(
            Vector3::new(state.position.x, state.position.y, state.position.z),
            state.velocity,
            vf.target_id,
        ))
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
