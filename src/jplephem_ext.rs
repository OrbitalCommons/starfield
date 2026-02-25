//! Extension methods for `SpiceKernel` that bridge starfield's `Time` type
//! to the standalone `starfield-jplephem` crate's raw Julian Date API.
//!
//! Import the trait to use `.compute_at()`, `.compute_km()`, and `.at()`:
//!
//! ```ignore
//! use starfield::jplephem_ext::SpiceKernelExt;
//! ```

use crate::jplephem::kernel::SpiceKernel;
use crate::jplephem::PlanetState;
use crate::positions::Position;
use crate::time::Time;
use nalgebra::Vector3;

/// Extension trait adding `&Time`-based convenience methods to `SpiceKernel`.
pub trait SpiceKernelExt {
    /// Compute a body's state at a given time.
    ///
    /// Returns `PlanetState` with position in AU and velocity in AU/day,
    /// relative to the Solar System Barycenter (SSB).
    fn compute_at(
        &mut self,
        name: &str,
        time: &Time,
    ) -> crate::jplephem::errors::Result<PlanetState>;

    /// Compute raw position and velocity in km and km/s.
    fn compute_km(
        &mut self,
        name: &str,
        time: &Time,
    ) -> crate::jplephem::errors::Result<(Vector3<f64>, Vector3<f64>)>;

    /// Compute a body's barycentric position at a given time.
    ///
    /// Returns a `Position` with `PositionKind::Barycentric`,
    /// position in AU and velocity in AU/day, relative to the SSB.
    fn at(&mut self, name: &str, time: &Time) -> crate::jplephem::errors::Result<Position>;
}

impl SpiceKernelExt for SpiceKernel {
    fn compute_at(
        &mut self,
        name: &str,
        time: &Time,
    ) -> crate::jplephem::errors::Result<PlanetState> {
        self.compute_at_jd(name, time.tdb())
    }

    fn compute_km(
        &mut self,
        name: &str,
        time: &Time,
    ) -> crate::jplephem::errors::Result<(Vector3<f64>, Vector3<f64>)> {
        self.compute_km_jd(name, time.tdb())
    }

    fn at(&mut self, name: &str, time: &Time) -> crate::jplephem::errors::Result<Position> {
        let state = self.compute_at_jd(name, time.tdb())?;
        let vf = self.get(name)?;
        Ok(Position::barycentric(
            Vector3::new(state.position.x, state.position.y, state.position.z),
            state.velocity,
            vf.target_id,
        ))
    }
}
