//! Planetary ephemeris calculations module

use nalgebra::{Point3, Vector3};
use thiserror::Error;

use crate::jplephem::kernel::SpiceKernel;
use crate::time::Time;

/// Error type for planetary calculations
#[derive(Debug, Error)]
pub enum PlanetError {
    #[error("Planet not found: {0}")]
    NotFound(String),

    #[error("Data error: {0}")]
    DataError(String),

    #[error("Invalid time: {0}")]
    TimeError(String),

    #[error("Ephemeris error: {0}")]
    EphemerisError(#[from] crate::jplephem::JplephemError),
}

/// Major solar system bodies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Body {
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
    Pluto,
}

impl Body {
    /// Get the body's name
    pub fn name(&self) -> &'static str {
        match self {
            Body::Sun => "Sun",
            Body::Mercury => "Mercury",
            Body::Venus => "Venus",
            Body::Earth => "Earth",
            Body::Moon => "Moon",
            Body::Mars => "Mars",
            Body::Jupiter => "Jupiter",
            Body::Saturn => "Saturn",
            Body::Uranus => "Uranus",
            Body::Neptune => "Neptune",
            Body::Pluto => "Pluto",
        }
    }

    /// Get the NAIF SPICE ID for this body
    pub fn naif_id(&self) -> i32 {
        match self {
            Body::Sun => 10,
            Body::Mercury => 199,
            Body::Venus => 299,
            Body::Earth => 399,
            Body::Moon => 301,
            Body::Mars => 499,
            Body::Jupiter => 599,
            Body::Saturn => 699,
            Body::Uranus => 799,
            Body::Neptune => 899,
            Body::Pluto => 999,
        }
    }

    /// Get the SPICE name used for kernel lookups
    fn spice_name(&self) -> &'static str {
        match self {
            Body::Sun => "sun",
            Body::Mercury => "mercury",
            Body::Venus => "venus",
            Body::Earth => "earth",
            Body::Moon => "moon",
            Body::Mars => "mars",
            Body::Jupiter => "jupiter barycenter",
            Body::Saturn => "saturn barycenter",
            Body::Uranus => "uranus barycenter",
            Body::Neptune => "neptune barycenter",
            Body::Pluto => "pluto barycenter",
        }
    }
}

/// A planet's state (position + velocity) at a point in time
#[derive(Debug, Clone)]
pub struct PlanetState {
    /// Position in AU (relative to SSB)
    pub position: Point3<f64>,
    /// Velocity in AU/day
    pub velocity: Vector3<f64>,
}

/// Planetary ephemeris backed by a SPICE kernel
pub struct Ephemeris {
    kernel: SpiceKernel,
}

impl Ephemeris {
    /// Create an ephemeris from a loaded SpiceKernel
    pub fn from_kernel(kernel: SpiceKernel) -> Self {
        Self { kernel }
    }

    /// Get a body's state at a given time
    pub fn get_state(&mut self, body: Body, time: &Time) -> Result<PlanetState, PlanetError> {
        Ok(self.kernel.compute_at(body.spice_name(), time)?)
    }

    /// Access the underlying SpiceKernel
    pub fn kernel(&self) -> &SpiceKernel {
        &self.kernel
    }

    /// Access the underlying SpiceKernel mutably
    pub fn kernel_mut(&mut self) -> &mut SpiceKernel {
        &mut self.kernel
    }
}
