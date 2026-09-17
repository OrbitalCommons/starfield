//! Celestial body definitions and calculations

use crate::coordinates::Equatorial;
use crate::time::Time;
use crate::Result;
use nalgebra::Point3;

/// A trait for objects that have a position in the sky
pub trait CelestialObject {
    /// Get the position of the object at a specific time
    fn position_at(&self, time: &Time) -> Result<Equatorial>;
}

/// A celestial body in the solar system
#[derive(Debug, Clone)]
pub struct CelestialBody {
    /// Name of the body
    pub name: String,
    /// Position in 3D space (AU)
    pub position: Point3<f64>,
}

impl CelestialBody {
    /// Create a new celestial body
    pub fn new(name: &str, position: Point3<f64>) -> Self {
        Self {
            name: name.to_string(),
            position,
        }
    }
}

impl CelestialObject for CelestialBody {
    fn position_at(&self, _time: &Time) -> Result<Equatorial> {
        // For now, just return a fixed position
        // In a real implementation, we would calculate the position based on the time
        Ok(Equatorial::new(0.0, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_celestial_body() {
        let body = CelestialBody::new("Test", Point3::new(1.0, 2.0, 3.0));
        assert_eq!(body.name, "Test");
        assert_eq!(body.position.x, 1.0);
        assert_eq!(body.position.y, 2.0);
        assert_eq!(body.position.z, 3.0);
    }
}
