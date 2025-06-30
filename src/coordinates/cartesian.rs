//! # Cartesian Coordinate System Module
//!
//! This module provides a 3D Cartesian coordinate representation that serves as
//! the fundamental intermediate format for coordinate transformations in astronomical
//! calculations.
//!
//! ## Design Philosophy
//!
//! The `Cartesian3` struct stores coordinates in a standard right-handed
//! Cartesian coordinate system, providing exact representation of 3D positions
//! without the mathematical singularities and complexities that can arise with
//! spherical coordinate systems.
//!
//! ## Coordinate System Convention
//!
//! This implementation follows astronomical conventions:
//! - **X-axis**: Points toward the vernal equinox (RA = 0°, Dec = 0°)
//! - **Y-axis**: Points toward RA = 90°, Dec = 0°
//! - **Z-axis**: Points toward the north celestial pole (Dec = +90°)
//!
//! ## Internal Storage
//!
//! Coordinates are stored as three `f64` values representing distances or
//! direction cosines along each axis:
//! - Values maintain full IEEE 754 double precision
//! - No conversion artifacts during storage
//! - Direct mathematical operations preserve accuracy
//!
//! ## Use as Intermediate Representation
//!
//! Cartesian coordinates serve as the preferred intermediate format because:
//! - Linear transformations (rotations, translations) are straightforward
//! - No singularities at poles unlike spherical systems
//! - Vector operations (dot products, cross products) are direct
//! - Coordinate frame transformations use simple matrix multiplication
//!
//! ## Examples
//!
//! ```rust
//! use starfield::coordinates::cartesian::Cartesian3;
//!
//! // Unit vector pointing toward vernal equinox
//! let vernal_equinox = Cartesian3::new(1.0, 0.0, 0.0);
//!
//! // Unit vector pointing toward north celestial pole
//! let north_pole = Cartesian3::new(0.0, 0.0, 1.0);
//!
//! // Calculate dot product (cosine of angle between vectors)
//! let dot_product = vernal_equinox.dot(&north_pole);
//! assert_eq!(dot_product, 0.0); // Perpendicular vectors
//! ```

use nalgebra::Vector3;
use std::f64::consts::PI;

/// Three-dimensional Cartesian coordinate representation
///
/// Represents a point or direction in 3D space using standard Cartesian
/// coordinates. This struct serves as the fundamental building block for
/// astronomical coordinate transformations and calculations.
///
/// # Coordinate System
///
/// The coordinate system follows astronomical conventions:
/// - **X**: Toward vernal equinox (RA = 0°, Dec = 0°)
/// - **Y**: Toward RA = 90°, Dec = 0°  
/// - **Z**: Toward north celestial pole (Dec = +90°)
///
/// # Storage Strategy
///
/// - Each component stored as `f64` for maximum precision
/// - No internal coordinate transformations or normalizations
/// - Direct storage preserves exact input values
/// - Compatible with nalgebra Vector3 for linear algebra operations
///
/// # Unit Vectors vs Position Vectors
///
/// This type can represent both:
/// - **Unit vectors**: Direction in space (magnitude = 1.0)
/// - **Position vectors**: Actual positions with distance information
/// - **Velocity vectors**: Rate of change in position
///
/// The interpretation depends on context and the specific use case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cartesian3 {
    /// X-component (toward vernal equinox)
    pub x: f64,
    /// Y-component (toward RA = 90°)  
    pub y: f64,
    /// Z-component (toward north celestial pole)
    pub z: f64,
}

impl Cartesian3 {
    /// Creates a new Cartesian coordinate
    ///
    /// # Arguments
    ///
    /// * `x` - X-component (toward vernal equinox)
    /// * `y` - Y-component (toward RA = 90°)
    /// * `z` - Z-component (toward north celestial pole)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    ///
    /// // Create a unit vector pointing toward vernal equinox
    /// let coord = Cartesian3::new(1.0, 0.0, 0.0);
    /// assert_eq!(coord.x, 1.0);
    /// assert_eq!(coord.y, 0.0);
    /// assert_eq!(coord.z, 0.0);
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Cartesian3 { x, y, z }
    }

    /// Creates a Cartesian coordinate from spherical coordinates
    ///
    /// Converts from spherical coordinates (right ascension, declination, distance)
    /// to Cartesian coordinates using standard astronomical conventions.
    ///
    /// # Arguments
    ///
    /// * `ra` - Right ascension in radians
    /// * `dec` - Declination in radians  
    /// * `distance` - Distance from origin (default 1.0 for unit vectors)
    ///
    /// # Mathematical Conversion
    ///
    /// - `x = distance * cos(dec) * cos(ra)`
    /// - `y = distance * cos(dec) * sin(ra)`
    /// - `z = distance * sin(dec)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    /// use std::f64::consts::PI;
    ///
    /// // North celestial pole (Dec = 90°)
    /// let north_pole = Cartesian3::from_spherical(0.0, PI / 2.0, 1.0);
    /// assert!((north_pole.x).abs() < 1e-15);
    /// assert!((north_pole.y).abs() < 1e-15);
    /// assert!((north_pole.z - 1.0).abs() < 1e-15);
    /// ```
    pub fn from_spherical(ra: f64, dec: f64, distance: f64) -> Self {
        let cos_dec = dec.cos();
        Cartesian3 {
            x: distance * cos_dec * ra.cos(),
            y: distance * cos_dec * ra.sin(),
            z: distance * dec.sin(),
        }
    }

    /// Converts to spherical coordinates
    ///
    /// Returns (right_ascension, declination, distance) tuple in radians.
    /// Right ascension is normalized to [0, 2π), declination to [-π/2, π/2].
    ///
    /// # Returns
    ///
    /// `(ra, dec, distance)` where:
    /// - `ra`: Right ascension in radians [0, 2π)
    /// - `dec`: Declination in radians [-π/2, π/2]
    /// - `distance`: Distance from origin
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    /// use std::f64::consts::PI;
    ///
    /// let coord = Cartesian3::new(1.0, 0.0, 0.0);
    /// let (ra, dec, dist) = coord.to_spherical();
    /// assert!((ra - 0.0).abs() < 1e-15);
    /// assert!((dec - 0.0).abs() < 1e-15);
    /// assert!((dist - 1.0).abs() < 1e-15);
    /// ```
    pub fn to_spherical(&self) -> (f64, f64, f64) {
        let distance = self.magnitude();

        if distance == 0.0 {
            return (0.0, 0.0, 0.0);
        }

        let dec = (self.z / distance).asin();
        let ra = if self.x == 0.0 && self.y == 0.0 {
            0.0 // Arbitrary choice at poles
        } else {
            let mut ra = self.y.atan2(self.x);
            if ra < 0.0 {
                ra += 2.0 * PI;
            }
            ra
        };

        (ra, dec, distance)
    }

    /// Calculates the magnitude (length) of the coordinate vector
    ///
    /// Returns the Euclidean distance from the origin to this point.
    ///
    /// # Mathematical Formula
    ///
    /// `magnitude = sqrt(x² + y² + z²)`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    ///
    /// let coord = Cartesian3::new(3.0, 4.0, 0.0);
    /// assert_eq!(coord.magnitude(), 5.0);
    ///
    /// let unit_vector = Cartesian3::new(1.0, 0.0, 0.0);
    /// assert_eq!(unit_vector.magnitude(), 1.0);
    /// ```
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Returns a normalized (unit) vector in the same direction
    ///
    /// Creates a new coordinate with magnitude 1.0 pointing in the same
    /// direction as this coordinate. Returns None if the magnitude is zero.
    ///
    /// # Returns
    ///
    /// `Some(Cartesian3)` with magnitude 1.0, or `None` if input magnitude is zero
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    ///
    /// let coord = Cartesian3::new(3.0, 4.0, 0.0);
    /// let unit = coord.normalize().unwrap();
    /// assert!((unit.magnitude() - 1.0).abs() < 1e-15);
    /// assert_eq!(unit.x, 0.6);
    /// assert_eq!(unit.y, 0.8);
    /// assert_eq!(unit.z, 0.0);
    /// ```
    pub fn normalize(&self) -> Option<Cartesian3> {
        let mag = self.magnitude();
        if mag == 0.0 {
            None
        } else {
            Some(Cartesian3 {
                x: self.x / mag,
                y: self.y / mag,
                z: self.z / mag,
            })
        }
    }

    /// Calculates the dot product with another coordinate
    ///
    /// The dot product represents the cosine of the angle between two
    /// unit vectors, or the projection of one vector onto another.
    ///
    /// # Arguments
    ///
    /// * `other` - The other coordinate to compute dot product with
    ///
    /// # Mathematical Formula
    ///
    /// `dot = x₁*x₂ + y₁*y₂ + z₁*z₂`
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    ///
    /// let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
    /// let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
    /// assert_eq!(x_axis.dot(&y_axis), 0.0); // Perpendicular
    ///
    /// let same_dir = Cartesian3::new(2.0, 0.0, 0.0);
    /// assert_eq!(x_axis.dot(&same_dir), 2.0); // Parallel
    /// ```
    pub fn dot(&self, other: &Cartesian3) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculates the cross product with another coordinate
    ///
    /// The cross product produces a vector perpendicular to both input vectors,
    /// with magnitude equal to the area of the parallelogram they form.
    ///
    /// # Arguments
    ///
    /// * `other` - The other coordinate to compute cross product with
    ///
    /// # Mathematical Formula
    ///
    /// ```text
    /// cross = (y₁*z₂ - z₁*y₂, z₁*x₂ - x₁*z₂, x₁*y₂ - y₁*x₂)
    /// ```
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    ///
    /// let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
    /// let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
    /// let z_axis = x_axis.cross(&y_axis);
    ///
    /// assert!((z_axis.x - 0.0).abs() < 1e-15);
    /// assert!((z_axis.y - 0.0).abs() < 1e-15);
    /// assert!((z_axis.z - 1.0).abs() < 1e-15);
    /// ```
    pub fn cross(&self, other: &Cartesian3) -> Cartesian3 {
        Cartesian3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Calculates angular distance to another coordinate
    ///
    /// Returns the angle between two position vectors in radians.
    /// Both coordinates are treated as directions from the origin.
    ///
    /// # Arguments
    ///
    /// * `other` - The other coordinate to measure angle to
    ///
    /// # Returns
    ///
    /// Angle in radians [0, π]
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    /// use std::f64::consts::PI;
    ///
    /// let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
    /// let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
    /// let angle = x_axis.angular_distance(&y_axis);
    /// assert!((angle - PI / 2.0).abs() < 1e-15);
    /// ```
    pub fn angular_distance(&self, other: &Cartesian3) -> f64 {
        let dot_product = self.dot(other);
        let mag_product = self.magnitude() * other.magnitude();

        if mag_product == 0.0 {
            return 0.0;
        }

        let cos_angle = dot_product / mag_product;

        // Handle numerical precision issues
        if cos_angle >= 1.0 {
            0.0
        } else if cos_angle <= -1.0 {
            PI
        } else {
            cos_angle.acos()
        }
    }

    /// Converts to nalgebra Vector3 for linear algebra operations
    ///
    /// Creates a nalgebra Vector3<f64> with the same components,
    /// enabling integration with nalgebra's linear algebra ecosystem.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    /// use nalgebra::Vector3;
    ///
    /// let coord = Cartesian3::new(1.0, 2.0, 3.0);
    /// let vec: Vector3<f64> = coord.to_vector3();
    /// assert_eq!(vec.x, 1.0);
    /// assert_eq!(vec.y, 2.0);
    /// assert_eq!(vec.z, 3.0);
    /// ```
    pub fn to_vector3(&self) -> Vector3<f64> {
        Vector3::new(self.x, self.y, self.z)
    }

    /// Creates from nalgebra Vector3
    ///
    /// Converts a nalgebra Vector3<f64> to Cartesian3,
    /// enabling easy integration with nalgebra operations.
    ///
    /// # Arguments
    ///
    /// * `vec` - The nalgebra Vector3 to convert from
    ///
    /// # Examples
    ///
    /// ```rust
    /// use starfield::coordinates::cartesian::Cartesian3;
    /// use nalgebra::Vector3;
    ///
    /// let vec = Vector3::new(1.0, 2.0, 3.0);
    /// let coord = Cartesian3::from_vector3(vec);
    /// assert_eq!(coord.x, 1.0);
    /// assert_eq!(coord.y, 2.0);
    /// assert_eq!(coord.z, 3.0);
    /// ```
    pub fn from_vector3(vec: Vector3<f64>) -> Self {
        Cartesian3 {
            x: vec.x,
            y: vec.y,
            z: vec.z,
        }
    }
}

// Arithmetic operations for convenience
impl std::ops::Add for Cartesian3 {
    type Output = Cartesian3;

    fn add(self, other: Cartesian3) -> Cartesian3 {
        Cartesian3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for Cartesian3 {
    type Output = Cartesian3;

    fn sub(self, other: Cartesian3) -> Cartesian3 {
        Cartesian3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Mul<f64> for Cartesian3 {
    type Output = Cartesian3;

    fn mul(self, scalar: f64) -> Cartesian3 {
        Cartesian3 {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Div<f64> for Cartesian3 {
    type Output = Cartesian3;

    fn div(self, scalar: f64) -> Cartesian3 {
        Cartesian3 {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_cartesian_creation() {
        let coord = Cartesian3::new(1.0, 2.0, 3.0);
        assert_eq!(coord.x, 1.0);
        assert_eq!(coord.y, 2.0);
        assert_eq!(coord.z, 3.0);
    }

    #[test]
    fn test_magnitude_calculation() {
        let coord = Cartesian3::new(3.0, 4.0, 0.0);
        assert_eq!(coord.magnitude(), 5.0);

        let unit_vector = Cartesian3::new(1.0, 0.0, 0.0);
        assert_eq!(unit_vector.magnitude(), 1.0);

        let zero_vector = Cartesian3::new(0.0, 0.0, 0.0);
        assert_eq!(zero_vector.magnitude(), 0.0);
    }

    #[test]
    fn test_normalize() {
        let coord = Cartesian3::new(3.0, 4.0, 0.0);
        let normalized = coord.normalize().unwrap();

        assert!((normalized.magnitude() - 1.0).abs() < 1e-15);
        assert!((normalized.x - 0.6).abs() < 1e-15);
        assert!((normalized.y - 0.8).abs() < 1e-15);
        assert_eq!(normalized.z, 0.0);

        // Test zero vector
        let zero = Cartesian3::new(0.0, 0.0, 0.0);
        assert!(zero.normalize().is_none());
    }

    #[test]
    fn test_dot_product() {
        let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
        let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
        let z_axis = Cartesian3::new(0.0, 0.0, 1.0);

        // Orthogonal vectors have dot product of 0
        assert_eq!(x_axis.dot(&y_axis), 0.0);
        assert_eq!(x_axis.dot(&z_axis), 0.0);
        assert_eq!(y_axis.dot(&z_axis), 0.0);

        // Parallel vectors
        let same_direction = Cartesian3::new(2.0, 0.0, 0.0);
        assert_eq!(x_axis.dot(&same_direction), 2.0);

        // Opposite vectors
        let opposite = Cartesian3::new(-1.0, 0.0, 0.0);
        assert_eq!(x_axis.dot(&opposite), -1.0);
    }

    #[test]
    fn test_cross_product() {
        let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
        let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
        let z_axis = Cartesian3::new(0.0, 0.0, 1.0);

        // Right-hand rule: x × y = z
        let cross_xy = x_axis.cross(&y_axis);
        assert!((cross_xy.x - 0.0).abs() < 1e-15);
        assert!((cross_xy.y - 0.0).abs() < 1e-15);
        assert!((cross_xy.z - 1.0).abs() < 1e-15);

        // y × z = x
        let cross_yz = y_axis.cross(&z_axis);
        assert!((cross_yz.x - 1.0).abs() < 1e-15);
        assert!((cross_yz.y - 0.0).abs() < 1e-15);
        assert!((cross_yz.z - 0.0).abs() < 1e-15);

        // z × x = y
        let cross_zx = z_axis.cross(&x_axis);
        assert!((cross_zx.x - 0.0).abs() < 1e-15);
        assert!((cross_zx.y - 1.0).abs() < 1e-15);
        assert!((cross_zx.z - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_spherical_conversions() {
        // Test vernal equinox (RA=0, Dec=0)
        let vernal_equinox = Cartesian3::from_spherical(0.0, 0.0, 1.0);
        assert!((vernal_equinox.x - 1.0).abs() < 1e-15);
        assert!((vernal_equinox.y - 0.0).abs() < 1e-15);
        assert!((vernal_equinox.z - 0.0).abs() < 1e-15);

        let (ra, dec, dist) = vernal_equinox.to_spherical();
        assert!((ra - 0.0).abs() < 1e-15);
        assert!((dec - 0.0).abs() < 1e-15);
        assert!((dist - 1.0).abs() < 1e-15);

        // Test north celestial pole (Dec = π/2)
        let north_pole = Cartesian3::from_spherical(0.0, PI / 2.0, 1.0);
        assert!((north_pole.x - 0.0).abs() < 1e-15);
        assert!((north_pole.y - 0.0).abs() < 1e-15);
        assert!((north_pole.z - 1.0).abs() < 1e-15);

        let (_, dec, dist) = north_pole.to_spherical();
        assert!((dec - PI / 2.0).abs() < 1e-15);
        assert!((dist - 1.0).abs() < 1e-15);

        // Test RA = π/2 (90°)
        let ra_90 = Cartesian3::from_spherical(PI / 2.0, 0.0, 1.0);
        assert!((ra_90.x - 0.0).abs() < 1e-15);
        assert!((ra_90.y - 1.0).abs() < 1e-15);
        assert!((ra_90.z - 0.0).abs() < 1e-15);

        let (ra, dec, dist) = ra_90.to_spherical();
        assert!((ra - PI / 2.0).abs() < 1e-15);
        assert!((dec - 0.0).abs() < 1e-15);
        assert!((dist - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_angular_distance() {
        let x_axis = Cartesian3::new(1.0, 0.0, 0.0);
        let y_axis = Cartesian3::new(0.0, 1.0, 0.0);
        let z_axis = Cartesian3::new(0.0, 0.0, 1.0);

        // 90° angles between coordinate axes
        let angle_xy = x_axis.angular_distance(&y_axis);
        assert!((angle_xy - PI / 2.0).abs() < 1e-15);

        let angle_xz = x_axis.angular_distance(&z_axis);
        assert!((angle_xz - PI / 2.0).abs() < 1e-15);

        // 180° angle between opposite directions
        let opposite_x = Cartesian3::new(-1.0, 0.0, 0.0);
        let angle_opposite = x_axis.angular_distance(&opposite_x);
        assert!((angle_opposite - PI).abs() < 1e-15);

        // 0° angle between same directions
        let same_direction = Cartesian3::new(2.0, 0.0, 0.0);
        let angle_same = x_axis.angular_distance(&same_direction);
        assert!((angle_same - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_arithmetic_operations() {
        let a = Cartesian3::new(1.0, 2.0, 3.0);
        let b = Cartesian3::new(4.0, 5.0, 6.0);

        // Addition
        let sum = a + b;
        assert_eq!(sum.x, 5.0);
        assert_eq!(sum.y, 7.0);
        assert_eq!(sum.z, 9.0);

        // Subtraction
        let diff = b - a;
        assert_eq!(diff.x, 3.0);
        assert_eq!(diff.y, 3.0);
        assert_eq!(diff.z, 3.0);

        // Scalar multiplication
        let scaled = a * 2.0;
        assert_eq!(scaled.x, 2.0);
        assert_eq!(scaled.y, 4.0);
        assert_eq!(scaled.z, 6.0);

        // Scalar division
        let divided = a / 2.0;
        assert_eq!(divided.x, 0.5);
        assert_eq!(divided.y, 1.0);
        assert_eq!(divided.z, 1.5);
    }

    #[test]
    fn test_vector3_conversions() {
        let coord = Cartesian3::new(1.0, 2.0, 3.0);
        let vec = coord.to_vector3();

        assert_eq!(vec.x, 1.0);
        assert_eq!(vec.y, 2.0);
        assert_eq!(vec.z, 3.0);

        let coord_back = Cartesian3::from_vector3(vec);
        assert_eq!(coord, coord_back);
    }

    #[test]
    fn test_round_trip_spherical_conversion() {
        let test_cases = vec![
            (0.0, 0.0),            // Vernal equinox
            (PI / 2.0, 0.0),       // RA = 90°
            (PI, 0.0),             // RA = 180°
            (3.0 * PI / 2.0, 0.0), // RA = 270°
            (0.0, PI / 4.0),       // Dec = 45°
            (0.0, -PI / 4.0),      // Dec = -45°
            (PI / 3.0, PI / 6.0),  // RA = 60°, Dec = 30°
        ];

        for (ra, dec) in test_cases {
            let original_coord = Cartesian3::from_spherical(ra, dec, 1.0);
            let (converted_ra, converted_dec, converted_dist) = original_coord.to_spherical();
            let round_trip_coord =
                Cartesian3::from_spherical(converted_ra, converted_dec, converted_dist);

            assert!(
                (original_coord.x - round_trip_coord.x).abs() < 1e-14,
                "X mismatch for RA={}, Dec={}",
                ra,
                dec
            );
            assert!(
                (original_coord.y - round_trip_coord.y).abs() < 1e-14,
                "Y mismatch for RA={}, Dec={}",
                ra,
                dec
            );
            assert!(
                (original_coord.z - round_trip_coord.z).abs() < 1e-14,
                "Z mismatch for RA={}, Dec={}",
                ra,
                dec
            );
        }
    }

    #[test]
    fn test_special_cases() {
        // Test zero vector
        let zero = Cartesian3::new(0.0, 0.0, 0.0);
        let (ra, dec, dist) = zero.to_spherical();
        assert_eq!(ra, 0.0);
        assert_eq!(dec, 0.0);
        assert_eq!(dist, 0.0);

        // Test very small values
        let tiny = Cartesian3::new(1e-15, 1e-15, 1e-15);
        assert!(tiny.magnitude() > 0.0);
        let normalized = tiny.normalize().unwrap();
        assert!((normalized.magnitude() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_precision_preservation() {
        // Test that very precise values are preserved
        let precise_coord =
            Cartesian3::new(0.123456789012345, 0.987654321098765, 0.555666777888999);

        assert_eq!(precise_coord.x, 0.123456789012345);
        assert_eq!(precise_coord.y, 0.987654321098765);
        assert_eq!(precise_coord.z, 0.555666777888999);

        // Test arithmetic preserves precision
        let doubled = precise_coord * 2.0;
        assert_eq!(doubled.x, 0.123456789012345 * 2.0);
        assert_eq!(doubled.y, 0.987654321098765 * 2.0);
        assert_eq!(doubled.z, 0.555666777888999 * 2.0);
    }
}
