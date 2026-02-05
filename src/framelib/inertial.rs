use super::frame_rotations;
use crate::coordinates::cartesian::Cartesian3;
use nalgebra::Matrix3;
use once_cell::sync::Lazy;

// Static transformation matrices

// These should be the transformation matrices FROM equatorial TO the other system
static EQ_TO_EC: Lazy<Matrix3<f64>> =
    Lazy::new(|| *frame_rotations::INERTIAL_FRAMES.get("ECLIPJ2000").unwrap());

static EC_TO_EQ: Lazy<Matrix3<f64>> = Lazy::new(|| {
    frame_rotations::INERTIAL_FRAMES
        .get("ECLIPJ2000")
        .unwrap()
        .try_inverse()
        .unwrap()
});

static EQ_TO_GAL: Lazy<Matrix3<f64>> =
    Lazy::new(|| *frame_rotations::INERTIAL_FRAMES.get("GALACTIC").unwrap());

static GAL_TO_EQ: Lazy<Matrix3<f64>> = Lazy::new(|| {
    frame_rotations::INERTIAL_FRAMES
        .get("GALACTIC")
        .unwrap()
        .try_inverse()
        .unwrap()
});

// Marker trait for inertial coordinate systems
pub trait InertialFrame: Sized {
    fn to_cartesian(&self) -> Cartesian3;
    fn from_cartesian(cart: Cartesian3) -> Self;

    fn angle_between(&self, other: &Self) -> f64 {
        let cart1 = self.to_cartesian();
        let cart2 = other.to_cartesian();

        let dot_product = cart1.dot(&cart2);
        let magnitude1 = cart1.magnitude();
        let magnitude2 = cart2.magnitude();

        let cos_angle = dot_product / (magnitude1 * magnitude2);

        // Handle numerical precision issues
        if cos_angle >= 1.0 {
            0.0
        } else if cos_angle <= -1.0 {
            std::f64::consts::PI
        } else {
            cos_angle.acos()
        }
    }
}

// Equatorial coordinates (RA/Dec)
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Equatorial {
    pub ra: f64,  // Right ascension in radians
    pub dec: f64, // Declination in radians
}

impl Equatorial {
    pub fn new(ra: f64, dec: f64) -> Self {
        let normalized_ra = ra.rem_euclid(2.0 * std::f64::consts::PI);
        Equatorial {
            ra: normalized_ra,
            dec,
        }
    }

    /// Create a new Equatorial coordinate with values in degrees
    pub fn from_degrees(ra_deg: f64, dec_deg: f64) -> Self {
        Self::new(
            ra_deg * std::f64::consts::PI / 180.0,
            dec_deg * std::f64::consts::PI / 180.0,
        )
    }

    /// Get right ascension in degrees
    pub fn ra_degrees(&self) -> f64 {
        self.ra * 180.0 / std::f64::consts::PI
    }

    /// Get declination in degrees
    pub fn dec_degrees(&self) -> f64 {
        self.dec * 180.0 / std::f64::consts::PI
    }

    /// Calculate angular distance to another Equatorial coordinate in radians
    pub fn angular_distance(&self, other: &Equatorial) -> f64 {
        self.angle_between(other)
    }
}

// Ecliptic coordinates
#[derive(Debug, Clone, Copy)]
pub struct Ecliptic {
    pub lon: f64, // Ecliptic longitude in radians
    pub lat: f64, // Ecliptic latitude in radians
}

// Galactic coordinates
#[derive(Debug, Clone, Copy)]
pub struct Galactic {
    pub lon: f64, // Galactic longitude in radians
    pub lat: f64, // Galactic latitude in radians
}

impl InertialFrame for Equatorial {
    fn to_cartesian(&self) -> Cartesian3 {
        let cos_dec = self.dec.cos();
        Cartesian3::new(
            cos_dec * self.ra.cos(),
            cos_dec * self.ra.sin(),
            self.dec.sin(),
        )
    }

    fn from_cartesian(cart: Cartesian3) -> Self {
        let r_xy = (cart.x * cart.x + cart.y * cart.y).sqrt();
        Equatorial::new(cart.y.atan2(cart.x), cart.z.atan2(r_xy))
    }
}

// Conversions FROM Equatorial
impl From<Equatorial> for Ecliptic {
    fn from(val: Equatorial) -> Self {
        // Convert equatorial cartesian to ecliptic cartesian
        let eq_cart = val.to_cartesian();
        let ec_cart = *EQ_TO_EC * eq_cart.to_vector3();
        Ecliptic::from_cartesian(Cartesian3::from_vector3(ec_cart))
    }
}

impl From<Equatorial> for Galactic {
    fn from(val: Equatorial) -> Self {
        // Convert equatorial cartesian to galactic cartesian
        let eq_cart = val.to_cartesian();
        let gal_cart = *EQ_TO_GAL * eq_cart.to_vector3();
        Galactic::from_cartesian(Cartesian3::from_vector3(gal_cart))
    }
}

impl InertialFrame for Ecliptic {
    fn to_cartesian(&self) -> Cartesian3 {
        let cos_lat = self.lat.cos();
        Cartesian3::new(
            cos_lat * self.lon.cos(),
            cos_lat * self.lon.sin(),
            self.lat.sin(),
        )
    }

    fn from_cartesian(cart: Cartesian3) -> Self {
        let r_xy = (cart.x * cart.x + cart.y * cart.y).sqrt();
        Ecliptic {
            lon: cart.y.atan2(cart.x),
            lat: cart.z.atan2(r_xy),
        }
    }
}

// Conversions FROM Ecliptic
impl From<Ecliptic> for Equatorial {
    fn from(val: Ecliptic) -> Self {
        // Convert ecliptic cartesian to equatorial cartesian
        let ec_cart = val.to_cartesian();
        let eq_cart = *EC_TO_EQ * ec_cart.to_vector3();
        Equatorial::from_cartesian(Cartesian3::from_vector3(eq_cart))
    }
}

impl From<Ecliptic> for Galactic {
    fn from(val: Ecliptic) -> Self {
        // First convert to equatorial, then to galactic
        let equatorial: Equatorial = val.into();
        equatorial.into()
    }
}

impl InertialFrame for Galactic {
    fn to_cartesian(&self) -> Cartesian3 {
        let cos_lat = self.lat.cos();
        Cartesian3::new(
            cos_lat * self.lon.cos(),
            cos_lat * self.lon.sin(),
            self.lat.sin(),
        )
    }

    fn from_cartesian(cart: Cartesian3) -> Self {
        let r_xy = (cart.x * cart.x + cart.y * cart.y).sqrt();
        Galactic {
            lon: cart.y.atan2(cart.x),
            lat: cart.z.atan2(r_xy),
        }
    }
}

// Conversions FROM Galactic
impl From<Galactic> for Equatorial {
    fn from(val: Galactic) -> Self {
        // Convert galactic cartesian to equatorial cartesian
        let gal_cart = val.to_cartesian();
        let eq_cart = *GAL_TO_EQ * gal_cart.to_vector3();
        Equatorial::from_cartesian(Cartesian3::from_vector3(eq_cart))
    }
}

impl From<Galactic> for Ecliptic {
    fn from(val: Galactic) -> Self {
        // First convert to equatorial, then to ecliptic
        let equatorial: Equatorial = val.into();
        equatorial.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::Vector3;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use std::f64::consts::PI;

    #[test]
    fn test_equatorial_to_cartesian_roundtrip() {
        let mut rng = StdRng::seed_from_u64(424242); // Use a fixed seed for reproducibility
        for i in 0..100 {
            let original_ra = rng.gen::<f64>() * 2.0 * PI;
            // Ensure dec is within -PI/2 to PI/2, avoiding poles where atan2 might be less stable
            // or where small cartesian errors can lead to large angle errors.
            // Let's restrict it slightly away from exact poles for more robust testing of the general case.
            let original_dec = (rng.gen::<f64>() * PI - PI / 2.0) * 0.99;

            let equatorial_original = Equatorial {
                ra: original_ra,
                dec: original_dec,
            };

            // Convert to Cartesian
            let cartesian = equatorial_original.to_cartesian();

            // Convert back to Equatorial
            let equatorial_roundtrip = Equatorial::from_cartesian(cartesian);

            println!(
                "Test {}: Original RA: {:.6} rad, Dec: {:.6} rad",
                i, original_ra, original_dec
            );
            println!(
                "           Cartesian X: {:.6}, Y: {:.6}, Z: {:.6}",
                cartesian.x, cartesian.y, cartesian.z
            );
            println!(
                "           Roundtrip RA: {:.6} rad, Dec: {:.6} rad",
                equatorial_roundtrip.ra, equatorial_roundtrip.dec
            );

            // We can compare sin and cos of the angles.
            assert_relative_eq!(
                original_ra.cos(),
                equatorial_roundtrip.ra.cos(),
                epsilon = 1e-9
            );
            assert_relative_eq!(
                original_ra.sin(),
                equatorial_roundtrip.ra.sin(),
                epsilon = 1e-9
            );
            assert_relative_eq!(original_dec, equatorial_roundtrip.dec, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_equatorial_to_cartesian_specific_cases() {
        // Case 1: RA = 0, Dec = 0 (Vernal Equinox direction)
        let eq1 = Equatorial { ra: 0.0, dec: 0.0 };
        let cart1 = eq1.to_cartesian();
        assert_relative_eq!(cart1.x, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.z, 0.0, epsilon = 1e-9);
        let eq1_rt = Equatorial::from_cartesian(cart1);
        assert_relative_eq!(eq1_rt.ra, 0.0, epsilon = 1e-9);
        assert_relative_eq!(eq1_rt.dec, 0.0, epsilon = 1e-9);

        // Case 2: North Celestial Pole
        let eq2 = Equatorial {
            ra: 0.0,
            dec: PI / 2.0,
        }; // RA can be anything here
        let cart2 = eq2.to_cartesian();
        assert_relative_eq!(cart2.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.z, 1.0, epsilon = 1e-9);
        let eq2_rt = Equatorial::from_cartesian(cart2);
        // RA is ill-defined at poles, atan2(0,0) is often 0.
        // We only care about dec here, or that the resulting cartesian is the same.
        assert_relative_eq!(eq2_rt.dec, PI / 2.0, epsilon = 1e-9);

        // Case 3: South Celestial Pole
        let eq3 = Equatorial {
            ra: 0.0,
            dec: -PI / 2.0,
        };
        let cart3 = eq3.to_cartesian();
        assert_relative_eq!(cart3.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.z, -1.0, epsilon = 1e-9);
        let eq3_rt = Equatorial::from_cartesian(cart3);
        assert_relative_eq!(eq3_rt.dec, -PI / 2.0, epsilon = 1e-9);

        // Case 4: RA = 90 deg (PI/2), Dec = 0
        let eq4 = Equatorial {
            ra: PI / 2.0,
            dec: 0.0,
        };
        let cart4 = eq4.to_cartesian();
        assert_relative_eq!(cart4.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.y, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.z, 0.0, epsilon = 1e-9);
        let eq4_rt = Equatorial::from_cartesian(cart4);
        assert_relative_eq!(eq4_rt.ra, PI / 2.0, epsilon = 1e-9);
        assert_relative_eq!(eq4_rt.dec, 0.0, epsilon = 1e-9);

        // Case 5: RA = 45 deg (PI/4), Dec = 45 deg (PI/4)
        let eq5 = Equatorial {
            ra: PI / 4.0,
            dec: PI / 4.0,
        };
        let cart5 = eq5.to_cartesian();
        // cos(PI/4) = sin(PI/4) = 1/sqrt(2)
        let val = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(cart5.x, val * val, epsilon = 1e-9); // cos(dec)*cos(ra) = (1/sqrt(2))*(1/sqrt(2)) = 1/2
        assert_relative_eq!(cart5.y, val * val, epsilon = 1e-9); // cos(dec)*sin(ra) = (1/sqrt(2))*(1/sqrt(2)) = 1/2
        assert_relative_eq!(cart5.z, val, epsilon = 1e-9); // sin(dec) = 1/sqrt(2)
        let eq5_rt = Equatorial::from_cartesian(cart5);
        assert_relative_eq!(eq5_rt.ra, PI / 4.0, epsilon = 1e-9);
        assert_relative_eq!(eq5_rt.dec, PI / 4.0, epsilon = 1e-9);
    }

    #[test]
    fn test_ecliptic_to_cartesian_roundtrip() {
        let mut rng = StdRng::seed_from_u64(424243); // Use a fixed seed for reproducibility
        for i in 0..100 {
            let original_lon = rng.gen::<f64>() * 2.0 * PI;
            // Ensure lat is within -PI/2 to PI/2, avoiding poles.
            let original_lat = (rng.gen::<f64>() * PI - PI / 2.0) * 0.99;

            let ecliptic_original = Ecliptic {
                lon: original_lon,
                lat: original_lat,
            };

            // Convert to Cartesian
            let cartesian = ecliptic_original.to_cartesian();

            // Convert back to Ecliptic
            let ecliptic_roundtrip = Ecliptic::from_cartesian(cartesian);

            println!(
                "Test {}: Original Lon: {:.6} rad, Lat: {:.6} rad",
                i, original_lon, original_lat
            );
            println!(
                "           Cartesian X: {:.6}, Y: {:.6}, Z: {:.6}",
                cartesian.x, cartesian.y, cartesian.z
            );
            println!(
                "           Roundtrip Lon: {:.6} rad, Lat: {:.6} rad",
                ecliptic_roundtrip.lon, ecliptic_roundtrip.lat
            );

            // Compare sin and cos of the angles for longitude, and direct value for latitude.
            assert_relative_eq!(
                original_lon.cos(),
                ecliptic_roundtrip.lon.cos(),
                epsilon = 1e-9
            );
            assert_relative_eq!(
                original_lon.sin(),
                ecliptic_roundtrip.lon.sin(),
                epsilon = 1e-9
            );
            assert_relative_eq!(original_lat, ecliptic_roundtrip.lat, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_ecliptic_to_cartesian_specific_cases() {
        // Case 1: Lon = 0, Lat = 0 (Direction of Vernal Equinox in Ecliptic plane)
        let ec1 = Ecliptic { lon: 0.0, lat: 0.0 };
        let cart1 = ec1.to_cartesian();
        assert_relative_eq!(cart1.x, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.z, 0.0, epsilon = 1e-9);
        let ec1_rt = Ecliptic::from_cartesian(cart1);
        assert_relative_eq!(ec1_rt.lon, 0.0, epsilon = 1e-9);
        assert_relative_eq!(ec1_rt.lat, 0.0, epsilon = 1e-9);

        // Case 2: North Ecliptic Pole
        let ec2 = Ecliptic {
            lon: 0.0,
            lat: PI / 2.0,
        }; // Lon can be anything here
        let cart2 = ec2.to_cartesian();
        assert_relative_eq!(cart2.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.z, 1.0, epsilon = 1e-9);
        let ec2_rt = Ecliptic::from_cartesian(cart2);
        // Lon is ill-defined at poles, atan2(0,0) is often 0.
        // We only care about lat here.
        assert_relative_eq!(ec2_rt.lat, PI / 2.0, epsilon = 1e-9);

        // Case 3: South Ecliptic Pole
        let ec3 = Ecliptic {
            lon: 0.0,
            lat: -PI / 2.0,
        };
        let cart3 = ec3.to_cartesian();
        assert_relative_eq!(cart3.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.z, -1.0, epsilon = 1e-9);
        let ec3_rt = Ecliptic::from_cartesian(cart3);
        assert_relative_eq!(ec3_rt.lat, -PI / 2.0, epsilon = 1e-9);

        // Case 4: Lon = 90 deg (PI/2), Lat = 0
        let ec4 = Ecliptic {
            lon: PI / 2.0,
            lat: 0.0,
        };
        let cart4 = ec4.to_cartesian();
        assert_relative_eq!(cart4.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.y, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.z, 0.0, epsilon = 1e-9);
        let ec4_rt = Ecliptic::from_cartesian(cart4);
        assert_relative_eq!(ec4_rt.lon, PI / 2.0, epsilon = 1e-9);
        assert_relative_eq!(ec4_rt.lat, 0.0, epsilon = 1e-9);

        // Case 5: Lon = 45 deg (PI/4), Lat = 45 deg (PI/4)
        let ec5 = Ecliptic {
            lon: PI / 4.0,
            lat: PI / 4.0,
        };
        let cart5 = ec5.to_cartesian();
        // cos(PI/4) = sin(PI/4) = 1/sqrt(2)
        let val = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(cart5.x, val * val, epsilon = 1e-9); // cos(lat)*cos(lon)
        assert_relative_eq!(cart5.y, val * val, epsilon = 1e-9); // cos(lat)*sin(lon)
        assert_relative_eq!(cart5.z, val, epsilon = 1e-9); // sin(lat)
        let ec5_rt = Ecliptic::from_cartesian(cart5);
        assert_relative_eq!(ec5_rt.lon, PI / 4.0, epsilon = 1e-9);
        assert_relative_eq!(ec5_rt.lat, PI / 4.0, epsilon = 1e-9);
    }

    #[test]
    fn test_galactic_to_cartesian_roundtrip() {
        let mut rng = StdRng::seed_from_u64(424244); // Use a fixed seed for reproducibility
        for i in 0..100 {
            let original_lon = rng.gen::<f64>() * 2.0 * PI;
            let original_lat = (rng.gen::<f64>() * PI - PI / 2.0) * 0.99;

            let galactic_original = Galactic {
                lon: original_lon,
                lat: original_lat,
            };

            let cartesian = galactic_original.to_cartesian();
            let galactic_roundtrip = Galactic::from_cartesian(cartesian);

            println!(
                "Test {}: Original Lon: {:.6} rad, Lat: {:.6} rad",
                i, original_lon, original_lat
            );
            println!(
                "           Cartesian X: {:.6}, Y: {:.6}, Z: {:.6}",
                cartesian.x, cartesian.y, cartesian.z
            );
            println!(
                "           Roundtrip Lon: {:.6} rad, Lat: {:.6} rad",
                galactic_roundtrip.lon, galactic_roundtrip.lat
            );

            assert_relative_eq!(
                original_lon.cos(),
                galactic_roundtrip.lon.cos(),
                epsilon = 1e-9
            );
            assert_relative_eq!(
                original_lon.sin(),
                galactic_roundtrip.lon.sin(),
                epsilon = 1e-9
            );
            assert_relative_eq!(original_lat, galactic_roundtrip.lat, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_galactic_to_cartesian_specific_cases() {
        // Case 1: Galactic Center (Lon = 0, Lat = 0)
        let gal1 = Galactic { lon: 0.0, lat: 0.0 };
        let cart1 = gal1.to_cartesian();
        assert_relative_eq!(cart1.x, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart1.z, 0.0, epsilon = 1e-9);
        let gal1_rt = Galactic::from_cartesian(cart1);
        assert_relative_eq!(gal1_rt.lon, 0.0, epsilon = 1e-9);
        assert_relative_eq!(gal1_rt.lat, 0.0, epsilon = 1e-9);

        // Case 2: North Galactic Pole
        let gal2 = Galactic {
            lon: 0.0,
            lat: PI / 2.0,
        };
        let cart2 = gal2.to_cartesian();
        assert_relative_eq!(cart2.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart2.z, 1.0, epsilon = 1e-9);
        let gal2_rt = Galactic::from_cartesian(cart2);
        assert_relative_eq!(gal2_rt.lat, PI / 2.0, epsilon = 1e-9);

        // Case 3: South Galactic Pole
        let gal3 = Galactic {
            lon: 0.0,
            lat: -PI / 2.0,
        };
        let cart3 = gal3.to_cartesian();
        assert_relative_eq!(cart3.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.y, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart3.z, -1.0, epsilon = 1e-9);
        let gal3_rt = Galactic::from_cartesian(cart3);
        assert_relative_eq!(gal3_rt.lat, -PI / 2.0, epsilon = 1e-9);

        // Case 4: Lon = 90 deg (PI/2), Lat = 0 (Galactic plane, direction of Galactic rotation)
        let gal4 = Galactic {
            lon: PI / 2.0,
            lat: 0.0,
        };
        let cart4 = gal4.to_cartesian();
        assert_relative_eq!(cart4.x, 0.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.y, 1.0, epsilon = 1e-9);
        assert_relative_eq!(cart4.z, 0.0, epsilon = 1e-9);
        let gal4_rt = Galactic::from_cartesian(cart4);
        assert_relative_eq!(gal4_rt.lon, PI / 2.0, epsilon = 1e-9);
        assert_relative_eq!(gal4_rt.lat, 0.0, epsilon = 1e-9);

        // Case 5: Lon = 45 deg (PI/4), Lat = 45 deg (PI/4)
        let gal5 = Galactic {
            lon: PI / 4.0,
            lat: PI / 4.0,
        };
        let cart5 = gal5.to_cartesian();
        let val = 1.0 / 2.0_f64.sqrt();
        assert_relative_eq!(cart5.x, val * val, epsilon = 1e-9);
        assert_relative_eq!(cart5.y, val * val, epsilon = 1e-9);
        assert_relative_eq!(cart5.z, val, epsilon = 1e-9);
        let gal5_rt = Galactic::from_cartesian(cart5);
        assert_relative_eq!(gal5_rt.lon, PI / 4.0, epsilon = 1e-9);
        assert_relative_eq!(gal5_rt.lat, PI / 4.0, epsilon = 1e-9);
    }

    #[test]
    fn test_matty_sanity() {
        let x_initial = Vector3::new(1.0, 7.0, 9.0);
        println!("x_initial: {:?}", x_initial);

        let xp = EC_TO_EQ.clone() * x_initial;

        let x_final = EQ_TO_EC.clone() * xp;
        println!("x_final: {:?}", x_final);

        assert_relative_eq!(x_initial.x, x_final.x, epsilon = 1e-10);
        assert_relative_eq!(x_initial.y, x_final.y, epsilon = 1e-10);
        assert_relative_eq!(x_initial.z, x_final.z, epsilon = 1e-10);
    }

    #[test]
    fn test_identity_eq() {
        // Test the identity transformation
        let eq1 = Equatorial {
            ra: 27.0 * PI / 180.0,
            dec: 24.0 * PI / 180.0,
        };

        let ec: Ecliptic = eq1.into();
        println!(
            "ec: lon={:.2}°, lat={:.2}°",
            ec.lon.to_degrees(),
            ec.lat.to_degrees()
        );

        let eq2: Equatorial = ec.into();

        assert_relative_eq!(eq1.ra, eq2.ra, epsilon = 1e-10);
        assert_relative_eq!(eq1.dec, eq2.dec, epsilon = 1e-10);
    }

    #[test]
    fn test_equatorial_to_ecliptic() {
        // Test Equatorial to Ecliptic conversion
        let eq = Equatorial {
            ra: 0.0 * PI / 180.0,
            dec: 0.0 * PI / 180.0,
        };

        let ec: Ecliptic = eq.into();
        println!(
            "Ecliptic: lon={:.2}°, lat={:.2}°",
            ec.lon.to_degrees(),
            ec.lat.to_degrees()
        );

        assert_relative_eq!(ec.lon, 0.0 * PI / 180.0, epsilon = 1e-4);
        assert_relative_eq!(ec.lat, 0.0 * PI / 180.0, epsilon = 1e-4);

        // Test Ecliptic to Equatorial conversion
        let ec = Equatorial {
            ra: 15.0 * PI / 180.0,
            dec: 0.0 * PI / 180.0,
        };

        let ec: Ecliptic = ec.into();
        println!(
            "Equatorial: RA={:.2}°, Dec={:.2}°",
            ec.lon.to_degrees(),
            ec.lat.to_degrees()
        );

        assert_relative_eq!(ec.lon, 13.811618 * PI / 180.0, epsilon = 1e-4);
        assert_relative_eq!(ec.lat, -5.909203 * PI / 180.0, epsilon = 1e-4);

        // Test Ecliptic to Equatorial conversion
        let ec = Equatorial {
            ra: 165.0 * PI / 180.0,
            dec: -12.0 * PI / 180.0,
        };

        let ec: Ecliptic = ec.into();
        println!(
            "Equatorial: RA={:.2}°, Dec={:.2}°",
            ec.lon.to_degrees(),
            ec.lat.to_degrees()
        );

        assert_relative_eq!(ec.lon, 171.004394 * PI / 180.0, epsilon = 1e-4);
        assert_relative_eq!(ec.lat, -16.945252 * PI / 180.0, epsilon = 1e-4);
    }

    #[test]
    fn test_ecliptic_to_equatorial_rt_rand() {
        let mut rng = StdRng::seed_from_u64(23423 as u64);
        for i in 0..100 {
            println!("Random test {}", i);
            let ra = rng.gen_range(0.0..(2.0 * PI));
            let dec = rng.gen_range(-PI / 2.1..PI / 2.1);
            let eq1 = Equatorial { ra, dec };
            let ec: Ecliptic = eq1.into();
            let eq2: Equatorial = ec.into();
            assert_relative_eq!(eq1.ra, eq2.ra, epsilon = 1e-4);
            assert_relative_eq!(eq1.dec, eq2.dec, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_angle_between() {
        // Test angle between same point (should be 0)
        let eq1 = Equatorial { ra: 0.0, dec: 0.0 };
        let eq2 = Equatorial { ra: 0.0, dec: 0.0 };
        assert_relative_eq!(eq1.angle_between(&eq2), 0.0, epsilon = 1e-9);

        // Test angle between opposite points (should be π)
        let eq3 = Equatorial { ra: 0.0, dec: 0.0 };
        let eq4 = Equatorial { ra: PI, dec: 0.0 };
        assert_relative_eq!(eq3.angle_between(&eq4), PI, epsilon = 1e-9);

        // Test angle between perpendicular points (should be π/2)
        let eq5 = Equatorial { ra: 0.0, dec: 0.0 };
        let eq6 = Equatorial {
            ra: 0.0,
            dec: PI / 2.0,
        };
        assert_relative_eq!(eq5.angle_between(&eq6), PI / 2.0, epsilon = 1e-9);
    }

    #[test]
    fn test_coordinate_conversions() {
        // Test data - assuming the comment at bottom has format:
        // RA(deg) Dec(deg) for Celestial (Equatorial)
        // Lon(deg) Lat(deg) for Ecliptic
        // Lon(deg) Lat(deg) for Galactic

        let equatorial = Equatorial {
            ra: 24.0 * PI / 180.0,
            dec: 27.0 * PI / 180.0,
        };

        let ecliptic = Ecliptic {
            lon: 32.22518 * PI / 180.0,
            lat: 15.80545 * PI / 180.0,
        };

        let galactic = Galactic {
            lon: 135.03726 * PI / 180.0, // Note: swapped based on comment
            lat: -34.82204 * PI / 180.0,
        };

        // Test Galactic to Equatorial
        let ga2eq: Equatorial = galactic.into();
        println!(
            "Galactic to Equatorial: RA={:.5}°, Dec={:.5}°",
            ga2eq.ra.to_degrees(),
            ga2eq.dec.to_degrees()
        );
        assert_relative_eq!(equatorial.ra, ga2eq.ra, epsilon = 1e-4);
        assert_relative_eq!(equatorial.dec, ga2eq.dec, epsilon = 1e-4);

        // Test Ecliptic to Equatorial
        let ec2eq: Equatorial = ecliptic.into();
        println!(
            "Ecliptic to Equatorial: RA={:.5}°, Dec={:.5}°",
            ec2eq.ra.to_degrees(),
            ec2eq.dec.to_degrees()
        );
        assert_relative_eq!(equatorial.ra, ec2eq.ra, epsilon = 1e-4);
        assert_relative_eq!(equatorial.dec, ec2eq.dec, epsilon = 1e-4);

        // Test Equatorial to Ecliptic
        let eq2ec: Ecliptic = equatorial.into();
        println!(
            "Equatorial to Ecliptic: Lon={:.5}°, Lat={:.5}°",
            eq2ec.lon.to_degrees(),
            eq2ec.lat.to_degrees()
        );
        assert_relative_eq!(ecliptic.lon, eq2ec.lon, epsilon = 1e-4);
        assert_relative_eq!(ecliptic.lat, eq2ec.lat, epsilon = 1e-4);

        // Test Galactic to Ecliptic
        let ga2ec: Ecliptic = galactic.into();
        println!(
            "Galactic to Ecliptic: Lon={:.5}°, Lat={:.5}°",
            ga2ec.lon.to_degrees(),
            ga2ec.lat.to_degrees()
        );
        assert_relative_eq!(ecliptic.lon, ga2ec.lon, epsilon = 1e-4);
        assert_relative_eq!(ecliptic.lat, ga2ec.lat, epsilon = 1e-4);

        // Test Equatorial to Galactic
        let eq2ga: Galactic = equatorial.into();
        println!(
            "Equatorial to Galactic: Lon={:.5}°, Lat={:.5}°",
            eq2ga.lon.to_degrees(),
            eq2ga.lat.to_degrees()
        );
        assert_relative_eq!(galactic.lon, eq2ga.lon, epsilon = 1e-4);
        assert_relative_eq!(galactic.lat, eq2ga.lat, epsilon = 1e-4);

        // Test Ecliptic to Galactic
        let ec2ga: Galactic = ecliptic.into();
        println!(
            "Ecliptic to Galactic: Lon={:.5}°, Lat={:.5}°",
            ec2ga.lon.to_degrees(),
            ec2ga.lat.to_degrees()
        );
        assert_relative_eq!(galactic.lon, ec2ga.lon, epsilon = 1e-4);
        assert_relative_eq!(galactic.lat, ec2ga.lat, epsilon = 1e-4);
    }
}
