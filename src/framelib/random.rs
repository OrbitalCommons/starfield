use super::inertial::Equatorial;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// An iterator that generates random Equatorial coordinates with uniform distribution
/// across the celestial sphere.
pub struct RandomEquatorial {
    rng: StdRng,
}

impl RandomEquatorial {
    /// Create a new RandomEquatorial iterator with a specific seed for deterministic output
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create a new RandomEquatorial iterator with a random seed
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
        }
    }
}

impl Default for RandomEquatorial {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for RandomEquatorial {
    type Item = Equatorial;

    fn next(&mut self) -> Option<Self::Item> {
        // RA is uniformly distributed from 0 to 2π
        let ra = self.rng.gen::<f64>() * 2.0 * PI;

        // For uniform distribution on a sphere, we need to account for the
        // changing area element at different declinations.
        // Using the inverse transform method: dec = arcsin(2u - 1)
        // where u is uniform in [0, 1]
        let u: f64 = self.rng.gen();
        let dec = (2.0 * u - 1.0).asin();

        Some(Equatorial::new(ra, dec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_deterministic_with_seed() {
        let mut iter1 = RandomEquatorial::with_seed(42);
        let mut iter2 = RandomEquatorial::with_seed(42);

        for _ in 0..10 {
            let coord1 = iter1.next().unwrap();
            let coord2 = iter2.next().unwrap();

            assert_relative_eq!(coord1.ra, coord2.ra, epsilon = 1e-10);
            assert_relative_eq!(coord1.dec, coord2.dec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_different_seeds_produce_different_values() {
        let mut iter1 = RandomEquatorial::with_seed(42);
        let mut iter2 = RandomEquatorial::with_seed(123);

        let coord1 = iter1.next().unwrap();
        let coord2 = iter2.next().unwrap();

        // These should be different (extremely unlikely to be the same)
        assert!(coord1.ra != coord2.ra || coord1.dec != coord2.dec);
    }

    #[test]
    fn test_bounds() {
        let mut iter = RandomEquatorial::with_seed(12345);

        for _ in 0..1000 {
            let coord = iter.next().unwrap();

            // RA should be in [0, 2π)
            assert!(coord.ra >= 0.0 && coord.ra < 2.0 * PI);

            // Dec should be in [-π/2, π/2]
            assert!(coord.dec >= -PI / 2.0 && coord.dec <= PI / 2.0);
        }
    }

    #[test]
    fn test_distribution_statistics() {
        let mut iter = RandomEquatorial::with_seed(999);
        let n = 10000;

        let mut ra_sum = 0.0;
        let mut dec_sin_sum = 0.0;

        for _ in 0..n {
            let coord = iter.next().unwrap();
            ra_sum += coord.ra;
            dec_sin_sum += coord.dec.sin();
        }

        // Mean RA should be approximately π (middle of [0, 2π])
        let mean_ra = ra_sum / n as f64;
        assert_relative_eq!(mean_ra, PI, epsilon = 0.1);

        // Mean sin(dec) should be approximately 0 for uniform sphere distribution
        let mean_sin_dec = dec_sin_sum / n as f64;
        assert_relative_eq!(mean_sin_dec, 0.0, epsilon = 0.05);
    }

    #[test]
    fn test_iterator_is_infinite() {
        let mut iter = RandomEquatorial::with_seed(777);

        // Take many values to ensure it doesn't terminate
        for _ in 0..10000 {
            assert!(iter.next().is_some());
        }
    }
}
