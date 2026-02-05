//! Chebyshev polynomial functionality for ephemeris interpolation
//!
//! Chebyshev polynomials are used in JPL ephemerides for interpolating
//! positions and velocities of celestial bodies from ephemeris data.

use crate::jplephem::errors::{JplephemError, Result};

/// Chebyshev polynomial representation and evaluation
///
/// Stores the coefficients of a Chebyshev polynomial expansion
/// and provides methods to evaluate the polynomial and its derivatives.
#[derive(Debug, Clone)]
pub struct ChebyshevPolynomial {
    coefficients: Vec<f64>,
}

impl ChebyshevPolynomial {
    /// Create a new Chebyshev polynomial with the given coefficients
    ///
    /// Coefficients are ordered from lowest to highest degree:
    /// `[c0, c1, c2, ..., cn]` where the polynomial is:
    /// `f(x) = c0*T0(x) + c1*T1(x) + c2*T2(x) + ... + cn*Tn(x)`
    pub fn new(coefficients: Vec<f64>) -> Self {
        Self { coefficients }
    }

    /// Evaluate the Chebyshev polynomial at point x in [-1, 1]
    ///
    /// Uses Clenshaw's recurrence for numerical stability.
    pub fn evaluate(&self, x: f64) -> f64 {
        if self.coefficients.is_empty() {
            return 0.0;
        }
        let n = self.coefficients.len();
        if n == 1 {
            return self.coefficients[0];
        }

        // Clenshaw recurrence: more stable than direct evaluation
        let mut b_k_plus_1 = 0.0;
        let mut b_k_plus_2 = 0.0;
        for i in (1..n).rev() {
            let b_k = 2.0 * x * b_k_plus_1 - b_k_plus_2 + self.coefficients[i];
            b_k_plus_2 = b_k_plus_1;
            b_k_plus_1 = b_k;
        }
        self.coefficients[0] + x * b_k_plus_1 - b_k_plus_2
    }

    /// Calculate the derivative of the Chebyshev polynomial at point x
    ///
    /// Uses the relation dT_n(x)/dx = n * U_{n-1}(x) where U is the
    /// Chebyshev polynomial of the second kind.
    pub fn derivative(&self, x: f64) -> f64 {
        if self.coefficients.len() <= 1 {
            return 0.0;
        }

        let mut result = 0.0;
        for i in 1..self.coefficients.len() {
            let n = i as f64;
            result += self.coefficients[i] * n * Self::chebyshev_u(i - 1, x);
        }
        result
    }

    /// Compute U_n(x), the Chebyshev polynomial of the second kind
    fn chebyshev_u(n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => 2.0 * x,
            _ => {
                let mut u_prev2 = 1.0;
                let mut u_prev1 = 2.0 * x;
                let mut u_n = 0.0;
                for _ in 2..=n {
                    u_n = 2.0 * x * u_prev1 - u_prev2;
                    u_prev2 = u_prev1;
                    u_prev1 = u_n;
                }
                u_n
            }
        }
    }

    /// Get the degree of the polynomial
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }

    /// Get a reference to the coefficients
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }
}

/// Normalize a time value to [-1, 1] for Chebyshev evaluation
///
/// # Arguments
/// * `time` - The time to normalize
/// * `midpoint` - The midpoint of the time interval
/// * `radius` - The half-length of the time interval
pub fn normalize_time(time: f64, midpoint: f64, radius: f64) -> Result<f64> {
    if radius <= 0.0 {
        return Err(JplephemError::Other(
            "Invalid radius for time normalization: must be positive".to_string(),
        ));
    }

    let normalized = (time - midpoint) / radius;

    if !(-1.0 - 1e-9..=1.0 + 1e-9).contains(&normalized) {
        return Err(JplephemError::OutOfRangeError {
            jd: time,
            start_jd: midpoint - radius,
            end_jd: midpoint + radius,
        });
    }

    // Clamp to [-1, 1] to handle floating point edge cases
    Ok(normalized.clamp(-1.0, 1.0))
}

/// Rescale a derivative from normalized time to physical time
///
/// When evaluating derivatives of Chebyshev polynomials, we get the derivative
/// with respect to normalized time. This rescales to physical time units.
pub fn rescale_derivative(deriv_normalized: f64, radius: f64) -> Result<f64> {
    if radius <= 0.0 {
        return Err(JplephemError::Other(
            "Invalid radius for derivative rescaling: must be positive".to_string(),
        ));
    }
    Ok(deriv_normalized / radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chebyshev_constant() {
        let poly = ChebyshevPolynomial::new(vec![5.0]);
        assert_eq!(poly.evaluate(-1.0), 5.0);
        assert_eq!(poly.evaluate(0.0), 5.0);
        assert_eq!(poly.evaluate(1.0), 5.0);
        assert_eq!(poly.derivative(0.0), 0.0);
    }

    #[test]
    fn test_chebyshev_linear() {
        // f(x) = 3 + 2x
        let poly = ChebyshevPolynomial::new(vec![3.0, 2.0]);
        assert!((poly.evaluate(-1.0) - 1.0).abs() < 1e-12);
        assert!((poly.evaluate(0.0) - 3.0).abs() < 1e-12);
        assert!((poly.evaluate(1.0) - 5.0).abs() < 1e-12);
        assert!((poly.derivative(0.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_chebyshev_quadratic() {
        // T2(x) = 2x² - 1, so f(x) = 3 + 2x + (2x² - 1) = 2 + 2x + 2x²
        let poly = ChebyshevPolynomial::new(vec![3.0, 2.0, 1.0]);
        assert!((poly.evaluate(-1.0) - 2.0).abs() < 1e-12);
        assert!((poly.evaluate(0.0) - 2.0).abs() < 1e-12);
        assert!((poly.evaluate(1.0) - 6.0).abs() < 1e-12);
        // f'(x) = 2 + 4x
        assert!((poly.derivative(-1.0) - (-2.0)).abs() < 1e-12);
        assert!((poly.derivative(0.0) - 2.0).abs() < 1e-12);
        assert!((poly.derivative(1.0) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_x_squared_approximation() {
        // x^2 = (T2(x) + 1)/2, coefficients [0.5, 0, 0.5]
        let poly = ChebyshevPolynomial::new(vec![0.5, 0.0, 0.5]);
        for i in 0..=10 {
            let x = -1.0 + i as f64 * 0.2;
            let expected = x * x;
            assert!(
                (expected - poly.evaluate(x)).abs() < 1e-12,
                "Bad approximation at x={x}"
            );
        }
    }

    #[test]
    fn test_time_normalization() {
        let mid = 100.0;
        let rad = 10.0;
        assert_eq!(normalize_time(100.0, mid, rad).unwrap(), 0.0);
        assert_eq!(normalize_time(90.0, mid, rad).unwrap(), -1.0);
        assert_eq!(normalize_time(110.0, mid, rad).unwrap(), 1.0);
        assert_eq!(normalize_time(95.0, mid, rad).unwrap(), -0.5);
        assert!(normalize_time(79.0, mid, rad).is_err());
        assert!(normalize_time(121.0, mid, rad).is_err());
        assert!(normalize_time(100.0, mid, 0.0).is_err());
    }

    #[test]
    fn test_derivative_rescaling() {
        assert_eq!(rescale_derivative(5.0, 10.0).unwrap(), 0.5);
        assert!(rescale_derivative(5.0, 0.0).is_err());
    }
}
