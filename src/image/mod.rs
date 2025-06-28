use ndarray::Array2;

pub mod daofind;

/// Performs sigma clipping on a 2D array of f64 values.
/// Values outside the clipping bounds are set to the bounds rather than NaN.
///
/// # Arguments
///
/// * `data` - Input 2D array to be sigma clipped
/// * `sigma` - Number of standard deviations to use for clipping
/// * `maxiters` - Maximum number of iterations to perform (None for unlimited)
/// * `use_median` - If true, use median as center function; otherwise use mean
///
/// # Returns
///
/// A new Array2<f64> with clipped values set to the bounds
pub fn sigma_clip(
    data: &Array2<f64>,
    sigma: f64,
    maxiters: Option<usize>,
    use_median: bool,
) -> Array2<f64> {
    let maxiters = maxiters.unwrap_or(usize::MAX);
    let mut result = data.clone();
    let mut iteration = 0;
    let mut n_changed = 1;

    // Always perform at least one iteration
    while (n_changed > 0 || iteration == 0) && iteration < maxiters {
        // Filter out NaN values for calculations
        let valid_data: Vec<f64> = result.iter().filter(|&&x| x.is_finite()).cloned().collect();

        if valid_data.is_empty() {
            break;
        }

        // Calculate center (mean or median)
        let center = if use_median {
            median(&valid_data)
        } else {
            valid_data.iter().sum::<f64>() / valid_data.len() as f64
        };

        // Calculate standard deviation
        let sum_sq: f64 = valid_data.iter().map(|&x| (x - center).powi(2)).sum();
        let std_dev = (sum_sq / valid_data.len() as f64).sqrt();

        // Define clipping bounds
        let lower_bound = center - (sigma * std_dev);
        let upper_bound = center + (sigma * std_dev);

        // Apply clipping
        n_changed = 0;

        for val in result.iter_mut() {
            if !val.is_finite() {
                continue;
            }

            if *val < lower_bound {
                *val = lower_bound;
                n_changed += 1;
            } else if *val > upper_bound {
                *val = upper_bound;
                n_changed += 1;
            }
        }

        iteration += 1;
    }

    result
}

/// Calculates the median of a vector of f64 values.
///
/// # Arguments
///
/// * `data` - Input vector of f64 values
///
/// # Returns
///
/// The median value
fn median(data: &[f64]) -> f64 {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let len = sorted_data.len();
    if len % 2 == 0 {
        (sorted_data[len / 2 - 1] + sorted_data[len / 2]) / 2.0
    } else {
        sorted_data[len / 2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        // Test odd number of elements
        let data_odd = vec![5.0, 3.0, 1.0, 4.0, 2.0];
        assert_eq!(median(&data_odd), 3.0);

        // Test even number of elements
        let data_even = vec![5.0, 3.0, 1.0, 4.0, 2.0, 6.0];
        assert_eq!(median(&data_even), 3.5);

        // Test with repeated values
        let data_repeated = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        assert_eq!(median(&data_repeated), 2.0);

        // Test with single element
        let data_single = vec![42.0];
        assert_eq!(median(&data_single), 42.0);
    }

    #[test]
    fn test_sigma_clip_mean() {
        // Create a test array with outliers
        // [[ 1.0, 2.0, 3.0 ],
        //  [ 4.0, 100.0, 6.0 ],
        //  [ 7.0, 8.0, 9.0 ]]
        // Mean = 15.56, StdDev = 31.79
        // With sigma=2: Lower bound = -48.03, Upper bound = 79.13
        let mut data = Array2::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 100.0; // Outlier
        data[[1, 2]] = 6.0;
        data[[2, 0]] = 7.0;
        data[[2, 1]] = 8.0;
        data[[2, 2]] = 9.0;

        // Apply sigma clipping with mean
        let clipped = sigma_clip(&data, 2.0, Some(1), false);

        // The value at [1, 1] should be clipped to the upper bound
        for row in 0..3 {
            for col in 0..3 {
                if row == 1 && col == 1 {
                    // The outlier value should be clipped, expected near 79.13
                    assert!(clipped[[row, col]] < 80.0);
                } else {
                    // All other values should remain unchanged
                    assert_eq!(clipped[[row, col]], data[[row, col]]);
                }
            }
        }
    }

    #[test]
    fn test_sigma_clip_median() {
        // Create a test array with outliers
        // [[ 1.0, 2.0, 3.0 ],
        //  [ 4.0, 100.0, 6.0 ],
        //  [ 7.0, 8.0, 9.0 ]]
        // Median = 6.0, StdDev = 31.79
        // With sigma=2: Lower bound = -57.58, Upper bound = 69.58
        let mut data = Array2::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 100.0; // Outlier
        data[[1, 2]] = 6.0;
        data[[2, 0]] = 7.0;
        data[[2, 1]] = 8.0;
        data[[2, 2]] = 9.0;

        // Apply sigma clipping with median
        let clipped = sigma_clip(&data, 2.0, Some(1), true);

        // The value at [1, 1] should be clipped to the upper bound
        for row in 0..3 {
            for col in 0..3 {
                if row == 1 && col == 1 {
                    // The outlier value should be clipped, expected near 69.58
                    assert!(clipped[[row, col]] < 70.0);
                } else {
                    // All other values should remain unchanged
                    assert_eq!(clipped[[row, col]], data[[row, col]]);
                }
            }
        }
    }

    #[test]
    fn test_sigma_clip_multiple_iterations() {
        // Create a test array with extreme outliers
        let mut data = Array2::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = 3.0;
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 30.0; // Smaller outlier
        data[[1, 2]] = 6.0;
        data[[2, 0]] = 7.0;
        data[[2, 1]] = 8.0;
        data[[2, 2]] = 1000.0; // Extreme outlier

        // Apply sigma clipping with multiple iterations
        let clipped = sigma_clip(&data, 2.0, Some(3), false);

        // At least the extreme outlier should be clipped
        assert!(
            clipped[[2, 2]] < 1000.0,
            "Expected clipped[2,2] < 1000.0, got {}",
            clipped[[2, 2]]
        );

        // Other values should remain unchanged
        assert_eq!(clipped[[0, 0]], 1.0);
        assert_eq!(clipped[[0, 1]], 2.0);
        assert_eq!(clipped[[0, 2]], 3.0);
        assert_eq!(clipped[[1, 0]], 4.0);
        assert_eq!(clipped[[1, 2]], 6.0);
        assert_eq!(clipped[[2, 0]], 7.0);
        assert_eq!(clipped[[2, 1]], 8.0);
    }

    #[test]
    fn test_sigma_clip_gaussian_data() {
        // Create a more realistic test array - normal data with a single outlier
        let mut data = Array2::zeros((3, 3));
        data[[0, 0]] = 10.0;
        data[[0, 1]] = 11.0;
        data[[0, 2]] = 9.0;
        data[[1, 0]] = 12.0;
        data[[1, 1]] = 10.5;
        data[[1, 2]] = 9.5;
        data[[2, 0]] = 11.5;
        data[[2, 1]] = 10.0;
        data[[2, 2]] = 50.0; // Outlier

        // Apply sigma clipping with mean and a smaller sigma value to ensure clipping
        let clipped = sigma_clip(&data, 2.0, Some(1), false);

        // The outlier should be clipped
        assert!(
            clipped[[2, 2]] < 50.0,
            "Expected outlier to be clipped, got {}",
            clipped[[2, 2]]
        );

        // Other values should remain unchanged
        for row in 0..3 {
            for col in 0..3 {
                if row == 2 && col == 2 {
                    continue; // Skip the outlier
                }
                assert_eq!(clipped[[row, col]], data[[row, col]]);
            }
        }
    }

    #[test]
    fn test_sigma_clip_with_nan_values() {
        // Create a test array with NaN values
        let mut data = Array2::zeros((3, 3));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 2.0;
        data[[0, 2]] = f64::NAN;
        data[[1, 0]] = 4.0;
        data[[1, 1]] = 100.0; // Outlier
        data[[1, 2]] = 6.0;
        data[[2, 0]] = 7.0;
        data[[2, 1]] = 8.0;
        data[[2, 2]] = 9.0;

        // Apply sigma clipping
        let clipped = sigma_clip(&data, 2.0, Some(1), false);

        // NaN values should remain NaN
        assert!(clipped[[0, 2]].is_nan());

        // Outliers should be clipped
        assert!(clipped[[1, 1]] < data[[1, 1]]);

        // Other values should remain unchanged
        assert_eq!(clipped[[0, 0]], 1.0);
        assert_eq!(clipped[[0, 1]], 2.0);
        assert_eq!(clipped[[1, 0]], 4.0);
        assert_eq!(clipped[[1, 2]], 6.0);
        assert_eq!(clipped[[2, 0]], 7.0);
        assert_eq!(clipped[[2, 1]], 8.0);
        assert_eq!(clipped[[2, 2]], 9.0);
    }
}
