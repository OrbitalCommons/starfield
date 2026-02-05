//! Search algorithms for finding events and extrema in time-dependent functions
//!
//! Ported from Python Skyfield's `searchlib.py`. Provides:
//! - [`find_discrete`] — find times when a discrete function changes value
//! - [`find_maxima`] — find local maxima of a continuous function
//! - [`find_minima`] — find local minima of a continuous function

use crate::constants::DAY_S;

/// Default epsilon for discrete event finding (0.001 seconds in days)
pub const EPSILON_DISCRETE: f64 = 0.001 / DAY_S;

/// Default epsilon for extrema finding (1.0 second in days)
pub const EPSILON_EXTREMA: f64 = 1.0 / DAY_S;

/// Default number of subdivisions per bracket refinement
pub const DEFAULT_NUM: usize = 12;

/// Find times at which a discrete function of time changes value.
///
/// Used for instantaneous events like sunrise, transits, and seasons.
///
/// # Arguments
/// * `jd_start` - Start Julian date (TT)
/// * `jd_end` - End Julian date (TT)
/// * `f` - Function mapping a slice of Julian dates to discrete integer values
/// * `step_days` - Sampling interval in days
/// * `epsilon` - Convergence threshold in days (default: 0.001/86400)
/// * `num` - Number of subdivisions per refinement (default: 12)
///
/// # Returns
/// Vector of `(jd, value)` pairs at each transition
pub fn find_discrete<F>(
    jd_start: f64,
    jd_end: f64,
    f: &mut F,
    step_days: f64,
    epsilon: f64,
    num: usize,
) -> Vec<(f64, i64)>
where
    F: FnMut(&[f64]) -> Vec<i64>,
{
    assert!(
        jd_start < jd_end,
        "start time must be earlier than end time"
    );

    let sample_count = ((jd_end - jd_start) / step_days) as usize + 2;
    let jd = linspace(jd_start, jd_end, sample_count);
    find_discrete_core(&jd, f, epsilon, num)
}

fn find_discrete_core<F>(initial_jd: &[f64], f: &mut F, epsilon: f64, num: usize) -> Vec<(f64, i64)>
where
    F: FnMut(&[f64]) -> Vec<i64>,
{
    let end_mask = linspace(0.0, 1.0, num);
    let start_mask: Vec<f64> = end_mask.iter().copied().rev().collect();

    let mut jd = initial_jd.to_vec();

    loop {
        let y = f(&jd);

        // Find indices where consecutive values differ
        let mut transition_indices = Vec::new();
        for i in 0..y.len() - 1 {
            if y[i] != y[i + 1] {
                transition_indices.push(i);
            }
        }

        if transition_indices.is_empty() {
            return Vec::new();
        }

        let starts: Vec<f64> = transition_indices.iter().map(|&i| jd[i]).collect();
        let ends: Vec<f64> = transition_indices.iter().map(|&i| jd[i + 1]).collect();

        // Check convergence
        let max_width = starts
            .iter()
            .zip(ends.iter())
            .map(|(s, e)| e - s)
            .fold(0.0_f64, f64::max);

        if max_width <= epsilon {
            let values: Vec<i64> = transition_indices.iter().map(|&i| y[i + 1]).collect();
            return ends.into_iter().zip(values).collect();
        }

        // Subdivide each bracket
        jd = outer_interp(&starts, &start_mask, &ends, &end_mask);
    }
}

/// Find the local maxima of a continuous function over a time range.
///
/// # Arguments
/// * `jd_start` - Start Julian date (TT)
/// * `jd_end` - End Julian date (TT)
/// * `f` - Function mapping a slice of Julian dates to f64 values
/// * `step_days` - Sampling interval in days
/// * `epsilon` - Convergence threshold in days (default: 1.0/86400)
/// * `num` - Number of subdivisions per refinement (default: 12)
///
/// # Returns
/// Vector of `(jd, value)` pairs at each maximum
pub fn find_maxima<F>(
    jd_start: f64,
    jd_end: f64,
    f: &F,
    step_days: f64,
    epsilon: f64,
    num: usize,
) -> Vec<(f64, f64)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    assert!(
        jd_start < jd_end,
        "start time must be earlier than end time"
    );

    // Add extra points beyond range to catch maxima near endpoints
    let steps = ((jd_end - jd_start) / step_days) as usize + 3;
    let real_step = (jd_end - jd_start) / steps as f64;
    let jd = linspace(jd_start - real_step, jd_end + real_step, steps + 2);

    let end_alpha = linspace(0.0, 1.0, num);
    let start_alpha: Vec<f64> = end_alpha.iter().copied().rev().collect();

    let mut jd = jd;

    loop {
        let y = f(&jd);

        // Check convergence using first interval
        if jd.len() >= 2 && (jd[1] - jd[0]) <= epsilon {
            let (mut max_jd, mut max_y) = identify_maxima(&jd, &y);

            // Filter out maxima outside our bounds
            let keep: Vec<usize> = (0..max_jd.len())
                .filter(|&i| max_jd[i] >= jd_start && max_jd[i] <= jd_end)
                .collect();
            max_jd = keep.iter().map(|&i| max_jd[i]).collect();
            max_y = keep.iter().map(|&i| max_y[i]).collect();

            // Deduplicate within epsilon
            if !max_jd.is_empty() {
                let mut deduped_jd = vec![max_jd[0]];
                let mut deduped_y = vec![max_y[0]];
                for i in 1..max_jd.len() {
                    if max_jd[i] - *deduped_jd.last().unwrap() > epsilon {
                        deduped_jd.push(max_jd[i]);
                        deduped_y.push(max_y[i]);
                    }
                }
                return deduped_jd.into_iter().zip(deduped_y).collect();
            }
            return Vec::new();
        }

        let (left, right) = choose_brackets(&y);

        if left.is_empty() {
            return Vec::new();
        }

        let starts: Vec<f64> = left.iter().map(|&i| jd[i]).collect();
        let ends: Vec<f64> = right.iter().map(|&i| jd[i]).collect();

        jd = outer_interp(&starts, &start_alpha, &ends, &end_alpha);
        jd = remove_adjacent_duplicates(&jd);
    }
}

/// Find the local minima of a continuous function over a time range.
///
/// Negates the function and delegates to [`find_maxima`].
pub fn find_minima<F>(
    jd_start: f64,
    jd_end: f64,
    f: &F,
    step_days: f64,
    epsilon: f64,
    num: usize,
) -> Vec<(f64, f64)>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let neg_f = |jd: &[f64]| -> Vec<f64> { f(jd).iter().map(|&v| -v).collect() };
    find_maxima(jd_start, jd_end, &neg_f, step_days, epsilon, num)
        .into_iter()
        .map(|(jd, v)| (jd, -v))
        .collect()
}

/// Generate `n` evenly spaced values from `start` to `end` (inclusive).
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

/// Interpolate between starts and ends using outer product masks.
///
/// For each bracket (start[i], end[i]), generate `num` points by
/// `start[i] * start_mask[j] + end[i] * end_mask[j]`.
fn outer_interp(starts: &[f64], start_mask: &[f64], ends: &[f64], end_mask: &[f64]) -> Vec<f64> {
    let num = start_mask.len();
    let mut result = Vec::with_capacity(starts.len() * num);
    for i in 0..starts.len() {
        for j in 0..num {
            result.push(starts[i] * start_mask[j] + ends[i] * end_mask[j]);
        }
    }
    result
}

/// Find bracket indices where maxima might exist.
///
/// Uses second-difference-of-sign to identify downward inflection points.
fn choose_brackets(y: &[f64]) -> (Vec<usize>, Vec<usize>) {
    if y.len() < 3 {
        return (Vec::new(), Vec::new());
    }

    // diff(sign(diff(y)))
    let dsd = diff_sign_diff(y);

    // Find indices where dsd < 0
    let mut indices = Vec::new();
    for (i, &v) in dsd.iter().enumerate() {
        if v < 0 {
            indices.push(i);
        }
    }

    // Expand each index to [i, i+1], flatten, deduplicate
    let mut left = Vec::new();
    for &idx in &indices {
        left.push(idx);
        left.push(idx + 1);
    }
    left = remove_adjacent_duplicates(&left);

    let right: Vec<usize> = left.iter().map(|&l| l + 1).collect();
    (left, right)
}

/// Identify exact maxima positions from converged x/y data.
fn identify_maxima(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    if x.len() < 3 {
        return (Vec::new(), Vec::new());
    }

    let dsd = diff_sign_diff(y);

    // Sharp peaks: dsd == -2
    let mut peak_x = Vec::new();
    let mut peak_y = Vec::new();
    for (i, &v) in dsd.iter().enumerate() {
        if v == -2 {
            let idx = i + 1;
            peak_x.push(x[idx]);
            peak_y.push(y[idx]);
        }
    }

    // Plateau maxima: find runs of zeros in dsd bordered by -1 values
    let nonzero_indices: Vec<usize> = dsd
        .iter()
        .enumerate()
        .filter(|(_, &v)| v != 0)
        .map(|(i, _)| i)
        .collect();
    let nonzero_values: Vec<i32> = nonzero_indices.iter().map(|&i| dsd[i]).collect();

    let mut plateau_x = Vec::new();
    let mut plateau_y = Vec::new();
    for i in 0..nonzero_values.len().saturating_sub(1) {
        if nonzero_values[i] == -1 && nonzero_values[i + 1] == -1 {
            let left_idx = nonzero_indices[i];
            let right_idx = nonzero_indices[i + 1] + 2;
            if right_idx < x.len() {
                plateau_x.push((x[left_idx] + x[right_idx]) / 2.0);
                plateau_y.push(y[left_idx + 1]);
            }
        }
    }

    // Combine and sort
    peak_x.extend(plateau_x);
    peak_y.extend(plateau_y);

    if peak_x.len() > 1 {
        let mut indices: Vec<usize> = (0..peak_x.len()).collect();
        indices.sort_by(|&a, &b| peak_x[a].partial_cmp(&peak_x[b]).unwrap());
        let sorted_x: Vec<f64> = indices.iter().map(|&i| peak_x[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| peak_y[i]).collect();
        (sorted_x, sorted_y)
    } else {
        (peak_x, peak_y)
    }
}

/// Compute diff(sign(diff(y))) as integer values.
fn diff_sign_diff(y: &[f64]) -> Vec<i32> {
    if y.len() < 2 {
        return Vec::new();
    }
    let sign_diff: Vec<i32> = y
        .windows(2)
        .map(|w| {
            let d = w[1] - w[0];
            if d > 0.0 {
                1
            } else if d < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect();

    sign_diff.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Remove consecutive duplicate values from a sorted slice.
fn remove_adjacent_duplicates<T: Copy + PartialEq>(a: &[T]) -> Vec<T> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut result = vec![a[0]];
    for i in 1..a.len() {
        if a[i] != a[i - 1] {
            result.push(a[i]);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_linspace() {
        let v = linspace(0.0, 1.0, 5);
        assert_eq!(v.len(), 5);
        assert_relative_eq!(v[0], 0.0);
        assert_relative_eq!(v[1], 0.25);
        assert_relative_eq!(v[4], 1.0);
    }

    #[test]
    fn test_linspace_single() {
        let v = linspace(5.0, 10.0, 1);
        assert_eq!(v.len(), 1);
        assert_relative_eq!(v[0], 5.0);
    }

    #[test]
    fn test_find_discrete_step_function() {
        // sign(sin(x)) transitions at multiples of pi
        let mut f = |jd: &[f64]| -> Vec<i64> {
            jd.iter()
                .map(|&x| {
                    let s = x.sin();
                    if s > 0.0 {
                        1
                    } else {
                        0
                    }
                })
                .collect()
        };

        let results = find_discrete(0.1, 4.0 * PI, &mut f, 0.1, EPSILON_DISCRETE, DEFAULT_NUM);

        // Should find transitions near pi, 2*pi, 3*pi
        assert!(
            results.len() >= 3,
            "Expected at least 3 transitions, got {}",
            results.len()
        );
        for (jd, _) in &results {
            // Each transition should be near a multiple of pi
            let nearest_pi = (*jd / PI).round() * PI;
            assert!(
                (jd - nearest_pi).abs() < 0.01,
                "Transition at {} not near a multiple of pi",
                jd
            );
        }
    }

    #[test]
    fn test_find_discrete_no_transitions() {
        let mut f = |jd: &[f64]| -> Vec<i64> { vec![1; jd.len()] };
        let results = find_discrete(0.0, 10.0, &mut f, 1.0, EPSILON_DISCRETE, DEFAULT_NUM);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_maxima_sin() {
        // sin(x) has maxima at pi/2, 5*pi/2, etc.
        let f = |jd: &[f64]| -> Vec<f64> { jd.iter().map(|&x| x.sin()).collect() };

        let results = find_maxima(0.0, 4.0 * PI, &f, 0.5, EPSILON_EXTREMA, DEFAULT_NUM);

        // Should find maxima near pi/2 and 5*pi/2
        assert!(
            results.len() >= 2,
            "Expected at least 2 maxima, got {}",
            results.len()
        );

        for (jd, val) in &results {
            // Value at maximum should be close to 1.0
            assert_relative_eq!(*val, 1.0, epsilon = 0.01);
            // Position should be near an odd multiple of pi/2
            let phase = (*jd / (PI / 2.0)).round();
            assert!(
                (phase as i64) % 2 == 1,
                "Maximum at {} not near odd multiple of pi/2",
                jd
            );
        }
    }

    #[test]
    fn test_find_minima_sin() {
        let f = |jd: &[f64]| -> Vec<f64> { jd.iter().map(|&x| x.sin()).collect() };

        let results = find_minima(0.0, 4.0 * PI, &f, 0.5, EPSILON_EXTREMA, DEFAULT_NUM);

        assert!(
            results.len() >= 2,
            "Expected at least 2 minima, got {}",
            results.len()
        );

        for (_, val) in &results {
            assert_relative_eq!(*val, -1.0, epsilon = 0.01);
        }
    }

    #[test]
    fn test_find_maxima_no_maxima() {
        // Monotonically increasing function
        let f = |jd: &[f64]| -> Vec<f64> { jd.to_vec() };
        let results = find_maxima(0.0, 10.0, &f, 1.0, EPSILON_EXTREMA, DEFAULT_NUM);
        assert!(results.is_empty());
    }

    #[test]
    fn test_diff_sign_diff() {
        // [1, 3, 2] -> sign(diff) = [+1, -1] -> diff = [-2]
        let y = vec![1.0, 3.0, 2.0];
        let dsd = diff_sign_diff(&y);
        assert_eq!(dsd, vec![-2]);
    }

    #[test]
    fn test_diff_sign_diff_plateau() {
        // [1, 3, 3, 2] -> sign(diff) = [+1, 0, -1] -> diff = [-1, -1]
        let y = vec![1.0, 3.0, 3.0, 2.0];
        let dsd = diff_sign_diff(&y);
        assert_eq!(dsd, vec![-1, -1]);
    }
}
