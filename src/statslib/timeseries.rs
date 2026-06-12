//! Robust time-series statistics.
//!
//! Diffusion estimation from a mean-squared-displacement fit and the median
//! absolute deviation as a robust scale estimator.
//!
//! The diffusion coefficient is measured from a least-squares fit of the
//! mean-squared displacement MSD(τ) = ⟨(x(t+τ) − x(t))²⟩_t over all time
//! origins (ensemble-of-origins average), D = MSD/τ slope. A max-over-t of
//! (Δx)²/t from a single origin overestimates — the running supremum of a
//! random walk grows like t·log log t.

/// Measure the diffusion coefficient (units of `series`² per unit time) from
/// a time series sampled every `dt` time units.
///
/// A boxcar filter of `window_size` samples first removes short-period
/// oscillations; the MSD is then averaged over all time origins for lags
/// up to a quarter of the series, and D is the least-squares slope of
/// MSD(τ) = D·τ through the origin.
///
/// Returns 0.0 for series shorter than 4 samples.
///
/// # Example
///
/// ```
/// use starfield::statslib::measure_diffusion;
///
/// // A constant series does not diffuse.
/// let series = vec![100.0; 100];
/// assert!(measure_diffusion(&series, 1.0, 5).abs() < 1e-10);
/// ```
pub fn measure_diffusion(series: &[f64], dt: f64, window_size: usize) -> f64 {
    if series.len() < 4 {
        return 0.0;
    }

    let smoothed = uniform_filter(series, window_size);
    let n = smoothed.len();
    let max_lag = (n / 4).max(1);

    let mut sum_msd_tau = 0.0;
    let mut sum_tau_sq = 0.0;
    for lag in 1..=max_lag {
        let tau = lag as f64 * dt;
        let mut acc = 0.0;
        let mut count = 0usize;
        for t in 0..(n - lag) {
            let dx = smoothed[t + lag] - smoothed[t];
            acc += dx * dx;
            count += 1;
        }
        let msd = acc / count as f64;
        sum_msd_tau += msd * tau;
        sum_tau_sq += tau * tau;
    }

    if sum_tau_sq <= 0.0 {
        return 0.0;
    }
    sum_msd_tau / sum_tau_sq
}

/// Simple uniform (boxcar) filter for a 1D series.
///
/// Each output sample is the mean of a window of `window` samples centered
/// on the input sample (truncated at the series boundaries). A window of 0,
/// 1, or longer than the series returns the input unchanged.
///
/// # Example
///
/// ```
/// use starfield::statslib::uniform_filter;
///
/// let data = [0.0, 0.0, 3.0, 0.0, 0.0];
/// let smoothed = uniform_filter(&data, 3);
/// assert!((smoothed[2] - 1.0).abs() < 1e-12);
/// ```
pub fn uniform_filter(data: &[f64], window: usize) -> Vec<f64> {
    if window == 0 || data.len() < window {
        return data.to_vec();
    }

    let half_w = window / 2;
    let n = data.len();
    let mut filtered = Vec::with_capacity(n);

    for i in 0..n {
        let lo = i.saturating_sub(half_w);
        let hi = (i + half_w + 1).min(n);
        let sum: f64 = data[lo..hi].iter().sum();
        filtered.push(sum / (hi - lo) as f64);
    }

    filtered
}

/// Median of a slice (averaging the two central elements for even n).
fn median(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// Median absolute deviation, scaled by 1.4826 to be a consistent
/// estimator of σ for Gaussian data.
///
/// Returns 0.0 for an empty slice.
///
/// # Example
///
/// ```
/// use starfield::statslib::median_absolute_deviation;
///
/// // median = 3, |dev| = [2, 1, 0, 1, 2] → median 1 → 1.4826
/// let mad = median_absolute_deviation(&[1.0, 2.0, 3.0, 4.0, 5.0]);
/// assert!((mad - 1.4826).abs() < 1e-12);
/// ```
pub fn median_absolute_deviation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = median(&sorted);
    let mut devs: Vec<f64> = values.iter().map(|&v| (v - med).abs()).collect();
    devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    1.4826 * median(&devs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    /// Standard normal draws via Box-Muller (avoids a rand_distr dependency).
    fn normal_draws(rng: &mut impl Rng, n: usize, mean: f64, sigma: f64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let u1: f64 = rng.random::<f64>().max(f64::MIN_POSITIVE);
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
                mean + sigma * z
            })
            .collect()
    }

    #[test]
    fn test_measure_diffusion_constant() {
        let series = vec![100.0; 100];
        let d = measure_diffusion(&series, 1e6, 5);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_measure_diffusion_recovers_random_walk() {
        // Random walk with step σ per sample: true D = σ²/dt
        // (MSD(τ) = σ²·τ/dt). The MSD fit must land within ~2× without
        // the systematic high bias of a max-over-t estimator.
        let mut rng = rand::rngs::StdRng::seed_from_u64(12345);
        let sigma = 0.05_f64;
        let dt = 1000.0;
        let steps = normal_draws(&mut rng, 4000, 0.0, sigma);
        let mut x = 150.0;
        let series: Vec<f64> = steps
            .iter()
            .map(|s| {
                x += s;
                x
            })
            .collect();
        let d_true = sigma * sigma / dt;
        let d = measure_diffusion(&series, dt, 1);
        assert!(
            d > 0.4 * d_true && d < 2.5 * d_true,
            "D = {d:.3e}, true = {d_true:.3e}"
        );
    }

    #[test]
    fn test_uniform_filter_identity() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let filtered = uniform_filter(&data, 1);
        for (a, b) in data.iter().zip(filtered.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn test_median_absolute_deviation() {
        // values: median = 3, |dev| = [2, 1, 0, 1, 2] → median 1 → 1.4826
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mad = median_absolute_deviation(&values);
        assert!((mad - 1.4826).abs() < 1e-12, "MAD = {mad}");
        // For Gaussian draws, MAD ≈ σ.
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let draws = normal_draws(&mut rng, 20_000, 10.0, 2.0);
        let mad_g = median_absolute_deviation(&draws);
        assert!((mad_g - 2.0).abs() < 0.1, "Gaussian MAD = {mad_g}");
    }

    #[test]
    fn test_median_absolute_deviation_empty() {
        assert_eq!(median_absolute_deviation(&[]), 0.0);
    }
}
