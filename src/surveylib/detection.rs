//! Per-epoch detection efficiency and multi-epoch linking.
//!
//! Two layers:
//!
//! 1. [`DetectionModel`] — the probability that a *single* exposure of a
//!    source at apparent magnitude `m` yields a catalogued detection,
//!    modeled as the logistic completeness curve used throughout the
//!    survey literature.
//! 2. [`LinkingRequirement`] — how many independent per-epoch detections
//!    a survey's pipeline needs before it links them into a discoverable
//!    object (a tracklet/orbit). The k-of-n probability is computed with
//!    the **exact** Poisson-binomial tail ([`poisson_binomial_tail`]),
//!    never a binomial approximation, so heterogeneous per-epoch
//!    probabilities (different bands, depths, or footprint gating) are
//!    handled exactly.
//!
//! # Every parameter here is fitted, not derived
//!
//! `m50`, `steepness`, and `ceiling` are calibration knobs measured by
//! injecting synthetic sources into a real pipeline (or tuned to
//! reproduce a published recovery fraction). They are not derivable from
//! first principles, and values fitted for one survey/pipeline are
//! meaningless for another. Keep your survey's numbers next to their
//! provenance, and flag any parameter tuned to match an end-to-end result
//! as *fitted* in its documentation.

/// Survival probability `P(X >= k)` of a Poisson-binomial random variable:
/// the number of successes among independent Bernoulli trials with
/// heterogeneous probabilities `probs`.
///
/// Computed by exact dynamic programming over the count distribution in
/// `O(n·k)` time and `O(k)` space: `dist[j]` tracks `P(exactly j successes
/// so far)` for `j < k`, and `tail` absorbs `P(>= k)` as trials are folded
/// in. No sampling, no normal/Poisson approximation.
///
/// Edge cases: `k = 0` returns 1 exactly; `k > probs.len()` returns 0.
///
/// # Example
///
/// ```
/// use starfield::surveylib::poisson_binomial_tail;
///
/// // Homogeneous case reduces to the binomial: n = 10, p = 0.5,
/// // P(X >= 6) = 386/1024.
/// let tail = poisson_binomial_tail(&[0.5; 10], 6);
/// assert!((tail - 386.0 / 1024.0).abs() < 1e-12);
///
/// // P(X >= 1) = 1 - prod(1 - p_i).
/// let any = poisson_binomial_tail(&[0.1, 0.4, 0.7], 1);
/// assert!((any - (1.0 - 0.9 * 0.6 * 0.3)).abs() < 1e-12);
/// ```
pub fn poisson_binomial_tail(probs: &[f64], k: u32) -> f64 {
    let k = k as usize;
    if k == 0 {
        return 1.0;
    }
    // dist[j] = P(exactly j successes so far) for j < k; `tail` absorbs
    // P(>= k successes).
    let mut dist = vec![0.0_f64; k];
    dist[0] = 1.0;
    let mut tail = 0.0_f64;
    for &p in probs {
        tail += dist[k - 1] * p;
        for j in (1..k).rev() {
            dist[j] = dist[j] * (1.0 - p) + dist[j - 1] * p;
        }
        dist[0] *= 1.0 - p;
    }
    tail
}

/// Logistic single-epoch detection completeness:
///
/// ```text
/// p(m) = ceiling / (1 + exp(steepness · (m − m50)))
/// ```
///
/// `m50` is the magnitude of 50%-of-ceiling completeness, `steepness`
/// sets the transition width (the curve falls from 0.73·ceiling to
/// 0.27·ceiling over `2/steepness` magnitudes), and `ceiling` is the
/// bright-end asymptote (real pipelines lose a few percent of even the
/// brightest sources to masking, blends, and chip gaps, so `ceiling < 1`).
///
/// **All three parameters are fitted calibration values** — see the
/// [module docs](self) — typically measured per band by synthetic-source
/// injection. Model one band per `DetectionModel`; multi-band surveys
/// carry one model (or one per-epoch [`m50`](crate::surveylib::Epoch)
/// override) per band's epochs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionModel {
    /// Magnitude at which completeness is half the ceiling (fitted).
    pub m50: f64,
    /// Logistic slope in 1/mag (fitted); larger is a sharper cutoff.
    pub steepness: f64,
    /// Bright-end asymptotic completeness in `[0, 1]` (fitted).
    pub ceiling: f64,
}

impl DetectionModel {
    /// New logistic model; see the field docs for parameter meanings.
    pub fn new(m50: f64, steepness: f64, ceiling: f64) -> Self {
        Self {
            m50,
            steepness,
            ceiling,
        }
    }

    /// Single-epoch detection probability at apparent magnitude `mag`.
    pub fn efficiency(&self, mag: f64) -> f64 {
        self.efficiency_at_depth(mag, self.m50)
    }

    /// Single-epoch detection probability with the 50%-completeness depth
    /// overridden to `m50` (e.g. a per-epoch depth from varying observing
    /// conditions or a different band); `steepness` and `ceiling` are
    /// retained from the model.
    pub fn efficiency_at_depth(&self, mag: f64, m50: f64) -> f64 {
        self.ceiling / (1.0 + (self.steepness * (mag - m50)).exp())
    }
}

/// How many per-epoch detections a survey pipeline requires before the
/// object counts as found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkingRequirement {
    /// Detections on at least `k` of the survey's epochs, with each epoch
    /// an independent Bernoulli trial at its own probability. `k = 1`
    /// degenerates to "any single detection suffices"; multi-night
    /// linking pipelines use larger `k` (a tracklet needs enough
    /// detections to constrain an orbit).
    KOfN {
        /// Minimum number of detected epochs.
        k: u32,
    },
}

impl LinkingRequirement {
    /// Probability that the requirement is met, given independent
    /// per-epoch detection probabilities (exact Poisson-binomial tail).
    pub fn probability(&self, epoch_probs: &[f64]) -> f64 {
        match self {
            LinkingRequirement::KOfN { k } => poisson_binomial_tail(epoch_probs, *k),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisson_binomial_tail_matches_binomial() {
        let probs = vec![0.5; 10];
        let tail = poisson_binomial_tail(&probs, 6);
        assert!((tail - 386.0 / 1024.0).abs() < 1e-12, "tail = {tail}");
        // Degenerate edges.
        assert_eq!(poisson_binomial_tail(&probs, 0), 1.0);
        assert!(poisson_binomial_tail(&probs, 11) < 1e-12);
        assert_eq!(poisson_binomial_tail(&[], 0), 1.0);
        assert_eq!(poisson_binomial_tail(&[], 1), 0.0);
    }

    #[test]
    fn poisson_binomial_tail_heterogeneous() {
        // P(X >= 1) = 1 - prod(1 - p_i).
        let probs = [0.1, 0.4, 0.7];
        let tail = poisson_binomial_tail(&probs, 1);
        let expected = 1.0 - 0.9 * 0.6 * 0.3;
        assert!((tail - expected).abs() < 1e-12);
        // P(X >= 3) = prod(p_i).
        let all = poisson_binomial_tail(&probs, 3);
        assert!((all - 0.1 * 0.4 * 0.7).abs() < 1e-12);
        // Exhaustive middle term: P(X >= 2) by enumeration.
        let p2 = 0.1 * 0.4 * 0.3 + 0.1 * 0.6 * 0.7 + 0.9 * 0.4 * 0.7 + 0.1 * 0.4 * 0.7;
        assert!((poisson_binomial_tail(&probs, 2) - p2).abs() < 1e-12);
    }

    #[test]
    fn poisson_binomial_tail_monotone_in_k() {
        let probs = [0.9, 0.2, 0.5, 0.8, 0.3];
        let mut prev = 1.0;
        for k in 0..=6 {
            let t = poisson_binomial_tail(&probs, k);
            assert!(t <= prev + 1e-15, "tail not monotone at k = {k}");
            assert!((0.0..=1.0).contains(&t));
            prev = t;
        }
    }

    #[test]
    fn logistic_efficiency_shape() {
        let model = DetectionModel::new(24.0, 3.0, 0.95);
        // Half the ceiling exactly at m50.
        assert!((model.efficiency(24.0) - 0.475).abs() < 1e-12);
        // Bright end approaches the ceiling, never exceeds it.
        assert!(model.efficiency(18.0) > 0.94);
        assert!(model.efficiency(18.0) <= 0.95);
        // Faint end falls to zero.
        assert!(model.efficiency(28.0) < 1e-4);
        // Monotone decreasing.
        let mut prev = model.efficiency(15.0);
        for tenth in 151..=300 {
            let p = model.efficiency(tenth as f64 / 10.0);
            assert!(p <= prev + 1e-15);
            prev = p;
        }
    }

    #[test]
    fn per_epoch_depth_override() {
        let model = DetectionModel::new(24.0, 3.0, 1.0);
        // Overriding the depth shifts the curve, keeping shape parameters.
        assert!((model.efficiency_at_depth(22.0, 22.0) - 0.5).abs() < 1e-12);
        assert_eq!(
            model.efficiency_at_depth(20.0, 24.0),
            model.efficiency(20.0)
        );
    }

    #[test]
    fn linking_k_of_n() {
        let probs = [0.5; 10];
        let link = LinkingRequirement::KOfN { k: 6 };
        assert!((link.probability(&probs) - 386.0 / 1024.0).abs() < 1e-12);
        // k = 1 is the complement of total non-detection.
        let any = LinkingRequirement::KOfN { k: 1 };
        assert!((any.probability(&[0.25, 0.25]) - (1.0 - 0.75 * 0.75)).abs() < 1e-12);
    }
}
