//! Statistical primitives for astronomical data reduction.
//!
//! Two submodules:
//!
//! * [`circular`] — directional statistics: circular mean and spread, the
//!   Rayleigh and Kuiper uniformity tests, and seeded Monte Carlo
//!   joint-significance helpers. Useful for position angles, longitudes, and
//!   any other quantity that lives on the circle, where arithmetic means are
//!   wrong (the average of 1° and 359° is 0°, not 180°).
//! * [`timeseries`] — robust time-series statistics: a multi-origin
//!   mean-squared-displacement diffusion estimator and the median absolute
//!   deviation scale estimator.
//!
//! The most common functions are re-exported at this level.

pub mod circular;
pub mod timeseries;

pub use circular::{
    circular_mean, circular_std, kuiper_p_value, kuiper_statistic, mean_resultant_length,
    monte_carlo_joint_significance, monte_carlo_joint_significance_with, rayleigh_p_value,
    rayleigh_z, wrap_to_pi, JointSignificance,
};
pub use timeseries::{measure_diffusion, median_absolute_deviation, uniform_filter};
