//! Survey definitions: footprint, epoch grid, and linking requirement as
//! plain data.

use crate::surveylib::detection::LinkingRequirement;
use crate::surveylib::footprint::Footprint;
use crate::time::Time;

/// One observing epoch of a survey: a time on the survey's per-position
/// visit grid, with an optional epoch-specific depth.
///
/// The grid is *per sky position*: each epoch is one visit of a given
/// field, so a survey definition describes the cadence a typical position
/// in the footprint receives (e.g. "ten nights per band over six years"),
/// not the survey's full exposure log.
#[derive(Debug, Clone)]
pub struct Epoch {
    /// Observation time.
    pub time: Time,
    /// Epoch-specific 50%-completeness magnitude, overriding the
    /// [`DetectionModel`](crate::surveylib::DetectionModel)'s `m50` —
    /// use this for per-band depths or per-night observing conditions.
    /// `None` falls back to the model's depth.
    pub m50: Option<f64>,
}

impl Epoch {
    /// An epoch observed at the detection model's default depth.
    pub fn new(time: Time) -> Self {
        Self { time, m50: None }
    }

    /// An epoch with its own 50%-completeness depth (e.g. a different
    /// band or degraded conditions).
    pub fn with_depth(time: Time, m50: f64) -> Self {
        Self {
            time,
            m50: Some(m50),
        }
    }
}

/// A survey as data: name, equatorial footprint, per-position epoch grid,
/// and the pipeline's linking requirement.
///
/// The detection efficiency model is deliberately *not* part of the
/// definition — it is a separate, explicitly-fitted
/// [`DetectionModel`](crate::surveylib::DetectionModel) passed alongside
/// (see the [module docs](crate::surveylib) on fitted parameters), so the
/// same survey geometry can be evaluated under different calibrations.
#[derive(Debug, Clone)]
pub struct SurveyDefinition {
    /// Human-readable survey name.
    pub name: String,
    /// Sky coverage in equatorial coordinates.
    pub footprint: Footprint,
    /// Observing epochs a typical footprint position receives.
    pub epochs: Vec<Epoch>,
    /// Detections required across the epochs to count as found.
    pub linking: LinkingRequirement,
}

impl SurveyDefinition {
    /// The epoch times, in grid order — the argument to hand to
    /// [`earth_positions_from_ephemeris`](crate::surveylib::geometry::earth_positions_from_ephemeris)
    /// or
    /// [`earth_positions_with`](crate::surveylib::geometry::earth_positions_with)
    /// when precomputing per-epoch Earth states (once per survey, shared
    /// by every orbit).
    pub fn epoch_times(&self) -> Vec<Time> {
        self.epochs.iter().map(|e| e.time.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Timescale;

    #[test]
    fn epoch_depth_override_is_optional() {
        let ts = Timescale::default();
        let t = ts.tt_jd(2_456_000.0, None);
        assert_eq!(Epoch::new(t.clone()).m50, None);
        assert_eq!(Epoch::with_depth(t, 23.5).m50, Some(23.5));
    }

    #[test]
    fn epoch_times_preserve_grid_order() {
        let ts = Timescale::default();
        let survey = SurveyDefinition {
            name: "toy".to_string(),
            footprint: Footprint::AllSky,
            epochs: (0..5)
                .map(|k| Epoch::new(ts.tt_jd(2_456_000.0 + 10.0 * k as f64, None)))
                .collect(),
            linking: LinkingRequirement::KOfN { k: 2 },
        };
        let times = survey.epoch_times();
        assert_eq!(times.len(), 5);
        for (k, t) in times.iter().enumerate() {
            assert!((t.tt() - (2_456_000.0 + 10.0 * k as f64)).abs() < 1e-9);
        }
    }
}
