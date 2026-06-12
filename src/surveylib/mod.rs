//! Survey-agnostic detection and completeness simulation.
//!
//! Answers the question *"would survey X have detected object Y?"* — and
//! its population form, *"what fraction of this population would have been
//! found?"* — for any imaging survey describable as a footprint, an epoch
//! grid, a magnitude-efficiency curve, and a linking rule. The machinery
//! was extracted from Planet Nine survey-exclusion analyses (ZTF, DES,
//! Pan-STARRS) but contains no survey-specific numbers; concrete survey
//! definitions (real footprints, fitted depths, cadences) belong
//! downstream, next to their provenance.
//!
//! # The pipeline
//!
//! 1. **Geometry** ([`geometry`]): per-epoch apparent geocentric
//!    equatorial positions from heliocentric *ecliptic* J2000 target and
//!    Earth states. Earth states come from a real ephemeris kernel
//!    ([`geometry::earth_positions_from_ephemeris`]) or any
//!    caller-supplied model ([`geometry::earth_positions_with`]),
//!    precomputed once per survey grid. The ecliptic→equatorial rotation
//!    is the typed [`framelib`](crate::framelib) transform — declination
//!    is never approximated by ecliptic latitude.
//! 2. **Footprint** ([`footprint`]): equatorial membership tests with
//!    exact solid angles (declination bands, box unions, custom
//!    predicates).
//! 3. **Detection** ([`detection`]): logistic per-epoch magnitude
//!    efficiency, then the pipeline's k-of-n linking requirement via the
//!    exact Poisson-binomial tail over the heterogeneous per-epoch
//!    probabilities.
//! 4. **Accumulation** (this module): [`detection_probability`] per
//!    orbit, [`expected_completeness`] over a population —
//!    *deterministic* sums of probabilities, never Bernoulli draws, so
//!    results carry no Monte-Carlo noise — and [`combined_detection`] /
//!    [`unique_contributions`] for the per-orbit OR across several
//!    surveys evaluated on one shared reference population.
//!
//! Apparent magnitudes are the caller's job: supply them per epoch in
//! [`EpochState`], e.g. from
//! [`magnitudelib::small_body`](crate::magnitudelib::small_body)
//! (mass→radius→H→`m = H + 5 log10(rΔ)`, optionally with the IAU H-G
//! phase term). No survey's depth or color model is hard-wired here.
//!
//! # Fitted parameters, not derived ones
//!
//! Completeness models are *calibrated*, not deduced: `m50`, `steepness`,
//! and `ceiling` in [`DetectionModel`] come from synthetic-source
//! injection into a real pipeline, and published recovery fractions are
//! often reproduced only after tuning an extra efficiency knob (per-night
//! usability, weather, focal-plane fill). When you build a concrete
//! survey definition downstream, document every such value as **fitted**,
//! with its source — a completeness number is only as honest as its least
//! documented calibration parameter.
//!
//! # Example
//!
//! ```
//! use starfield::surveylib::{
//!     detection_probability, expected_completeness, DetectionModel, Epoch, EpochState,
//!     Footprint, LinkingRequirement, SurveyDefinition,
//! };
//! use starfield::time::Timescale;
//!
//! let ts = Timescale::default();
//! // A toy survey: northern sky, four epochs, detections on >= 2 epochs
//! // required for linking. (Real definitions live downstream, with
//! // documented fitted parameters.)
//! let survey = SurveyDefinition {
//!     name: "toy-north".to_string(),
//!     footprint: Footprint::DecRange {
//!         dec_min_deg: 0.0,
//!         dec_max_deg: 90.0,
//!     },
//!     epochs: (0..4)
//!         .map(|k| Epoch::new(ts.tt_jd(2_456_000.0 + 90.0 * k as f64, None)))
//!         .collect(),
//!     linking: LinkingRequirement::KOfN { k: 2 },
//! };
//! let model = DetectionModel::new(23.0, 3.0, 0.95);
//!
//! // Per-epoch apparent states of one bright northern object (normally
//! // computed via surveylib::geometry from orbit + Earth states).
//! let states: Vec<EpochState> = (0..4)
//!     .map(|k| EpochState {
//!         ra_deg: 30.0 + 0.1 * k as f64,
//!         dec_deg: 20.0,
//!         apparent_mag: 19.0,
//!     })
//!     .collect();
//! let p = detection_probability(&survey, &model, &states);
//! assert!(p > 0.99);
//!
//! // Deterministic population completeness: the mean per-orbit
//! // probability, no Bernoulli sampling.
//! let completeness = expected_completeness([p, 0.0, 0.5]);
//! assert!((completeness - (p + 0.5) / 3.0).abs() < 1e-12);
//! ```

pub mod detection;
pub mod footprint;
pub mod geometry;
pub mod survey;

pub use detection::{poisson_binomial_tail, DetectionModel, LinkingRequirement};
pub use footprint::{Footprint, SkyBox, FULL_SKY_DEG2};
pub use geometry::{
    apparent_equatorial, apparent_equatorial_light_time, earth_positions_from_ephemeris,
    earth_positions_with, ApparentPosition,
};
pub use survey::{Epoch, SurveyDefinition};

/// The apparent state of one target at one survey epoch: equatorial
/// position (degrees) and apparent magnitude in the epoch's band.
///
/// Produce these per epoch from your orbit model and Earth states (see
/// [`geometry`]) and your photometry (see
/// [`magnitudelib::small_body`](crate::magnitudelib::small_body)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochState {
    /// Apparent right ascension (degrees).
    pub ra_deg: f64,
    /// Apparent declination (degrees).
    pub dec_deg: f64,
    /// Apparent magnitude in the band this epoch is observed in.
    pub apparent_mag: f64,
}

/// Per-epoch detection probabilities of one target: the logistic
/// magnitude efficiency (with any per-epoch depth override) gated by
/// footprint membership of the epoch's *apparent* position — an epoch
/// whose position falls outside the footprint contributes exactly 0.
///
/// `states` must align one-to-one with `survey.epochs`.
///
/// # Panics
///
/// Panics if `states.len() != survey.epochs.len()`.
pub fn epoch_detection_probabilities(
    survey: &SurveyDefinition,
    model: &DetectionModel,
    states: &[EpochState],
) -> Vec<f64> {
    assert_eq!(
        states.len(),
        survey.epochs.len(),
        "one EpochState per survey epoch (survey '{}' has {} epochs)",
        survey.name,
        survey.epochs.len()
    );
    survey
        .epochs
        .iter()
        .zip(states)
        .map(|(epoch, state)| {
            if survey.footprint.contains(state.ra_deg, state.dec_deg) {
                let m50 = epoch.m50.unwrap_or(model.m50);
                model.efficiency_at_depth(state.apparent_mag, m50)
            } else {
                0.0
            }
        })
        .collect()
}

/// End-to-end probability that the survey detects *and links* the target:
/// the linking requirement's probability (exact Poisson-binomial tail for
/// k-of-n) over the per-epoch probabilities of
/// [`epoch_detection_probabilities`].
///
/// # Panics
///
/// Panics if `states.len() != survey.epochs.len()`.
pub fn detection_probability(
    survey: &SurveyDefinition,
    model: &DetectionModel,
    states: &[EpochState],
) -> f64 {
    let probs = epoch_detection_probabilities(survey, model, states);
    survey.linking.probability(&probs)
}

/// Expected completeness over a population: the mean of per-orbit
/// detection probabilities, accumulated **deterministically** (a sum of
/// probabilities, not Bernoulli realizations), so the result is exact for
/// the given population — no Monte-Carlo noise term. Returns 0 for an
/// empty population.
///
/// Feed it [`detection_probability`] evaluated per orbit; for a
/// non-detection survey this same number is the expected *excluded*
/// fraction of the population.
pub fn expected_completeness(orbit_probabilities: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut n = 0u64;
    for p in orbit_probabilities {
        sum += p;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Per-orbit OR across surveys: the probability that *at least one*
/// survey detects the orbit, `1 − Π(1 − pᵢ)`, from the same orbit's
/// detection probability in each survey.
///
/// This is the rigorous multi-survey combination — evaluate every survey
/// on one shared reference population and average this per-orbit OR —
/// as opposed to combining published population-level fractions, which is
/// only valid when they are already disjoint ("unique") contributions.
pub fn combined_detection(per_survey_probabilities: &[f64]) -> f64 {
    1.0 - per_survey_probabilities
        .iter()
        .map(|p| 1.0 - p)
        .product::<f64>()
}

/// Decompose the per-orbit OR into *unique* survey contributions in the
/// given order: element `i` is `pᵢ · Π_{j<i}(1 − pⱼ)`, the probability
/// that survey `i` detects the orbit and no earlier survey does. The
/// elements sum to [`combined_detection`] exactly.
pub fn unique_contributions(per_survey_probabilities: &[f64]) -> Vec<f64> {
    let mut none_so_far = 1.0;
    per_survey_probabilities
        .iter()
        .map(|&p| {
            let unique = p * none_so_far;
            none_so_far *= 1.0 - p;
            unique
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalogs::synthetic::orbits::{InclinationModel, SyntheticOrbitPopulation};
    use crate::constants::GM_SUN;
    use crate::keplerlib::KeplerOrbit;
    use crate::time::{Time, Timescale};
    use nalgebra::Vector3;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Heliocentric *ecliptic* Keplerian orbit (no equatorial rotation):
    /// `at(t).position` stays in the frame the elements are given in.
    fn ecliptic_kepler(
        a: f64,
        e: f64,
        i_deg: f64,
        omega_big_deg: f64,
        omega_deg: f64,
        m_deg: f64,
        epoch: &Time,
    ) -> KeplerOrbit {
        KeplerOrbit::from_mean_anomaly(
            a * (1.0 - e * e),
            e,
            i_deg,
            omega_big_deg,
            omega_deg,
            m_deg,
            epoch,
            GM_SUN,
            Some(10),
            None,
        )
    }

    /// Analytic circular coplanar 1-AU Earth (phase zero at `t0_tt`) —
    /// a caller-supplied model, exercising the closure path.
    fn circular_earth(t0_tt: f64) -> impl FnMut(&Time) -> Vector3<f64> {
        move |t: &Time| {
            let theta = std::f64::consts::TAU * (t.tt() - t0_tt) / 365.25;
            Vector3::new(theta.cos(), theta.sin(), 0.0)
        }
    }

    fn toy_survey(
        ts: &Timescale,
        footprint: Footprint,
        n_epochs: usize,
        k: u32,
    ) -> SurveyDefinition {
        SurveyDefinition {
            name: "toy".to_string(),
            footprint,
            epochs: (0..n_epochs)
                .map(|j| Epoch::new(ts.tt_jd(2_456_000.0 + 180.0 * j as f64, None)))
                .collect(),
            linking: LinkingRequirement::KOfN { k },
        }
    }

    /// Per-epoch apparent states of a synthetic orbit at constant
    /// apparent magnitude, through the real geometry chain.
    fn states_for_orbit(
        orbit: &KeplerOrbit,
        survey: &SurveyDefinition,
        earth_positions: &[Vector3<f64>],
        mag: f64,
    ) -> Vec<EpochState> {
        survey
            .epochs
            .iter()
            .zip(earth_positions)
            .map(|(epoch, earth)| {
                let helio = orbit.at(&epoch.time).position;
                let pos = apparent_equatorial(&helio, earth);
                EpochState {
                    ra_deg: pos.ra_deg,
                    dec_deg: pos.dec_deg,
                    apparent_mag: mag,
                }
            })
            .collect()
    }

    /// End-to-end: a full-sky survey deep enough to see everything finds
    /// (essentially) the whole synthetic population.
    #[test]
    fn full_sky_deep_survey_complete_on_bright_population() {
        let ts = Timescale::default();
        let survey = toy_survey(&ts, Footprint::AllSky, 8, 3);
        let model = DetectionModel::new(24.0, 3.0, 1.0);
        let earths = earth_positions_with(circular_earth(2_456_000.0), &survey.epoch_times());

        let mut rng = StdRng::seed_from_u64(42);
        let population = SyntheticOrbitPopulation::scattered_disk()
            .with_count(50)
            .generate(&mut rng)
            .unwrap();

        let epoch0 = ts.tt_jd(2_456_000.0, None);
        let completeness = expected_completeness(population.iter().map(|o| {
            let orbit = ecliptic_kepler(
                o.a,
                o.e,
                o.i_deg,
                o.omega_big_deg,
                o.omega_deg,
                o.mean_anomaly_deg,
                &epoch0,
            );
            let states = states_for_orbit(&orbit, &survey, &earths, 18.0);
            detection_probability(&survey, &model, &states)
        }));
        assert!(completeness > 0.999, "completeness = {completeness}");
    }

    /// End-to-end: a magnitude-limited survey sees (essentially) none of
    /// a population fainter than its depth.
    #[test]
    fn shallow_survey_blind_to_faint_population() {
        let ts = Timescale::default();
        let survey = toy_survey(&ts, Footprint::AllSky, 8, 3);
        let model = DetectionModel::new(20.0, 3.0, 1.0);
        let earths = earth_positions_with(circular_earth(2_456_000.0), &survey.epoch_times());

        let mut rng = StdRng::seed_from_u64(42);
        let population = SyntheticOrbitPopulation::scattered_disk()
            .with_count(50)
            .generate(&mut rng)
            .unwrap();

        let epoch0 = ts.tt_jd(2_456_000.0, None);
        let completeness = expected_completeness(population.iter().map(|o| {
            let orbit = ecliptic_kepler(
                o.a,
                o.e,
                o.i_deg,
                o.omega_big_deg,
                o.omega_deg,
                o.mean_anomaly_deg,
                &epoch0,
            );
            let states = states_for_orbit(&orbit, &survey, &earths, 26.0);
            detection_probability(&survey, &model, &states)
        }));
        assert!(completeness < 1e-6, "completeness = {completeness}");
    }

    /// k-of-n with a hand-computable Poisson-binomial expectation: at
    /// mag = m50 with ceiling 1 every epoch is an exact fair coin, so a
    /// 6-of-10 requirement gives the binomial tail 386/1024.
    #[test]
    fn k_of_n_matches_hand_computed_binomial_tail() {
        let ts = Timescale::default();
        let survey = toy_survey(&ts, Footprint::AllSky, 10, 6);
        let model = DetectionModel::new(22.0, 3.0, 1.0);
        let states: Vec<EpochState> = (0..10)
            .map(|_| EpochState {
                ra_deg: 100.0,
                dec_deg: 0.0,
                apparent_mag: 22.0,
            })
            .collect();
        let p = detection_probability(&survey, &model, &states);
        assert!((p - 386.0 / 1024.0).abs() < 1e-12, "p = {p}");

        // One epoch outside the footprint becomes a structural zero:
        // 6 of the 9 remaining fair coins, P = C(9,>=6)/2^9 = 130/512.
        let mut survey_north = survey;
        survey_north.footprint = Footprint::DecRange {
            dec_min_deg: -10.0,
            dec_max_deg: 90.0,
        };
        let mut states_one_out = states;
        states_one_out[3].dec_deg = -45.0;
        let p9 = detection_probability(&survey_north, &model, &states_one_out);
        assert!((p9 - 130.0 / 512.0).abs() < 1e-12, "p = {p9}");
    }

    /// Declination-footprint regression for the beta-for-delta bug class:
    /// an ecliptic-plane (i = 0) object has ecliptic latitude ~0
    /// everywhere, but its *declination* swings +/- the obliquity, so a
    /// northern declination cut must pass it near ecliptic longitude 90
    /// deg and reject it near 270 deg. Conflating latitude with
    /// declination zeroes both (and the planar-population completeness).
    #[test]
    fn declination_footprint_distinguishes_ecliptic_longitudes() {
        let ts = Timescale::default();
        let footprint = Footprint::DecRange {
            dec_min_deg: 5.0,
            dec_max_deg: 90.0,
        };
        let survey = toy_survey(&ts, footprint, 1, 1);
        let model = DetectionModel::new(24.0, 3.0, 1.0);
        let epoch0 = ts.tt_jd(2_456_000.0, None);
        // Heliocentric view (Earth at the Sun) isolates the frame
        // transform from parallax.
        let earths = vec![Vector3::zeros()];

        // Near-circular planar orbit at ecliptic longitude ~90 deg:
        // dec ~ +23.4. (e is tiny but nonzero: the universal-variable
        // Kepler solver is singular at exactly e = 0.)
        let north = ecliptic_kepler(400.0, 1e-6, 0.0, 0.0, 0.0, 90.0, &epoch0);
        let states = states_for_orbit(&north, &survey, &earths, 18.0);
        assert!(states[0].dec_deg > 20.0, "dec = {}", states[0].dec_deg);
        assert!(detection_probability(&survey, &model, &states) > 0.99);

        // Same orbit at longitude ~270 deg: dec ~ -23.4, outside the cut.
        let south = ecliptic_kepler(400.0, 1e-6, 0.0, 0.0, 0.0, 270.0, &epoch0);
        let states = states_for_orbit(&south, &survey, &earths, 18.0);
        assert!(states[0].dec_deg < -20.0, "dec = {}", states[0].dec_deg);
        assert_eq!(detection_probability(&survey, &model, &states), 0.0);

        // Population level: a planar population is found at the fraction
        // of longitudes whose declination clears the cut, ~0.43 for
        // dec > 5 deg — emphatically not 0 (the conflated result, beta = 0
        // for every planar orbit) and not 1.
        let mut rng = StdRng::seed_from_u64(7);
        let population = SyntheticOrbitPopulation::scattered_disk()
            .with_count(400)
            .with_inclination(InclinationModel::Planar)
            .generate(&mut rng)
            .unwrap();
        let completeness = expected_completeness(population.iter().map(|o| {
            let orbit = ecliptic_kepler(
                o.a,
                o.e,
                o.i_deg,
                o.omega_big_deg,
                o.omega_deg,
                o.mean_anomaly_deg,
                &epoch0,
            );
            let states = states_for_orbit(&orbit, &survey, &earths, 18.0);
            detection_probability(&survey, &model, &states)
        }));
        assert!(
            (0.3..0.55).contains(&completeness),
            "planar-population completeness = {completeness}"
        );
    }

    #[test]
    fn per_epoch_depth_overrides_apply() {
        let ts = Timescale::default();
        let mut survey = toy_survey(&ts, Footprint::AllSky, 2, 1);
        // Second epoch is two magnitudes shallower.
        survey.epochs[1].m50 = Some(20.0);
        let model = DetectionModel::new(22.0, 3.0, 1.0);
        let state = EpochState {
            ra_deg: 10.0,
            dec_deg: 10.0,
            apparent_mag: 22.0,
        };
        let probs = epoch_detection_probabilities(&survey, &model, &[state, state]);
        assert!((probs[0] - 0.5).abs() < 1e-12);
        assert!(probs[1] < 0.01, "shallow epoch p = {}", probs[1]);
    }

    #[test]
    fn expected_completeness_is_deterministic_mean() {
        assert_eq!(expected_completeness(std::iter::empty::<f64>()), 0.0);
        assert!((expected_completeness([0.2, 0.4, 0.9]) - 0.5).abs() < 1e-12);
        // Repeated evaluation is bit-identical (no RNG anywhere).
        let probs = [0.123, 0.456, 0.789, 0.0, 1.0];
        assert_eq!(
            expected_completeness(probs).to_bits(),
            expected_completeness(probs).to_bits()
        );
    }

    #[test]
    fn combined_detection_is_per_orbit_or() {
        assert_eq!(combined_detection(&[]), 0.0);
        let p = combined_detection(&[0.5, 0.5]);
        assert!((p - 0.75).abs() < 1e-12);
        // Certain detection by one survey dominates.
        assert!((combined_detection(&[1.0, 0.1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn unique_contributions_sum_to_combined() {
        let probs = [0.564, 0.3, 0.5];
        let unique = unique_contributions(&probs);
        assert_eq!(unique.len(), 3);
        // First survey keeps its full probability.
        assert!((unique[0] - 0.564).abs() < 1e-12);
        // Later surveys are credited only with orbits no earlier survey saw.
        assert!((unique[1] - 0.3 * (1.0 - 0.564)).abs() < 1e-12);
        let sum: f64 = unique.iter().sum();
        assert!((sum - combined_detection(&probs)).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "one EpochState per survey epoch")]
    fn mismatched_state_count_panics() {
        let ts = Timescale::default();
        let survey = toy_survey(&ts, Footprint::AllSky, 3, 1);
        let model = DetectionModel::new(22.0, 3.0, 1.0);
        let state = EpochState {
            ra_deg: 0.0,
            dec_deg: 0.0,
            apparent_mag: 20.0,
        };
        detection_probability(&survey, &model, &[state]);
    }

    /// The geometry chain against a real DE421 Earth: ephemeris and
    /// analytic circular Earth paths must agree to the parallax scale.
    #[test]
    fn ephemeris_and_analytic_earth_agree_on_detection() {
        let Ok(kernel) = crate::jplephem::SpiceKernel::open("test_data/de421.bsp") else {
            eprintln!("skipping: test_data/de421.bsp not available");
            return;
        };
        let mut ephemeris = crate::planetlib::Ephemeris::from_kernel(kernel);
        let ts = Timescale::default();
        let survey = toy_survey(&ts, Footprint::AllSky, 6, 2);
        let times = survey.epoch_times();
        let earths_eph = earth_positions_from_ephemeris(&mut ephemeris, &times).unwrap();
        let earths_circ = earth_positions_with(circular_earth(2_456_000.0), &times);

        let epoch0 = ts.tt_jd(2_456_000.0, None);
        let orbit = ecliptic_kepler(400.0, 0.2, 17.0, 100.0, 150.0, 60.0, &epoch0);
        let helio = orbit.at(&epoch0).position;
        for (e_eph, e_circ) in earths_eph.iter().zip(&earths_circ) {
            let p_eph = apparent_equatorial(&helio, e_eph);
            let p_circ = apparent_equatorial(&helio, e_circ);
            // The analytic Earth's phase is arbitrary, so the apparent
            // positions can differ by up to the full parallax amplitude,
            // 2 asin(1/r) ~ 0.3 deg at r >~ 320 AU - but no more.
            let dra = (p_eph.ra_deg - p_circ.ra_deg).abs().min(360.0);
            let ddec = (p_eph.dec_deg - p_circ.dec_deg).abs();
            assert!(dra < 0.4 && ddec < 0.4, "dra = {dra:.3}, ddec = {ddec:.3}");
        }
    }
}
