//! Ephemeris and photometry for MPC minor planets
//!
//! Bridges [`MpcOrbRecord`] to starfield's Keplerian propagator and provides
//! the IAU HG magnitude model for computing apparent magnitudes.

use std::f64::consts::PI;

use nalgebra::Vector3;
use starfield_core::catalogs::StarData;
use starfield_core::constants::GM_SUN;
use starfield_core::keplerlib::{mpcorb_orbit, KeplerOrbit};
use starfield_core::positions::Position;
use starfield_core::time::{Time, Timescale};

use crate::mpc::mpcorb::{unpack_epoch, MpcOrbRecord};

impl MpcOrbRecord {
    /// Build a [`KeplerOrbit`] from this record's orbital elements.
    ///
    /// The orbit is heliocentric in the ecliptic plane, with the standard
    /// ECLIPJ2000-to-equatorial rotation applied (matching `mpcorb_orbit`).
    pub fn to_kepler_orbit(&self, ts: &Timescale) -> Option<KeplerOrbit> {
        let epoch = self.epoch_time(ts)?;
        Some(mpcorb_orbit(
            self.semimajor_axis,
            self.eccentricity,
            self.inclination,
            self.long_asc_node,
            self.arg_perihelion,
            self.mean_anomaly,
            &epoch,
            GM_SUN,
            Some(&self.readable_designation),
        ))
    }

    /// Compute the heliocentric equatorial position at a given time.
    ///
    /// Returns a [`Position`] in AU (ICRF equatorial frame, centered on the Sun).
    pub fn heliocentric_position(&self, ts: &Timescale, time: &Time) -> Option<Position> {
        let orbit = self.to_kepler_orbit(ts)?;
        Some(orbit.at(time))
    }

    /// Compute the apparent magnitude using the IAU HG phase function.
    ///
    /// # Arguments
    /// * `r_au` — heliocentric distance in AU
    /// * `delta_au` — observer distance in AU
    /// * `phase_angle_rad` — Sun-target-observer angle in radians
    ///
    /// Returns `None` if `h_magnitude` is not available.
    pub fn apparent_magnitude(
        &self,
        r_au: f64,
        delta_au: f64,
        phase_angle_rad: f64,
    ) -> Option<f64> {
        let h = self.h_magnitude?;
        let g = self.g_slope.unwrap_or(0.15);
        Some(hg_apparent_magnitude(h, g, r_au, delta_au, phase_angle_rad))
    }

    /// Compute RA/Dec and apparent magnitude as seen from an observer.
    ///
    /// Returns a [`StarData`] suitable for rendering pipelines, or `None`
    /// if the epoch cannot be unpacked or H magnitude is missing.
    ///
    /// # Arguments
    /// * `ts` — timescale for time conversions
    /// * `time` — observation epoch
    /// * `observer_pos_au` — observer's heliocentric equatorial position in AU
    pub fn to_star_data(
        &self,
        ts: &Timescale,
        time: &Time,
        observer_pos_au: &Vector3<f64>,
    ) -> Option<StarData> {
        let target = self.heliocentric_position(ts, time)?;
        let target_pos = target.position;

        // Vector from observer to target
        let rel = target_pos - observer_pos_au;
        let delta = rel.norm();
        let r = target_pos.norm();

        // RA/Dec from the relative vector
        let (ra_rad, dec_rad) = cartesian_to_radec(&rel);
        let ra_deg = ra_rad.to_degrees();
        let dec_deg = dec_rad.to_degrees();

        // Phase angle: angle at the target between Sun and observer
        let phase_angle = phase_angle_rad(&target_pos, observer_pos_au);

        let mag = self.apparent_magnitude(r, delta, phase_angle)?;

        // Use packed designation as a numeric ID (hash it)
        let id = designation_to_id(&self.designation);

        Some(StarData::new(id, ra_deg, dec_deg, mag, None))
    }

    /// Parse the packed epoch into a [`Time`] object.
    fn epoch_time(&self, ts: &Timescale) -> Option<Time> {
        let (year, month, day) = unpack_epoch(&self.epoch_packed)?;
        Some(ts.tt((year, month, day as u32)))
    }
}

/// IAU HG apparent magnitude model (Bowell et al. 1989).
///
/// `V = H + 5*log10(r * delta) - 2.5*log10((1-G)*Phi1(alpha) + G*Phi2(alpha))`
pub fn hg_apparent_magnitude(
    h: f64,
    g: f64,
    r_au: f64,
    delta_au: f64,
    phase_angle_rad: f64,
) -> f64 {
    let phi1 = bowell_phi(1, phase_angle_rad);
    let phi2 = bowell_phi(2, phase_angle_rad);
    let phase_correction = -2.5 * ((1.0 - g) * phi1 + g * phi2).log10();
    h + 5.0 * (r_au * delta_au).log10() + phase_correction
}

/// Bowell basis functions Phi_1 and Phi_2 for the HG system.
///
/// Phi_i(alpha) = exp(-A_i * tan(alpha/2)^B_i)
///
/// Coefficients from Bowell et al. (1989), Table III.
fn bowell_phi(index: u8, alpha: f64) -> f64 {
    let (a, b) = match index {
        1 => (3.332, 0.631),
        _ => (1.862, 1.218),
    };
    let tan_half = (alpha / 2.0).tan();
    (-a * tan_half.powf(b)).exp()
}

/// Convert a Cartesian ICRF vector to RA/Dec (radians).
fn cartesian_to_radec(v: &Vector3<f64>) -> (f64, f64) {
    let r = v.norm();
    let dec = (v.z / r).asin();
    let ra = v.y.atan2(v.x);
    let ra = if ra < 0.0 { ra + 2.0 * PI } else { ra };
    (ra, dec)
}

/// Phase angle: angle at the target between Sun direction and observer direction.
fn phase_angle_rad(target_helio: &Vector3<f64>, observer_helio: &Vector3<f64>) -> f64 {
    let sun_dir = -target_helio; // Sun as seen from target
    let obs_dir = observer_helio - target_helio; // Observer as seen from target
    let cos_phase = sun_dir.dot(&obs_dir) / (sun_dir.norm() * obs_dir.norm());
    cos_phase.clamp(-1.0, 1.0).acos()
}

/// Convert a packed MPC designation to a numeric ID.
fn designation_to_id(designation: &str) -> u64 {
    // Try to parse as a simple number first (numbered asteroids like "00001")
    if let Ok(n) = designation.trim().parse::<u64>() {
        return n;
    }
    // Fall back to a simple hash for provisional designations
    let mut hash: u64 = 5381;
    for b in designation.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mpc::mpcorb::parse_mpcorb_line;

    // Main belt — low ecc, low-to-moderate inclination
    const CERES_LINE: &str = "00001    3.35  0.15 K25BL 231.53975   73.29974   80.24963   10.58789  0.0795763  0.21429712   2.7656157  0 MPO964264  7384 126 1801-2026 0.69 M-v 30k MPCORBFIT  4000      (1) Ceres              20260103";
    const PALLAS_LINE: &str = "00002    4.11  0.15 K25BL 211.52977  310.93340  172.88859   34.92833  0.2306430  0.21379713   2.7699258  0 MPO964264  9023 124 1804-2025 0.64 M-c 28k MPCORBFIT  4000      (2) Pallas             20251214";
    const VESTA_LINE: &str = "00004    3.25  0.15 K25BL  26.80969  151.53711  103.70232    7.14406  0.0901676  0.27158812   2.3615413  0 MPO964264  7603 112 1821-2025 0.69 M-p 18k MPCORBFIT  4000      (4) Vesta              20250624";
    // NEA, very high eccentricity (e=0.827)
    const ICARUS_LINE: &str = "01566   16.53  0.15 K25BL 153.07893   31.43821   87.95243   22.80320  0.8270057  0.88054806   1.0780378  0 MPO950970  1842  43 1949-2025 0.63 M-v 3Ek MPCLINUX   9803   (1566) Icarus             20250818";
    // NEA, high inclination (i=64.0°)
    const TANTALUS_LINE: &str = "02102   16.00  0.15 K25BL 347.35567   61.49853   94.35226   64.00516  0.2993501  0.67264934   1.2900610  0 MPO964302  1837  25 1975-2025 0.62 M-v 3Ek MPCORBFIT  9803   (2102) Tantalus           20251209";

    fn ts() -> Timescale {
        Timescale::default()
    }

    #[test]
    fn test_ceres_kepler_orbit() {
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        let orbit = rec.to_kepler_orbit(&ts()).unwrap();
        assert!(orbit.target_name.as_deref().unwrap().contains("Ceres"));
    }

    #[test]
    fn test_ceres_heliocentric_distance_at_epoch() {
        let ts = ts();
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        let epoch = rec.epoch_time(&ts).unwrap();
        let pos = rec.heliocentric_position(&ts, &epoch).unwrap();
        let dist = pos.position.norm();
        // Ceres orbits at ~2.56-2.98 AU
        assert!(
            dist > 2.4 && dist < 3.1,
            "Ceres heliocentric distance at epoch should be ~2.5-3.0 AU, got {dist}"
        );
    }

    #[test]
    fn test_pallas_heliocentric_distance() {
        let ts = ts();
        let rec = parse_mpcorb_line(PALLAS_LINE).unwrap();
        let epoch = rec.epoch_time(&ts).unwrap();
        let pos = rec.heliocentric_position(&ts, &epoch).unwrap();
        let dist = pos.position.norm();
        // Pallas: a=2.77, e=0.23, so r ranges ~2.13-3.41
        assert!(
            dist > 2.0 && dist < 3.5,
            "Pallas heliocentric distance should be ~2.0-3.5 AU, got {dist}"
        );
    }

    #[test]
    fn test_hg_magnitude_at_opposition() {
        // At opposition: phase angle ≈ 0, r ≈ a-1, delta ≈ a-1
        // For Ceres: H=3.35, a=2.77 → r≈2.77, delta≈1.77
        let mag = hg_apparent_magnitude(3.35, 0.15, 2.77, 1.77, 0.0);
        // At opposition, Ceres reaches ~6.6-7.5 magnitude
        assert!(
            mag > 5.0 && mag < 9.0,
            "Ceres opposition magnitude should be ~6-8, got {mag}"
        );
    }

    #[test]
    fn test_hg_magnitude_increases_with_distance() {
        let mag_near = hg_apparent_magnitude(3.35, 0.15, 2.0, 1.0, 0.1);
        let mag_far = hg_apparent_magnitude(3.35, 0.15, 4.0, 3.0, 0.1);
        assert!(
            mag_far > mag_near,
            "Farther object should be dimmer: near={mag_near}, far={mag_far}"
        );
    }

    #[test]
    fn test_hg_magnitude_increases_with_phase() {
        let mag_low = hg_apparent_magnitude(3.35, 0.15, 2.77, 1.77, 0.1);
        let mag_high = hg_apparent_magnitude(3.35, 0.15, 2.77, 1.77, 1.0);
        assert!(
            mag_high > mag_low,
            "Higher phase should be dimmer: low={mag_low}, high={mag_high}"
        );
    }

    #[test]
    fn test_to_star_data() {
        let ts = ts();
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        let epoch = rec.epoch_time(&ts).unwrap();

        // Observer at Earth-like position (1 AU along x-axis)
        let observer = Vector3::new(1.0, 0.0, 0.0);
        let star = rec.to_star_data(&ts, &epoch, &observer).unwrap();

        assert!(star.magnitude > 0.0 && star.magnitude < 20.0);
        assert_eq!(star.id, 1); // Ceres = numbered asteroid 1
    }

    #[test]
    fn test_designation_to_id_numbered() {
        assert_eq!(designation_to_id("00001"), 1);
        assert_eq!(designation_to_id("00004"), 4);
        assert_eq!(designation_to_id("12345"), 12345);
    }

    #[test]
    fn test_phase_angle_opposition() {
        // Target at (3, 0, 0), observer at (1, 0, 0) → phase ≈ 0
        let target = Vector3::new(3.0, 0.0, 0.0);
        let observer = Vector3::new(1.0, 0.0, 0.0);
        let angle = phase_angle_rad(&target, &observer);
        assert!(
            angle < 0.01,
            "Opposition phase angle should be ~0, got {angle}"
        );
    }

    #[test]
    fn test_phase_angle_quadrature() {
        // For 90° phase angle, Sun-target-observer angle must be 90°.
        // Target at (1, 1, 0), observer at (2, 0, 0):
        //   Sun-dir from target = (-1, -1, 0)
        //   Obs-dir from target = (1, -1, 0)
        //   dot = -1 + 1 = 0 → angle = 90°
        let target = Vector3::new(1.0, 1.0, 0.0);
        let observer = Vector3::new(2.0, 0.0, 0.0);
        let angle = phase_angle_rad(&target, &observer);
        let expected = PI / 2.0;
        assert!(
            (angle - expected).abs() < 0.01,
            "Quadrature phase should be ~90°, got {:.1}°",
            angle.to_degrees()
        );
    }

    // -----------------------------------------------------------------------
    // HORIZONS comparison tests
    //
    // These compare our Keplerian propagation against JPL HORIZONS ephemeris
    // for known objects at specific dates. Two-body propagation is expected
    // to agree to ~0.01 AU over short arcs (weeks) and diverge over longer
    // periods due to perturbations from Jupiter et al.
    // -----------------------------------------------------------------------

    /// Compare Ceres position on 2026-01-01 against HORIZONS.
    ///
    /// HORIZONS vectors query (JPL solution #48, DE441):
    /// Target: 1 Ceres  Center: Sun [500@10]  Frame: ICRF
    /// JD 2461041.5 TDB = 2026-Jan-01 00:00:00.0000 TDB
    #[test]
    fn test_ceres_vs_horizons_2026_jan_01() {
        let ts = ts();
        let rec = parse_mpcorb_line(CERES_LINE).unwrap();
        let t = ts.tt((2026, 1, 1));
        let pos = rec.heliocentric_position(&ts, &t).unwrap();

        let horizons_x = 2.547982816654031;
        let horizons_y = 1.352191737738587;
        let horizons_z = 0.1190652834647933;

        // Two-body from epoch (~Nov 21 2025) to Jan 1 2026 is ~41 days.
        let tol = 0.005; // 0.005 AU ≈ 750,000 km
        assert!(
            (pos.position.x - horizons_x).abs() < tol,
            "Ceres X: got {}, expected {}, diff={}",
            pos.position.x,
            horizons_x,
            (pos.position.x - horizons_x).abs()
        );
        assert!(
            (pos.position.y - horizons_y).abs() < tol,
            "Ceres Y: got {}, expected {}, diff={}",
            pos.position.y,
            horizons_y,
            (pos.position.y - horizons_y).abs()
        );
        assert!(
            (pos.position.z - horizons_z).abs() < tol,
            "Ceres Z: got {}, expected {}, diff={}",
            pos.position.z,
            horizons_z,
            (pos.position.z - horizons_z).abs()
        );
    }

    /// Compare Vesta position on 2026-01-01 against HORIZONS.
    ///
    /// HORIZONS vectors query (JPL solution #36, DE441):
    /// Target: 4 Vesta  Center: Sun [500@10]  Frame: ICRF
    /// JD 2461041.5 TDB = 2026-Jan-01 00:00:00.0000 TDB
    #[test]
    fn test_vesta_vs_horizons_2026_jan_01() {
        let ts = ts();
        let rec = parse_mpcorb_line(VESTA_LINE).unwrap();
        let t = ts.tt((2026, 1, 1));
        let pos = rec.heliocentric_position(&ts, &t).unwrap();

        let horizons_x = 1.100732664743188;
        let horizons_y = -1.717210308829650;
        let horizons_z = -0.8289402050303827;

        // Vesta epoch is Nov 21 2025, so ~41 days propagation.
        let tol = 0.005;
        assert!(
            (pos.position.x - horizons_x).abs() < tol,
            "Vesta X: got {}, expected {}, diff={}",
            pos.position.x,
            horizons_x,
            (pos.position.x - horizons_x).abs()
        );
        assert!(
            (pos.position.y - horizons_y).abs() < tol,
            "Vesta Y: got {}, expected {}, diff={}",
            pos.position.y,
            horizons_y,
            (pos.position.y - horizons_y).abs()
        );
        assert!(
            (pos.position.z - horizons_z).abs() < tol,
            "Vesta Z: got {}, expected {}, diff={}",
            pos.position.z,
            horizons_z,
            (pos.position.z - horizons_z).abs()
        );
    }

    /// Compare Pallas position on 2026-01-01 against HORIZONS.
    ///
    /// HORIZONS vectors query (JPL solution #72, DE441):
    /// Target: 2 Pallas  Center: Sun [500@10]  Frame: ICRF
    /// JD 2461041.5 TDB = 2026-Jan-01 00:00:00.0000 TDB
    #[test]
    fn test_pallas_vs_horizons_2026_jan_01() {
        let ts = ts();
        let rec = parse_mpcorb_line(PALLAS_LINE).unwrap();
        let t = ts.tt((2026, 1, 1));
        let pos = rec.heliocentric_position(&ts, &t).unwrap();

        let horizons_x = 2.898022065781879;
        let horizons_y = -1.585772601027873;
        let horizons_z = 0.1063451739399872;

        // Pallas epoch is Nov 21 2025, ~41 days propagation.
        let tol = 0.005;
        assert!(
            (pos.position.x - horizons_x).abs() < tol,
            "Pallas X: got {}, expected {}, diff={}",
            pos.position.x,
            horizons_x,
            (pos.position.x - horizons_x).abs()
        );
        assert!(
            (pos.position.y - horizons_y).abs() < tol,
            "Pallas Y: got {}, expected {}, diff={}",
            pos.position.y,
            horizons_y,
            (pos.position.y - horizons_y).abs()
        );
        assert!(
            (pos.position.z - horizons_z).abs() < tol,
            "Pallas Z: got {}, expected {}, diff={}",
            pos.position.z,
            horizons_z,
            (pos.position.z - horizons_z).abs()
        );
    }

    /// Icarus: NEA with e=0.827 — exercises highly eccentric orbit propagation.
    /// Perihelion ~0.19 AU, aphelion ~1.97 AU.
    #[test]
    fn test_icarus_heliocentric_distance() {
        let ts = ts();
        let rec = parse_mpcorb_line(ICARUS_LINE).unwrap();
        let epoch = rec.epoch_time(&ts).unwrap();
        let pos = rec.heliocentric_position(&ts, &epoch).unwrap();
        let dist = pos.position.norm();
        // a=1.078, e=0.827 → perihelion=0.186, aphelion=1.970
        assert!(
            dist > 0.15 && dist < 2.1,
            "Icarus distance should be 0.19-1.97 AU, got {dist}"
        );
    }

    /// Tantalus: NEA with i=64° — exercises high-inclination orbit.
    #[test]
    fn test_tantalus_heliocentric_distance() {
        let ts = ts();
        let rec = parse_mpcorb_line(TANTALUS_LINE).unwrap();
        let epoch = rec.epoch_time(&ts).unwrap();
        let pos = rec.heliocentric_position(&ts, &epoch).unwrap();
        let dist = pos.position.norm();
        // a=1.290, e=0.299 → perihelion=0.904, aphelion=1.676
        assert!(
            dist > 0.8 && dist < 1.8,
            "Tantalus distance should be 0.9-1.7 AU, got {dist}"
        );
    }

    /// Compare Icarus (e=0.827) on 2026-01-01 against HORIZONS.
    ///
    /// HORIZONS vectors query (JPL solution #194, DE441):
    /// Target: 1566 Icarus  Center: Sun [500@10]  Frame: ICRF
    /// JD 2461041.5 TDB = 2026-Jan-01 00:00:00.0000 TDB
    #[test]
    fn test_icarus_vs_horizons_2026_jan_01() {
        let ts = ts();
        let rec = parse_mpcorb_line(ICARUS_LINE).unwrap();
        let t = ts.tt((2026, 1, 1));
        let pos = rec.heliocentric_position(&ts, &t).unwrap();

        let horizons_x = 0.9272316329869779;
        let horizons_y = -1.379476203634930;
        let horizons_z = -1.050260897170735;

        // High-ecc orbit: two-body should still be close over 41 days,
        // but perturbations may be larger near perihelion.
        let tol = 0.01;
        assert!(
            (pos.position.x - horizons_x).abs() < tol,
            "Icarus X: got {}, expected {}, diff={}",
            pos.position.x,
            horizons_x,
            (pos.position.x - horizons_x).abs()
        );
        assert!(
            (pos.position.y - horizons_y).abs() < tol,
            "Icarus Y: got {}, expected {}, diff={}",
            pos.position.y,
            horizons_y,
            (pos.position.y - horizons_y).abs()
        );
        assert!(
            (pos.position.z - horizons_z).abs() < tol,
            "Icarus Z: got {}, expected {}, diff={}",
            pos.position.z,
            horizons_z,
            (pos.position.z - horizons_z).abs()
        );
    }

    /// Compare Tantalus (i=64°) on 2026-01-01 against HORIZONS.
    ///
    /// HORIZONS vectors query (JPL solution #274, DE441):
    /// Target: 2102 Tantalus  Center: Sun [500@10]  Frame: ICRF
    /// JD 2461041.5 TDB = 2026-Jan-01 00:00:00.0000 TDB
    #[test]
    fn test_tantalus_vs_horizons_2026_jan_01() {
        let ts = ts();
        let rec = parse_mpcorb_line(TANTALUS_LINE).unwrap();
        let t = ts.tt((2026, 1, 1));
        let pos = rec.heliocentric_position(&ts, &t).unwrap();

        let horizons_x = -0.4063748762567138;
        let horizons_y = -0.3604771474080175;
        let horizons_z = 0.7546149775701417;

        let tol = 0.01;
        assert!(
            (pos.position.x - horizons_x).abs() < tol,
            "Tantalus X: got {}, expected {}, diff={}",
            pos.position.x,
            horizons_x,
            (pos.position.x - horizons_x).abs()
        );
        assert!(
            (pos.position.y - horizons_y).abs() < tol,
            "Tantalus Y: got {}, expected {}, diff={}",
            pos.position.y,
            horizons_y,
            (pos.position.y - horizons_y).abs()
        );
        assert!(
            (pos.position.z - horizons_z).abs() < tol,
            "Tantalus Z: got {}, expected {}, diff={}",
            pos.position.z,
            horizons_z,
            (pos.position.z - horizons_z).abs()
        );
    }
}
