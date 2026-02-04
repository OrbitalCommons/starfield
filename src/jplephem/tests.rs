//! Tests for the jplephem module

#[cfg(test)]
mod tests {
    use crate::jplephem::daf::DAF;
    use crate::jplephem::errors::Result;
    use crate::jplephem::kernel::SpiceKernel;
    use crate::jplephem::spk::SPK;

    fn test_data_path(filename: &str) -> String {
        format!("src/jplephem/test_data/{filename}")
    }

    #[test]
    fn test_daf_open_de421() -> Result<()> {
        let daf = DAF::open(test_data_path("de421.bsp"))?;
        assert_eq!(daf.locidw, "DAF/SPK");
        assert_eq!(daf.nd, 2);
        assert_eq!(daf.ni, 6);
        Ok(())
    }

    #[test]
    fn test_daf_summaries() -> Result<()> {
        let daf = DAF::open(test_data_path("de421.bsp"))?;
        let summaries = daf.summaries()?;
        assert_eq!(summaries.len(), 15, "DE421 should have 15 segments");
        for (_name, values) in &summaries {
            assert_eq!(
                values.len(),
                8,
                "Each summary should have 8 values (2 doubles + 6 ints)"
            );
        }
        Ok(())
    }

    #[test]
    fn test_daf_read_array() -> Result<()> {
        let daf = DAF::open(test_data_path("de421.bsp"))?;
        let array = daf.read_array(1, 10)?;
        assert_eq!(array.len(), 10);
        // Values should not all be zero (real data)
        assert!(
            array.iter().any(|&v| v != 0.0),
            "Array should contain non-zero data"
        );
        Ok(())
    }

    #[test]
    fn test_daf_map_array_matches_read_array() -> Result<()> {
        let daf = DAF::open(test_data_path("de421.bsp"))?;
        let arr1 = daf.read_array(1, 100)?;
        let arr2 = daf.map_array(1, 100)?;
        assert_eq!(arr1.len(), arr2.len());
        for (a, b) in arr1.iter().zip(arr2.iter()) {
            assert_eq!(a, b);
        }
        Ok(())
    }

    #[test]
    fn test_de421_load() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;
        assert_eq!(spk.daf.locidw, "DAF/SPK");
        assert_eq!(spk.segments.len(), 15, "DE421 should have 15 segments");
        Ok(())
    }

    #[test]
    fn test_de421_date_range() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;
        let min_jd = spk
            .segments
            .iter()
            .map(|s| s.start_jd)
            .fold(f64::MAX, f64::min);
        let max_jd = spk
            .segments
            .iter()
            .map(|s| s.end_jd)
            .fold(f64::MIN, f64::max);
        assert!(
            (min_jd - 2414864.50).abs() < 0.01,
            "Expected start ~2414864.50, got {min_jd}"
        );
        assert!(
            (max_jd - 2471184.50).abs() < 0.01,
            "Expected end ~2471184.50, got {max_jd}"
        );
        Ok(())
    }

    #[test]
    fn test_de421_segment_ids() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;

        let expected_pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (3, 301),
            (3, 399),
            (1, 199),
            (2, 299),
            (4, 499),
        ];

        for (center, target) in expected_pairs {
            assert!(
                spk.segments
                    .iter()
                    .any(|s| s.center == center && s.target == target),
                "Missing segment center={center}, target={target}"
            );
        }
        assert_eq!(spk.segments.len(), expected_pairs.len());
        Ok(())
    }

    #[test]
    fn test_get_segment() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;
        let seg = spk.get_segment(3, 301)?;
        assert_eq!(seg.center, 3);
        assert_eq!(seg.target, 301);
        assert!(spk.get_segment(999, 999).is_err());
        Ok(())
    }

    #[test]
    fn test_compute_returns_nonzero() -> Result<()> {
        let mut spk = SPK::open(test_data_path("de421.bsp"))?;
        // Earth barycenter at J2000 (TDB seconds = 0)
        let seg = spk.get_segment_mut(0, 3)?;
        let pos = seg.compute(0.0, 0.0)?;
        assert!(
            pos.x != 0.0 || pos.y != 0.0 || pos.z != 0.0,
            "Position should be nonzero"
        );
        assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
        Ok(())
    }

    #[test]
    fn test_compute_and_differentiate_returns_nonzero() -> Result<()> {
        let mut spk = SPK::open(test_data_path("de421.bsp"))?;
        let seg = spk.get_segment_mut(0, 3)?;
        let (pos, vel) = seg.compute_and_differentiate(0.0, 0.0)?;
        assert!(pos.norm() > 0.0, "Position should be nonzero");
        assert!(vel.norm() > 0.0, "Velocity should be nonzero");
        assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
        assert!(vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite());
        Ok(())
    }

    #[test]
    fn test_out_of_range_error() -> Result<()> {
        let mut spk = SPK::open(test_data_path("de421.bsp"))?;
        let seg = spk.get_segment_mut(0, 3)?;
        // Way in the future — should fail
        let result = seg.compute(1e15, 0.0);
        assert!(result.is_err());
        Ok(())
    }

    #[test]
    fn test_major_segments_compute_at_j2000() -> Result<()> {
        let mut spk = SPK::open(test_data_path("de421.bsp"))?;
        // Test the main planet barycenter segments (these all cover J2000)
        let pairs = [
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 9),
            (0, 10),
            (3, 301),
            (3, 399),
        ];
        for (center, target) in pairs {
            let seg = spk.get_segment_mut(center, target)?;
            let (pos, vel) = seg.compute_and_differentiate(0.0, 0.0)?;
            assert!(
                pos.norm() > 0.0,
                "Segment ({center}->{target}) position should be nonzero at J2000"
            );
            assert!(
                vel.norm() > 0.0,
                "Segment ({center}->{target}) velocity should be nonzero at J2000"
            );
        }
        Ok(())
    }

    #[test]
    fn test_kernel_chain_resolution() -> Result<()> {
        let kernel = SpiceKernel::open(test_data_path("de421.bsp"))?;

        // Earth should have a 2-segment chain: SSB->EMB + EMB->Earth
        let earth = kernel.get("earth")?;
        assert_eq!(earth.chain.len(), 2);
        assert_eq!(earth.chain[0], (0, 3));
        assert_eq!(earth.chain[1], (3, 399));

        // Moon: SSB->EMB + EMB->Moon
        let moon = kernel.get("moon")?;
        assert_eq!(moon.chain.len(), 2);
        assert_eq!(moon.chain[0], (0, 3));
        assert_eq!(moon.chain[1], (3, 301));

        // Sun: directly SSB->Sun
        let sun = kernel.get("sun")?;
        assert_eq!(sun.chain.len(), 1);
        assert_eq!(sun.chain[0], (0, 10));

        // Mars: SSB->Mars Barycenter + Mars Barycenter->Mars
        let mars = kernel.get("mars")?;
        assert_eq!(mars.chain.len(), 2);
        assert_eq!(mars.chain[0], (0, 4));
        assert_eq!(mars.chain[1], (4, 499));

        Ok(())
    }

    #[test]
    fn test_kernel_compute_at() -> Result<()> {
        use crate::time::Timescale;

        let mut kernel = SpiceKernel::open(test_data_path("de421.bsp"))?;
        let ts = Timescale::default();
        let t = ts.tdb_jd(2451545.0); // J2000

        let state = kernel.compute_at("earth", &t)?;

        // Earth should be roughly 1 AU from SSB
        let dist_au =
            (state.position.x.powi(2) + state.position.y.powi(2) + state.position.z.powi(2)).sqrt();
        assert!(
            dist_au > 0.9 && dist_au < 1.1,
            "Earth should be ~1 AU from SSB, got {dist_au}"
        );

        // Velocity should be roughly 30 km/s ≈ 0.0173 AU/day
        let speed_au_day = state.velocity.norm();
        assert!(
            speed_au_day > 0.01 && speed_au_day < 0.03,
            "Earth orbital speed should be ~0.017 AU/day, got {speed_au_day}"
        );

        Ok(())
    }

    #[test]
    fn test_kernel_numeric_id_lookup() -> Result<()> {
        let kernel = SpiceKernel::open(test_data_path("de421.bsp"))?;
        let earth = kernel.get("399")?;
        assert_eq!(earth.target_id, 399);
        Ok(())
    }

    #[test]
    fn test_segment_describe() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;
        let desc = spk.segments[0].describe(false);
        // Should contain date range and body names
        assert!(!desc.is_empty());
        assert!(desc.contains(".."));
        Ok(())
    }

    #[test]
    fn test_comments() -> Result<()> {
        let spk = SPK::open(test_data_path("de421.bsp"))?;
        // Just verify it doesn't crash — comment content varies
        let _ = spk.comments();
        Ok(())
    }
}
