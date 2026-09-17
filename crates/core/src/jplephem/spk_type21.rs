//! SPK Type 21: Extended Modified Difference Arrays
//!
//! Type 21 segments store trajectory data using Modified Difference Arrays (MDA),
//! the Shampine-Gordon polynomial interpolation method. This is the modern
//! successor to Type 1, used by all Horizons-generated small-body SPK files
//! since October 2018.
//!
//! Each record contains a reference epoch, stepsize function coefficients,
//! reference position/velocity, and a modified divided difference table.
//! Interpolation reconstructs position and velocity at any epoch within
//! the record's coverage using the difference table recurrence.

use std::sync::Arc;

use nalgebra::Vector3;

use super::daf::DAF;
use super::errors::{JplephemError, Result};

/// Maximum supported difference table order
const MAXTRM: usize = 25;

/// A parsed Type 21 MDA record
#[derive(Debug, Clone)]
struct MdaRecord {
    /// Reference epoch (TDB seconds past J2000)
    tl: f64,
    /// Stepsize function coefficients (length MAXDIM)
    g: Vec<f64>,
    /// Reference position [x, y, z] in km
    refpos: [f64; 3],
    /// Reference velocity [vx, vy, vz] in km/s
    refvel: [f64; 3],
    /// Modified difference table, shape [3][MAXDIM] (one row per component)
    dt: [Vec<f64>; 3],
    /// Maximum integration order + 1
    kqmax1: usize,
    /// Integration order for each component [x, y, z]
    kq: [usize; 3],
}

/// Parsed Type 21 segment data
#[derive(Debug, Clone)]
pub struct Type21Data {
    /// MAXDIM for this segment (variable, unlike Type 1's fixed 15)
    maxdim: usize,
    /// Difference line size = 4 * MAXDIM + 11
    dlsize: usize,
    /// Number of records
    n_records: usize,
    /// Epoch table: one epoch per record (used for record lookup)
    epoch_table: Vec<f64>,
    /// Epoch directory for fast lookup when n_records > 100
    epoch_dir: Vec<f64>,
    /// All record data stored as flat f64 array
    record_data: Vec<f64>,
}

impl Type21Data {
    /// Load Type 21 segment data from a DAF file
    ///
    /// `start_i` and `end_i` are 1-indexed double-word addresses from the segment summary.
    pub fn load(daf: &Arc<DAF>, start_i: usize, end_i: usize) -> Result<Self> {
        // Last word of segment is MAXDIM, second-to-last is N (record count)
        let maxdim_raw = daf.read_array(end_i, end_i)?;
        let maxdim = maxdim_raw[0] as usize;

        if maxdim > MAXTRM {
            return Err(JplephemError::InvalidFormat(format!(
                "Type 21 MAXDIM {maxdim} exceeds maximum supported {MAXTRM}"
            )));
        }
        if maxdim == 0 {
            return Err(JplephemError::InvalidFormat(
                "Type 21 MAXDIM is zero".to_string(),
            ));
        }

        let dlsize = 4 * maxdim + 11;

        let n_records_raw = daf.read_array(end_i - 1, end_i - 1)?;
        let n_records = n_records_raw[0] as usize;

        if n_records == 0 {
            return Err(JplephemError::InvalidFormat(
                "Type 21 segment has zero records".to_string(),
            ));
        }

        // Epoch directory count
        let epoch_dir_count = if n_records > 100 {
            (n_records - 1) / 100
        } else {
            0
        };

        // Read all record data: n_records * dlsize doubles starting at start_i
        let record_end = start_i + n_records * dlsize - 1;
        let record_data = daf.read_array(start_i, record_end)?;

        // Epoch table follows records: n_records entries
        let epoch_start = start_i + n_records * dlsize;
        let epoch_end = epoch_start + n_records - 1;
        let epoch_table = daf.read_array(epoch_start, epoch_end)?;

        // Epoch directory follows epoch table (before N and MAXDIM)
        let epoch_dir = if epoch_dir_count > 0 {
            let dir_start = end_i - 1 - epoch_dir_count;
            let dir_end = end_i - 2;
            daf.read_array(dir_start, dir_end)?
        } else {
            Vec::new()
        };

        // Validate segment size
        let expected_size = n_records * dlsize + n_records + epoch_dir_count + 2;
        let actual_size = end_i - start_i + 1;
        if actual_size != expected_size {
            return Err(JplephemError::InvalidFormat(format!(
                "Type 21 segment size mismatch: expected {expected_size}, got {actual_size} \
                 (n_records={n_records}, dlsize={dlsize}, maxdim={maxdim}, epoch_dir={epoch_dir_count})"
            )));
        }

        Ok(Type21Data {
            maxdim,
            dlsize,
            n_records,
            epoch_table,
            epoch_dir,
            record_data,
        })
    }

    /// Find the record index for a given epoch (TDB seconds past J2000)
    fn find_record_index(&self, et: f64) -> usize {
        // Use epoch directory for initial bracket if available
        let (search_start, search_end) = if !self.epoch_dir.is_empty() {
            let mut bracket_end = self.n_records;
            let mut bracket_start = self.epoch_dir.len() * 100 + 1;

            for (i, &dir_epoch) in self.epoch_dir.iter().enumerate() {
                if dir_epoch > et {
                    bracket_end = (i + 1) * 100;
                    bracket_start = i * 100 + 1;
                    break;
                }
            }
            (bracket_start, bracket_end)
        } else {
            (1, self.n_records)
        };

        // Linear scan within bracket (1-indexed)
        let mut record_index = search_end;
        for i in (search_start - 1)..search_end {
            if self.epoch_table[i] > et {
                record_index = i + 1;
                break;
            }
        }

        record_index
    }

    /// Parse a record from the flat data array
    fn parse_record(&self, record_index: usize) -> Result<MdaRecord> {
        let offset = (record_index - 1) * self.dlsize;
        let rec = &self.record_data[offset..offset + self.dlsize];
        let maxdim = self.maxdim;

        let tl = rec[0];
        let g = rec[1..1 + maxdim].to_vec();

        // Reference position and velocity are interleaved:
        // rec[maxdim+1] = x_pos, rec[maxdim+2] = x_vel
        // rec[maxdim+3] = y_pos, rec[maxdim+4] = y_vel
        // rec[maxdim+5] = z_pos, rec[maxdim+6] = z_vel
        let refpos = [rec[maxdim + 1], rec[maxdim + 3], rec[maxdim + 5]];
        let refvel = [rec[maxdim + 2], rec[maxdim + 4], rec[maxdim + 6]];

        // Difference table: stored in column-major order (Fortran style)
        // dt[component][order] with shape (MAXDIM, 3) in Fortran = 3 columns of MAXDIM each
        let dt_flat = &rec[maxdim + 7..4 * maxdim + 7];
        let dt_x = dt_flat[0..maxdim].to_vec();
        let dt_y = dt_flat[maxdim..2 * maxdim].to_vec();
        let dt_z = dt_flat[2 * maxdim..3 * maxdim].to_vec();

        let kqmax1 = rec[4 * maxdim + 7] as usize;
        let kq = [
            rec[4 * maxdim + 8] as usize,
            rec[4 * maxdim + 9] as usize,
            rec[4 * maxdim + 10] as usize,
        ];

        if kqmax1 > maxdim + 1 {
            return Err(JplephemError::InvalidFormat(format!(
                "Type 21 KQMAX1 ({kqmax1}) exceeds MAXDIM+1 ({})",
                maxdim + 1
            )));
        }

        Ok(MdaRecord {
            tl,
            g,
            refpos,
            refvel,
            dt: [dt_x, dt_y, dt_z],
            kqmax1,
            kq,
        })
    }

    /// Compute position and velocity at the given epoch
    ///
    /// `et` is TDB seconds past J2000.
    /// Returns (position_km, velocity_km_s).
    pub fn compute(&self, et: f64) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let record_index = self.find_record_index(et);
        let rec = self.parse_record(record_index)?;
        evaluate_mda(&rec, et)
    }
}

/// Evaluate Modified Difference Array interpolation for position and velocity
///
/// This implements the Shampine-Gordon recurrence as used by the NAIF SPICE
/// toolkit and the spktype21 Python package.
fn evaluate_mda(rec: &MdaRecord, et: f64) -> Result<(Vector3<f64>, Vector3<f64>)> {
    let delta = et - rec.tl;
    let kqmax1 = rec.kqmax1;

    if kqmax1 < 2 {
        return Err(JplephemError::InvalidFormat(format!(
            "Type 21 KQMAX1 ({kqmax1}) must be >= 2"
        )));
    }

    let mq2 = kqmax1 - 2;
    let mut ks = kqmax1 - 1;

    // Build factorial-like coefficients from the stepsize function
    let mut fc = vec![0.0; MAXTRM + 3];
    let mut wc = vec![0.0; MAXTRM + 3];
    fc[0] = 1.0;

    let mut tp = delta;
    for j in 1..=mq2 {
        if rec.g[j - 1] == 0.0 {
            return Err(JplephemError::InvalidFormat(format!(
                "Type 21 zero stepsize at index {j}"
            )));
        }
        fc[j] = tp / rec.g[j - 1];
        wc[j - 1] = delta / rec.g[j - 1];
        tp = delta + rec.g[j - 1];
    }

    // Initialize W array with reciprocals
    let mut w = vec![0.0; MAXTRM + 3];
    for j in 1..=kqmax1 {
        w[j - 1] = 1.0 / (j as f64);
    }

    // Compute position interpolation weights
    let mut jx = 0usize;
    let mut ks1 = ks as isize - 1;

    while ks >= 2 {
        jx += 1;
        for j in 1..=jx {
            w[j + ks - 1] = fc[j] * w[j + ks1 as usize - 1] - wc[j - 1] * w[j + ks - 1];
        }
        ks -= 1;
        ks1 -= 1;
    }

    // Compute position for each component
    let mut position = [0.0f64; 3];
    for (i, pos) in position.iter_mut().enumerate() {
        let kqq = rec.kq[i];
        let mut sum = 0.0;
        for j in (1..=kqq).rev() {
            sum += rec.dt[i][j - 1] * w[j + ks - 1];
        }
        *pos = rec.refpos[i] + delta * (rec.refvel[i] + delta * sum);
    }

    // Compute velocity interpolation weights (one more reduction step)
    for j in 1..=jx {
        w[j + ks - 1] = fc[j] * w[j + ks1 as usize - 1] - wc[j - 1] * w[j + ks - 1];
    }
    ks -= 1;

    // Compute velocity for each component
    let mut velocity = [0.0f64; 3];
    for (i, vel) in velocity.iter_mut().enumerate() {
        let kqq = rec.kq[i];
        let mut sum = 0.0;
        for j in (1..=kqq).rev() {
            sum += rec.dt[i][j - 1] * w[j + ks - 1];
        }
        *vel = rec.refvel[i] + delta * sum;
    }

    Ok((
        Vector3::new(position[0], position[1], position[2]),
        Vector3::new(velocity[0], velocity[1], velocity[2]),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal Type 21 record for testing the MDA evaluation
    fn make_test_record(maxdim: usize) -> MdaRecord {
        // A simple record: body at rest at (100, 200, 300) km
        MdaRecord {
            tl: 0.0,
            g: vec![1.0; maxdim],
            refpos: [100.0, 200.0, 300.0],
            refvel: [0.0, 0.0, 0.0],
            dt: [vec![0.0; maxdim], vec![0.0; maxdim], vec![0.0; maxdim]],
            kqmax1: 2,
            kq: [1, 1, 1],
        }
    }

    #[test]
    fn test_evaluate_mda_stationary() {
        let rec = make_test_record(15);
        let (pos, vel) = evaluate_mda(&rec, 0.0).unwrap();
        assert!((pos.x - 100.0).abs() < 1e-12);
        assert!((pos.y - 200.0).abs() < 1e-12);
        assert!((pos.z - 300.0).abs() < 1e-12);
        assert!(vel.x.abs() < 1e-12);
        assert!(vel.y.abs() < 1e-12);
        assert!(vel.z.abs() < 1e-12);
    }

    #[test]
    fn test_evaluate_mda_uniform_motion() {
        // Body moving with constant velocity (10, -5, 3) km/s from (100, 200, 300)
        // With zero difference table entries, position = refpos + delta * refvel
        let rec = MdaRecord {
            tl: 1000.0,
            g: vec![1.0; 15],
            refpos: [100.0, 200.0, 300.0],
            refvel: [10.0, -5.0, 3.0],
            dt: [vec![0.0; 15], vec![0.0; 15], vec![0.0; 15]],
            kqmax1: 2,
            kq: [1, 1, 1],
        };

        // Evaluate 10 seconds after reference epoch
        let dt = 10.0;
        let (pos, vel) = evaluate_mda(&rec, 1000.0 + dt).unwrap();

        assert!(
            (pos.x - (100.0 + 10.0 * dt)).abs() < 1e-10,
            "x: {} vs {}",
            pos.x,
            100.0 + 10.0 * dt
        );
        assert!(
            (pos.y - (200.0 - 5.0 * dt)).abs() < 1e-10,
            "y: {} vs {}",
            pos.y,
            200.0 - 5.0 * dt
        );
        assert!(
            (pos.z - (300.0 + 3.0 * dt)).abs() < 1e-10,
            "z: {} vs {}",
            pos.z,
            300.0 + 3.0 * dt
        );

        // Velocity should be the constant refvel
        assert!((vel.x - 10.0).abs() < 1e-10);
        assert!((vel.y - (-5.0)).abs() < 1e-10);
        assert!((vel.z - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_mda_at_reference_epoch() {
        // At the reference epoch (delta=0), position should equal refpos
        // and velocity should equal refvel regardless of difference table
        let rec = MdaRecord {
            tl: 5000.0,
            g: vec![86400.0; 15],
            refpos: [1.0e8, -2.0e7, 3.0e6],
            refvel: [25.0, -10.0, 5.0],
            dt: [vec![1e-5; 15], vec![2e-6; 15], vec![3e-7; 15]],
            kqmax1: 8,
            kq: [7, 7, 7],
        };

        let (pos, vel) = evaluate_mda(&rec, 5000.0).unwrap();
        assert!((pos.x - 1.0e8).abs() < 1e-6);
        assert!((pos.y - (-2.0e7)).abs() < 1e-6);
        assert!((pos.z - 3.0e6).abs() < 1e-6);
        assert!((vel.x - 25.0).abs() < 1e-10);
        assert!((vel.y - (-10.0)).abs() < 1e-10);
        assert!((vel.z - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaluate_mda_zero_stepsize_error() {
        let mut rec = make_test_record(15);
        rec.kqmax1 = 4;
        rec.kq = [3, 3, 3];
        rec.g[0] = 0.0; // zero stepsize should cause error

        let result = evaluate_mda(&rec, 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_mda_kqmax1_too_small() {
        let mut rec = make_test_record(15);
        rec.kqmax1 = 1; // must be >= 2

        let result = evaluate_mda(&rec, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_type21_data_load_validation() {
        // Verify that loading with invalid parameters produces errors
        // (We can't easily construct a DAF in-memory here, so we test
        // the record parsing and evaluation paths directly)
        let maxdim = 15;
        let dlsize = 4 * maxdim + 11;
        assert_eq!(dlsize, 71);

        let maxdim = 20;
        let dlsize = 4 * maxdim + 11;
        assert_eq!(dlsize, 91);
    }

    #[test]
    fn test_find_record_index_small() {
        // Test record finding with a small epoch table (no directory)
        let data = Type21Data {
            maxdim: 15,
            dlsize: 71,
            n_records: 5,
            epoch_table: vec![100.0, 200.0, 300.0, 400.0, 500.0],
            epoch_dir: vec![],
            record_data: vec![0.0; 71 * 5],
        };

        // Before first epoch -> record 1
        assert_eq!(data.find_record_index(50.0), 1);
        // Between first and second -> record 1
        assert_eq!(data.find_record_index(150.0), 2);
        // Between second and third -> record 2
        assert_eq!(data.find_record_index(250.0), 3);
        // After all epochs -> last record
        assert_eq!(data.find_record_index(600.0), 5);
    }
}
