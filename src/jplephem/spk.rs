//! Spacecraft Planet Kernel (SPK) format reader
//!
//! Reads NASA SPICE SPK files containing position and velocity data
//! for solar system bodies, stored as Chebyshev polynomial coefficients.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nalgebra::Vector3;

use crate::jplephem::calendar::calendar_date_from_float;
use crate::jplephem::chebyshev::{normalize_time, rescale_derivative, ChebyshevPolynomial};
use crate::jplephem::daf::DAF;
use crate::jplephem::errors::{JplephemError, Result};
use crate::jplephem::names::get_target_name;

/// Type 2 record coefficients: (midpoint, radius, x_coeffs, y_coeffs, z_coeffs)
type Type2Coeffs = (f64, f64, Vec<f64>, Vec<f64>, Vec<f64>);
/// Type 3 record coefficients: (midpoint, radius, (pos_xyz), (vel_xyz))
type Type3Coeffs = (
    f64,
    f64,
    (Vec<f64>, Vec<f64>, Vec<f64>),
    (Vec<f64>, Vec<f64>, Vec<f64>),
);

/// J2000 epoch as Julian date
const T0: f64 = 2451545.0;
/// Seconds per day
const S_PER_DAY: f64 = 86400.0;

/// Convert seconds since J2000 to Julian date
pub fn seconds_to_jd(seconds: f64) -> f64 {
    T0 + seconds / S_PER_DAY
}

/// Convert Julian date to seconds since J2000
pub fn jd_to_seconds(jd: f64) -> f64 {
    (jd - T0) * S_PER_DAY
}

/// Spacecraft Planet Kernel (SPK) file reader
pub struct SPK {
    /// The underlying DAF file
    pub daf: Arc<DAF>,
    /// Segments in the file
    pub segments: Vec<Segment>,
    /// Map of (center, target) pairs to segment indices
    pairs: HashMap<(i32, i32), usize>,
}

/// A segment containing ephemeris data for a specific body pair
pub struct Segment {
    daf: Arc<DAF>,
    /// Segment source label
    pub source: String,
    /// Start epoch in TDB seconds since J2000
    pub start_second: f64,
    /// End epoch in TDB seconds since J2000
    pub end_second: f64,
    /// Target body NAIF ID
    pub target: i32,
    /// Center body NAIF ID
    pub center: i32,
    /// Reference frame ID
    pub frame: i32,
    /// SPK data type (2=position only, 3=position+velocity)
    pub data_type: i32,
    /// Start index in file (1-indexed double-words)
    pub start_i: usize,
    /// End index in file (1-indexed double-words)
    pub end_i: usize,
    /// Start Julian date
    pub start_jd: f64,
    /// End Julian date
    pub end_jd: f64,
    /// Cached segment data
    data: Option<SegmentData>,
}

/// Cached coefficient data for a segment
#[derive(Clone)]
struct SegmentData {
    init: f64,
    intlen: f64,
    coefficients: Vec<f64>,
    /// (n_records, n_components, n_coeffs_per_component)
    shape: (usize, usize, usize),
    record_size: usize,
    data_type: i32,
}

impl SPK {
    /// Open an SPK file at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let daf = Arc::new(DAF::open(path)?);

        let mut spk = SPK {
            daf,
            segments: Vec::new(),
            pairs: HashMap::new(),
        };

        spk.parse_segments()?;
        Ok(spk)
    }

    fn parse_segments(&mut self) -> Result<()> {
        let summaries = self.daf.summaries()?;

        for (name, values) in summaries.iter() {
            if values.len() < (self.daf.nd + self.daf.ni) as usize {
                continue;
            }

            let source = String::from_utf8_lossy(name).trim_end().to_string();

            let start_second = values[0];
            let end_second = values[1];
            let target = values[2] as i32;
            let center = values[3] as i32;
            let frame = values[4] as i32;
            let data_type = values[5] as i32;
            let start_i = values[6] as usize;
            let end_i = values[7] as usize;

            // Basic validation
            if start_i == 0 || end_i < start_i {
                continue;
            }
            if data_type != 2 && data_type != 3 {
                continue;
            }

            let start_jd = seconds_to_jd(start_second);
            let end_jd = seconds_to_jd(end_second);

            let segment = Segment {
                daf: Arc::clone(&self.daf),
                source,
                start_second,
                end_second,
                target,
                center,
                frame,
                data_type,
                start_i,
                end_i,
                start_jd,
                end_jd,
                data: None,
            };

            let idx = self.segments.len();
            self.pairs.insert((center, target), idx);
            self.segments.push(segment);
        }

        Ok(())
    }

    /// Get the segment for the given center and target body IDs
    pub fn get_segment(&self, center: i32, target: i32) -> Result<&Segment> {
        self.pairs
            .get(&(center, target))
            .map(|&idx| &self.segments[idx])
            .ok_or(JplephemError::BodyNotFound { center, target })
    }

    /// Get a mutable reference to the segment for the given body pair
    pub fn get_segment_mut(&mut self, center: i32, target: i32) -> Result<&mut Segment> {
        let idx = *self
            .pairs
            .get(&(center, target))
            .ok_or(JplephemError::BodyNotFound { center, target })?;
        Ok(&mut self.segments[idx])
    }

    /// Read comments from the underlying DAF file
    pub fn comments(&self) -> Result<String> {
        self.daf.comments()
    }
}

impl Segment {
    /// Compute position (km) at the given time
    ///
    /// `tdb` and `tdb2` are TDB seconds since J2000, split for precision.
    pub fn compute(&mut self, tdb: f64, tdb2: f64) -> Result<Vector3<f64>> {
        let (position, _) = self.compute_and_differentiate(tdb, tdb2)?;
        Ok(position)
    }

    /// Compute position (km) and velocity (km/s) at the given time
    ///
    /// `tdb` and `tdb2` are TDB seconds since J2000, split for precision.
    pub fn compute_and_differentiate(
        &mut self,
        tdb: f64,
        tdb2: f64,
    ) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let et = tdb + tdb2;

        if et < self.start_second || et > self.end_second {
            return Err(JplephemError::OutOfRangeError {
                jd: seconds_to_jd(et),
                start_jd: self.start_jd,
                end_jd: self.end_jd,
            });
        }

        let data = self.load_data()?;
        let record_index = Self::find_record_index(et, data.init, data.intlen, data.shape.0)?;

        match data.data_type {
            2 => {
                let (record_mid, record_radius, coeffs_x, coeffs_y, coeffs_z) =
                    Self::get_record_coefficients_type2(
                        &data.coefficients,
                        record_index,
                        data.record_size,
                        data.shape.2,
                    )?;

                let t = normalize_time(et, record_mid, record_radius)?;

                let poly_x = ChebyshevPolynomial::new(coeffs_x);
                let poly_y = ChebyshevPolynomial::new(coeffs_y);
                let poly_z = ChebyshevPolynomial::new(coeffs_z);

                let position =
                    Vector3::new(poly_x.evaluate(t), poly_y.evaluate(t), poly_z.evaluate(t));

                let velocity = Vector3::new(
                    rescale_derivative(poly_x.derivative(t), record_radius)?,
                    rescale_derivative(poly_y.derivative(t), record_radius)?,
                    rescale_derivative(poly_z.derivative(t), record_radius)?,
                );

                Ok((position, velocity))
            }
            3 => {
                let (record_mid, record_radius, pos_coeffs, vel_coeffs) =
                    Self::get_record_coefficients_type3(
                        &data.coefficients,
                        record_index,
                        data.record_size,
                        data.shape.2,
                    )?;

                let t = normalize_time(et, record_mid, record_radius)?;

                let poly_x = ChebyshevPolynomial::new(pos_coeffs.0);
                let poly_y = ChebyshevPolynomial::new(pos_coeffs.1);
                let poly_z = ChebyshevPolynomial::new(pos_coeffs.2);

                let poly_vx = ChebyshevPolynomial::new(vel_coeffs.0);
                let poly_vy = ChebyshevPolynomial::new(vel_coeffs.1);
                let poly_vz = ChebyshevPolynomial::new(vel_coeffs.2);

                let position =
                    Vector3::new(poly_x.evaluate(t), poly_y.evaluate(t), poly_z.evaluate(t));

                let velocity = Vector3::new(
                    rescale_derivative(poly_vx.evaluate(t), record_radius)?,
                    rescale_derivative(poly_vy.evaluate(t), record_radius)?,
                    rescale_derivative(poly_vz.evaluate(t), record_radius)?,
                );

                Ok((position, velocity))
            }
            _ => Err(JplephemError::UnsupportedDataType(data.data_type)),
        }
    }

    fn find_record_index(et: f64, init: f64, intlen: f64, n_records: usize) -> Result<usize> {
        let elapsed = et - init;
        if elapsed < 0.0 {
            return Err(JplephemError::OutOfRangeError {
                jd: seconds_to_jd(et),
                start_jd: seconds_to_jd(init),
                end_jd: seconds_to_jd(init + intlen * n_records as f64),
            });
        }
        let mut index = (elapsed / intlen).floor() as usize;
        // Clamp to last record for times at the exact end of the range
        if index >= n_records {
            if index == n_records {
                index = n_records - 1;
            } else {
                return Err(JplephemError::OutOfRangeError {
                    jd: seconds_to_jd(et),
                    start_jd: seconds_to_jd(init),
                    end_jd: seconds_to_jd(init + intlen * n_records as f64),
                });
            }
        }
        Ok(index)
    }

    fn get_record_coefficients_type2(
        coefficients: &[f64],
        record_index: usize,
        record_size: usize,
        n_coeffs: usize,
    ) -> Result<Type2Coeffs> {
        let record_start = record_index * record_size;
        if record_start + 2 + 3 * n_coeffs > coefficients.len() {
            return Err(JplephemError::InvalidFormat(
                "Record index out of bounds".to_string(),
            ));
        }

        let record_mid = coefficients[record_start];
        let record_radius = coefficients[record_start + 1];

        let x_start = record_start + 2;
        let y_start = x_start + n_coeffs;
        let z_start = y_start + n_coeffs;

        let coeffs_x = coefficients[x_start..x_start + n_coeffs].to_vec();
        let coeffs_y = coefficients[y_start..y_start + n_coeffs].to_vec();
        let coeffs_z = coefficients[z_start..z_start + n_coeffs].to_vec();

        Ok((record_mid, record_radius, coeffs_x, coeffs_y, coeffs_z))
    }

    fn get_record_coefficients_type3(
        coefficients: &[f64],
        record_index: usize,
        record_size: usize,
        n_coeffs: usize,
    ) -> Result<Type3Coeffs> {
        let record_start = record_index * record_size;
        if record_start + 2 + 6 * n_coeffs > coefficients.len() {
            return Err(JplephemError::InvalidFormat(
                "Record index out of bounds".to_string(),
            ));
        }

        let record_mid = coefficients[record_start];
        let record_radius = coefficients[record_start + 1];

        let x_start = record_start + 2;
        let y_start = x_start + n_coeffs;
        let z_start = y_start + n_coeffs;
        let vx_start = z_start + n_coeffs;
        let vy_start = vx_start + n_coeffs;
        let vz_start = vy_start + n_coeffs;

        let pos = (
            coefficients[x_start..x_start + n_coeffs].to_vec(),
            coefficients[y_start..y_start + n_coeffs].to_vec(),
            coefficients[z_start..z_start + n_coeffs].to_vec(),
        );
        let vel = (
            coefficients[vx_start..vx_start + n_coeffs].to_vec(),
            coefficients[vy_start..vy_start + n_coeffs].to_vec(),
            coefficients[vz_start..vz_start + n_coeffs].to_vec(),
        );

        Ok((record_mid, record_radius, pos, vel))
    }

    fn load_data(&mut self) -> Result<&SegmentData> {
        if self.data.is_some() {
            return Ok(self.data.as_ref().unwrap());
        }

        let array = self.daf.read_array(self.start_i, self.end_i)?;

        match self.data_type {
            2 => self.load_data_type_2(&array),
            3 => self.load_data_type_3(&array),
            _ => Err(JplephemError::UnsupportedDataType(self.data_type)),
        }
    }

    fn load_data_type_2(&mut self, array: &[f64]) -> Result<&SegmentData> {
        if array.len() < 4 {
            return Err(JplephemError::InvalidFormat(
                "Segment data too small for Type 2".to_string(),
            ));
        }

        let n = array.len();
        let init = array[n - 4];
        let intlen = array[n - 3];
        let rsize = array[n - 2] as usize;
        let n_rec = array[n - 1] as usize;

        if rsize < 5 {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid record size for Type 2: {rsize}"
            )));
        }

        let n_coeffs = (rsize - 2) / 3;

        let expected_size = n_rec * rsize + 4;
        if array.len() != expected_size {
            return Err(JplephemError::InvalidFormat(format!(
                "Inconsistent array size: expected {expected_size}, got {}",
                array.len()
            )));
        }

        let coefficients = array[0..(n - 4)].to_vec();

        self.data = Some(SegmentData {
            init,
            intlen,
            coefficients,
            shape: (n_rec, 3, n_coeffs),
            record_size: rsize,
            data_type: self.data_type,
        });
        Ok(self.data.as_ref().unwrap())
    }

    fn load_data_type_3(&mut self, array: &[f64]) -> Result<&SegmentData> {
        if array.len() < 4 {
            return Err(JplephemError::InvalidFormat(
                "Segment data too small for Type 3".to_string(),
            ));
        }

        let n = array.len();
        let init = array[n - 4];
        let intlen = array[n - 3];
        let rsize = array[n - 2] as usize;
        let n_rec = array[n - 1] as usize;

        if rsize < 8 {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid record size for Type 3: {rsize}"
            )));
        }

        let n_coeffs = (rsize - 2) / 6;

        let expected_size = n_rec * rsize + 4;
        if array.len() != expected_size {
            return Err(JplephemError::InvalidFormat(format!(
                "Inconsistent array size: expected {expected_size}, got {}",
                array.len()
            )));
        }

        let coefficients = array[0..(n - 4)].to_vec();

        self.data = Some(SegmentData {
            init,
            intlen,
            coefficients,
            shape: (n_rec, 6, n_coeffs),
            record_size: rsize,
            data_type: self.data_type,
        });
        Ok(self.data.as_ref().unwrap())
    }

    /// Return a human-readable description of this segment
    pub fn describe(&self, verbose: bool) -> String {
        let start_date = calendar_date_from_float(self.start_jd);
        let end_date = calendar_date_from_float(self.end_jd);
        let start = format!("{}-{:02}-{:02}", start_date.0, start_date.1, start_date.2);
        let end = format!("{}-{:02}-{:02}", end_date.0, end_date.1, end_date.2);

        let center_name = get_target_name(self.center).unwrap_or("Unknown");
        let target_name = get_target_name(self.target).unwrap_or("Unknown");

        let mut text = format!(
            "{start}..{end}  Type {}  {center_name} ({}) -> {target_name} ({})",
            self.data_type, self.center, self.target
        );

        if verbose {
            text.push_str(&format!("\n  frame={} source={}", self.frame, self.source));
        }
        text
    }
}

impl std::fmt::Display for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe(false))
    }
}

impl std::fmt::Debug for Segment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe(true))
    }
}
