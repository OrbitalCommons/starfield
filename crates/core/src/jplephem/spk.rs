//! Spacecraft Planet Kernel (SPK) format reader
//!
//! Reads NASA SPICE SPK files containing position and velocity data
//! for solar system bodies, stored as Chebyshev polynomial coefficients
//! or Modified Difference Arrays.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nalgebra::Vector3;

use super::calendar::calendar_date_from_float;
use super::chebyshev::{rescale_derivative, ChebyshevRecords};
use super::daf::DAF;
use super::errors::{JplephemError, Result};
use super::names::get_target_name;
use super::spk_type21::Type21Data;

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

/// SPK data types this reader can evaluate: Chebyshev position (2), Chebyshev
/// position and velocity (3), and extended modified difference arrays (21).
///
/// Segments of any other type are skipped when a file is parsed, so a body
/// carried only by such segments is not reachable; `Position::from_spk_target`
/// reports that as [`JplephemError::UnsupportedDataType`].
pub const SUPPORTED_DATA_TYPES: [i32; 3] = [2, 3, 21];

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
    /// SPK data type (2=Chebyshev position, 3=Chebyshev pos+vel, 21=MDA)
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
enum SegmentData {
    /// Type 2/3 Chebyshev polynomial data
    Chebyshev {
        records: ChebyshevRecords,
        data_type: i32,
    },
    /// Type 21 Modified Difference Array data
    Type21(Type21Data),
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

    /// Create an SPK from an in-memory byte buffer
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let daf = Arc::new(DAF::from_bytes(data)?);

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
            if !SUPPORTED_DATA_TYPES.contains(&data_type) {
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

        match data {
            SegmentData::Chebyshev { records, data_type } => {
                let record = records.record_at(et)?;
                let t = record.normalized_time(et)?;
                let radius = record.radius();

                match data_type {
                    2 => {
                        let x = record.polynomial(0)?;
                        let y = record.polynomial(1)?;
                        let z = record.polynomial(2)?;

                        let position = Vector3::new(x.evaluate(t), y.evaluate(t), z.evaluate(t));
                        let velocity = Vector3::new(
                            rescale_derivative(x.derivative(t), radius)?,
                            rescale_derivative(y.derivative(t), radius)?,
                            rescale_derivative(z.derivative(t), radius)?,
                        );

                        Ok((position, velocity))
                    }
                    3 => {
                        let position = Vector3::new(
                            record.polynomial(0)?.evaluate(t),
                            record.polynomial(1)?.evaluate(t),
                            record.polynomial(2)?.evaluate(t),
                        );
                        let velocity = Vector3::new(
                            rescale_derivative(record.polynomial(3)?.evaluate(t), radius)?,
                            rescale_derivative(record.polynomial(4)?.evaluate(t), radius)?,
                            rescale_derivative(record.polynomial(5)?.evaluate(t), radius)?,
                        );

                        Ok((position, velocity))
                    }
                    _ => Err(JplephemError::UnsupportedDataType(*data_type)),
                }
            }
            SegmentData::Type21(type21_data) => type21_data.compute(et),
        }
    }

    fn load_data(&mut self) -> Result<&SegmentData> {
        if let Some(ref data) = self.data {
            return Ok(data);
        }

        match self.data_type {
            2 | 3 => {
                let array = self.daf.read_array(self.start_i, self.end_i)?;
                match self.data_type {
                    2 => self.load_data_type_2(&array),
                    3 => self.load_data_type_3(&array),
                    _ => unreachable!(),
                }
            }
            21 => self.load_data_type_21(),
            _ => Err(JplephemError::UnsupportedDataType(self.data_type)),
        }
    }

    fn load_data_type_2(&mut self, array: &[f64]) -> Result<&SegmentData> {
        self.data = Some(SegmentData::Chebyshev {
            records: ChebyshevRecords::load(array, 3)?,
            data_type: self.data_type,
        });
        Ok(self.data.as_ref().unwrap())
    }

    fn load_data_type_3(&mut self, array: &[f64]) -> Result<&SegmentData> {
        self.data = Some(SegmentData::Chebyshev {
            records: ChebyshevRecords::load(array, 6)?,
            data_type: self.data_type,
        });
        Ok(self.data.as_ref().unwrap())
    }

    fn load_data_type_21(&mut self) -> Result<&SegmentData> {
        let type21_data = Type21Data::load(&self.daf, self.start_i, self.end_i)?;
        self.data = Some(SegmentData::Type21(type21_data));
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
