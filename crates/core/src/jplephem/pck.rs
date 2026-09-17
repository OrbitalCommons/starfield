//! Binary Planetary Constants Kernel (PCK) reader
//!
//! A binary PCK — a `.bpc` file — is a DAF whose segments hold the orientation
//! of a body-fixed frame rather than the position of a body. Each type 2
//! segment stores Chebyshev coefficients for three Euler angles, in radians,
//! that carry a reference frame (almost always J2000) into the body-fixed
//! frame of a NAIF *frame class id*: 31006 for `MOON_PA_DE421`, 31008 for
//! `MOON_PA_DE440`, 3000 for `ITRF93`.
//!
//! This is a port of python `jplephem/pck.py`. Only data type 2 is defined for
//! binary PCK files, and only data type 2 is supported here; any other type
//! reports [`JplephemError::UnsupportedDataType`].
//!
//! The angles are not pole right ascension, declination and prime meridian:
//! they are the SPICE Euler angles of the rotation, and are turned into a
//! matrix by [`crate::planetarylib::pck_frame::PckFrame`].
//!
//! # Example
//!
//! ```no_run
//! use starfield_core::jplephem::pck::PCK;
//!
//! let pck = PCK::open("moon_pa_de421_1900-2050.bpc").unwrap();
//! let segment = pck.segment_for(31006).unwrap();
//! let (angles, rates) = segment.compute(2451545.0).unwrap();
//! println!("{angles} rad, {rates} rad/day");
//! ```

use std::path::Path;
use std::sync::Arc;

use nalgebra::Vector3;
use once_cell::sync::OnceCell;

use super::calendar::calendar_date_from_float;
use super::chebyshev::ChebyshevRecords;
use super::daf::DAF;
use super::errors::{JplephemError, Result};
use super::kernel::S_PER_DAY;
use super::names::get_target_name;
use super::spk::{jd_to_seconds, seconds_to_jd};

/// The only data type a binary PCK segment may use.
const CHEBYSHEV_TYPE: i32 = 2;

/// The frame class ids that NAIF's published binary PCK kernels carry.
const FRAME_NAMES: [(i32, &str); 3] = [
    (3000, "ITRF93"),
    (31006, "MOON_PA_DE421"),
    (31008, "MOON_PA_DE440"),
];

/// The name of a well-known NAIF frame class id.
///
/// Covers the frames the binary PCK kernels NAIF publishes orient: `ITRF93`
/// (3000) and the lunar principal-axes frames `MOON_PA_DE421` (31006) and
/// `MOON_PA_DE440` (31008). A text frame kernel is what names any other id.
pub fn frame_name(frame_class_id: i32) -> Option<&'static str> {
    FRAME_NAMES
        .iter()
        .find(|(id, _)| *id == frame_class_id)
        .map(|(_, name)| *name)
}

/// A binary Planetary Constants Kernel (`.bpc`) file.
///
/// Open one with [`PCK::open`] or [`PCK::from_bytes`], then reach its
/// orientation data through [`segments`](Self::segments) or
/// [`segment_for`](Self::segment_for). Printing the kernel lists its segments,
/// in the style of [`crate::jplephem::SPK`].
pub struct PCK {
    /// The underlying DAF file.
    pub daf: Arc<DAF>,
    /// Every segment found in the file, in file order.
    segments: Vec<PckSegment>,
}

impl PCK {
    /// Open a binary PCK file at the given path.
    ///
    /// # Errors
    ///
    /// Returns [`JplephemError::FileError`] if the file cannot be read and
    /// [`JplephemError::InvalidFormat`] if it is not a DAF.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::from_daf(Arc::new(DAF::open(path)?))
    }

    /// Read a binary PCK from an in-memory byte buffer.
    ///
    /// # Errors
    ///
    /// Returns [`JplephemError::InvalidFormat`] if the bytes are not a DAF.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_daf(Arc::new(DAF::from_bytes(data)?))
    }

    /// Build a kernel around an already-opened DAF.
    fn from_daf(daf: Arc<DAF>) -> Result<Self> {
        let mut segments = Vec::new();

        for (name, values) in daf.summaries()?.iter() {
            // A DAF/PCK summary is two doubles and five integers.
            if values.len() < 7 {
                return Err(JplephemError::InvalidFormat(format!(
                    "A binary PCK summary needs 7 values, found {}",
                    values.len()
                )));
            }

            let start_second = values[0];
            let end_second = values[1];

            segments.push(PckSegment {
                daf: Arc::clone(&daf),
                source: String::from_utf8_lossy(name).trim_end().to_string(),
                start_second,
                end_second,
                start_jd: seconds_to_jd(start_second),
                end_jd: seconds_to_jd(end_second),
                body: values[2] as i32,
                frame: values[3] as i32,
                data_type: values[4] as i32,
                start_i: values[5] as usize,
                end_i: values[6] as usize,
                records: OnceCell::new(),
            });
        }

        Ok(PCK { daf, segments })
    }

    /// Every segment in the file, in file order.
    pub fn segments(&self) -> &[PckSegment] {
        &self.segments
    }

    /// Take ownership of the segments, dropping the kernel wrapper.
    ///
    /// Each segment keeps its own reference to the underlying DAF, so the
    /// segments stay usable once the [`PCK`] is gone.
    pub fn into_segments(self) -> Vec<PckSegment> {
        self.segments
    }

    /// The first segment whose frame class id is `body`, if there is one.
    ///
    /// The id is the NAIF frame class id the kernel orients, such as 31006 for
    /// `MOON_PA_DE421`.
    pub fn segment_for(&self, body: i32) -> Option<&PckSegment> {
        self.segments.iter().find(|s| s.body == body)
    }

    /// Read the comment area of the underlying DAF file.
    ///
    /// # Errors
    ///
    /// Returns [`JplephemError::FileError`] if the comment records cannot be
    /// read.
    pub fn comments(&self) -> Result<String> {
        self.daf.comments()
    }
}

impl std::fmt::Display for PCK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "File type {} with {} segments:",
            self.daf.locidw,
            self.segments.len()
        )?;
        for segment in &self.segments {
            writeln!(f, "{segment}")?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for PCK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// One segment of a binary PCK file: the orientation of a single frame over a
/// single stretch of time.
pub struct PckSegment {
    /// The file the coefficients are read from.
    daf: Arc<DAF>,
    /// Segment source label, such as `DE-0421LE-0421`.
    pub source: String,
    /// Start epoch, in TDB seconds since J2000.
    pub start_second: f64,
    /// End epoch, in TDB seconds since J2000.
    pub end_second: f64,
    /// NAIF frame class id of the body-fixed frame this segment orients.
    pub body: i32,
    /// NAIF id of the reference frame the angles rotate away from; 1 is J2000.
    pub frame: i32,
    /// Segment data type; only 2 is defined for binary PCK files.
    pub data_type: i32,
    /// Start index in the file, as a 1-indexed double-word address.
    pub start_i: usize,
    /// End index in the file, as a 1-indexed double-word address.
    pub end_i: usize,
    /// Start epoch, as a Julian date.
    pub start_jd: f64,
    /// End epoch, as a Julian date.
    pub end_jd: f64,
    /// Coefficients, read from the file the first time they are needed.
    records: OnceCell<ChebyshevRecords>,
}

impl PckSegment {
    /// The three Euler angles and their rates at TDB Julian date `tdb_jd`.
    ///
    /// Returns the angles in radians and their rates in radians per day. The
    /// angles are the SPICE Euler angles of the rotation from the reference
    /// frame to the body-fixed frame, in the order the kernel stores them.
    ///
    /// # Errors
    ///
    /// Returns [`JplephemError::UnsupportedDataType`] unless the segment is
    /// type 2, and [`JplephemError::OutOfRangeError`] if `tdb_jd` lies outside
    /// the span the segment covers.
    pub fn compute(&self, tdb_jd: f64) -> Result<(Vector3<f64>, Vector3<f64>)> {
        let et = jd_to_seconds(tdb_jd);

        if et < self.start_second || et > self.end_second {
            return Err(JplephemError::OutOfRangeError {
                jd: tdb_jd,
                start_jd: self.start_jd,
                end_jd: self.end_jd,
            });
        }

        let record = self.records()?.record_at(et)?;
        let t = record.normalized_time(et)?;
        // The Chebyshev variable spans the record's radius in seconds, so a
        // derivative with respect to it becomes radians per day this way.
        let rate_scale = S_PER_DAY / record.radius();

        let mut angles = Vector3::zeros();
        let mut rates = Vector3::zeros();
        for i in 0..3 {
            let polynomial = record.polynomial(i)?;
            angles[i] = polynomial.evaluate(t);
            rates[i] = polynomial.derivative(t) * rate_scale;
        }

        Ok((angles, rates))
    }

    /// The coefficients of this segment, read from the file on first use.
    ///
    /// # Errors
    ///
    /// Returns [`JplephemError::UnsupportedDataType`] unless the segment is
    /// type 2, and [`JplephemError::InvalidFormat`] if the coefficient array
    /// is malformed.
    fn records(&self) -> Result<&ChebyshevRecords> {
        if self.data_type != CHEBYSHEV_TYPE {
            return Err(JplephemError::UnsupportedDataType(self.data_type));
        }
        self.records.get_or_try_init(|| {
            let array = self.daf.read_array(self.start_i, self.end_i)?;
            ChebyshevRecords::load(&array, 3)
        })
    }

    /// A human-readable description of this segment.
    ///
    /// The verbose form adds the data type and the source label, matching the
    /// description [`crate::jplephem::spk::Segment`] prints.
    pub fn describe(&self, verbose: bool) -> String {
        let start = calendar_date_from_float(self.start_jd);
        let end = calendar_date_from_float(self.end_jd);
        let body_name = frame_name(self.body)
            .or_else(|| get_target_name(self.body))
            .unwrap_or("Unknown");

        let mut text = format!(
            "{}-{:02}-{:02}..{}-{:02}-{:02}  frame={}  {} ({})",
            start.0, start.1, start.2, end.0, end.1, end.2, self.frame, body_name, self.body
        );

        if verbose {
            text.push_str(&format!(
                "\n  data_type={} source={}",
                self.data_type, self.source
            ));
        }
        text
    }
}

impl std::fmt::Display for PckSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe(false))
    }
}

impl std::fmt::Debug for PckSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe(true))
    }
}

/// Synthetic binary PCK files, shared by the tests of this module and of
/// [`crate::planetarylib::pck_frame`].
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Frame class id of the lunar principal-axes frame of DE421.
    pub const MOON_PA_DE421: i32 = 31006;

    /// Build a DAF/PCK file in memory around one type 2 segment.
    ///
    /// `records` is one entry per record: its midpoint and radius in TDB
    /// seconds since J2000, then the Chebyshev coefficients of each of the
    /// three angles.
    pub fn synthetic_pck(
        body: i32,
        data_type: i32,
        records: &[(f64, f64, [Vec<f64>; 3])],
    ) -> Result<Vec<u8>> {
        let n_coeffs = records[0].2[0].len();
        let record_size = 2 + 3 * n_coeffs;
        let intlen = 2.0 * records[0].1;
        let init = records[0].0 - records[0].1;

        let mut array: Vec<f64> = Vec::new();
        for (midpoint, radius, angles) in records {
            array.push(*midpoint);
            array.push(*radius);
            for angle in angles {
                assert_eq!(angle.len(), n_coeffs, "records must be the same size");
                array.extend_from_slice(angle);
            }
        }
        array.extend_from_slice(&[init, intlen, record_size as f64, records.len() as f64]);

        // Records 1..=3 hold the header, the summary record and the name
        // record; the coefficients start at the first double-word after them.
        const RECORD_SIZE: usize = 1024;
        let start_i = 3 * RECORD_SIZE / 8 + 1;
        let end_i = start_i + array.len() - 1;

        let mut file = vec![0u8; 3 * RECORD_SIZE];
        file[0..8].copy_from_slice(b"DAF/PCK ");
        file[8..12].copy_from_slice(&2u32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&5u32.to_le_bytes()); // NI
        file[16..76].copy_from_slice(format!("{:<60}", "SYNTHETIC").as_bytes());
        file[76..80].copy_from_slice(&2u32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2u32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&((end_i + 1) as u32).to_le_bytes()); // FREE

        // Summary record: NEXT, PREV, NSUM, then the one summary itself.
        let summary = RECORD_SIZE;
        let put = |file: &mut Vec<u8>, at: usize, value: f64| {
            file[at..at + 8].copy_from_slice(&value.to_le_bytes());
        };
        put(&mut file, summary, 0.0);
        put(&mut file, summary + 8, 0.0);
        put(&mut file, summary + 16, 1.0);
        put(&mut file, summary + 24, records[0].0 - records[0].1);
        put(
            &mut file,
            summary + 32,
            records[records.len() - 1].0 + records[records.len() - 1].1,
        );
        // Five integers packed two to a double-word.
        let ints = [body, 1, data_type, start_i as i32, end_i as i32];
        for (i, value) in ints.iter().enumerate() {
            let at = summary + 40 + (i / 2) * 8 + (i % 2) * 4;
            file[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }

        // Name record.
        file[2 * RECORD_SIZE..2 * RECORD_SIZE + 40]
            .copy_from_slice(format!("{:<40}", "SYNTHETIC PCK").as_bytes());

        for value in &array {
            file.extend_from_slice(&value.to_le_bytes());
        }

        Ok(file)
    }

    /// One day of coverage, centred on J2000, with easily checked angles.
    pub fn one_day_pck(data_type: i32) -> Vec<u8> {
        synthetic_pck(
            MOON_PA_DE421,
            data_type,
            &[(
                0.0,
                S_PER_DAY / 2.0,
                [
                    vec![0.5, 0.25, 0.125],
                    vec![1.0, 0.0, 0.0],
                    vec![0.0, 2.0, 0.0],
                ],
            )],
        )
        .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{one_day_pck, synthetic_pck, MOON_PA_DE421};
    use super::*;
    use crate::jplephem::chebyshev::ChebyshevPolynomial;

    #[test]
    fn test_segment_metadata_is_parsed() {
        let pck = PCK::from_bytes(&one_day_pck(2)).unwrap();
        assert_eq!(pck.segments().len(), 1);

        let segment = &pck.segments()[0];
        assert_eq!(segment.body, MOON_PA_DE421);
        assert_eq!(segment.frame, 1);
        assert_eq!(segment.data_type, 2);
        assert_eq!(segment.source, "SYNTHETIC PCK");
        assert_eq!(segment.start_second, -S_PER_DAY / 2.0);
        assert_eq!(segment.end_second, S_PER_DAY / 2.0);
        assert!((segment.start_jd - 2451544.5).abs() < 1e-9);
        assert!((segment.end_jd - 2451545.5).abs() < 1e-9);
        assert!(pck.segment_for(MOON_PA_DE421).is_some());
        assert!(pck.segment_for(3000).is_none());
    }

    #[test]
    fn test_compute_matches_the_polynomials() {
        let pck = PCK::from_bytes(&one_day_pck(2)).unwrap();
        let segment = pck.segment_for(MOON_PA_DE421).unwrap();

        // A quarter of a day past J2000 is halfway from the record's midpoint
        // to its end, so the Chebyshev variable is exactly 1/2.
        let tdb_jd = 2451545.25;
        let s = 0.5;
        let (angles, rates) = segment.compute(tdb_jd).unwrap();

        let expected = [
            ChebyshevPolynomial::new(vec![0.5, 0.25, 0.125]),
            ChebyshevPolynomial::new(vec![1.0, 0.0, 0.0]),
            ChebyshevPolynomial::new(vec![0.0, 2.0, 0.0]),
        ];
        for (i, polynomial) in expected.iter().enumerate() {
            assert!((angles[i] - polynomial.evaluate(s)).abs() < 1e-12);
            // The record spans half a day either side of its midpoint, so a
            // derivative per unit of the variable is twice that per day.
            assert!((rates[i] - polynomial.derivative(s) * 2.0).abs() < 1e-12);
        }

        // And the same values worked out by hand: T0 = 1, T1 = s, T2 = 2s²-1.
        assert!((angles[0] - (0.5 + 0.25 * s + 0.125 * (2.0 * s * s - 1.0))).abs() < 1e-12);
        assert!((angles[1] - 1.0).abs() < 1e-12);
        assert!((angles[2] - 2.0 * s).abs() < 1e-12);
        assert!((rates[0] - (0.25 + 0.125 * 4.0 * s) * 2.0).abs() < 1e-12);
        assert_eq!(rates[1], 0.0);
        assert!((rates[2] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_compute_at_the_ends_of_the_record() {
        let pck = PCK::from_bytes(&one_day_pck(2)).unwrap();
        let segment = pck.segment_for(MOON_PA_DE421).unwrap();

        let (start, _) = segment.compute(2451544.5).unwrap();
        let (end, _) = segment.compute(2451545.5).unwrap();
        // Angle 2 is 2·T1(s), which runs from -2 to 2 across the record.
        assert!((start[2] + 2.0).abs() < 1e-9);
        assert!((end[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_picks_the_right_record() {
        // Two consecutive one-day records, each holding a constant angle.
        let bytes = synthetic_pck(
            MOON_PA_DE421,
            2,
            &[
                (
                    0.0,
                    S_PER_DAY / 2.0,
                    [vec![7.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
                ),
                (
                    S_PER_DAY,
                    S_PER_DAY / 2.0,
                    [vec![9.0, 0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
                ),
            ],
        )
        .unwrap();

        let pck = PCK::from_bytes(&bytes).unwrap();
        let segment = pck.segment_for(MOON_PA_DE421).unwrap();
        assert_eq!(segment.compute(2451545.0).unwrap().0[0], 7.0);
        assert_eq!(segment.compute(2451546.0).unwrap().0[0], 9.0);
    }

    #[test]
    fn test_compute_outside_the_segment_is_an_error() {
        let pck = PCK::from_bytes(&one_day_pck(2)).unwrap();
        let segment = pck.segment_for(MOON_PA_DE421).unwrap();

        assert!(matches!(
            segment.compute(2451540.0),
            Err(JplephemError::OutOfRangeError { .. })
        ));
        assert!(matches!(
            segment.compute(2451550.0),
            Err(JplephemError::OutOfRangeError { .. })
        ));
    }

    #[test]
    fn test_other_data_types_are_unsupported() {
        let pck = PCK::from_bytes(&one_day_pck(3)).unwrap();
        let segment = pck.segment_for(MOON_PA_DE421).unwrap();
        assert_eq!(segment.data_type, 3);
        assert!(matches!(
            segment.compute(2451545.0),
            Err(JplephemError::UnsupportedDataType(3))
        ));
    }

    #[test]
    fn test_describe_and_display() {
        let pck = PCK::from_bytes(&one_day_pck(2)).unwrap();
        let text = format!("{pck}");
        assert!(text.contains("DAF/PCK"));
        assert!(text.contains("1 segments"));
        assert!(text.contains("frame=1"));
        assert!(text.contains("MOON_PA_DE421 (31006)"));
        assert!(pck.segments()[0].describe(true).contains("data_type=2"));
    }

    #[test]
    fn test_well_known_frame_names() {
        assert_eq!(frame_name(31006), Some("MOON_PA_DE421"));
        assert_eq!(frame_name(31008), Some("MOON_PA_DE440"));
        assert_eq!(frame_name(3000), Some("ITRF93"));
        assert_eq!(frame_name(301), None);
    }

    #[test]
    fn test_from_bytes_rejects_a_non_daf() {
        assert!(PCK::from_bytes(&[0u8; 16]).is_err());
    }
}
