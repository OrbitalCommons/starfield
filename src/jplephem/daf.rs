//! Double Array File (DAF) format reader for SPICE files
//!
//! Handles reading NAIF's DAF binary format, used for SPK and PCK files.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use byteorder::{BigEndian, ByteOrder, LittleEndian};
use memmap2::{Mmap, MmapOptions};

use crate::jplephem::errors::{io_err, JplephemError, Result};

const RECORD_SIZE: usize = 1024;
const DOUBLE_SIZE: usize = 8;

/// DAF file endianness
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Endian {
    Big,
    Little,
}

/// Double Array File (DAF) reader
pub struct DAF {
    pub path: PathBuf,
    file: Mutex<File>,
    /// File ID word (e.g. "DAF/SPK", "DAF/PCK")
    pub locidw: String,
    /// Number of double-precision components per summary
    pub nd: u32,
    /// Number of integer components per summary
    pub ni: u32,
    /// Forward pointer to first summary record
    pub fward: u32,
    /// Backward pointer to last summary record
    pub bward: u32,
    /// First free address
    pub free: u32,
    /// Internal file name
    pub ifname: String,
    /// Byte order
    pub endian: Endian,
    /// Memory map for efficient access
    map: Option<Mmap>,
    /// Size of each summary entry in bytes
    summary_step: usize,
    /// Size of each summary entry in double-words
    summary_length: usize,
}

impl DAF {
    /// Open a DAF file at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let file = File::open(&path_buf).map_err(|e| io_err(&path_buf, e))?;

        let mut daf = DAF {
            path: path_buf,
            file: Mutex::new(file),
            locidw: String::new(),
            nd: 0,
            ni: 0,
            fward: 0,
            bward: 0,
            free: 0,
            ifname: String::new(),
            endian: Endian::Little,
            map: None,
            summary_step: 0,
            summary_length: 0,
        };

        daf.read_header()?;
        daf.setup_memory_map()?;

        daf.summary_length = daf.nd as usize + (daf.ni as usize).div_ceil(2);
        daf.summary_step = 8 * daf.summary_length;

        Ok(daf)
    }

    fn read_header(&mut self) -> Result<()> {
        let header = self.read_record(1)?;

        let locidw = String::from_utf8_lossy(&header[0..8])
            .trim_end()
            .to_string();

        // Determine endianness: ND and NI should be small values (1-10)
        let nd_le = LittleEndian::read_u32(&header[8..12]);
        let ni_le = LittleEndian::read_u32(&header[12..16]);
        let nd_be = BigEndian::read_u32(&header[8..12]);
        let ni_be = BigEndian::read_u32(&header[12..16]);

        let endian = if nd_le > 0 && nd_le < 10 && ni_le > 0 && ni_le < 10 {
            Endian::Little
        } else if nd_be > 0 && nd_be < 10 && ni_be > 0 && ni_be < 10 {
            Endian::Big
        } else {
            return Err(JplephemError::InvalidFormat(format!(
                "Cannot determine endianness: LE({nd_le},{ni_le}) BE({nd_be},{ni_be})"
            )));
        };

        // DAF header layout:
        //   0..8    LOCIDW (8 bytes, ASCII)
        //   8..12   ND (u32)
        //   12..16  NI (u32)
        //   16..76  LOCIFN (60 bytes, internal filename)
        //   76..80  FWARD (u32, first summary record)
        //   80..84  BWARD (u32, last summary record)
        //   84..88  FREE (u32, first free address)
        let (nd, ni, fward, bward, free) = match endian {
            Endian::Little => (
                LittleEndian::read_u32(&header[8..12]),
                LittleEndian::read_u32(&header[12..16]),
                LittleEndian::read_u32(&header[76..80]),
                LittleEndian::read_u32(&header[80..84]),
                LittleEndian::read_u32(&header[84..88]),
            ),
            Endian::Big => (
                BigEndian::read_u32(&header[8..12]),
                BigEndian::read_u32(&header[12..16]),
                BigEndian::read_u32(&header[76..80]),
                BigEndian::read_u32(&header[80..84]),
                BigEndian::read_u32(&header[84..88]),
            ),
        };

        let ifname = String::from_utf8_lossy(&header[16..76])
            .trim_end()
            .to_string();

        self.locidw = locidw;
        self.nd = nd;
        self.ni = ni;
        self.fward = fward;
        self.bward = bward;
        self.free = free;
        self.ifname = ifname;
        self.endian = endian;

        if self.fward == 0 || self.bward == 0 {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid DAF header: nd={nd}, ni={ni}, fward={fward}, bward={bward}"
            )));
        }

        Ok(())
    }

    fn setup_memory_map(&mut self) -> Result<()> {
        let file = self.file.get_mut().unwrap();
        if let Ok(file_clone) = file.try_clone() {
            if let Ok(mmap) = unsafe { MmapOptions::new().map(&file_clone) } {
                self.map = Some(mmap);
            }
        }
        Ok(())
    }

    fn get_file(&self) -> Result<std::sync::MutexGuard<'_, std::fs::File>> {
        self.file
            .lock()
            .map_err(|_| JplephemError::Other("Failed to lock file".to_string()))
    }

    /// Read a 1024-byte record at the given record number (1-indexed)
    pub fn read_record(&self, record_number: usize) -> Result<Vec<u8>> {
        if record_number < 1 {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid record number: {record_number}"
            )));
        }
        let offset = (record_number - 1) * RECORD_SIZE;

        // Try memory map first
        if let Some(ref map) = self.map {
            if offset + RECORD_SIZE <= map.len() {
                return Ok(map[offset..offset + RECORD_SIZE].to_vec());
            }
        }

        // Fall back to file I/O
        let mut file = self.get_file()?;
        let mut buffer = vec![0u8; RECORD_SIZE];
        file.seek(SeekFrom::Start(offset as u64))
            .map_err(|e| io_err(&self.path, e))?;
        file.read_exact(&mut buffer)
            .map_err(|e| io_err(&self.path, e))?;
        Ok(buffer)
    }

    /// Read comments from the comment area (records 2..fward)
    pub fn comments(&self) -> Result<String> {
        let fward = self.fward as usize;
        if fward <= 2 {
            return Ok(String::new());
        }

        let mut comments = String::new();
        for record_number in 2..fward {
            let record = self.read_record(record_number)?;
            let text = String::from_utf8_lossy(&record);
            comments.push_str(&text);
        }

        Ok(comments
            .trim_end_matches(|c: char| c == '\0' || c.is_whitespace())
            .to_string())
    }

    /// Read summary records and extract segment metadata
    ///
    /// Returns pairs of (name_bytes, summary_values) where summary_values
    /// contains ND doubles followed by NI integers (as f64).
    pub fn summaries(&self) -> Result<Vec<(Vec<u8>, Vec<f64>)>> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        if self.fward == 0 || self.fward > 10000 {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid forward pointer: {}",
                self.fward
            )));
        }

        let mut record_number = self.fward as usize;

        while record_number > 0 {
            if !visited.insert(record_number) {
                break; // Cycle detected
            }

            let summary_data = self.read_record(record_number)?;
            let name_data = self.read_record(record_number + 1)?;

            // First 24 bytes: NEXT (f64), PREV (f64), NSUM (f64)
            let next = self.read_f64_from_bytes(&summary_data[0..8]) as usize;
            let _prev = self.read_f64_from_bytes(&summary_data[8..16]) as usize;
            let n_summaries = self.read_f64_from_bytes(&summary_data[16..24]) as usize;

            let max_summaries = (RECORD_SIZE - 24) / self.summary_step.max(1);
            if n_summaries > max_summaries {
                return Err(JplephemError::InvalidFormat(format!(
                    "Too many summaries in record {record_number}: {n_summaries} > {max_summaries}"
                )));
            }

            for i in 0..n_summaries {
                let name_start = i * self.summary_step;
                let name_end = (name_start + self.summary_step).min(name_data.len());
                let name = name_data[name_start..name_end].to_vec();

                let summary_start = 24 + i * self.summary_step;

                let mut values = Vec::with_capacity(self.nd as usize + self.ni as usize);

                // Read ND double-precision values
                for j in 0..self.nd as usize {
                    let pos = summary_start + j * 8;
                    if pos + 8 <= summary_data.len() {
                        values.push(self.read_f64_from_bytes(&summary_data[pos..pos + 8]));
                    }
                }

                // Read NI integer values (packed as pairs of i32 into 8-byte slots)
                let int_start = summary_start + (self.nd as usize * 8);
                for j in 0..self.ni as usize {
                    let double_idx = j / 2;
                    let int_offset = j % 2;
                    let pos = int_start + double_idx * 8 + int_offset * 4;

                    if pos + 4 <= summary_data.len() {
                        let value = match self.endian {
                            Endian::Big => BigEndian::read_i32(&summary_data[pos..pos + 4]) as f64,
                            Endian::Little => {
                                LittleEndian::read_i32(&summary_data[pos..pos + 4]) as f64
                            }
                        };
                        values.push(value);
                    }
                }

                result.push((name, values));
            }

            if next == 0 || next == record_number {
                break;
            }
            record_number = next;
        }

        if result.is_empty() {
            return Err(JplephemError::InvalidFormat(
                "No summaries found in DAF file".to_string(),
            ));
        }

        Ok(result)
    }

    /// Read an array of f64 values from the file (1-indexed addresses)
    pub fn read_array(&self, start: usize, end: usize) -> Result<Vec<f64>> {
        if start < 1 || end < start {
            return Err(JplephemError::InvalidFormat(format!(
                "Invalid array bounds: start={start}, end={end}"
            )));
        }

        let length = end - start + 1;

        // Try memory map first — read directly from mapped bytes
        if let Some(ref map) = self.map {
            let byte_start = (start - 1) * DOUBLE_SIZE;
            let byte_end = byte_start + length * DOUBLE_SIZE;

            if byte_end <= map.len() {
                let mut result = Vec::with_capacity(length);
                for i in 0..length {
                    let pos = byte_start + i * DOUBLE_SIZE;
                    let value = match self.endian {
                        Endian::Big => BigEndian::read_f64(&map[pos..pos + DOUBLE_SIZE]),
                        Endian::Little => LittleEndian::read_f64(&map[pos..pos + DOUBLE_SIZE]),
                    };
                    result.push(value);
                }
                return Ok(result);
            }
        }

        // Fall back to file I/O
        let mut file = self.get_file()?;
        let byte_offset = (start - 1) * DOUBLE_SIZE;
        let bytes_to_read = length * DOUBLE_SIZE;

        file.seek(SeekFrom::Start(byte_offset as u64))
            .map_err(|e| io_err(&self.path, e))?;

        let mut buffer = vec![0u8; bytes_to_read];
        file.read_exact(&mut buffer)
            .map_err(|e| io_err(&self.path, e))?;

        let mut result = Vec::with_capacity(length);
        for i in 0..length {
            let pos = i * DOUBLE_SIZE;
            let value = match self.endian {
                Endian::Big => BigEndian::read_f64(&buffer[pos..pos + DOUBLE_SIZE]),
                Endian::Little => LittleEndian::read_f64(&buffer[pos..pos + DOUBLE_SIZE]),
            };
            result.push(value);
        }

        Ok(result)
    }

    /// Map an array of f64 values (alias for read_array, uses mmap when available)
    pub fn map_array(&self, start: usize, end: usize) -> Result<Vec<f64>> {
        self.read_array(start, end)
    }

    /// Read an f64 from a byte slice using the file's endianness
    fn read_f64_from_bytes(&self, bytes: &[u8]) -> f64 {
        match self.endian {
            Endian::Big => BigEndian::read_f64(bytes),
            Endian::Little => LittleEndian::read_f64(bytes),
        }
    }
}
