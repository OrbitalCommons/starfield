//! Planetary Constants Kernel (PCK) format stub
//!
//! PCK files contain rotation and orientation data for planets and moons.
//! This is a placeholder for future implementation.

use std::path::Path;
use std::sync::Arc;

use crate::jplephem::daf::DAF;
use crate::jplephem::errors::Result;

/// Planetary Constants Kernel (PCK) file reader
pub struct PCK {
    /// The underlying DAF file
    pub daf: Arc<DAF>,
}

impl PCK {
    /// Open a PCK file at the given path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let daf = Arc::new(DAF::open(path)?);
        Ok(PCK { daf })
    }
}
