//! Helpers for pulling typed values out of an Arrow [`RecordBatch`] row.
//!
//! These wrap the usual downcast dance (`col.as_any().downcast_ref::<Float64Array>()`)
//! into short call sites used inside per-release `build_entry` functions.

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt32Array, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use starfield_core::{Result, StarfieldError};

fn col(batch: &RecordBatch, idx: usize) -> &dyn Array {
    batch.column(idx).as_ref()
}

fn schema_name(batch: &RecordBatch, idx: usize) -> String {
    batch.schema().field(idx).name().clone()
}

fn type_err(batch: &RecordBatch, idx: usize, want: &str) -> StarfieldError {
    StarfieldError::DataError(format!(
        "gaia parse: column {} expected {}, found {:?}",
        schema_name(batch, idx),
        want,
        batch.schema().field(idx).data_type(),
    ))
}

/// Required `f64`. Errors if the column type is wrong or the value is null.
pub fn req_f64(batch: &RecordBatch, idx: usize, row: usize) -> Result<f64> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| type_err(batch, idx, "Float64"))?;
    if arr.is_null(row) {
        return Err(StarfieldError::DataError(format!(
            "gaia parse: required column {} is null at row {}",
            schema_name(batch, idx),
            row
        )));
    }
    Ok(arr.value(row))
}

/// Required `u64`. Handles both Int64 (arrow-csv may parse signed) and UInt64.
pub fn req_u64(batch: &RecordBatch, idx: usize, row: usize) -> Result<u64> {
    let any = col(batch, idx).as_any();
    if let Some(arr) = any.downcast_ref::<UInt64Array>() {
        if arr.is_null(row) {
            return Err(null_err(batch, idx, row));
        }
        return Ok(arr.value(row));
    }
    if let Some(arr) = any.downcast_ref::<Int64Array>() {
        if arr.is_null(row) {
            return Err(null_err(batch, idx, row));
        }
        return Ok(arr.value(row) as u64);
    }
    Err(type_err(batch, idx, "UInt64 or Int64"))
}

pub fn opt_f64(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<f64>> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| type_err(batch, idx, "Float64"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

pub fn opt_f32(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<f32>> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| type_err(batch, idx, "Float32"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

/// Required `f32`.
pub fn req_f32(batch: &RecordBatch, idx: usize, row: usize) -> Result<f32> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| type_err(batch, idx, "Float32"))?;
    if arr.is_null(row) {
        return Err(null_err(batch, idx, row));
    }
    Ok(arr.value(row))
}

pub fn opt_u32(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<u32>> {
    let any = col(batch, idx).as_any();
    if let Some(arr) = any.downcast_ref::<UInt32Array>() {
        return Ok(if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        });
    }
    if let Some(arr) = any.downcast_ref::<Int32Array>() {
        return Ok(if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row) as u32)
        });
    }
    if let Some(arr) = any.downcast_ref::<Int16Array>() {
        return Ok(if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row) as u32)
        });
    }
    Err(type_err(batch, idx, "UInt32 / Int32 / Int16"))
}

pub fn opt_u64(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<u64>> {
    let any = col(batch, idx).as_any();
    if let Some(arr) = any.downcast_ref::<UInt64Array>() {
        return Ok(if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        });
    }
    if let Some(arr) = any.downcast_ref::<Int64Array>() {
        return Ok(if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row) as u64)
        });
    }
    Err(type_err(batch, idx, "UInt64 or Int64"))
}

pub fn opt_bool(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<bool>> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| type_err(batch, idx, "Boolean"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

pub fn opt_str(batch: &RecordBatch, idx: usize, row: usize) -> Result<Option<&str>> {
    let arr = col(batch, idx)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| type_err(batch, idx, "Utf8"))?;
    Ok(if arr.is_null(row) {
        None
    } else {
        Some(arr.value(row))
    })
}

fn null_err(batch: &RecordBatch, idx: usize, row: usize) -> StarfieldError {
    StarfieldError::DataError(format!(
        "gaia parse: required column {} is null at row {}",
        schema_name(batch, idx),
        row
    ))
}
