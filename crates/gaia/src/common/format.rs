//! Tiny formatting helpers used by per-release `format_csv_row` implementations.
//!
//! All helpers stringify in a way that round-trips through Rust's standard
//! `FromStr` impls (and therefore through the arrow CSV reader's typed parsers).
//! `Option::None` always becomes the empty string.

use crate::common::core::VarFlag;
use std::fmt::Display;

#[inline]
pub fn fopt<T: Display>(v: Option<T>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => String::new(),
    }
}

#[inline]
pub fn fopt_bool(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "true",
        Some(false) => "false",
        None => "",
    }
}

#[inline]
pub fn fvar(v: VarFlag) -> &'static str {
    v.as_str()
}

#[inline]
pub fn fopt_str(v: Option<&str>) -> &str {
    v.unwrap_or("")
}
