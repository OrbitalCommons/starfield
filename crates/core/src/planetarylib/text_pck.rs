//! Parser for JPL/NAIF text kernels.
//!
//! Text kernels are the plain-text half of the SPICE kernel family: planetary
//! constants kernels (`.tpc`) such as `pck00011.tpc`, and frame kernels
//! (`.tf`) such as `moon_080317.tf`. Both are sequences of `name = value`
//! assignments wrapped in `\begindata` / `\begintext` markers, and both are
//! parsed by the code in this module into a
//! [`HashMap`]`<`[`String`]`, `[`KernelValue`]`>`.
//!
//! This is a port of `skyfield/data/text_pck.py`. For an (incomplete) summary
//! of the file format, look for the heading "NAIF Text Kernel Format" in the
//! [PCK Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html).
//!
//! The grammar the parser accepts:
//!
//! * Only text between a line reading `\begindata` and a line reading
//!   `\begintext` is data; everything else is commentary and is skipped.
//! * `name = value` assigns, `name += value` appends to an earlier assignment.
//! * A value is either a single token or a parenthesised sequence of tokens,
//!   which may run across as many lines as it likes.
//! * Numbers may use `D` in place of `E` for the exponent, as in `-1.4D-12`.
//! * Strings are single quoted, as in `'MOON_PA_DE421'`.
//!
//! # Example
//!
//! ```
//! use starfield_core::planetarylib::text_pck::{parse, KernelValue};
//!
//! let text = "KPL/PCK\n\\begindata\nBODY499_RADII = ( 3396.19 3396.19 3376.20 )\n\\begintext\n";
//! let variables = parse(text).unwrap();
//! assert_eq!(
//!     variables["BODY499_RADII"],
//!     KernelValue::Numbers(vec![3396.19, 3396.19, 3376.20]),
//! );
//! ```

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use crate::{Result, StarfieldError};

/// A value assigned to a name inside a text kernel.
///
/// Following the convention of the SPICE readers, an `=` assignment of one
/// value becomes a scalar ([`KernelValue::Number`] or [`KernelValue::String`])
/// whether or not it was parenthesised, while two or more values become a
/// vector ([`KernelValue::Numbers`] or [`KernelValue::Strings`]). A `+=`
/// assignment always produces a vector, since appending is only meaningful for
/// one.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelValue {
    /// A single number. Integers in the kernel are widened to `f64`.
    Number(f64),
    /// A parenthesised sequence of numbers.
    Numbers(Vec<f64>),
    /// A single quoted string, without its quotes.
    String(String),
    /// A parenthesised sequence of quoted strings, without their quotes.
    Strings(Vec<String>),
}

impl KernelValue {
    /// The value as a scalar number, or `None` if it is not a scalar number.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            KernelValue::Number(x) => Some(*x),
            _ => None,
        }
    }

    /// The value as a slice of numbers, or `None` if it is not a number vector.
    pub fn as_numbers(&self) -> Option<&[f64]> {
        match self {
            KernelValue::Numbers(v) => Some(v),
            _ => None,
        }
    }

    /// The value as a scalar string, or `None` if it is not a scalar string.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            KernelValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// The value as a slice of strings, or `None` if it is not a string vector.
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            KernelValue::Strings(v) => Some(v),
            _ => None,
        }
    }

    /// Every number in the value, whether it is stored as a scalar or a vector.
    ///
    /// Returns `None` for string values.
    pub fn to_numbers(&self) -> Option<Vec<f64>> {
        match self {
            KernelValue::Number(x) => Some(vec![*x]),
            KernelValue::Numbers(v) => Some(v.clone()),
            _ => None,
        }
    }
}

/// A single evaluated token: the atom out of which [`KernelValue`]s are built.
#[derive(Debug, Clone, PartialEq)]
enum Atom {
    Number(f64),
    Text(String),
}

/// The assignment operator that separated a name from its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Operator {
    /// `=`, which replaces any earlier value.
    Assign,
    /// `+=`, which appends to any earlier value.
    Append,
}

/// The token grammar, transcribed from `skyfield/data/text_pck.py`.
///
/// The alternatives are tried in order at each position, so `=` and `+=` win
/// over the catch-all, and the catch-all never swallows a parenthesis.
static TOKEN_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"[A-Za-z]\w+|=|\+=|\(|\)|'[^']*'|[^), \t]+").expect("token regex is valid")
});

/// Parse a text kernel, returning every name it assigns.
///
/// Later assignments to the same name replace earlier ones, and `+=`
/// assignments append, exactly as the SPICE readers do.
///
/// This is the unchecked tokenizer, a port of Skyfield's `text_pck.load`: it
/// does not look for the `KPL/…` magic number, and input with no
/// `\begindata` block — an HTML error page, say — parses as an empty map.
/// [`PlanetaryConstants::read_text`](crate::planetarylib::PlanetaryConstants::read_text)
/// is the validated entry point; it checks
/// [`TEXT_MAGIC_NUMBERS`](crate::planetarylib::TEXT_MAGIC_NUMBERS) first.
///
/// # Errors
///
/// Returns [`StarfieldError::DataError`] if an assignment is malformed, if a
/// number cannot be parsed, if a sequence mixes numbers with strings, or if a
/// value uses the calendar-date syntax (`@01-MAY-1991`), which is not
/// supported.
pub fn parse(text: &str) -> Result<HashMap<String, KernelValue>> {
    let mut variables = HashMap::new();
    load(text, &mut variables)?;
    Ok(variables)
}

/// Parse a text kernel, merging its assignments into an existing map.
///
/// This is the incremental form of [`parse`], for loading several kernels into
/// one set of variables.
///
/// # Errors
///
/// As [`parse`].
pub fn load(text: &str, variables: &mut HashMap<String, KernelValue>) -> Result<()> {
    for (name, operator, atoms) in assignments(text)? {
        match operator {
            Operator::Assign => {
                variables.insert(name, combine(atoms)?);
            }
            Operator::Append => match variables.remove(&name) {
                None => {
                    variables.insert(name, combine_as_vector(atoms)?);
                }
                Some(old) => {
                    let mut merged = unbox(old);
                    merged.extend(atoms);
                    variables.insert(name, combine_as_vector(merged)?);
                }
            },
        }
    }
    Ok(())
}

/// Split a text kernel into its raw `(name, operator, values)` assignments.
fn assignments(text: &str) -> Result<Vec<(String, Operator, Vec<Atom>)>> {
    let tokens = tokenize(text);
    let mut out = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        let name = tokens[i];
        i += 1;

        let operator = match tokens.get(i) {
            Some(&"=") => Operator::Assign,
            Some(&"+=") => Operator::Append,
            _ => {
                return Err(StarfieldError::DataError(format!(
                    "text kernel: an equals sign is expected after {:?}",
                    name
                )))
            }
        };
        i += 1;

        let first = tokens.get(i).ok_or_else(|| {
            StarfieldError::DataError(format!("text kernel: no value assigned to {:?}", name))
        })?;
        i += 1;

        let mut atoms = Vec::new();
        if *first == "(" {
            loop {
                let token = tokens.get(i).ok_or_else(|| {
                    StarfieldError::DataError(format!(
                        "text kernel: unterminated value for {:?}",
                        name
                    ))
                })?;
                i += 1;
                if *token == ")" {
                    break;
                }
                atoms.push(evaluate(token)?);
            }
        } else {
            atoms.push(evaluate(first)?);
        }

        out.push((name.to_string(), operator, atoms));
    }

    Ok(out)
}

/// Yield every token inside the `\begindata` blocks of a text kernel.
fn tokenize(text: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut in_data = false;

    for line in text.lines() {
        if !in_data {
            // Most lines are commentary; skip the trim() on them.
            if line.contains("\\begindata") && line.trim() == "\\begindata" {
                in_data = true;
            }
            continue;
        }
        if line.trim() == "\\begintext" {
            in_data = false;
            continue;
        }
        for m in TOKEN_RE.find_iter(line) {
            tokens.push(m.as_str());
        }
    }

    tokens
}

/// Turn one token into a number or a string.
fn evaluate(token: &str) -> Result<Atom> {
    if let Some(rest) = token.strip_prefix('\'') {
        return Ok(Atom::Text(
            rest.strip_suffix('\'').unwrap_or(rest).to_string(),
        ));
    }
    if token.starts_with('@') {
        return Err(StarfieldError::DataError(format!(
            "text kernel: calendar dates such as {:?} are not supported",
            token
        )));
    }
    let normalized = token.replace('D', "E");
    normalized.parse::<f64>().map(Atom::Number).map_err(|_| {
        StarfieldError::DataError(format!("text kernel: cannot parse {:?} as a number", token))
    })
}

/// Build the value of an `=` assignment, unboxing a lone token to a scalar.
fn combine(atoms: Vec<Atom>) -> Result<KernelValue> {
    if atoms.len() == 1 {
        return Ok(match atoms.into_iter().next() {
            Some(Atom::Number(x)) => KernelValue::Number(x),
            Some(Atom::Text(s)) => KernelValue::String(s),
            None => unreachable!("length was just checked"),
        });
    }
    combine_as_vector(atoms)
}

/// Build a vector value, rejecting sequences that mix numbers with strings.
fn combine_as_vector(atoms: Vec<Atom>) -> Result<KernelValue> {
    let numeric = atoms
        .iter()
        .filter(|a| matches!(a, Atom::Number(_)))
        .count();
    if numeric == atoms.len() {
        return Ok(KernelValue::Numbers(
            atoms
                .into_iter()
                .map(|a| match a {
                    Atom::Number(x) => x,
                    Atom::Text(_) => unreachable!("all atoms are numbers"),
                })
                .collect(),
        ));
    }
    if numeric == 0 {
        return Ok(KernelValue::Strings(
            atoms
                .into_iter()
                .map(|a| match a {
                    Atom::Text(s) => s,
                    Atom::Number(_) => unreachable!("no atom is a number"),
                })
                .collect(),
        ));
    }
    Err(StarfieldError::DataError(
        "text kernel: a value may not mix numbers with strings".to_string(),
    ))
}

/// Turn an existing value back into the list of atoms it was built from.
fn unbox(value: KernelValue) -> Vec<Atom> {
    match value {
        KernelValue::Number(x) => vec![Atom::Number(x)],
        KernelValue::Numbers(v) => v.into_iter().map(Atom::Number).collect(),
        KernelValue::String(s) => vec![Atom::Text(s)],
        KernelValue::Strings(v) => v.into_iter().map(Atom::Text).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_only_data_blocks_are_read() {
        let text = "\
BODY000_IGNORED = ( 1.0 )
\\begindata
BODY499_RADII = ( 3396.19 3396.19 3376.20 )
\\begintext
BODY000_ALSO_IGNORED = ( 2.0 )
";
        let v = parse(text).unwrap();
        assert_eq!(v.len(), 1);
        assert_eq!(
            v["BODY499_RADII"],
            KernelValue::Numbers(vec![3396.19, 3396.19, 3376.20])
        );
    }

    #[test]
    fn test_indented_begindata_marker() {
        // pck00011.tpc indents one of its markers; the SPICE readers strip the
        // line before comparing it, and so must we.
        let text = "Current values:\n\n        \\begindata\nA_NAME = 1\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(v["A_NAME"], KernelValue::Number(1.0));
    }

    #[test]
    fn test_a_lone_value_is_unboxed_to_a_scalar() {
        let text = "\\begindata\nA = 1.5\nB = ( 1.5 )\nC = ( 1.5 2.5 )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(v["A"], KernelValue::Number(1.5));
        assert_eq!(v["B"], KernelValue::Number(1.5));
        assert_eq!(v["C"], KernelValue::Numbers(vec![1.5, 2.5]));
    }

    #[test]
    fn test_d_exponent_marker() {
        let text = "\\begindata\nBODY301_PM = ( 38.3213 13.17635815 -1.4D-12 )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(
            v["BODY301_PM"],
            KernelValue::Numbers(vec![38.3213, 13.17635815, -1.4e-12])
        );
    }

    #[test]
    fn test_e_exponent_and_leading_plus() {
        let text = "\\begindata\nX = ( 0.14947253587500003E+06 +350.891982443297 )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(
            v["X"],
            KernelValue::Numbers(vec![149472.53587500003, 350.891982443297])
        );
    }

    #[test]
    fn test_value_continues_across_lines() {
        let text = "\\begindata\nX = (\n 1\n 2\n\n 3 )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(v["X"], KernelValue::Numbers(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_quoted_strings() {
        let text = "\\begindata\nFRAME_31006_NAME = 'MOON_PA_DE421'\n\
                    NAMES = ( 'A B' 'C' )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(
            v["FRAME_31006_NAME"],
            KernelValue::String("MOON_PA_DE421".to_string())
        );
        assert_eq!(
            v["NAMES"],
            KernelValue::Strings(vec!["A B".to_string(), "C".to_string()])
        );
    }

    #[test]
    fn test_append_operator() {
        let text = "\\begindata\nX = ( 1 2 )\nX += ( 3 )\nY = 1\nY += 2\nZ += ( 9 )\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(v["X"], KernelValue::Numbers(vec![1.0, 2.0, 3.0]));
        assert_eq!(v["Y"], KernelValue::Numbers(vec![1.0, 2.0]));
        assert_eq!(v["Z"], KernelValue::Numbers(vec![9.0]));
    }

    #[test]
    fn test_later_assignment_replaces_earlier() {
        let text = "\\begindata\nX = 1\nX = 2\n\\begintext\n";
        let v = parse(text).unwrap();
        assert_eq!(v["X"], KernelValue::Number(2.0));
    }

    #[test]
    fn test_missing_equals_is_an_error() {
        let text = "\\begindata\nX 1\n\\begintext\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn test_unterminated_vector_is_an_error() {
        let text = "\\begindata\nX = ( 1 2\n\\begintext\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn test_calendar_date_is_rejected() {
        let text = "\\begindata\nX = @01-MAY-1991\n\\begintext\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn test_mixed_sequence_is_rejected() {
        let text = "\\begindata\nX = ( 1 'two' )\n\\begintext\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn test_kernel_value_accessors() {
        assert_eq!(KernelValue::Number(1.0).as_number(), Some(1.0));
        assert_eq!(KernelValue::Number(1.0).as_numbers(), None);
        assert_eq!(
            KernelValue::Numbers(vec![1.0, 2.0]).as_numbers(),
            Some(&[1.0, 2.0][..])
        );
        assert_eq!(KernelValue::Number(1.0).to_numbers(), Some(vec![1.0]));
        assert_eq!(KernelValue::String("a".into()).as_string(), Some("a"));
        assert_eq!(
            KernelValue::Strings(vec!["a".into()]).as_strings(),
            Some(&["a".to_string()][..])
        );
        assert_eq!(KernelValue::String("a".into()).to_numbers(), None);
    }
}
