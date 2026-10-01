//! Serde support for [`Time`].
//!
//! A `Time` serialises as a map that carries its internal split-day
//! representation exactly, plus two informational fields:
//!
//! ```json
//! {
//!   "whole": 2461557.0,
//!   "tt_fraction": 0.5008007407407407,
//!   "tai_fraction": 0.5004282407407407,
//!   "tdb_fraction": 0.5008007323281066,
//!   "jd_tdb": 2461557.5008007325,
//!   "utc": "2027-06-01T00:00:00.000000Z"
//! }
//! ```
//!
//! * `whole` and `tt_fraction` are the Julian day number and TT day
//!   fraction `Time` stores. Keeping them apart preserves the
//!   sub-microsecond resolution a single `f64` Julian date (about 40 us
//!   near the present) would lose. Both are required on input unless one
//!   of the fallbacks below is used.
//! * `tai_fraction` (present when the time was built from UTC or TAI) and
//!   `tdb_fraction` are written so that [`Time::tai`] and [`Time::tdb`]
//!   return bit-identical values after a round trip. Both are optional on
//!   input and are derived from TT when absent.
//! * `leap_second` is written only when `true`.
//! * `jd_tdb` (TDB Julian date as one `f64`) and `utc` (ISO 8601, to the
//!   microsecond; omitted if the UTC conversion fails or the time is inside a leap
//!   second, which the calendar formatter cannot print as second 60) are for human
//!   readers and other tools. They are ignored when `whole` and
//!   `tt_fraction` are present.
//!
//! Formats that are not human-readable (bincode, CBOR, ...) get a
//! fixed-layout struct of `whole`, `tt_fraction`, `tai_fraction`
//! (`Option`), `tdb_fraction` and `leap_second`, with no informational
//! fields or optional entries, since such formats may encode structs
//! positionally.
//!
//! Round trips through any serde format that preserves `f64` exactly
//! compare equal and give identical TT, TAI and TDB. The [`Timescale`] is not serialised: a deserialised time uses
//! [`Timescale::default`], so UT1, delta-T and polar motion are
//! recomputed from its tables.
//!
//! Deserialisation fails if any numeric component (`whole`, a fraction,
//! or `jd_tdb`) is NaN or infinite.
//!
//! For hand-written input in human-readable formats, two shorter forms
//! are also accepted:
//!
//! * a string, parsed with [`Time::from_str`](std::str::FromStr), such as
//!   `"2027-06-01T00:00:00Z"` or `"JD 2461558.5 TDB"`;
//! * a map with only `jd_tdb`, read with [`Timescale::tdb_jd`].

use std::fmt;
use std::sync::OnceLock;

use serde::de::{self, Deserializer, IgnoredAny, MapAccess, Visitor};
use serde::ser::{self, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

use super::{once_with, Time, Timescale};

const FIELDS: &[&str] = &[
    "whole",
    "tt_fraction",
    "tai_fraction",
    "tdb_fraction",
    "leap_second",
    "jd_tdb",
    "utc",
];

/// Fixed-layout form for non-human-readable formats, which may encode
/// structs positionally and cannot skip fields.
#[derive(Serialize, Deserialize)]
#[serde(rename = "Time")]
struct Compact {
    whole: f64,
    tt_fraction: f64,
    tai_fraction: Option<f64>,
    tdb_fraction: f64,
    leap_second: bool,
}

impl Time {
    /// TDB day fraction, computing and caching it if needed.
    fn tdb_fraction_value(&self) -> f64 {
        let jd_tdb = self.tdb();
        self.tdb_fraction
            .get()
            .copied()
            .unwrap_or(jd_tdb - self.whole)
    }

    /// Rebuild a time from its serialised parts, rejecting any component
    /// that is NaN or infinite.
    fn from_parts(
        whole: f64,
        tt_fraction: f64,
        tai_fraction: Option<f64>,
        tdb_fraction: Option<f64>,
        leap_second: bool,
    ) -> Result<Time, String> {
        require_finite("whole", whole)?;
        require_finite("tt_fraction", tt_fraction)?;
        if let Some(tai) = tai_fraction {
            require_finite("tai_fraction", tai)?;
        }
        if let Some(tdb) = tdb_fraction {
            require_finite("tdb_fraction", tdb)?;
        }
        Ok(Time {
            ts: Timescale::default(),
            whole,
            tt_fraction,
            tai_fraction,
            ut1_fraction: OnceLock::new(),
            tdb_fraction: tdb_fraction.map_or_else(OnceLock::new, once_with),
            delta_t: OnceLock::new(),
            shape: None,
            leap_second,
        })
    }
}

fn require_finite(field: &str, value: f64) -> Result<(), String> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(format!("`{field}` must be finite, got {value}"))
    }
}

impl Serialize for Time {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Refuse non-finite parts before deriving anything from them: the
        // output could not be read back, and the UTC lookup assumes a
        // finite date.
        require_finite("whole", self.whole).map_err(ser::Error::custom)?;
        require_finite("tt_fraction", self.tt_fraction).map_err(ser::Error::custom)?;
        if let Some(tai) = self.tai_fraction {
            require_finite("tai_fraction", tai).map_err(ser::Error::custom)?;
        }
        let tdb_fraction = self.tdb_fraction_value();
        require_finite("tdb_fraction", tdb_fraction).map_err(ser::Error::custom)?;
        if !serializer.is_human_readable() {
            return Compact {
                whole: self.whole,
                tt_fraction: self.tt_fraction,
                tai_fraction: self.tai_fraction,
                tdb_fraction,
                leap_second: self.leap_second,
            }
            .serialize(serializer);
        }
        let jd_tdb = self.whole + tdb_fraction;
        // The calendar formatter cannot print second 60, so a leap-second
        // instant omits `utc` rather than show the following midnight.
        let utc = if self.leap_second {
            None
        } else {
            self.utc_iso('T', 6).ok()
        };

        let mut state = serializer.serialize_struct("Time", FIELDS.len())?;
        state.serialize_field("whole", &self.whole)?;
        state.serialize_field("tt_fraction", &self.tt_fraction)?;
        match self.tai_fraction {
            Some(tai) => state.serialize_field("tai_fraction", &tai)?,
            None => state.skip_field("tai_fraction")?,
        }
        state.serialize_field("tdb_fraction", &tdb_fraction)?;
        if self.leap_second {
            state.serialize_field("leap_second", &true)?;
        } else {
            state.skip_field("leap_second")?;
        }
        state.serialize_field("jd_tdb", &jd_tdb)?;
        match utc {
            Some(utc) => state.serialize_field("utc", &utc)?,
            None => state.skip_field("utc")?,
        }
        state.end()
    }
}

struct TimeVisitor;

impl<'de> Visitor<'de> for TimeVisitor {
    type Value = Time;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "a map with `whole` and `tt_fraction` (or `jd_tdb`), \
             or an ISO 8601 / `JD <number> <scale>` string",
        )
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<Time, E> {
        Timescale::default().parse(value).map_err(E::custom)
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Time, M::Error> {
        let mut whole: Option<f64> = None;
        let mut tt_fraction: Option<f64> = None;
        let mut tai_fraction: Option<f64> = None;
        let mut tdb_fraction: Option<f64> = None;
        let mut leap_second: Option<bool> = None;
        let mut jd_tdb: Option<f64> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "whole" => whole = Some(map.next_value()?),
                "tt_fraction" => tt_fraction = Some(map.next_value()?),
                "tai_fraction" => tai_fraction = Some(map.next_value()?),
                "tdb_fraction" => tdb_fraction = Some(map.next_value()?),
                "leap_second" => leap_second = Some(map.next_value()?),
                "jd_tdb" => jd_tdb = Some(map.next_value()?),
                _ => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }

        match (whole, tt_fraction) {
            (Some(whole), Some(tt_fraction)) => Time::from_parts(
                whole,
                tt_fraction,
                tai_fraction,
                tdb_fraction,
                leap_second.unwrap_or(false),
            )
            .map_err(de::Error::custom),
            (Some(_), None) => Err(de::Error::missing_field("tt_fraction")),
            (None, Some(_)) => Err(de::Error::missing_field("whole")),
            (None, None) => {
                let jd = jd_tdb.ok_or_else(|| de::Error::missing_field("whole"))?;
                require_finite("jd_tdb", jd).map_err(de::Error::custom)?;
                Ok(Timescale::default().tdb_jd(jd))
            }
        }
    }
}

impl<'de> Deserialize<'de> for Time {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        if deserializer.is_human_readable() {
            deserializer.deserialize_any(TimeVisitor)
        } else {
            let c = Compact::deserialize(deserializer)?;
            Time::from_parts(
                c.whole,
                c.tt_fraction,
                c.tai_fraction,
                Some(c.tdb_fraction),
                c.leap_second,
            )
            .map_err(de::Error::custom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serde_json::Value;

    #[test]
    fn non_finite_times_refuse_to_serialise() {
        use serde_test::{assert_ser_tokens_error, Configure};
        let ts = Timescale::default();
        for (t, message) in [
            (ts.tt_jd(f64::NAN, None), "`whole` must be finite, got NaN"),
            (
                ts.tt_jd(f64::INFINITY, None),
                "`whole` must be finite, got inf",
            ),
            (
                ts.tt_jd(2_461_558.0, Some(f64::NAN)),
                "`tt_fraction` must be finite, got NaN",
            ),
        ] {
            let err = serde_json::to_string(&t).unwrap_err().to_string();
            assert_eq!(err, message);
            assert_ser_tokens_error(&t.clone().compact(), &[], message);
            assert_ser_tokens_error(&t.readable(), &[], message);
        }
    }

    #[test]
    fn leap_second_omits_utc_and_round_trips_its_flag() {
        let t: Time = "2016-12-31T23:59:60Z".parse().unwrap();
        assert!(t.is_leap_second());
        let v: Value = serde_json::to_value(&t).unwrap();
        assert!(v.get("utc").is_none(), "{v}");
        assert_eq!(v["leap_second"], true);
        let back: Time = serde_json::from_value(v).unwrap();
        assert!(back.is_leap_second());
        assert_eq!(back.tt().to_bits(), t.tt().to_bits());
        assert_eq!(back.tai().to_bits(), t.tai().to_bits());

        let ordinary: Time = "2016-12-31T23:59:59Z".parse().unwrap();
        let v: Value = serde_json::to_value(&ordinary).unwrap();
        assert!(v["utc"]
            .as_str()
            .is_some_and(|u| u.starts_with("2016-12-31T23:59:59")));
    }

    fn round_trip(t: &Time) -> Time {
        let json = serde_json::to_string(t).unwrap();
        serde_json::from_str(&json).unwrap()
    }

    fn assert_bit_identical(a: &Time, b: &Time) {
        assert_eq!(a, b);
        assert_eq!(a.whole.to_bits(), b.whole.to_bits());
        assert_eq!(a.tt_fraction.to_bits(), b.tt_fraction.to_bits());
        assert_eq!(a.tt().to_bits(), b.tt().to_bits());
        assert_eq!(a.tai().to_bits(), b.tai().to_bits());
        assert_eq!(a.tdb().to_bits(), b.tdb().to_bits());
        assert_eq!(a.is_leap_second(), b.is_leap_second());
    }

    #[test]
    fn round_trip_is_exact_for_every_constructor() {
        let ts = Timescale::default();
        let times = [
            ts.parse("2027-06-01T00:00:00Z").unwrap(),
            ts.parse("2016-12-31T23:59:60.5Z").unwrap(),
            ts.parse("1987-03-14T01:59:26.535897Z").unwrap(),
            ts.tt_jd(2_451_545.0, Some(0.123_456_789_012_345)),
            ts.tdb_jd(2_461_558.5),
            ts.tai_jd(2_440_000.25, None),
            ts.ut1_jd(2_455_000.75),
            ts.j(2100.5),
            ts.tdb_jd(2_461_558.5).add_seconds(3_600.0),
        ];
        for t in &times {
            assert_bit_identical(t, &round_trip(t));
        }
    }

    #[test]
    fn compact_form_is_fixed_layout_and_exact() {
        use serde_test::{assert_tokens, Configure, Token};
        let t = Timescale::default().tdb_jd(2_461_558.5);
        let tdb_fraction = t.tdb_fraction_value();
        assert_tokens(
            &t.clone().compact(),
            &[
                Token::Struct {
                    name: "Time",
                    len: 5,
                },
                Token::Str("whole"),
                Token::F64(t.whole),
                Token::Str("tt_fraction"),
                Token::F64(t.tt_fraction),
                Token::Str("tai_fraction"),
                Token::Some,
                Token::F64(t.tai_fraction.unwrap()),
                Token::Str("tdb_fraction"),
                Token::F64(tdb_fraction),
                Token::Str("leap_second"),
                Token::Bool(false),
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn serialised_fields_are_documented_shape() {
        let t: Time = "2027-06-01T00:00:00Z".parse().unwrap();
        let v: Value = serde_json::to_value(&t).unwrap();
        assert_eq!(v["whole"], 2_461_557.0);
        assert!(v["tt_fraction"].is_f64());
        assert!(v["tai_fraction"].is_f64());
        assert!(v["tdb_fraction"].is_f64());
        assert!(v.get("leap_second").is_none());
        assert_eq!(v["utc"], "2027-06-01T00:00:00.000000Z");
        assert_abs_diff_eq!(v["jd_tdb"].as_f64().unwrap(), t.tdb(), epsilon = 0.0);

        let leap: Time = "2016-12-31T23:59:60Z".parse().unwrap();
        let v: Value = serde_json::to_value(&leap).unwrap();
        assert_eq!(v["leap_second"], true);
    }

    #[test]
    fn tdb_round_trip_returns_the_input_julian_date() {
        let t = Timescale::default().tdb_jd(2_461_558.5);
        assert_eq!(round_trip(&t).tdb(), 2_461_558.5);
    }

    #[test]
    fn informational_fields_are_ignored_when_parts_are_present() {
        let json = r#"{"whole": 2451545.0, "tt_fraction": 0.0,
                       "jd_tdb": 1.0, "utc": "garbage", "extra": [1, 2]}"#;
        let t: Time = serde_json::from_str(json).unwrap();
        assert_eq!(t.tt(), 2_451_545.0);
    }

    #[test]
    fn jd_tdb_alone_is_accepted() {
        let t: Time = serde_json::from_str(r#"{"jd_tdb": 2461558.5, "utc": "x"}"#).unwrap();
        assert_eq!(t.tdb(), 2_461_558.5);
    }

    #[test]
    fn strings_are_parsed() {
        let a: Time = serde_json::from_str(r#""2027-06-01T00:00:00Z""#).unwrap();
        let b: Time = "2027-06-01T00:00:00Z".parse().unwrap();
        assert_eq!(a, b);
        let c: Time = serde_json::from_str(r#""JD 2461558.5 TDB""#).unwrap();
        assert_eq!(c.tdb(), 2_461_558.5);
        assert!(serde_json::from_str::<Time>(r#""soon""#).is_err());
    }

    #[test]
    fn incomplete_maps_are_rejected() {
        for json in [
            r#"{"whole": 2451545.0}"#,
            r#"{"tt_fraction": 0.5}"#,
            r#"{}"#,
        ] {
            assert!(serde_json::from_str::<Time>(json).is_err(), "{json}");
        }
    }

    #[test]
    fn non_finite_map_components_are_rejected() {
        use serde_test::{assert_de_tokens_error, Token};
        let cases: [(&[(&str, f64)], &str); 6] = [
            (
                &[("whole", f64::NAN), ("tt_fraction", 0.0)],
                "`whole` must be finite, got NaN",
            ),
            (
                &[("whole", 2_451_545.0), ("tt_fraction", f64::INFINITY)],
                "`tt_fraction` must be finite, got inf",
            ),
            (
                &[
                    ("whole", 2_451_545.0),
                    ("tt_fraction", 0.0),
                    ("tai_fraction", f64::NAN),
                ],
                "`tai_fraction` must be finite, got NaN",
            ),
            (
                &[
                    ("whole", 2_451_545.0),
                    ("tt_fraction", 0.0),
                    ("tdb_fraction", f64::NEG_INFINITY),
                ],
                "`tdb_fraction` must be finite, got -inf",
            ),
            (
                &[("jd_tdb", f64::INFINITY)],
                "`jd_tdb` must be finite, got inf",
            ),
            (&[("jd_tdb", f64::NAN)], "`jd_tdb` must be finite, got NaN"),
        ];
        for (fields, message) in cases {
            let mut tokens = vec![Token::Map {
                len: Some(fields.len()),
            }];
            for (key, value) in fields {
                tokens.push(Token::Str(key));
                tokens.push(Token::F64(*value));
            }
            tokens.push(Token::MapEnd);
            assert_de_tokens_error::<serde_test::Readable<Time>>(&tokens, message);
        }
    }

    #[test]
    fn non_finite_compact_components_are_rejected() {
        use serde_test::{assert_de_tokens_error, Token};
        let compact = |whole: f64, tt: f64, tai: Option<f64>, tdb: f64| {
            let mut tokens = vec![
                Token::Struct {
                    name: "Time",
                    len: 5,
                },
                Token::Str("whole"),
                Token::F64(whole),
                Token::Str("tt_fraction"),
                Token::F64(tt),
                Token::Str("tai_fraction"),
            ];
            match tai {
                Some(v) => tokens.extend([Token::Some, Token::F64(v)]),
                None => tokens.push(Token::None),
            }
            tokens.extend([
                Token::Str("tdb_fraction"),
                Token::F64(tdb),
                Token::Str("leap_second"),
                Token::Bool(false),
                Token::StructEnd,
            ]);
            tokens
        };
        for (tokens, message) in [
            (
                compact(f64::INFINITY, 0.0, None, 0.0),
                "`whole` must be finite, got inf",
            ),
            (
                compact(2_451_545.0, f64::NAN, None, 0.0),
                "`tt_fraction` must be finite, got NaN",
            ),
            (
                compact(2_451_545.0, 0.0, Some(f64::NAN), 0.0),
                "`tai_fraction` must be finite, got NaN",
            ),
            (
                compact(2_451_545.0, 0.0, None, f64::INFINITY),
                "`tdb_fraction` must be finite, got inf",
            ),
        ] {
            assert_de_tokens_error::<serde_test::Compact<Time>>(&tokens, message);
        }
    }

    #[test]
    fn embeds_in_derived_structs() {
        #[derive(Serialize, Deserialize)]
        struct Record {
            label: String,
            epoch: Time,
        }
        let rec = Record {
            label: "obs".into(),
            epoch: "2027-06-01T12:34:56.789Z".parse().unwrap(),
        };
        let back: Record = serde_json::from_str(&serde_json::to_string(&rec).unwrap()).unwrap();
        assert_eq!(back.label, "obs");
        assert_bit_identical(&back.epoch, &rec.epoch);
    }
}
