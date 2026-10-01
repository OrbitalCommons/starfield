//! Parsing [`Time`] from text.
//!
//! Two forms are accepted, both case-insensitive and tolerant of
//! surrounding whitespace:
//!
//! ```text
//! time      = calendar / julian
//!
//! calendar  = date [ sep clock [ zone ] ]
//! date      = [ "+" / "-" ] 4*DIGIT "-" 2DIGIT "-" 2DIGIT
//! sep       = "T" / " "
//! clock     = 2DIGIT ":" 2DIGIT [ ":" 2DIGIT [ ( "." / "," ) 1*DIGIT ] ]
//! zone      = "Z" / [ " " ] "UTC" / ( "+" / "-" ) 2DIGIT [ [ ":" ] 2DIGIT ]
//!
//! julian    = "JD" [ " " ] number " " scale
//! number    = [ "+" / "-" ] 1*DIGIT [ "." *DIGIT ]
//! scale     = "TT" / "TDB" / "TAI" / "UT1"
//! ```
//!
//! A calendar string is an ISO 8601 / RFC 3339 instant in **UTC**: a
//! string without a zone designator is read as UTC, and a numeric offset
//! (`+05:30`) is removed to give UTC. A date with no clock is midnight
//! UTC. Seconds may be `60` during a leap second (`2016-12-31T23:59:60Z`),
//! which sets [`Time::is_leap_second`].
//!
//! A Julian date literal must name its timescale (`JD 2461558.5 TDB`):
//! TT and TDB differ by up to 1.7 ms and UT1 by about a minute, so a bare
//! number would be ambiguous. The integer and fractional digits are
//! converted separately, so a literal keeps sub-microsecond resolution
//! that a single `f64` Julian date could not hold.

use std::str::FromStr;
use std::sync::OnceLock;

use chrono::{Datelike, Duration, NaiveDate, Timelike};
use regex::Regex;

use super::{Result, Time, TimeError, Timescale};

/// ISO 8601 calendar instant, captured as date, clock and zone groups.
fn calendar_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?ix)^
            (?P<year>[+-]?\d{4,})-(?P<month>\d{2})-(?P<day>\d{2})
            (?:
                [T\x20]
                (?P<hour>\d{2}):(?P<minute>\d{2})
                (?::(?P<second>\d{2}(?:[.,]\d+)?))?
                (?P<zone>
                    Z
                    | \x20?UTC
                    | (?P<sign>[+-])(?P<zh>\d{2})(?::?(?P<zm>\d{2}))?
                )?
            )?
            $",
        )
        .expect("calendar pattern is valid")
    })
}

/// Julian date literal with an explicit timescale.
fn julian_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(
            r"(?ix)^
            JD\x20?
            (?P<sign>[+-])?(?P<int>\d+)(?:\.(?P<frac>\d*))?
            \x20+
            (?P<scale>TT|TDB|TAI|UT1)
            $",
        )
        .expect("julian pattern is valid")
    })
}

fn parse_error(input: &str, reason: &str) -> TimeError {
    TimeError::ParseError(format!(
        "cannot parse {input:?} as a time ({reason}); expected an ISO 8601 UTC instant \
         such as 2027-06-01T00:00:00Z or a Julian date such as 'JD 2461558.5 TDB'"
    ))
}

impl Timescale {
    /// Parse an ISO 8601 / RFC 3339 UTC instant or a `JD <number> <scale>`
    /// literal into a [`Time`] on this timescale.
    ///
    /// See the [module documentation](self) for the accepted grammar.
    /// [`Time::from_str`] is the same parser on [`Timescale::default`].
    ///
    /// # Example
    ///
    /// ```
    /// use starfield_core::time::Timescale;
    ///
    /// let ts = Timescale::default();
    /// let a = ts.parse("2000-01-01T11:58:55.816Z").unwrap();
    /// let b = ts.parse("JD 2451545.0 TT").unwrap();
    /// assert!(a.seconds_since(&b).abs() < 1e-3);
    /// ```
    pub fn parse(&self, input: &str) -> Result<Time> {
        let trimmed = input.trim();
        if let Some(caps) = julian_pattern().captures(trimmed) {
            return Ok(self.parse_julian(&caps));
        }
        if let Some(caps) = calendar_pattern().captures(trimmed) {
            return self.parse_calendar(input, &caps);
        }
        Err(parse_error(input, "unrecognised format"))
    }

    fn parse_julian(&self, caps: &regex::Captures<'_>) -> Time {
        let negative = caps.name("sign").is_some_and(|m| m.as_str() == "-");
        let sign = if negative { -1.0 } else { 1.0 };
        // `\d+` always parses; the fractional digits are read as `0.<digits>`
        // so the day fraction keeps full f64 resolution.
        let whole: f64 = caps["int"].parse::<f64>().unwrap_or(0.0) * sign;
        let fraction = caps
            .name("frac")
            .filter(|m| !m.as_str().is_empty())
            .map(|m| format!("0.{}", m.as_str()).parse::<f64>().unwrap_or(0.0))
            .unwrap_or(0.0)
            * sign;

        match caps["scale"].to_ascii_uppercase().as_str() {
            "TT" => self.tt_jd(whole, Some(fraction)),
            "TAI" => self.tai_jd(whole, Some(fraction)),
            "TDB" => self.tdb_jd_parts(whole, fraction),
            _ => self.ut1_jd(whole + fraction),
        }
    }

    fn parse_calendar(&self, input: &str, caps: &regex::Captures<'_>) -> Result<Time> {
        let field = |name: &str| caps.name(name).map(|m| m.as_str());
        let int = |name: &str| -> Result<i64> {
            field(name)
                .unwrap_or("0")
                .parse::<i64>()
                .map_err(|_| parse_error(input, &format!("bad {name} field")))
        };

        let year =
            i32::try_from(int("year")?).map_err(|_| parse_error(input, "year out of range"))?;
        let month = int("month")? as u32;
        let day = int("day")? as u32;
        let hour = int("hour")? as u32;
        let minute = int("minute")? as u32;
        let second: f64 = field("second")
            .unwrap_or("0")
            .replace(',', ".")
            .parse()
            .map_err(|_| parse_error(input, "bad second field"))?;

        if hour > 23 || minute > 59 || second >= 61.0 {
            return Err(parse_error(input, "clock field out of range"));
        }
        let leap = second >= 60.0;
        if leap && (hour != 23 || minute != 59) && caps.name("sign").is_none() {
            return Err(parse_error(input, "second 60 only occurs at 23:59 UTC"));
        }

        let offset_minutes = match caps.name("sign") {
            Some(sign) => {
                let zh = int("zh")?;
                let zm = int("zm")?;
                if zh > 23 || zm > 59 {
                    return Err(parse_error(input, "zone offset out of range"));
                }
                let magnitude = zh * 60 + zm;
                if sign.as_str() == "-" {
                    -magnitude
                } else {
                    magnitude
                }
            }
            None => 0,
        };

        // Normalise the wall-clock fields to UTC with chrono. The leap
        // second, if any, is held back and re-added after normalising so
        // the timescale sees `second = 60.x` on the UTC calendar.
        let whole_second = second.floor();
        let clamped_second = whole_second.min(59.0);
        let carried = second - clamped_second;
        let local = NaiveDate::from_ymd_opt(year, month, day)
            .and_then(|d| d.and_hms_opt(hour, minute, clamped_second as u32))
            .ok_or_else(|| parse_error(input, "no such calendar date"))?;
        let utc = local
            .checked_sub_signed(Duration::minutes(offset_minutes))
            .ok_or_else(|| parse_error(input, "date out of range"))?;

        if leap && (utc.hour() != 23 || utc.minute() != 59) {
            return Err(parse_error(input, "second 60 only occurs at 23:59 UTC"));
        }

        Ok(self.utc((
            utc.year(),
            utc.month(),
            utc.day(),
            utc.hour(),
            utc.minute(),
            utc.second() as f64 + carried,
        )))
    }
}

impl FromStr for Time {
    type Err = TimeError;

    /// Parse on [`Timescale::default`]; see [`Timescale::parse`].
    fn from_str(s: &str) -> Result<Self> {
        Timescale::default().parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{DAY_S, J2000};
    use approx::assert_abs_diff_eq;

    #[test]
    fn j2000_utc_string_matches_known_tt() {
        // 2000-01-01T12:00:00 UTC = TT 12:01:04.184 (32 leap seconds + 32.184 s).
        let t: Time = "2000-01-01T12:00:00Z".parse().unwrap();
        assert_abs_diff_eq!((t.tt() - J2000) * DAY_S, 64.184, epsilon = 1e-4);
    }

    #[test]
    fn zone_forms_agree() {
        let z: Time = "2027-06-01T00:00:00Z".parse().unwrap();
        for s in [
            "2027-06-01T00:00:00",
            "2027-06-01 00:00:00 UTC",
            "2027-06-01t00:00:00z",
            "2027-06-01T00:00",
            "2027-06-01",
            "2027-06-01T05:30:00+05:30",
            "2027-06-01T05:30:00+0530",
            "2027-05-31T19:00:00-05",
            "  2027-06-01T00:00:00.000Z  ",
            "2027-06-01T00:00:00,000Z",
        ] {
            let t: Time = s.parse().unwrap_or_else(|e| panic!("{s}: {e}"));
            assert_abs_diff_eq!(t.seconds_since(&z), 0.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn fractional_seconds_are_kept() {
        let a: Time = "2027-06-01T00:00:00Z".parse().unwrap();
        let b: Time = "2027-06-01T00:00:01.25Z".parse().unwrap();
        assert_abs_diff_eq!(b.seconds_since(&a), 1.25, epsilon = 1e-4);
    }

    #[test]
    fn leap_second_is_flagged_and_counted() {
        let before: Time = "2016-12-31T23:59:59Z".parse().unwrap();
        let leap: Time = "2016-12-31T23:59:60Z".parse().unwrap();
        let after: Time = "2017-01-01T00:00:00Z".parse().unwrap();
        assert!(leap.is_leap_second());
        assert!(!before.is_leap_second());
        assert_abs_diff_eq!(leap.seconds_since(&before), 1.0, epsilon = 1e-4);
        assert_abs_diff_eq!(after.seconds_since(&before), 2.0, epsilon = 1e-4);
    }

    #[test]
    fn leap_second_through_an_offset() {
        let utc: Time = "2016-12-31T23:59:60Z".parse().unwrap();
        let local: Time = "2017-01-01T00:59:60+01:00".parse().unwrap();
        assert!(local.is_leap_second());
        assert_eq!(utc, local);
    }

    #[test]
    fn julian_literals_respect_their_scale() {
        let ts = Timescale::default();
        let tt = ts.parse("JD 2451545.0 TT").unwrap();
        assert_eq!(tt.tt(), 2_451_545.0);

        let tdb = ts.parse("jd2461558.5 tdb").unwrap();
        assert_eq!(tdb.tdb(), 2_461_558.5);

        let tai = ts.parse("JD 2451545 TAI").unwrap();
        assert_eq!(tai.tai(), 2_451_545.0);

        let ut1 = ts.parse("JD 2451545.0 UT1").unwrap();
        assert_abs_diff_eq!(ut1.ut1(), 2_451_545.0, epsilon = 1e-9);

        let negative = ts.parse("JD -0.25 TT").unwrap();
        assert_eq!(negative.tt(), -0.25);
    }

    #[test]
    fn julian_literal_keeps_sub_microsecond_fraction() {
        let t: Time = "JD 2461558.123456789012 TT".parse().unwrap();
        assert_eq!(t.whole, 2_461_558.0);
        assert_abs_diff_eq!(t.tt_fraction, 0.123456789012, epsilon = 1e-17);
    }

    #[test]
    fn rejects_malformed_input() {
        for s in [
            "",
            "next tuesday",
            "JD 2461558.5",
            "JD 2461558.5 UTC",
            "2027-13-01T00:00:00Z",
            "2027-02-30T00:00:00Z",
            "2027-06-01T24:00:00Z",
            "2027-06-01T12:00:60Z",
            "2027-06-01T00:00:00+25:00",
            "2027-06-01T00:00:00Q",
        ] {
            assert!(
                matches!(s.parse::<Time>(), Err(TimeError::ParseError(_))),
                "{s:?} should not parse"
            );
        }
    }

    #[test]
    fn parse_keeps_the_calling_timescale() {
        let ts = Timescale::default();
        let t = ts.parse("2027-06-01T00:00:00Z").unwrap();
        assert!(std::sync::Arc::ptr_eq(&t.ts.0, &ts.0));
    }
}
