//! Calendar date and Julian date conversion functions

/// Convert Julian day integer to calendar date (year, month, day)
///
/// Uses the proleptic Gregorian calendar unless `julian_before` is set to a
/// specific Julian day, in which case the Julian calendar is used for dates
/// older than that.
pub fn compute_calendar_date(jd_integer: i32, julian_before: Option<i32>) -> (i32, i32, i32) {
    let use_gregorian = match julian_before {
        None => true,
        Some(jb) => jd_integer >= jb,
    };

    // See the Explanatory Supplement to the Astronomical Almanac 15.11.
    let f = jd_integer + 1401;
    let f = if use_gregorian {
        f + ((4 * jd_integer + 274277) / 146097 * 3 / 4 - 38)
    } else {
        f
    };

    let e = 4 * f + 3;
    let g = (e % 1461) / 4;
    let h = 5 * g + 2;
    let day = (h % 153) / 5 + 1;
    let month = (h / 153 + 2) % 12 + 1;
    let year = e / 1461 - 4716 + (12 + 2 - month) / 12;

    (year, month, day)
}

/// Convert (year, month, day) to Julian date float
pub fn compute_julian_date(year: i32, month: i32, day: f64) -> f64 {
    let day_int = day.floor() as i32;
    let jd_noon = compute_julian_day(year, month, day_int);
    (jd_noon as f64 - 0.5) + day.fract()
}

/// Convert (year, month, day) to Julian day integer (proleptic Gregorian)
pub fn compute_julian_day(year: i32, month: i32, day: i32) -> i32 {
    let janfeb = month < 3;
    1461 * (year + 4800 - if janfeb { 1 } else { 0 }) / 4
        + 367 * (month - 2 + if janfeb { 12 } else { 0 }) / 12
        - 3 * ((year + 4900 - if janfeb { 1 } else { 0 }) / 100) / 4
        - 32075
        + day
}

/// Format a Julian date as YYYY-MM-DD
pub fn format_date(jd: f64) -> String {
    let (year, month, day) = calendar_date_from_float(jd);
    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Convert a floating-point Julian date to a calendar date
pub fn calendar_date_from_float(jd: f64) -> (i32, i32, i32) {
    let i = (jd + 0.5) as i32;
    compute_calendar_date(i, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_julian_day_conversion() {
        assert_eq!(compute_julian_day(2000, 1, 1), 2451545);
        assert_eq!(compute_julian_day(2020, 1, 1), 2458850);
        assert_eq!(compute_julian_day(1969, 7, 20), 2440423);
        assert_eq!(compute_julian_day(1900, 1, 1), 2415021);
    }

    #[test]
    fn test_calendar_date_conversion() {
        assert_eq!(compute_calendar_date(2451545, None), (2000, 1, 1));
        assert_eq!(compute_calendar_date(2458850, None), (2020, 1, 1));
        assert_eq!(compute_calendar_date(2440423, None), (1969, 7, 20));
        assert_eq!(compute_calendar_date(2415021, None), (1900, 1, 1));
    }

    #[test]
    fn test_julian_date_conversion() {
        assert_eq!(compute_julian_date(2000, 1, 1.0), 2451544.5);
        assert_eq!(compute_julian_date(2020, 1, 1.5), 2458850.0);
        assert_eq!(compute_julian_date(1969, 7, 20.0), 2440422.5);
        assert_eq!(compute_julian_date(1900, 1, 1.0), 2415020.5);
    }
}
