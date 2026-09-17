use chrono::{DateTime, TimeZone, Utc};
use starfield::Timescale;

fn main() {
    println!("Starfield Timescale Demonstration");
    println!("=================================\n");

    // Create a timescale
    let ts = Timescale::default();
    println!("Created default timescale");

    // Get the current time
    let now = ts.now();
    println!("\nCurrent time: {}", now);

    // Create a time at J2000
    let j2000 = ts.tt_jd(2451545.0, None);
    println!("\nJ2000 epoch: {}", j2000);

    // Create a time from a specific date (UTC)
    let millennium = ts.utc((2000, 1, 1, 0, 0, 0.0));
    println!("\nMillennium (2000-01-01 00:00:00 UTC): {}", millennium);

    // Get time in different scales
    println!("\nTime scales for millennium:");
    println!("  UTC:  {}", millennium.utc_iso('T', 3).unwrap());
    println!("  TAI:  {:.6} (Julian date)", millennium.tai());
    println!("  TT:   {:.6} (Julian date)", millennium.tt());
    println!("  TDB:  {:.6} (Julian date)", millennium.tdb());
    println!("  UT1:  {:.6} (Julian date)", millennium.ut1());

    // Convert to calendar representations
    let tt_cal = millennium.tt_calendar();
    println!(
        "\nTT calendar: {:04}-{:02}-{:02} {:02}:{:02}:{:09.6}",
        tt_cal.year, tt_cal.month, tt_cal.day, tt_cal.hour, tt_cal.minute, tt_cal.second
    );

    // Show differences between time scales
    println!("\nTime scale differences:");
    println!(
        "  TT - TAI = {:.6} seconds",
        (millennium.tt() - millennium.tai()) * 86400.0
    );
    println!("  TT - UT1 = {:.6} seconds (Delta T)", millennium.delta_t());
    println!(
        "  TDB - TT = {:.6} seconds",
        (millennium.tdb() - millennium.tt()) * 86400.0
    );

    // Demonstrate time math
    let later = millennium.clone() + 1.0; // Add 1 day
    println!("\nOne day after millennium: {}", later);
    println!("Difference: {:.6} days", later - millennium);

    // Future time
    let future = ts.utc((2050, 1, 1, 0, 0, 0.0));
    println!("\nFuture date (2050-01-01): {}", future);
    println!("Delta T in 2050: {:.6} seconds", future.delta_t());

    // Historical time with delta T
    let historical = ts.utc((1600, 1, 1, 0, 0, 0.0));
    println!("\nHistorical date (1600-01-01): {}", historical);
    println!("Delta T in 1600: {:.6} seconds", historical.delta_t());

    // Time span with linspace
    println!("\nCreating a time span with 5 points from J2000 to J2000+10 days:");
    let span = ts.linspace(&j2000, &(j2000.clone() + 10.0), 5);
    for (i, t) in span.iter().enumerate() {
        println!("  Point {}: {}", i, t);
    }

    // Create from chrono DateTime
    let chrono_date = Utc.with_ymd_and_hms(2020, 1, 1, 0, 0, 0).unwrap();
    let from_chrono = ts.from_datetime(chrono_date);
    println!(
        "\nFrom chrono DateTime (2020-01-01 00:00:00 UTC): {}",
        from_chrono
    );

    // Convert back to chrono DateTime
    let to_chrono: DateTime<Utc> = from_chrono.utc_datetime().unwrap();
    println!("Back to chrono DateTime: {}", to_chrono);
}
