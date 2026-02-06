//! Tool for filtering Gaia catalog files to keep only bright stars
//!
//! This utility filters Gaia catalog files to keep only stars brighter than
//! a specified magnitude threshold (default: 20.0) and saves a smaller file
//! containing only essential fields: source_id, ra, dec, and phot_g_mean_mag.

use std::collections::VecDeque;
use std::fmt::Display;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use clap::{Parser, Subcommand};
use flate2::read::GzDecoder;
use flate2::write;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use starfield::catalogs::{MinimalCatalog, StarData};
use starfield::data::list_cached_gaia_files;

/// Command line arguments for the Gaia Catalog Filter Tool
#[derive(Parser)]
#[command(name = "gaia_filter")]
#[command(about = "Filters Gaia catalog files to keep only bright stars")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Output file path (.bin extension)
    #[arg(long, global = true)]
    output: Option<String>,

    /// Maximum magnitude threshold
    #[arg(long, global = true, default_value_t = 20.0)]
    magnitude: f64,

    /// Number of threads for parallel processing
    #[arg(long, global = true, default_value_t = 1)]
    threads: usize,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a single Gaia catalog file
    Single {
        /// Input Gaia catalog file (CSV or gzipped CSV)
        #[arg(long)]
        input: String,
    },

    /// Process all cached Gaia files
    All {
        /// Maximum number of files to process
        #[arg(long)]
        max_files: Option<usize>,
    },

    /// List cached Gaia catalog files
    List,
}

/// Parses CSV headers to find column indices for required fields
fn parse_headers(
    headers: &[&str],
) -> Result<(usize, usize, usize, usize), Box<dyn std::error::Error>> {
    let find_column = |name: &str| -> Result<usize, Box<dyn std::error::Error>> {
        headers
            .iter()
            .position(|&h| h == name)
            .ok_or_else(|| format!("Missing column: {}", name).into())
    };

    // Find required column indices
    let source_id_idx = find_column("source_id")?;
    let ra_idx = find_column("ra")?;
    let dec_idx = find_column("dec")?;
    let g_mag_idx = find_column("phot_g_mean_mag")?;

    Ok((source_id_idx, ra_idx, dec_idx, g_mag_idx))
}

/// Prints an error message to stderr in red text
fn print_error<T: Display, M: Display>(message: M, error: T) {
    let stderr = std::io::stderr();
    let mut handle = stderr.lock();
    let _ = writeln!(handle, "\x1b[31m{}: {}\x1b[0m", message, error);
}

/// Create a reader for a file, handling gzip if needed
fn create_reader(path: &Path) -> Result<Box<dyn BufRead>, Box<dyn std::error::Error>> {
    let input_file = File::open(path)?;

    // Determine if the file is gzipped or not
    let path_str = path.to_string_lossy().to_string();
    let is_gzipped = path_str.ends_with(".gz");

    // Create appropriate reader
    let reader: Box<dyn BufRead> = if is_gzipped {
        let decoder = GzDecoder::new(input_file);
        Box::new(BufReader::new(decoder))
    } else {
        Box::new(BufReader::new(input_file))
    };

    Ok(reader)
}

/// Creates an iterator over StarData from a CSV reader
struct GaiaFileIterator<R: BufRead> {
    reader: R,
    source_id_idx: usize,
    ra_idx: usize,
    dec_idx: usize,
    g_mag_idx: usize,
    magnitude_limit: f64,
    progress_bar: Option<ProgressBar>,
}

impl<R: BufRead> GaiaFileIterator<R> {
    fn new(
        reader: R,
        source_id_idx: usize,
        ra_idx: usize,
        dec_idx: usize,
        g_mag_idx: usize,
        magnitude_limit: f64,
        progress_bar: Option<ProgressBar>,
    ) -> Self {
        Self {
            reader,
            source_id_idx,
            ra_idx,
            dec_idx,
            g_mag_idx,
            magnitude_limit,
            progress_bar,
        }
    }
}

impl<R: BufRead> Iterator for GaiaFileIterator<R> {
    type Item = StarData;

    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();

        // Read lines until we find a valid star or reach EOF
        loop {
            line.clear();
            match self.reader.read_line(&mut line) {
                Ok(0) => return None, // End of file
                Ok(_) => {
                    // Update progress bar if available
                    if let Some(pb) = &self.progress_bar {
                        pb.inc(1);
                    }

                    // Skip empty lines
                    if line.trim().is_empty() {
                        continue;
                    }

                    // Split line into fields
                    let fields: Vec<&str> = line.trim().split(',').collect();

                    // Skip lines with insufficient fields
                    if fields.len() <= self.g_mag_idx
                        || fields.len() <= self.ra_idx
                        || fields.len() <= self.dec_idx
                        || fields.len() <= self.source_id_idx
                    {
                        let needed = 1 + [
                            self.g_mag_idx,
                            self.ra_idx,
                            self.dec_idx,
                            self.source_id_idx,
                        ]
                        .iter()
                        .max()
                        .unwrap();

                        print_error(
                            format!(
                                "Insufficient fields in line (found {}, needed at least {})",
                                fields.len(),
                                needed
                            ),
                            line.trim(),
                        );
                        continue;
                    }

                    // Parse the magnitude first for early filtering
                    let g_mag = match fields[self.g_mag_idx].parse::<f64>() {
                        Ok(mag) => mag,
                        Err(e) => {
                            print_error(
                                format!(
                                    "Error parsing magnitude from '{}'",
                                    fields[self.g_mag_idx]
                                ),
                                e,
                            );
                            continue;
                        }
                    };

                    // Skip stars fainter than magnitude limit
                    if g_mag > self.magnitude_limit {
                        continue;
                    }

                    // Parse required fields
                    let source_id = match fields[self.source_id_idx].parse::<u64>() {
                        Ok(id) => id,
                        Err(e) => {
                            print_error(
                                format!(
                                    "Error parsing source_id from '{}'",
                                    fields[self.source_id_idx]
                                ),
                                e,
                            );
                            continue;
                        }
                    };

                    let ra = match fields[self.ra_idx].parse::<f64>() {
                        Ok(ra) => ra,
                        Err(e) => {
                            print_error(
                                format!("Error parsing RA from '{}'", fields[self.ra_idx]),
                                e,
                            );
                            continue;
                        }
                    };

                    let dec = match fields[self.dec_idx].parse::<f64>() {
                        Ok(dec) => dec,
                        Err(e) => {
                            print_error(
                                format!("Error parsing DEC from '{}'", fields[self.dec_idx]),
                                e,
                            );
                            continue;
                        }
                    };

                    // Return a valid star
                    return Some(StarData::new(source_id, ra, dec, g_mag, None));
                }
                Err(e) => {
                    print_error("Error reading line", e);
                    continue;
                }
            }
        }
    }
}

/// Thread-safe collector for star data that implements Iterator
struct ThreadSafeStarCollector {
    /// Queue to hold star data from processing threads
    queue: Arc<Mutex<VecDeque<StarData>>>,
    /// Condition variable to signal when new data is available
    condvar: Arc<Condvar>,
    /// List of thread handles that are producing data
    thread_handles: Vec<JoinHandle<Result<u64, Box<dyn std::error::Error + Send + 'static>>>>,
    /// Flag to track if all threads have completed
    threads_complete: Arc<Mutex<bool>>,
    /// Total star count across all threads
    star_count: Arc<Mutex<u64>>,
}

impl ThreadSafeStarCollector {
    /// Create a new empty collector
    fn new() -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            condvar: Arc::new(Condvar::new()),
            thread_handles: Vec::new(),
            threads_complete: Arc::new(Mutex::new(false)),
            star_count: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a handle for adding stars from a thread
    fn get_producer(&self) -> StarCollectorProducer {
        StarCollectorProducer {
            queue: Arc::clone(&self.queue),
            condvar: Arc::clone(&self.condvar),
            star_count: Arc::clone(&self.star_count),
        }
    }

    /// Add a thread handle to track
    fn add_thread_handle(
        &mut self,
        handle: JoinHandle<Result<u64, Box<dyn std::error::Error + Send + 'static>>>,
    ) {
        self.thread_handles.push(handle);
    }

    /// Gets the total number of stars processed
    fn get_star_count(&self) -> u64 {
        *self.star_count.lock().unwrap()
    }

    /// Create an iterator that applies a function to each item
    fn inspect<F>(self, f: F) -> impl Iterator<Item = StarData>
    where
        F: FnMut(&StarData),
    {
        // The inspect adapter applies a function to a reference of each item
        InspectAdapter { collector: self, f }
    }

    /// Checks if any thread handles have completed and removes them
    fn poll_handles(&mut self) -> bool {
        // Use retain to keep only threads that haven't completed yet
        let mut all_complete = true;
        self.thread_handles.retain(|handle| {
            if handle.is_finished() {
                false // Don't keep finished threads
            } else {
                all_complete = false;
                true // Keep threads that are still running
            }
        });

        if all_complete && self.thread_handles.is_empty() {
            // All threads are complete, mark the collector as done
            let mut complete = self.threads_complete.lock().unwrap();
            *complete = true;
            // Notify any waiting iterators
            self.condvar.notify_all();
            true
        } else {
            false
        }
    }

    /// Wait for threads to complete and join them
    fn join_all(&mut self) -> Result<u64, Box<dyn std::error::Error>> {
        let mut final_count = 0;

        // Join all threads and collect results
        while !self.thread_handles.is_empty() {
            // Take a thread handle from the vec and join it
            if let Some(handle) = self.thread_handles.pop() {
                match handle.join() {
                    Ok(result) => match result {
                        Ok(count) => final_count += count,
                        Err(e) => return Err(e),
                    },
                    Err(e) => return Err(format!("Thread panicked: {:?}", e).into()),
                }
            }
        }

        // Mark all threads as complete
        {
            let mut complete = self.threads_complete.lock().unwrap();
            *complete = true;
            self.condvar.notify_all();
        }

        Ok(final_count)
    }
}

/// Adapter struct for inspect functionality
struct InspectAdapter<F> {
    collector: ThreadSafeStarCollector,
    f: F,
}

impl<F> Iterator for InspectAdapter<F>
where
    F: FnMut(&StarData),
{
    type Item = StarData;

    fn next(&mut self) -> Option<Self::Item> {
        match self.collector.next() {
            Some(item) => {
                (self.f)(&item);
                Some(item)
            }
            None => None,
        }
    }
}

/// Handle for adding stars to the collector from a thread
#[derive(Clone)]
struct StarCollectorProducer {
    queue: Arc<Mutex<VecDeque<StarData>>>,
    condvar: Arc<Condvar>,
    star_count: Arc<Mutex<u64>>,
}

impl StarCollectorProducer {
    /// Add a star to the collector and notify waiting consumers
    fn add_star(&self, star: StarData) {
        let mut queue = self.queue.lock().unwrap();
        queue.push_back(star);

        // Update the star count
        {
            let mut count = self.star_count.lock().unwrap();
            *count += 1;
        }

        // Notify a waiting consumer
        self.condvar.notify_one();
    }
}

/// Iterator implementation for the thread-safe collector
impl Iterator for ThreadSafeStarCollector {
    type Item = StarData;

    fn next(&mut self) -> Option<Self::Item> {
        // Lock the queue
        let mut queue = self.queue.lock().unwrap();

        // First check if we have data ready to consume
        if let Some(star) = queue.pop_front() {
            return Some(star);
        }

        // No data available, check thread status
        let complete = self.threads_complete.lock().unwrap();
        if *complete && queue.is_empty() {
            // All threads are done and queue is empty, we're done iterating
            return None;
        }

        // Poll thread handles to check for completion
        drop(queue); // Release the lock before polling
        drop(complete);
        self.poll_handles();

        // Try again for data, this time with waiting if threads are still running
        let mut queue = self.queue.lock().unwrap();

        // If queue is still empty, wait for more data or until all threads complete
        while queue.is_empty() {
            let complete = self.threads_complete.lock().unwrap();
            if *complete {
                // All threads done and queue is empty
                return None;
            }

            // Wait for notification that more data is available
            let (new_queue, _) = self
                .condvar
                .wait_timeout(queue, std::time::Duration::from_millis(100))
                .unwrap();
            queue = new_queue;

            // Check if we received data
            if !queue.is_empty() {
                break;
            }

            // No data yet, release locks and poll threads again
            drop(queue);
            drop(complete);
            self.poll_handles();
            queue = self.queue.lock().unwrap();
        }

        // Return data if available
        queue.pop_front()
    }
}

/// Process a single Gaia catalog file, adding stars to the collector
fn process_file<P: AsRef<Path>>(
    input_path: P,
    magnitude_limit: f64,
    progress_bar: Option<ProgressBar>,
    collector_producer: StarCollectorProducer,
) -> Result<u64, Box<dyn std::error::Error>> {
    // Create reader for the file
    let mut reader = create_reader(input_path.as_ref())?;

    // Read header line to determine column positions
    let mut header = String::new();
    reader.read_line(&mut header)?;

    // Parse header to find column indices
    let headers: Vec<&str> = header.trim().split(',').collect();
    let (source_id_idx, ra_idx, dec_idx, g_mag_idx) = parse_headers(&headers)?;

    // Create iterator over the file's stars
    let iterator = GaiaFileIterator::new(
        reader,
        source_id_idx,
        ra_idx,
        dec_idx,
        g_mag_idx,
        magnitude_limit,
        progress_bar.clone(),
    );

    // Collect all stars from the file and count them
    let mut count = 0;

    // Process stars from the file and add them to the shared collector
    for star in iterator {
        collector_producer.add_star(star);
        count += 1;
    }

    // If we have a progress bar, mark it as finished
    if let Some(pb) = progress_bar {
        pb.set_length(pb.position());
        pb.finish_with_message("Complete");
        pb.reset_elapsed();
    }

    Ok(count)
}

/// Process multiple files in parallel and stream stars directly to output file
fn process_files_and_stream<P: AsRef<Path>>(
    input_files: Vec<PathBuf>,
    output_path: P,
    magnitude_limit: f64,
    num_threads: usize,
    multi_progress: &MultiProgress,
    total_progress: &ProgressBar,
) -> Result<u64, Box<dyn std::error::Error>> {
    let total_files = input_files.len();

    if total_files == 0 {
        return Err("No input files provided".into());
    }

    // Create a style for file progress bars
    let file_style = ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap()
        .progress_chars("#>-");

    // Create our thread-safe star collector
    let mut star_collector = ThreadSafeStarCollector::new();

    // Create shared collection for file line counts
    let file_line_counts = Arc::new(Mutex::new(Vec::new()));
    let pending_files = Arc::new(Mutex::new(input_files));

    // Create threads for processing files
    for thread_id in 0..num_threads {
        // Clone the shared resources for this thread
        let pending_files_clone = Arc::clone(&pending_files);
        let total_progress_clone = total_progress.clone();
        let file_line_counts_clone = Arc::clone(&file_line_counts);

        // Get a producer handle from the collector
        let producer = star_collector.get_producer();

        // Create a progress bar for this thread
        let thread_bar = multi_progress.add(ProgressBar::new(0));
        thread_bar.set_style(file_style.clone());
        thread_bar.set_message(format!("Thread {}", thread_id + 1));

        // Spawn a new thread
        let handle = thread::spawn(
            move || -> Result<u64, Box<dyn std::error::Error + Send + 'static>> {
                let mut total_count = 0;

                loop {
                    // Get the next file to process
                    let next_file = {
                        let mut files = pending_files_clone.lock().unwrap();
                        if files.is_empty() {
                            break; // No more files to process
                        }
                        files.pop().unwrap()
                    };

                    // Update the progress bar message
                    let file_name = next_file.file_name().unwrap_or_default().to_string_lossy();
                    thread_bar.set_message(format!("Thread {} - {}", thread_id + 1, file_name));

                    // Calculate estimated line count based on previously processed files
                    let estimated_lines = {
                        let line_counts = file_line_counts_clone.lock().unwrap();
                        if line_counts.is_empty() {
                            1_000_000 // Default estimate if no files processed yet
                        } else {
                            let total: u64 = line_counts.iter().sum();
                            let avg = total / line_counts.len() as u64;
                            avg
                        }
                    };

                    // Set the progress bar length to the estimated count
                    thread_bar.set_length(estimated_lines);
                    thread_bar.set_position(0);

                    // Process the file
                    match process_file(
                        &next_file,
                        magnitude_limit,
                        Some(thread_bar.clone()),
                        producer.clone(),
                    ) {
                        Ok(processed_count) => {
                            // Get the final position of the progress bar to estimate file size
                            let processed_lines = thread_bar.position();

                            // Store line count for future estimates
                            {
                                let mut line_counts = file_line_counts_clone.lock().unwrap();
                                line_counts.push(processed_lines);
                            }

                            // Add to this thread's total count
                            total_count += processed_count;

                            // Update the total progress bar
                            total_progress_clone.inc(1);
                        }
                        Err(e) => {
                            let err = format!("Error processing file {:?}: {}", next_file, e);
                            eprintln!("{}", err);
                            return Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                err,
                            )));
                        }
                    }
                }

                // Finish the thread's progress bar
                thread_bar.finish_with_message(format!("Thread {} - Complete", thread_id + 1));

                Ok(total_count)
            },
        );

        // Add the handle to the collector
        star_collector.add_thread_handle(handle);
    }

    // Create description for the catalog
    let desc = format!("Gaia catalog filtered to magnitude {}", magnitude_limit);

    // Create a progress bar for writing to the minimal catalog
    let catalog_style = ProgressStyle::default_bar()
        .template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] Writing stars ({eta})",
        )
        .unwrap()
        .progress_chars("=>-");

    let write_progress = multi_progress.add(ProgressBar::new_spinner());
    write_progress.set_style(catalog_style);
    write_progress.set_message("Writing catalog");
    write_progress.enable_steady_tick(std::time::Duration::from_millis(100));

    // Create a progress-tracking iterator - start streaming even before all threads complete
    let counting_iterator = star_collector.inspect(|_| {
        write_progress.inc(1);
    });

    // Write catalog directly from the iterator interface of our collector
    let written_count = MinimalCatalog::write_from_star_data(
        &output_path,
        counting_iterator,
        &desc,
        None, // Use dynamic counting since we're streaming
    )?;

    write_progress.finish_with_message(format!(
        "Catalog writing complete - {} stars",
        written_count
    ));

    Ok(written_count)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    println!("Gaia Catalog Filter Tool");
    println!("=======================");

    match &cli.command {
        Some(Commands::List) => {
            println!("Listing cached Gaia catalog files:");
            let files = list_cached_gaia_files()?;

            if files.is_empty() {
                println!("No files found in cache.");
            } else {
                for (i, path) in files.iter().enumerate() {
                    println!("  {}. {}", i + 1, path.display());
                }
                println!("Total: {} files", files.len());
            }
            return Ok(());
        }

        Some(Commands::All { max_files }) => {
            if cli.output.is_none() {
                return Err("Missing required output path".into());
            }
            let output_file = cli.output.unwrap();

            if !output_file.ends_with(".bin") {
                return Err("Output file must have .bin extension".into());
            }

            // Process all cached files (or up to max_files)
            let mut files = list_cached_gaia_files()?;

            if files.is_empty() {
                return Err("No Gaia files found in cache.".into());
            }

            // Limit number of files if max_files is specified
            if let Some(limit) = max_files {
                if *limit < files.len() {
                    println!(
                        "Limiting to {} files (out of {} available)",
                        limit,
                        files.len()
                    );
                    files.truncate(*limit);
                }
            }

            println!("Starting parallel processing with {} threads", cli.threads);

            // Setup progress indicators
            let multi_progress = MultiProgress::new();

            // Create a style for the overall progress bar
            let total_style = ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.green/blue}] {pos}/{len} files ({eta})",
                )
                .unwrap()
                .progress_chars("█▓▒░-");

            // Create a progress bar for overall progress
            let total_progress = multi_progress.add(ProgressBar::new(files.len() as u64));
            total_progress.set_style(total_style);
            total_progress.set_message("Processing files");

            // Process the files in parallel and stream directly to output

            let star_count = process_files_and_stream(
                files,
                &output_file,
                cli.magnitude,
                cli.threads,
                &multi_progress,
                &total_progress,
            )?;

            println!("Completed filtering:");
            println!(
                "  Kept {} stars with magnitude <= {}",
                star_count, cli.magnitude
            );
            println!("  Output written to: {}", output_file);
        }

        Some(Commands::Single { input }) => {
            if cli.output.is_none() {
                return Err("Missing required output path".into());
            }
            let output_file = cli.output.unwrap();

            if !output_file.ends_with(".bin") {
                return Err("Output file must have .bin extension".into());
            }

            // Setup progress indicators
            let multi_progress = MultiProgress::new();

            // Create a progress bar for the single file
            let file_style = ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} lines ({eta})")
                .unwrap()
                .progress_chars("#>-");

            let progress_bar = multi_progress.add(ProgressBar::new(1000000)); // Initial estimate
            progress_bar.set_style(file_style);
            let file_name = Path::new(&input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy();
            progress_bar.set_message(format!("Processing {}", file_name));

            // Create a thread-safe collector
            let mut star_collector = ThreadSafeStarCollector::new();
            let producer = star_collector.get_producer();

            // Process single file in a separate thread to allow streaming
            println!("Processing single file: {}", input);

            // Make a copy of the values we need for the thread
            let input_copy = input.clone();
            let magnitude = cli.magnitude;

            // Spawn a thread for processing the file
            let handle = thread::spawn(
                move || -> Result<u64, Box<dyn std::error::Error + Send + 'static>> {
                    match process_file(input_copy, magnitude, Some(progress_bar.clone()), producer)
                    {
                        Ok(count) => Ok(count),
                        Err(e) => {
                            let err = format!("Error processing file: {}", e);
                            Err(Box::new(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                err,
                            )))
                        }
                    }
                },
            );

            // Add the handle to the collector
            star_collector.add_thread_handle(handle);

            // Create description for the catalog
            let desc = format!("Gaia catalog filtered to magnitude {}", cli.magnitude);

            // Create a progress bar for writing to the minimal catalog
            let catalog_style = ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.yellow/blue}] Writing stars ({eta})",
                )
                .unwrap()
                .progress_chars("=>-");

            let write_progress = multi_progress.add(ProgressBar::new_spinner());
            write_progress.set_style(catalog_style);
            write_progress.set_message("Writing catalog");
            write_progress.enable_steady_tick(std::time::Duration::from_millis(100));
            write_progress.set_length(0); // Set to 0 initially, will be updated by the iterator

            // Create a progress-tracking iterator
            let counting_iterator = star_collector.inspect(|_| {
                write_progress.inc(1);
            });

            // Write catalog directly from the iterator interface of our collector
            let star_count = MinimalCatalog::write_from_star_data(
                &output_file,
                counting_iterator,
                &desc,
                None, // Use dynamic counting since we're streaming
            )?;

            write_progress
                .finish_with_message(format!("Catalog writing complete - {} stars", star_count));

            println!("Completed filtering:");
            println!(
                "  Kept {} stars with magnitude <= {}",
                star_count, cli.magnitude
            );
            println!("  Output written to: {}", output_file);
        }

        None => {
            println!("No command specified. Use --help for usage information.");
            return Err("No command specified".into());
        }
    }

    Ok(())
}
