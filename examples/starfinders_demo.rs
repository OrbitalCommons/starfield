use clap::{Parser, ValueEnum};
use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::Array2;
use starfield::image::starfinders::{
    DAOStarFinder, DAOStarFinderConfig, IRAFStarFinder, IRAFStarFinderConfig, StellarSource,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "daofind_demo")]
#[command(about = "Star detection demo using DAO or IRAF algorithms")]
struct Args {
    /// Input PNG image file
    input: PathBuf,

    /// Output PNG image file
    output: PathBuf,

    /// Detection method to use
    #[arg(short, long, default_value = "dao")]
    method: DetectionMethod,

    /// Full-width half-maximum of stars (default: 4.0)
    #[arg(long, default_value = "4.0")]
    fwhm: f64,

    /// Detection threshold
    #[arg(long, default_value = "5.0")]
    threshold: f64,
}

#[derive(Clone, ValueEnum)]
enum DetectionMethod {
    /// Use DAOStarFinder algorithm
    Dao,
    /// Use IRAFStarFinder algorithm
    Iraf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let input_path = &args.input;
    let output_path = &args.output;
    let fwhm = args.fwhm;
    let threshold = args.threshold;

    println!("Loading image: {}", input_path.display());

    // Load the input image
    let img = image::open(input_path)?;
    let gray_img = img.to_luma8();

    // Convert to ndarray for processing
    let (width, height) = gray_img.dimensions();
    let mut data = Array2::zeros((height as usize, width as usize));

    for (x, y, pixel) in gray_img.enumerate_pixels() {
        data[[y as usize, x as usize]] = pixel[0] as f64;
    }

    println!("Image dimensions: {}x{}", width, height);

    // Detect stars using the specified method
    let stars: Vec<Box<dyn StellarSource>> = match args.method {
        DetectionMethod::Dao => {
            println!(
                "Using DAOStarFinder with threshold={}, fwhm={}",
                threshold, fwhm
            );

            let config = DAOStarFinderConfig {
                threshold,
                fwhm,
                ratio: 1.0,
                theta: 0.0,
                sigma_radius: 1.5,
                sharpness: -10.0..=10.0,
                roundness: -10.0..=10.0,
                exclude_border: false,
                brightest: Some(500),
                peakmax: None,
                min_separation: 1.0,
            };

            let star_finder = DAOStarFinder::new(config)?;
            star_finder
                .find_stars(&data, None)
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn StellarSource>)
                .collect()
        }
        DetectionMethod::Iraf => {
            println!(
                "Using IRAFStarFinder with threshold={}, fwhm={}",
                threshold, fwhm
            );

            let config = IRAFStarFinderConfig {
                threshold,
                fwhm,
                sigma_radius: 1.5,
                minsep_fwhm: 1.0,
                sharpness: 0.2..=1.0,
                roundness: -1.0..=1.0,
                exclude_border: false,
                brightest: Some(500),
                peakmax: None,
                min_separation: Some(1.0),
            };

            let star_finder = IRAFStarFinder::new(config)?;
            star_finder
                .find_stars(&data, None)
                .into_iter()
                .map(|s| Box::new(s) as Box<dyn StellarSource>)
                .collect()
        }
    };

    println!("Found {} stars", stars.len());

    // Create output image at 2x resolution for sharper overlays
    let output_width = width * 2;
    let output_height = height * 2;
    let mut output_img: RgbImage = ImageBuffer::new(output_width, output_height);

    // Copy grayscale image to RGB at 2x scale
    for (x, y, pixel) in gray_img.enumerate_pixels() {
        let gray_val = pixel[0];
        let rgb_pixel = Rgb([gray_val, gray_val, gray_val]);

        // Each original pixel becomes a 2x2 block
        output_img.put_pixel(x * 2, y * 2, rgb_pixel);
        output_img.put_pixel(x * 2 + 1, y * 2, rgb_pixel);
        output_img.put_pixel(x * 2, y * 2 + 1, rgb_pixel);
        output_img.put_pixel(x * 2 + 1, y * 2 + 1, rgb_pixel);
    }

    // Use soft blue color for all overlays
    let soft_blue = Rgb([100, 150, 255]);

    // Draw detected stars using the common StellarSource trait
    for (i, star) in stars.iter().enumerate() {
        let (x_pos, y_pos) = star.get_centroid();
        let x = (x_pos * 2.0) as u32;
        let y = (y_pos * 2.0) as u32;

        // Draw a thin circle around each detected star (preserve center)
        draw_circle_hollow(&mut output_img, x, y, 20, soft_blue, 1);

        // Draw a smaller cross closer to the centroid (preserve center)
        draw_cross_hollow(&mut output_img, x, y, 6, soft_blue, 1);

        if i < 10 {
            println!(
                "Star {}: x={:.1}, y={:.1}, mag={:.2}, flux={:.1}",
                star.id(),
                x_pos,
                y_pos,
                star.mag(),
                star.flux()
            );
        }
    }

    // Save the output image
    output_img.save(output_path)?;
    println!("Saved output image: {}", output_path.display());

    Ok(())
}

/// Draw a hollow circle outline at the specified position, preserving center
fn draw_circle_hollow(
    img: &mut RgbImage,
    cx: u32,
    cy: u32,
    radius: u32,
    color: Rgb<u8>,
    thickness: u32,
) {
    let (width, height) = img.dimensions();
    let center_preserve_radius = 6; // Don't overwrite pixels within 6 pixels of center

    for angle in 0..360 {
        let rad = (angle as f64) * std::f64::consts::PI / 180.0;

        // Draw multiple concentric circles for thickness
        for t in 0..thickness {
            let r = radius + t;
            let x = cx as i32 + (r as f64 * rad.cos()) as i32;
            let y = cy as i32 + (r as f64 * rad.sin()) as i32;

            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let dist_from_center = ((x - cx as i32).pow(2) + (y - cy as i32).pow(2)) as f64;
                if dist_from_center.sqrt() > center_preserve_radius as f64 {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

/// Draw a hollow cross at the specified position, preserving center
fn draw_cross_hollow(
    img: &mut RgbImage,
    cx: u32,
    cy: u32,
    size: u32,
    color: Rgb<u8>,
    thickness: u32,
) {
    let (width, height) = img.dimensions();
    let center_preserve_radius = 4; // Don't overwrite pixels within 4 pixels of center

    // Horizontal line
    for dx in -(size as i32)..=(size as i32) {
        for t in 0..thickness {
            let x = cx as i32 + dx;
            let y = cy as i32 + t as i32 - (thickness as i32 / 2);

            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let dist_from_center = ((x - cx as i32).pow(2) + (y - cy as i32).pow(2)) as f64;
                if dist_from_center.sqrt() > center_preserve_radius as f64 {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }

    // Vertical line
    for dy in -(size as i32)..=(size as i32) {
        for t in 0..thickness {
            let x = cx as i32 + t as i32 - (thickness as i32 / 2);
            let y = cy as i32 + dy;

            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let dist_from_center = ((x - cx as i32).pow(2) + (y - cy as i32).pow(2)) as f64;
                if dist_from_center.sqrt() > center_preserve_radius as f64 {
                    img.put_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}
