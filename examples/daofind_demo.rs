use image::{ImageBuffer, Rgb, RgbImage};
use ndarray::Array2;
use starfield::image::daofind::{DAOStarFinder, DAOStarFinderConfig};
/// DAOStarFinder demonstration with PNG image processing
///
/// This example loads a PNG image, detects stars using the DAOStarFinder algorithm,
/// and outputs an overlay image with detected stars marked.
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 || args.len() > 4 {
        eprintln!("Usage: {} <input.png> <output.png> [fwhm]", args[0]);
        eprintln!("Detects stars in the input PNG and creates an overlay in the output PNG");
        eprintln!("  fwhm: Full-width half-maximum of stars (default: 4.0, use larger values like 8-12 for broader PSFs)");
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];
    let fwhm = if args.len() == 4 {
        args[3].parse::<f64>().unwrap_or_else(|_| {
            eprintln!("Error: FWHM must be a valid number");
            std::process::exit(1);
        })
    } else {
        4.0
    };

    println!("Loading image: {}", input_path);

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

    // Configure star finder with extremely broad acceptance criteria
    let config = DAOStarFinderConfig {
        threshold: 5.0,    // Very low threshold for maximum faint source detection
        fwhm,              // Full-width half-maximum of stars (from CLI)
        ratio: 1.0,        // Circular PSF
        theta: 0.0,        // No rotation
        sigma_radius: 1.5, // Kernel truncation
        sharplo: -10.0,    // Extremely relaxed sharpness bounds - accept almost anything
        sharphi: 10.0,     // Allow very sharp and very broad sources
        roundlo: -10.0,    // Extremely relaxed roundness bounds - accept very elongated
        roundhi: 10.0,     // Accept all roundness values
        exclude_border: false, // Include border detections for maximum coverage
        brightest: Some(500),  // Keep many more sources
        peakmax: None,         // No peak limit
        min_separation: 1.0,   // Minimal separation to pack in more sources
    };

    println!(
        "Detecting stars with threshold={}, fwhm={}",
        config.threshold, config.fwhm
    );

    // Create star finder and detect sources
    let star_finder = DAOStarFinder::new(config)?;
    let stars = star_finder.find_stars(&data, None);

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

    // Draw detected stars
    for (i, star) in stars.iter().enumerate() {
        let x = (star.x_centroid * 2.0) as u32;
        let y = (star.y_centroid * 2.0) as u32;

        // Draw a thin circle around each detected star (preserve center)
        draw_circle_hollow(&mut output_img, x, y, 20, soft_blue, 1);

        // Draw a smaller cross closer to the centroid (preserve center)
        draw_cross_hollow(&mut output_img, x, y, 6, soft_blue, 1);

        if i < 10 {
            println!(
                "Star {}: x={:.1}, y={:.1}, mag={:.2}, peak={:.1}, sharpness={:.3}",
                i + 1,
                star.x_centroid,
                star.y_centroid,
                star.mag,
                star.peak,
                star.sharpness
            );
        }
    }

    // Save the output image
    output_img.save(output_path)?;
    println!("Saved output image: {}", output_path);

    Ok(())
}

/// Draw a hollow circle outline at the specified position, preserving center
fn draw_circle_hollow(img: &mut RgbImage, cx: u32, cy: u32, radius: u32, color: Rgb<u8>, thickness: u32) {
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
fn draw_cross_hollow(img: &mut RgbImage, cx: u32, cy: u32, size: u32, color: Rgb<u8>, thickness: u32) {
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
