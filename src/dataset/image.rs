//! Image preprocessing as the LFM2-VL processor does it: one smart-resized tile for small images, or a grid of
//! 512 px tiles plus a smart-resized thumbnail for large ones; every tile is cut into 16 px patches, rescaled and
//! normalized. The HF fast image processor is not published as plain code, so the resize and grid choice follow
//! `processing_lfm2_vl.py` + config semantics (documented as approximations in training/README.md): sides are rounded
//! to the 32 px encoder-patch x downsample factor, scaled down/up into the min/max image-token budget, and the grid is
//! the InternVL-style closest aspect ratio within min_tiles..=max_tiles.
use std::path::Path;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::RgbImage;

use crate::config::ImageSettings;

/// One tile's patches: `pixels` is (h*w) patches x (patch*patch*3) values, row-major over an h x w patch grid.
#[derive(Debug, Clone)]
pub struct Tile {
    pub pixels: Vec<f32>,
    pub h: usize,
    pub w: usize,
}

impl Tile {
    /// Image tokens this tile becomes after the 2x2 unshuffle.
    pub fn tokens(&self, downsample: usize) -> usize {
        (self.h / downsample) * (self.w / downsample)
    }
}

#[derive(Debug, Clone)]
pub struct Processed {
    /// Grid tiles row-major, then the thumbnail when `grid` is set; a single tile otherwise.
    pub tiles: Vec<Tile>,
    pub grid: Option<(usize, usize)>,
}

pub fn load(path: &Path, s: &ImageSettings) -> Result<Processed> {
    let img = image::open(path).with_context(|| format!("image {}", path.display()))?.to_rgb8();
    Ok(process(&img, s))
}

pub fn process(img: &RgbImage, s: &ImageSettings) -> Processed {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let factor = s.patch * s.downsample;
    let rounded = round_by(h, s.patch) * round_by(w, s.patch);
    let too_large = rounded as f64 > s.max_pixels() as f64 * s.tolerance as f64;
    if s.split && too_large {
        let (rows, cols) = grid(w, h, s);
        let full = image::imageops::resize(img, (cols * s.tile) as u32, (rows * s.tile) as u32, FilterType::Triangle);
        let mut tiles = Vec::with_capacity(rows * cols + 1);
        for r in 0..rows {
            for c in 0..cols {
                let crop = image::imageops::crop_imm(&full, (c * s.tile) as u32, (r * s.tile) as u32, s.tile as u32, s.tile as u32);
                tiles.push(patchify(&crop.to_image(), s));
            }
        }
        if s.thumbnail {
            tiles.push(patchify(&smart_resize(img, factor, s), s));
        }
        return Processed { tiles, grid: Some((rows, cols)) };
    }
    Processed { tiles: vec![patchify(&smart_resize(img, factor, s), s)], grid: None }
}

fn round_by(x: usize, f: usize) -> usize {
    (((x as f64) / f as f64).round() as usize * f).max(f)
}

/// Sides multiple of `factor` whose area lies in the token budget, aspect ratio kept as far as rounding allows.
pub fn smart_dims(w: usize, h: usize, factor: usize, s: &ImageSettings) -> (usize, usize) {
    let (mut wb, mut hb) = (round_by(w, factor), round_by(h, factor));
    let area = (w * h) as f64;
    if wb * hb > s.max_pixels() {
        let beta = (area / s.max_pixels() as f64).sqrt();
        hb = (((h as f64 / beta) / factor as f64).floor() as usize * factor).max(factor);
        wb = (((w as f64 / beta) / factor as f64).floor() as usize * factor).max(factor);
    } else if wb * hb < s.min_pixels() {
        let beta = (s.min_pixels() as f64 / area).sqrt();
        hb = ((h as f64 * beta) / factor as f64).ceil() as usize * factor;
        wb = ((w as f64 * beta) / factor as f64).ceil() as usize * factor;
    }
    (wb, hb)
}

fn smart_resize(img: &RgbImage, factor: usize, s: &ImageSettings) -> RgbImage {
    let (wb, hb) = smart_dims(img.width() as usize, img.height() as usize, factor, s);
    image::imageops::resize(img, wb as u32, hb as u32, FilterType::Triangle)
}

/// (rows, cols): the grid whose cols/rows ratio is closest to the image's; a tie goes to the larger grid when the
/// image has the pixels to fill it.
pub fn grid(w: usize, h: usize, s: &ImageSettings) -> (usize, usize) {
    let aspect = w as f64 / h as f64;
    let mut ratios: Vec<(usize, usize)> = (1..=s.max_tiles)
        .flat_map(|r| (1..=s.max_tiles).map(move |c| (r, c)))
        .filter(|(r, c)| (s.min_tiles..=s.max_tiles).contains(&(r * c)))
        .collect();
    ratios.sort_by_key(|(r, c)| r * c);
    let mut best = (1, s.min_tiles.max(1));
    let mut best_diff = f64::INFINITY;
    let area = (w * h) as f64;
    for (r, c) in ratios {
        let diff = (aspect - c as f64 / r as f64).abs();
        if diff < best_diff {
            best = (r, c);
            best_diff = diff;
        } else if diff == best_diff && area > 0.5 * (s.tile * s.tile * r * c) as f64 {
            best = (r, c);
        }
    }
    best
}

/// Rows of patches in (patch row, patch col, channel) order, normalized as (v * rescale - mean) / std.
fn patchify(img: &RgbImage, s: &ImageSettings) -> Tile {
    let p = s.patch;
    let (h, w) = (img.height() as usize / p, img.width() as usize / p);
    let mut pixels = Vec::with_capacity(h * w * p * p * 3);
    for gy in 0..h {
        for gx in 0..w {
            for py in 0..p {
                for px in 0..p {
                    let rgb = img.get_pixel((gx * p + px) as u32, (gy * p + py) as u32).0;
                    for c in 0..3 {
                        pixels.push((rgb[c] as f32 * s.rescale - s.mean[c]) / s.std[c]);
                    }
                }
            }
        }
    }
    Tile { pixels, h, w }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{ProcessorConfig, VlConfig};

    fn settings() -> ImageSettings {
        ImageSettings::new(&VlConfig::default(), &ProcessorConfig::default(), None)
    }

    #[test]
    fn small_image_is_one_tile_in_budget() {
        let s = settings();
        let p = process(&RgbImage::new(300, 200), &s);
        assert_eq!(p.tiles.len(), 1);
        let t = &p.tiles[0];
        assert!(t.tokens(2) >= s.min_tokens && t.tokens(2) <= s.max_tokens);
    }

    #[test]
    fn large_wide_image_splits_with_thumbnail() {
        let s = settings();
        let p = process(&RgbImage::new(2048, 1024), &s);
        let (r, c) = p.grid.unwrap();
        assert_eq!((r, c), (1, 2));
        assert_eq!(p.tiles.len(), r * c + 1);
        assert_eq!(p.tiles[0].tokens(2), 256);
    }
}
