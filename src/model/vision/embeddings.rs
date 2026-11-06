//! SigLIP2 NaFlex patch + position embeddings. Patches arrive already flattened (16*16*3 values in (row, col,
//! channel) order) and go through a Linear; the learned 16x16 position grid is resized to each image's own patch grid
//! with antialiased bilinear interpolation, as torch's `interpolate(mode="bilinear", antialias=True,
//! align_corners=False)` does. The resize is a fixed linear map, so it is built as a (h*w, 256) weight matrix and
//! applied with one matmul.
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::lora::LoraLinear;

pub struct Embeddings {
    patch: LoraLinear,
    position: Tensor,
    grid: usize,
}

impl Embeddings {
    pub fn load(vb: VarBuilder, hidden: usize, patch: usize, channels: usize, num_patches: usize) -> anyhow::Result<Self> {
        let grid = (num_patches as f64).sqrt().round() as usize;
        Ok(Self {
            patch: LoraLinear::frozen(vb.pp("patch_embedding"), patch * patch * channels, hidden, true)?,
            position: vb.pp("position_embedding").get((num_patches, hidden), "weight")?,
            grid,
        })
    }

    /// `pixels` (h*w, patch_dim) for one image whose patch grid is h rows by w columns.
    pub fn forward(&self, pixels: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let patches = self.patch.forward(pixels, false)?;
        let rows = resize_weights(self.grid, h);
        let cols = resize_weights(self.grid, w);
        let g = self.grid;
        let mut m = vec![0f32; h * w * g * g];
        for (i, ri) in rows.iter().enumerate() {
            for (j, cj) in cols.iter().enumerate() {
                let out = (i * w + j) * g * g;
                for &(a, wa) in ri {
                    for &(b, wb) in cj {
                        m[out + a * g + b] = wa * wb;
                    }
                }
            }
        }
        let m = Tensor::from_vec(m, (h * w, g * g), pixels.device())?;
        patches + m.matmul(&self.position)?
    }
}

/// For each of `output` positions, the (input index, weight) pairs of torch's antialiased bilinear resize from
/// `input` positions (align_corners=False). When upsampling this is ordinary bilinear with edge clamping; when
/// downsampling the triangle filter widens by the scale so every input contributes.
pub fn resize_weights(input: usize, output: usize) -> Vec<Vec<(usize, f32)>> {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { scale } else { 1.0 };
    let inv = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            // torch truncates toward zero here; the arguments are >= -0.5 so `as i64` matches.
            let lo = ((center - support + 0.5) as i64).max(0) as usize;
            let hi = ((center + support + 0.5) as i64).min(input as i64) as usize;
            let mut taps: Vec<(usize, f64)> = (lo..hi)
                .map(|j| (j, (1.0 - ((j as f64 - center + 0.5) * inv).abs()).max(0.0)))
                .collect();
            let total: f64 = taps.iter().map(|t| t.1).sum();
            if total > 0.0 {
                for t in &mut taps {
                    t.1 /= total;
                }
            }
            taps.into_iter().filter(|t| t.1 > 0.0).map(|(j, w)| (j, w as f32)).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::resize_weights;

    #[test]
    fn identity_resize_is_one_hot() {
        for (i, taps) in resize_weights(16, 16).iter().enumerate() {
            assert_eq!(taps, &vec![(i, 1.0)]);
        }
    }

    #[test]
    fn weights_sum_to_one() {
        for (input, output) in [(16, 32), (16, 7), (16, 64), (16, 3)] {
            for taps in resize_weights(input, output) {
                let s: f32 = taps.iter().map(|t| t.1).sum();
                assert!((s - 1.0).abs() < 1e-5, "{input}->{output}: {s}");
            }
        }
    }
}
