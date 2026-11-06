//! Vision -> text bridge: 2x2 pixel unshuffle of the patch grid (1152 * 4 = 4608 channels per merged token), then
//! Linear -> exact (erf) GELU -> Linear into the text width. No LayerNorm (`projector_use_layernorm` is false).
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::lora::{LoraBuilder, LoraLinear};

pub struct Projector {
    linear_1: LoraLinear,
    linear_2: LoraLinear,
    factor: usize,
}

impl Projector {
    pub fn load(vb: VarBuilder, lora: &mut LoraBuilder, vision: usize, hidden: usize, text: usize, factor: usize, bias: bool) -> anyhow::Result<Self> {
        Ok(Self {
            linear_1: lora.linear(vb.pp("linear_1"), vision * factor * factor, hidden, bias)?,
            linear_2: lora.linear(vb.pp("linear_2"), hidden, text, bias)?,
            factor,
        })
    }

    /// `features` (h*w, C) of one tile -> (h/f * w/f, text hidden), row-major over the merged grid.
    pub fn forward(&self, features: &Tensor, h: usize, w: usize, train: bool) -> Result<Tensor> {
        let merged = pixel_unshuffle(features, h, w, self.factor)?;
        self.linear_2.forward(&self.linear_1.forward(&merged, train)?.gelu_erf()?, train)
    }
}

/// Merged token (i, j) concatenates the 2x2 block in the order (2i, 2j), (2i, 2j+1), (2i+1, 2j), (2i+1, 2j+1): the
/// channel order the HF reshape/permute sequence produces, which the projector's 4608 input columns were trained on.
pub fn pixel_unshuffle(x: &Tensor, h: usize, w: usize, f: usize) -> Result<Tensor> {
    let c = x.dim(1)?;
    x.reshape((h / f, f, w / f, f, c))?.permute((0, 2, 1, 3, 4))?.contiguous()?.reshape(((h / f) * (w / f), f * f * c))
}
