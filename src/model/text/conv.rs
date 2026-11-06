//! LFM2's gated short convolution: in_proj splits into gates B, C and the signal x; B·x goes through a causal
//! depthwise conv of width L_cache (3), then C gates it before out_proj. The conv is unrolled into L shifted
//! multiply-adds because candle's conv1d has no grouped (depthwise) backward on CPU, and three shifts are cheaper than a
//! general conv for kernel 3 anyway.
use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::lora::{LoraBuilder, LoraLinear};

pub struct ShortConv {
    in_proj: LoraLinear,
    out_proj: LoraLinear,
    /// One (C,) weight per kernel position; tap k multiplies the input L-1-k steps back.
    taps: Vec<Tensor>,
    bias: Option<Tensor>,
}

impl ShortConv {
    pub fn load(vb: VarBuilder, lora: &mut LoraBuilder, hidden: usize, kernel: usize, bias: bool) -> anyhow::Result<Self> {
        let in_proj = lora.linear(vb.pp("in_proj"), hidden, 3 * hidden, bias)?;
        let out_proj = lora.linear(vb.pp("out_proj"), hidden, hidden, bias)?;
        // Stored like torch's depthwise Conv1d weight: (C, 1, L).
        let w = vb.pp("conv").get((hidden, 1, kernel), "weight")?.squeeze(1)?;
        let taps = (0..kernel).map(|k| w.narrow(1, k, 1)?.squeeze(1)).collect::<Result<Vec<_>>>()?;
        let bias = if bias { Some(vb.pp("conv").get(hidden, "bias")?) } else { None };
        Ok(Self { in_proj, out_proj, taps, bias })
    }

    /// `x` is (1, T, H) after operator_norm.
    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let len = x.dim(1)?;
        let bcx = self.in_proj.forward(x, train)?;
        let b = bcx.narrow(D::Minus1, 0, hidden)?;
        let c = bcx.narrow(D::Minus1, hidden, hidden)?;
        let s = bcx.narrow(D::Minus1, 2 * hidden, hidden)?;
        let u = (b * s)?;
        let kernel = self.taps.len();
        // Left zero padding of L-1 makes output t depend on inputs t-L+1..=t only (torch pads both sides and trims).
        let padded = u.pad_with_zeros(1, kernel - 1, 0)?;
        let mut y = padded.narrow(1, 0, len)?.broadcast_mul(&self.taps[0])?;
        for (k, tap) in self.taps.iter().enumerate().skip(1) {
            y = (y + padded.narrow(1, k, len)?.broadcast_mul(tap)?)?;
        }
        if let Some(bias) = &self.bias {
            y = y.broadcast_add(bias)?;
        }
        self.out_proj.forward(&(c * y)?, train)
    }
}
