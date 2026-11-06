//! SwiGLU feed-forward: w2(silu(w1 x) * w3 x).
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::lora::{LoraBuilder, LoraLinear};

pub struct Mlp {
    w1: LoraLinear,
    w2: LoraLinear,
    w3: LoraLinear,
}

impl Mlp {
    /// `ffn` comes from the checkpoint's w1 shape: LFM2 derives it from config values with an auto-adjust rule, and
    /// reading the tensor avoids re-implementing (and mis-rounding) that rule.
    pub fn load(vb: VarBuilder, lora: &mut LoraBuilder, hidden: usize, ffn: usize) -> anyhow::Result<Self> {
        Ok(Self {
            w1: lora.linear(vb.pp("w1"), hidden, ffn, false)?,
            w2: lora.linear(vb.pp("w2"), ffn, hidden, false)?,
            w3: lora.linear(vb.pp("w3"), hidden, ffn, false)?,
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> Result<Tensor> {
        let gate = self.w1.forward(x, train)?.silu()?;
        self.w2.forward(&(gate * self.w3.forward(x, train)?)?, train)
    }
}
