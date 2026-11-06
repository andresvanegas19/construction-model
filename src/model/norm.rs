//! Norms written from elementary tensor ops instead of candle_nn's fused kernels: the fused rms_norm / layer_norm
//! have no backward pass, and the LoRA gradients must flow through every norm between adapted layers.
use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(vb: VarBuilder, dim: usize, eps: f64) -> anyhow::Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let inv = (x.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?.recip()?;
        x.broadcast_mul(&inv)?.broadcast_mul(&self.weight)
    }
}

pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn load(vb: VarBuilder, dim: usize, eps: f64) -> anyhow::Result<Self> {
        Ok(Self { weight: vb.get(dim, "weight")?, bias: vb.get(dim, "bias")?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let centered = x.broadcast_sub(&x.mean_keepdim(D::Minus1)?)?;
        let inv = (centered.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?.sqrt()?.recip()?;
        centered.broadcast_mul(&inv)?.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}
