//! Rotary embedding in the "rotate half" layout LFM2 uses (the two halves of each head are the pair, not adjacent
//! lanes). Done with plain ops: candle_nn's fused rope has no backward.
use candle_core::{DType, Device, Result, Tensor, D};

pub struct Rope {
    inv_freq: Vec<f32>,
    device: Device,
}

impl Rope {
    pub fn new(head_dim: usize, theta: f64, device: &Device) -> Self {
        let inv_freq = (0..head_dim / 2).map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32).collect();
        Self { inv_freq, device: device.clone() }
    }

    /// (cos, sin), each (T, head_dim): the frequencies repeated for both halves.
    pub fn tables(&self, len: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut angles = Vec::with_capacity(len * half * 2);
        for t in 0..len {
            let row: Vec<f32> = self.inv_freq.iter().map(|f| t as f32 * f).collect();
            angles.extend_from_slice(&row);
            angles.extend_from_slice(&row);
        }
        let angles = Tensor::from_vec(angles, (len, half * 2), &self.device)?.to_dtype(DType::F32)?;
        Ok((angles.cos()?, angles.sin()?))
    }
}

/// `x` is (1, heads, T, head_dim).
pub fn apply(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
}
