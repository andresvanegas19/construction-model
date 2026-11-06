//! LFM2 grouped-query attention: per-head RMSNorm on q and k before RoPE (LFM2's q/k layernorm), 32 query heads
//! sharing 8 key/value heads, causal over the full sequence.
use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use super::rope;
use crate::config::TextConfig;
use crate::lora::{LoraBuilder, LoraLinear};
use crate::model::norm::RmsNorm;

pub struct Attention {
    q_proj: LoraLinear,
    k_proj: LoraLinear,
    v_proj: LoraLinear,
    out_proj: LoraLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn load(vb: VarBuilder, lora: &mut LoraBuilder, cfg: &TextConfig) -> anyhow::Result<Self> {
        let (h, d) = (cfg.hidden_size, cfg.head_dim());
        let kv = cfg.num_key_value_heads * d;
        Ok(Self {
            q_proj: lora.linear(vb.pp("q_proj"), h, h, false)?,
            k_proj: lora.linear(vb.pp("k_proj"), h, kv, false)?,
            v_proj: lora.linear(vb.pp("v_proj"), h, kv, false)?,
            out_proj: lora.linear(vb.pp("out_proj"), h, h, false)?,
            q_norm: RmsNorm::load(vb.pp("q_layernorm"), d, cfg.norm_eps)?,
            k_norm: RmsNorm::load(vb.pp("k_layernorm"), d, cfg.norm_eps)?,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: d,
        })
    }

    /// `x` (1, T, H); `mask` (T, T) additive causal mask; `cos`/`sin` (T, head_dim).
    pub fn forward(&self, x: &Tensor, mask: &Tensor, cos: &Tensor, sin: &Tensor, train: bool) -> Result<Tensor> {
        let len = x.dim(1)?;
        let split = |t: Tensor, n: usize| t.reshape((1, len, n, self.head_dim));
        let q = self.q_norm.forward(&split(self.q_proj.forward(x, train)?, self.heads)?)?.transpose(1, 2)?;
        let k = self.k_norm.forward(&split(self.k_proj.forward(x, train)?, self.kv_heads)?)?.transpose(1, 2)?;
        let v = split(self.v_proj.forward(x, train)?, self.kv_heads)?.transpose(1, 2)?;
        let q = rope::apply(&q.contiguous()?, cos, sin)?;
        let k = repeat_kv(&rope::apply(&k.contiguous()?, cos, sin)?, self.heads / self.kv_heads)?;
        let v = repeat_kv(&v.contiguous()?, self.heads / self.kv_heads)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.t()?.contiguous()?)? * scale)?.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let out = probs.matmul(&v)?.transpose(1, 2)?.contiguous()?.reshape((1, len, self.heads * self.head_dim))?;
        self.out_proj.forward(&out, train)
    }
}

/// (1, kv, T, d) -> (1, kv * n, T, d), each kv head repeated n times in place (head i reads kv head i / n).
fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?.broadcast_as((b, kv, n, t, d))?.contiguous()?.reshape((b, kv * n, t, d))
}

/// Additive causal mask: 0 on and below the diagonal, a large negative above. -1e9 instead of -inf keeps a fully
/// masked row (never produced here, but cheap insurance) from turning softmax into NaN.
pub fn causal_mask(len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let values: Vec<f32> = (0..len).flat_map(|i| (0..len).map(move |j| if j > i { -1e9 } else { 0.0 })).collect();
    Tensor::from_vec(values, (len, len), device)
}
