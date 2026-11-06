//! SigLIP2 encoder layers: pre-LayerNorm, bidirectional multi-head attention with biases, and a tanh-GELU MLP.
//! Images run one tile at a time, so every patch is real and no padding mask is needed (the HF batch pads to 1024
//! patches and masks the padding out, which gives the same result for the real patches).
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::VisionConfig;
use crate::lora::LoraLinear;
use crate::model::norm::LayerNorm;
use crate::model::{tap, Point, Probe, Tower};

pub struct EncoderLayer {
    ln1: LayerNorm,
    ln2: LayerNorm,
    q: LoraLinear,
    k: LoraLinear,
    v: LoraLinear,
    out: LoraLinear,
    fc1: LoraLinear,
    fc2: LoraLinear,
    heads: usize,
}
impl EncoderLayer {
    pub fn load(vb: VarBuilder, cfg: &VisionConfig) -> anyhow::Result<Self> {
        let (h, f) = (cfg.hidden_size, cfg.intermediate_size);
        let att = vb.pp("self_attn");
        Ok(Self {
            ln1: LayerNorm::load(vb.pp("layer_norm1"), h, cfg.layer_norm_eps)?,
            ln2: LayerNorm::load(vb.pp("layer_norm2"), h, cfg.layer_norm_eps)?,
            q: LoraLinear::frozen(att.pp("q_proj"), h, h, true)?,
            k: LoraLinear::frozen(att.pp("k_proj"), h, h, true)?,
            v: LoraLinear::frozen(att.pp("v_proj"), h, h, true)?,
            out: LoraLinear::frozen(att.pp("out_proj"), h, h, true)?,
            fc1: LoraLinear::frozen(vb.pp("mlp").pp("fc1"), h, f, true)?,
            fc2: LoraLinear::frozen(vb.pp("mlp").pp("fc2"), f, h, true)?,
            heads: cfg.num_attention_heads,
        })
    }

    /// `x` (N, H) patch states of one tile.
    pub fn forward(&self, x: &Tensor, layer: usize, probe: &mut Option<&mut dyn Probe>) -> Result<Tensor> {
        tap(probe, Tower::Vision, layer, Point::BlockInput, x)?;
        let normed = self.ln1.forward(x)?;
        tap(probe, Tower::Vision, layer, Point::OperatorNorm, &normed)?;
        let att = self.attention(&normed)?;
        tap(probe, Tower::Vision, layer, Point::OperatorOut, &att)?;
        let h = (x + att)?;
        tap(probe, Tower::Vision, layer, Point::ResidualMid, &h)?;
        let normed = self.ln2.forward(&h)?;
        tap(probe, Tower::Vision, layer, Point::FfnNorm, &normed)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&normed, false)?.gelu()?, false)?;
        tap(probe, Tower::Vision, layer, Point::MlpOut, &mlp)?;
        let out = (h + mlp)?;
        tap(probe, Tower::Vision, layer, Point::BlockOutput, &out)?;
        Ok(out)
    }

    fn attention(&self, x: &Tensor) -> Result<Tensor> {
        let (n, hidden) = x.dims2()?;
        let d = hidden / self.heads;
        let split = |t: Tensor| -> Result<Tensor> { t.reshape((n, self.heads, d))?.transpose(0, 1)?.contiguous() };
        let q = split(self.q.forward(x, false)?)?;
        let k = split(self.k.forward(x, false)?)?;
        let v = split(self.v.forward(x, false)?)?;
        let scores = (q.matmul(&k.t()?.contiguous()?)? * (1.0 / (d as f64).sqrt()))?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?.transpose(0, 1)?.contiguous()?.reshape((n, hidden))?;
        self.out.forward(&out, false)
    }
}
