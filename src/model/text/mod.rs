//! The LFM2 text backbone. Pre-norm blocks: h = x + op(operator_norm(x)); out = h + ffn(ffn_norm(h)), where op is
//! the short conv or attention per `layer_types`. `embedding_norm` is LFM2's final norm (applied after the last block,
//! despite the name). The LM head is tied to `embed_tokens`.
pub mod attention;
pub mod conv;
pub mod mlp;
pub mod rope;

use std::collections::BTreeMap;

use anyhow::Context;
use candle_core::{Device, Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::TextConfig;
use crate::lora::LoraBuilder;
use crate::model::norm::RmsNorm;
use crate::model::{tap, Point, Probe, Tower};

pub enum Operator {
    Conv(conv::ShortConv),
    Attention(attention::Attention),
}

pub struct Block {
    pub operator: Operator,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
    mlp: mlp::Mlp,
}

pub struct TextModel {
    pub embed: Tensor,
    pub blocks: Vec<Block>,
    final_norm: RmsNorm,
    rope: rope::Rope,
    pub cfg: TextConfig,
}

impl TextModel {
    /// `vb` is rooted at `model.language_model`; `shapes` are the checkpoint's tensor shapes (for the FFN width).
    pub fn load(vb: VarBuilder, lora: &mut LoraBuilder, cfg: &TextConfig, shapes: &BTreeMap<String, Vec<usize>>) -> anyhow::Result<Self> {
        let h = cfg.hidden_size;
        let embed = vb.pp("embed_tokens").get((cfg.vocab_size, h), "weight")?;
        let ffn_key = format!("{}.layers.0.feed_forward.w1.weight", vb.prefix());
        let ffn = shapes.get(&ffn_key).and_then(|s| s.first().copied()).with_context(|| format!("missing {ffn_key}"))?;
        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lb = vb.pp("layers").pp(i);
            let operator = if cfg.is_attention(i) {
                Operator::Attention(attention::Attention::load(lb.pp("self_attn"), lora, cfg)?)
            } else {
                Operator::Conv(conv::ShortConv::load(lb.pp("conv"), lora, h, cfg.conv_l_cache, cfg.conv_bias)?)
            };
            blocks.push(Block {
                operator,
                operator_norm: RmsNorm::load(lb.pp("operator_norm"), h, cfg.norm_eps)?,
                ffn_norm: RmsNorm::load(lb.pp("ffn_norm"), h, cfg.norm_eps)?,
                mlp: mlp::Mlp::load(lb.pp("feed_forward"), lora, h, ffn)?,
            });
        }
        Ok(Self {
            embed,
            blocks,
            final_norm: RmsNorm::load(vb.pp("embedding_norm"), h, cfg.norm_eps)?,
            rope: rope::Rope::new(cfg.head_dim(), cfg.rope_theta, vb.device()),
            cfg: cfg.clone(),
        })
    }

    pub fn embed_ids(&self, ids: &[u32], device: &Device) -> Result<Tensor> {
        let ids = Tensor::new(ids, device)?;
        self.embed.index_select(&ids, 0)
    }

    /// `x` (1, T, H) input embeddings -> (1, T, H) final-normed hidden states.
    pub fn forward(&self, x: &Tensor, train: bool, mut probe: Option<&mut dyn Probe>) -> Result<Tensor> {
        let len = x.dim(1)?;
        let mask = attention::causal_mask(len, x.device())?;
        let (cos, sin) = self.rope.tables(len)?;
        let mut x = x.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            tap(&mut probe, Tower::Text, i, Point::BlockInput, &x)?;
            let normed = block.operator_norm.forward(&x)?;
            tap(&mut probe, Tower::Text, i, Point::OperatorNorm, &normed)?;
            let op = match &block.operator {
                Operator::Conv(c) => c.forward(&normed, train)?,
                Operator::Attention(a) => a.forward(&normed, &mask, &cos, &sin, train)?,
            };
            tap(&mut probe, Tower::Text, i, Point::OperatorOut, &op)?;
            let h = (x + op)?;
            tap(&mut probe, Tower::Text, i, Point::ResidualMid, &h)?;
            let normed = block.ffn_norm.forward(&h)?;
            tap(&mut probe, Tower::Text, i, Point::FfnNorm, &normed)?;
            let ffn = block.mlp.forward(&normed, train)?;
            tap(&mut probe, Tower::Text, i, Point::MlpOut, &ffn)?;
            x = (h + ffn)?;
            tap(&mut probe, Tower::Text, i, Point::BlockOutput, &x)?;
        }
        self.final_norm.forward(&x)
    }

    /// Logits through the tied head, only for the rows asked for: the full (T, 65536) matrix would dominate memory,
    /// and the loss needs only the labelled positions.
    pub fn logits_at(&self, hidden: &Tensor, rows: &[u32]) -> Result<Tensor> {
        let hidden = hidden.squeeze(0)?;
        let picked = hidden.index_select(&Tensor::new(rows, hidden.device())?, 0)?;
        picked.matmul(&self.embed.t()?)
    }

    pub fn layer_type(&self, layer: usize) -> &'static str {
        if self.cfg.is_attention(layer) {
            "attn"
        } else {
            "conv"
        }
    }
}
