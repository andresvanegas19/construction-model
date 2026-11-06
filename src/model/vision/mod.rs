//! The SigLIP2 vision tower as LFM2-VL uses it: embeddings, 27 encoder layers and `post_layernorm` (applied: the
//! VL model reads `last_hidden_state`, which includes it); no pooling head.
pub mod embeddings;
pub mod encoder;

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::config::VisionConfig;
use crate::model::norm::LayerNorm;
use crate::model::Probe;

pub struct VisionTower {
    embeddings: embeddings::Embeddings,
    layers: Vec<encoder::EncoderLayer>,
    post_norm: LayerNorm,
    pub cfg: VisionConfig,
}

impl VisionTower {
    /// `vb` is rooted at `model.vision_tower.vision_model`.
    pub fn load(vb: VarBuilder, cfg: &VisionConfig) -> anyhow::Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| encoder::EncoderLayer::load(vb.pp("encoder").pp("layers").pp(i), cfg))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Self {
            embeddings: embeddings::Embeddings::load(
                vb.pp("embeddings"),
                cfg.hidden_size,
                cfg.patch_size,
                cfg.num_channels,
                cfg.num_patches,
            )?,
            layers,
            post_norm: LayerNorm::load(vb.pp("post_layernorm"), cfg.hidden_size, cfg.layer_norm_eps)?,
            cfg: cfg.clone(),
        })
    }

    /// One tile: `pixels` (h*w, 768) -> (h*w, 1152).
    pub fn forward(&self, pixels: &Tensor, h: usize, w: usize, mut probe: Option<&mut dyn Probe>) -> Result<Tensor> {
        let mut x = self.embeddings.forward(pixels, h, w)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, i, &mut probe)?;
        }
        self.post_norm.forward(&x)
    }
}
