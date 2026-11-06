//! The assembled LFM2.5-VL: image tiles -> vision tower -> projector -> features scattered into the `<image>` token
//! positions of the text embeddings -> LFM2 -> tied head. Batch size is one sample with no padding: on CPU the
//! sequences differ wildly in length (a sketch-stage row vs a 10-tile image), so padding would waste most of the work.
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::Rng;

use crate::config::{ImageSettings, VlConfig};
use crate::dataset::chat::Encoded;
use crate::dataset::image::Tile;
use crate::lora::{AdapterRole, LoraBuilder};
use crate::model::projector::Projector;
use crate::model::text::TextModel;
use crate::model::vision::VisionTower;
use crate::model::Probe;
use crate::weights::{self, ModelFiles};

pub const IGNORE: i64 = -100;

pub struct Lfm2Vl {
    pub text: TextModel,
    pub vision: VisionTower,
    pub projector: Projector,
    pub cfg: VlConfig,
    pub images: ImageSettings,
    pub device: Device,
    lora_on: Arc<AtomicBool>,
    role: Arc<AtomicU8>,
}

impl Lfm2Vl {
    pub fn load(files: &ModelFiles, lora: &mut LoraBuilder, max_tiles: Option<usize>, device: &Device) -> Result<Self> {
        let (cfg, processor) = weights::read_configs(files)?;
        let shapes = weights::shapes(files)?;
        let vb = weights::var_builder(files, device)?;
        let root = vb.pp("model");
        let text = TextModel::load(root.pp("language_model"), lora, &cfg.text_config, &shapes)?;
        let vision = VisionTower::load(root.pp("vision_tower").pp("vision_model"), &cfg.vision_config)?;
        let projector = Projector::load(
            root.pp("multi_modal_projector"),
            lora,
            cfg.vision_config.hidden_size,
            cfg.projector_hidden_size,
            cfg.text_config.hidden_size,
            cfg.downsample_factor,
            cfg.projector_bias,
        )?;
        let images = ImageSettings::new(&cfg, &processor, max_tiles);
        Ok(Self {
            text,
            vision,
            projector,
            cfg,
            images,
            device: device.clone(),
            lora_on: lora.switch(),
            role: lora.role_switch(),
        })
    }

    /// Turns every adapter on or off at once (base vs tuned in the `lora_delta` experiment).
    pub fn set_lora(&self, on: bool) {
        self.lora_on.store(on, Ordering::Relaxed);
        if on && self.role.load(Ordering::Relaxed) == AdapterRole::Off as u8 {
            self.role.store(AdapterRole::Student as u8, Ordering::Relaxed);
        }
    }

    /// Switches every adapted layer between base (`Off`), `Student`, and `Teacher` adapters in-place.
    pub fn set_role(&self, role: AdapterRole) {
        self.lora_on.store(role != AdapterRole::Off, Ordering::Relaxed);
        self.role.store(role as u8, Ordering::Relaxed);
    }

    pub fn role(&self) -> AdapterRole {
        if !self.lora_on.load(Ordering::Relaxed) {
            AdapterRole::Off
        } else {
            AdapterRole::from_u8(self.role.load(Ordering::Relaxed))
        }
    }

    /// Vision tower output of one tile, (h*w, 1152). Detached: the tower is frozen and its graph would only cost
    /// memory during backprop.
    pub fn tile_states(&self, tile: &Tile, probe: Option<&mut dyn Probe>) -> candle_core::Result<Tensor> {
        let pixels = Tensor::from_slice(&tile.pixels, (tile.h * tile.w, tile.pixels.len() / (tile.h * tile.w)), &self.device)?;
        Ok(self.vision.forward(&pixels, tile.h, tile.w, probe)?.detach())
    }

    /// All image tokens of a sample in token order (tiles row-major, then the thumbnail), (M, text hidden).
    pub fn image_features(&self, tiles: &[Tile], train: bool) -> candle_core::Result<Option<Tensor>> {
        let mut parts = Vec::with_capacity(tiles.len());
        for tile in tiles {
            let states = self.tile_states(tile, None)?;
            parts.push(self.projector.forward(&states, tile.h, tile.w, train)?);
        }
        if parts.is_empty() {
            return Ok(None);
        }
        Ok(Some(Tensor::cat(&parts, 0)?))
    }

    fn embed_with_features(&self, ids: &[u32], feats: Option<&Tensor>, strict_image_count: bool) -> Result<Tensor> {
        let text = self.text.embed_ids(ids, &self.device)?;
        let Some(feats) = feats else {
            if strict_image_count && ids.iter().any(|&t| t == self.cfg.image_token_id) {
                bail!("sample has image tokens but no image tiles");
            }
            return Ok(text.unsqueeze(0)?);
        };
        let t = ids.len() as u32;
        let m = feats.dim(0)? as u32;
        let mut next = 0u32;
        let index: Vec<u32> = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                if id == self.cfg.image_token_id && (strict_image_count || next < m) {
                    next += 1;
                    t + next - 1
                } else {
                    i as u32
                }
            })
            .collect();
        if next != m {
            bail!("{next} image tokens in the text but {m} image features: the token layout and the tiling disagree");
        }
        let table = Tensor::cat(&[text, feats.clone()], 0)?;
        Ok(table.index_select(&Tensor::new(index.as_slice(), &self.device)?, 0)?.unsqueeze(0)?)
    }

    /// (1, T, H) input embeddings with image features in place of the image-token rows. A gather over
    /// cat(text rows, image rows) keeps the projector differentiable when it carries LoRA.
    pub fn embed(&self, enc: &Encoded, train: bool) -> Result<Tensor> {
        let feats = self.image_features(&enc.tiles, train)?;
        self.embed_with_features(&enc.ids, feats.as_ref(), true)
    }

    /// Next-token logits `[T_comp, vocab_size]` in F32 and target token IDs at the labelled (assistant) positions.
    pub fn completion_logits(&self, enc: &Encoded, train: bool) -> Result<(Tensor, Vec<u32>)> {
        let x = self.embed(enc, train)?;
        let hidden = self.text.forward(&x, train, None)?;
        let (rows, targets): (Vec<u32>, Vec<u32>) = enc
            .labels
            .iter()
            .enumerate()
            .skip(1)
            .filter(|(_, &l)| l != IGNORE)
            .map(|(i, &l)| (i as u32 - 1, l as u32))
            .unzip();
        if rows.is_empty() {
            bail!("sample has no labelled tokens");
        }
        let logits = self.text.logits_at(&hidden, &rows)?.to_dtype(DType::F32)?;
        Ok((logits, targets))
    }

    /// Mean next-token cross-entropy over the labelled (assistant) positions, and how many there were.
    pub fn loss(&self, enc: &Encoded, train: bool) -> Result<(Tensor, usize)> {
        let (logits, targets) = self.completion_logits(enc, train)?;
        let n = targets.len();
        let targets = Tensor::new(targets.as_slice(), &self.device)?;
        Ok((candle_nn::loss::cross_entropy(&logits, &targets)?, n))
    }

    /// Autoregressive rollout generation from `prompt` (`train = false`). Image tile features are computed once
    /// outside the token loop so multi-tile vision encoding is not repeated at every generated step.
    pub fn generate(
        &self,
        prompt: &Encoded,
        max_new_tokens: usize,
        temperature: f64,
        eos_id: u32,
        rng: &mut StdRng,
    ) -> Result<Vec<u32>> {
        if prompt.ids.is_empty() {
            bail!("cannot generate from an empty prompt");
        }
        if max_new_tokens == 0 {
            return Ok(vec![]);
        }
        let feats = self.image_features(&prompt.tiles, false)?;
        let mut ids = prompt.ids.clone();
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let x = self.embed_with_features(&ids, feats.as_ref(), false)?;
            let hidden = self.text.forward(&x, false, None)?;
            let last = (ids.len() - 1) as u32;
            let logits = self.text.logits_at(&hidden, &[last])?.squeeze(0)?.to_dtype(DType::F32)?;
            let vals: Vec<f32> = logits.to_vec1()?;
            let next_id = sample_token(&vals, temperature, rng);
            ids.push(next_id);
            generated.push(next_id);
            if next_id == eos_id {
                break;
            }
        }
        Ok(generated)
    }

    /// Forward only, for analysis: final hidden states with an optional probe on every text block.
    pub fn hidden(&self, enc: &Encoded, probe: Option<&mut dyn Probe>) -> Result<Tensor> {
        let x = self.embed(enc, false)?;
        Ok(self.text.forward(&x, false, probe)?)
    }
}

fn sample_token(logits: &[f32], temperature: f64, rng: &mut StdRng) -> u32 {
    if temperature <= 1e-5 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let inv_t = 1.0 / temperature;
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut weights = Vec::with_capacity(logits.len());
    let mut sum = 0.0f64;
    for &v in logits {
        let w = ((v as f64 - max_l) * inv_t).exp();
        let w = if w.is_finite() && w > 0.0 { w } else { 0.0 };
        sum += w;
        weights.push(w);
    }
    if sum <= 0.0 {
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }
    let mut draw = rng.gen::<f64>() * sum;
    for (i, &w) in weights.iter().enumerate() {
        draw -= w;
        if draw <= 0.0 {
            return i as u32;
        }
    }
    logits.len().saturating_sub(1) as u32
}
