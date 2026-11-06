//! `train` / `eval`: LoRA SFT of LFM2.5-VL on the prepared rows. CPU only, one sample
//! per micro-batch, gradients summed over `--grad-accum` samples, AdamW on the adapter vars only, cosine LR with
//! linear warmup.
mod r#loop;
pub mod metrics;

use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::Device;
use clap::Args;

use crate::config::MODEL_ID;
use crate::dataset::chat::{ChatEncoder, Encoded};
use crate::dataset::{read_jsonl, Row};
use crate::lora::{LoraBuilder, LoraConfig, DEFAULT_TARGETS};
use crate::model::vl::Lfm2Vl;
use crate::weights;

pub use r#loop::{add_into, clip, eval_loss, run};

#[derive(Debug, Clone, Args)]
pub struct ModelArgs {
    /// Hub id or a local snapshot directory.
    #[arg(long, default_value = MODEL_ID)]
    pub model: String,
    /// The training/ folder (data/, out/).
    #[arg(long, default_value = env!("CARGO_MANIFEST_DIR"))]
    pub root: PathBuf,
    /// Caps the tiles per image (each tile is 1024 patches through 27 vision layers + 256 text tokens on CPU).
    #[arg(long, default_value_t = 4)]
    pub max_tiles: usize,
    /// Longer samples are skipped rather than truncated: a cut assistant turn teaches the model to stop mid-answer.
    #[arg(long, default_value_t = 3072)]
    pub max_len: usize,
}

#[derive(Debug, Clone, Args)]
pub struct TrainArgs {
    #[command(flatten)]
    pub m: ModelArgs,
    #[arg(long)]
    pub run_id: String,
    #[arg(long, default_value_t = 200)]
    pub steps: u64,
    #[arg(long, default_value_t = 2e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 20)]
    pub warmup: u64,
    #[arg(long, default_value_t = 8)]
    pub grad_accum: usize,
    #[arg(long, default_value_t = 16)]
    pub rank: usize,
    #[arg(long, default_value_t = 32.0)]
    pub alpha: f64,
    #[arg(long, default_value_t = 0.0)]
    pub dropout: f32,
    /// Comma list of module names (last path segment), or `projector`; see lora::DEFAULT_TARGETS.
    #[arg(long, value_delimiter = ',')]
    pub targets: Vec<String>,
    #[arg(long, default_value_t = 0.0)]
    pub weight_decay: f64,
    #[arg(long, default_value_t = 1.0)]
    pub clip: f64,
    #[arg(long, default_value_t = 50)]
    pub eval_every: u64,
    /// Eval samples per periodic eval (all of them at the end).
    #[arg(long, default_value_t = 32)]
    pub eval_max: usize,
    #[arg(long, default_value_t = 100)]
    pub save_every: u64,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// Continue from a saved adapter (its rank / alpha / targets win over the flags).
    #[arg(long)]
    pub resume: Option<PathBuf>,
}

impl TrainArgs {
    pub fn lora(&self) -> LoraConfig {
        let targets = if self.targets.is_empty() { DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect() } else { self.targets.clone() };
        LoraConfig { rank: self.rank, alpha: self.alpha, dropout: self.dropout, targets }
    }
}

/// Linear warmup to `peak`, then cosine to 10 % of it at `total` (not to zero: the last steps still move).
pub fn lr_at(step: u64, total: u64, warmup: u64, peak: f64) -> f64 {
    if warmup > 0 && step <= warmup {
        return peak * step as f64 / warmup as f64;
    }
    let span = total.saturating_sub(warmup).max(1) as f64;
    let t = ((step - warmup.min(step)) as f64 / span).min(1.0);
    let floor = 0.1 * peak;
    floor + (peak - floor) * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
}

/// A loaded model plus everything needed to encode rows for it.
pub struct Session {
    pub files: weights::ModelFiles,
    pub model: Lfm2Vl,
    pub lora: LoraBuilder,
    pub tokenizer: tokenizers::Tokenizer,
}

impl Session {
    pub fn open(m: &ModelArgs, lora: LoraBuilder) -> Result<Self> {
        let device = Device::Cpu;
        let files = weights::fetch(&m.model)?;
        let tokenizer = weights::tokenizer(&files)?;
        let mut lora = lora;
        let model = Lfm2Vl::load(&files, &mut lora, Some(m.max_tiles), &device)?;
        Ok(Self { files, model, lora, tokenizer })
    }

    /// Encodes rows, dropping (and counting) the ones that fail or exceed `max_len`.
    pub fn encode(&self, rows: &[Row], root: &Path, max_len: usize) -> Result<(Vec<(Row, Encoded)>, usize)> {
        let enc = ChatEncoder::new(&self.tokenizer, self.model.images.clone())?;
        let mut out = Vec::with_capacity(rows.len());
        let mut dropped = 0;
        for row in rows {
            match enc.encode(&row.messages, &row.images, root) {
                Ok(e) if e.ids.len() <= max_len && e.trained_tokens() > 0 => out.push((row.clone(), e)),
                Ok(_) => dropped += 1,
                Err(err) => {
                    eprintln!("skip row ({}/{}): {err:#}", row.source, row.kind);
                    dropped += 1;
                }
            }
        }
        Ok((out, dropped))
    }
}

pub fn prepared(root: &Path, split: &str) -> Result<Vec<Row>> {
    read_jsonl(&root.join(format!("out/prepared/{split}.jsonl")))
}

#[cfg(test)]
mod tests {
    use super::lr_at;

    #[test]
    fn schedule_shape() {
        assert_eq!(lr_at(0, 100, 10, 1.0), 0.0);
        assert!((lr_at(10, 100, 10, 1.0) - 1.0).abs() < 1e-12);
        assert!((lr_at(100, 100, 10, 1.0) - 0.1).abs() < 1e-12);
        assert!(lr_at(50, 100, 10, 1.0) < 1.0 && lr_at(50, 100, 10, 1.0) > 0.1);
    }
}
