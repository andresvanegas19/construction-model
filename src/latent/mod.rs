//! `analyze`: the measurements of "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction"
//! (arXiv 2511.21594) on LFM2.5-VL. Hidden states are captured at the paper's in-block points of every text layer
//! (and of the vision tower), reduced with PCA / UMAP, and each experiment writes its part of report.json. Mapping
//! from the paper's findings to experiments and the decisions they inform: training/README.md.
pub mod capture;
pub mod experiments;
pub mod linalg;
pub mod pca;
pub mod report;
pub mod stats;
pub mod umap;

use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::dataset::chat::{ChatEncoder, Encoded};
use crate::lora::LoraBuilder;
use crate::model::vl::IGNORE;
use crate::train::{prepared, ModelArgs, Session};
use capture::States;

#[derive(Debug, Clone, Args)]
pub struct AnalyzeArgs {
    #[command(flatten)]
    pub m: ModelArgs,
    /// Output folder name under out/analysis/.
    #[arg(long)]
    pub id: String,
    /// Run every experiment (the default when `--only` is empty).
    #[arg(long)]
    pub all: bool,
    /// Comma list of experiment names (see experiments::NAMES).
    #[arg(long, value_delimiter = ',')]
    pub only: Vec<String>,
    /// One text per line instead of the prepared eval rows.
    #[arg(long)]
    pub text_file: Option<PathBuf>,
    /// Prepared split(s) used as the corpus: eval, train or both.
    #[arg(long, default_value = "eval")]
    pub corpus: String,
    #[arg(long, default_value_t = 64)]
    pub max_samples: usize,
    /// Text-only samples are cut to this many tokens (image samples are kept whole or skipped).
    #[arg(long, default_value_t = 512)]
    pub max_seq: usize,
    /// Total captured token positions across the corpus (memory: tokens x 16 layers x 7 points x 2048 x 4 B).
    #[arg(long, default_value_t = 4096)]
    pub max_tokens: usize,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
    /// PEFT-named adapter dir (adapter_model.safetensors + adapter_config.json) for `lora_delta`.
    #[arg(long)]
    pub adapter: Option<PathBuf>,
    /// With an adapter, run the corpus experiments on the tuned model instead of the base.
    #[arg(long)]
    pub tuned: bool,
    #[arg(long, default_value = "e")]
    pub repeat_token: String,
    #[arg(long, default_value_t = 128)]
    pub repeat_len: usize,
    /// Probe labels: `kind` (stage / image category) or `source`.
    #[arg(long, default_value = "kind")]
    pub probe_label: String,
    #[arg(long)]
    pub plan_dir: Option<PathBuf>,
    #[arg(long)]
    pub other_dir: Option<PathBuf>,
    #[arg(long, default_value_t = 40)]
    pub max_images: usize,
    /// Labeled embedding pairs for `cache_threshold` (README: how to export them).
    #[arg(long)]
    pub pairs: Option<PathBuf>,
    #[arg(long, default_value_t = 0.98)]
    pub cache_min: f64,
    /// Points per per-layer UMAP side file (the report's own point lists are capped at 2000).
    #[arg(long, default_value_t = 1000)]
    pub umap_points: usize,
}

pub struct Sample {
    pub enc: Encoded,
    pub source: String,
    pub kind: String,
}

pub struct Ctx<'a> {
    pub args: &'a AnalyzeArgs,
    pub session: Session,
    pub corpus: Vec<Sample>,
    states: Option<States>,
}

impl<'a> Ctx<'a> {
    pub fn samples(&self) -> Vec<&Encoded> {
        self.corpus.iter().map(|s| &s.enc).collect()
    }

    /// The shared capture of the corpus at every point of every text layer, taken once on first use.
    pub fn ensure(&mut self) -> Result<()> {
        if self.states.is_none() {
            let samples: Vec<&Encoded> = self.corpus.iter().map(|s| &s.enc).collect();
            let st = capture::capture(&self.session.model, &samples, &crate::model::Point::ALL, self.args.max_tokens, self.args.seed)?;
            self.states = Some(st);
        }
        Ok(())
    }

    /// The shared capture; call `ensure` first (split so experiments can hold it next to other borrows of ctx).
    pub fn st(&self) -> &States {
        self.states.as_ref().expect("Ctx::ensure not called")
    }

    pub fn encoder(&self) -> Result<ChatEncoder<'_>> {
        ChatEncoder::new(&self.session.tokenizer, self.session.model.images.clone())
    }

    /// Base model (adapters off) unless `--tuned`.
    pub fn default_lora(&self) {
        self.session.model.set_lora(self.args.tuned && self.args.adapter.is_some());
    }
}

pub fn run(a: &AnalyzeArgs) -> Result<()> {
    let mut report = report::Report::new(&a.m.root, &a.id)?;
    let result = analyze(a, &mut report);
    if let Err(e) = &result {
        eprintln!("analyze failed: {e:#}");
    }
    report.finish(result.is_ok())?;
    result
}

fn analyze(a: &AnalyzeArgs, report: &mut report::Report) -> Result<()> {
    let lora = match &a.adapter {
        Some(dir) => LoraBuilder::with_adapter(dir, &candle_core::Device::Cpu)?,
        None => LoraBuilder::new(None, 0),
    };
    let session = Session::open(&a.m, lora)?;
    let corpus = build_corpus(a, &session)?;
    eprintln!("corpus: {} samples", corpus.len());
    let mut ctx = Ctx { args: a, session, corpus, states: None };
    ctx.default_lora();
    let names: Vec<&str> = if a.only.is_empty() || a.all { experiments::NAMES.to_vec() } else { a.only.iter().map(|s| s.as_str()).collect() };
    for name in names {
        eprintln!("experiment {name}");
        // One failing experiment is reported in its own findings; the others still run.
        let exp = match experiments::run(name, &mut ctx, report) {
            Ok(e) => e,
            Err(err) => {
                let mut e = report::Experiment::default();
                e.finding(format!("failed: {err:#}"));
                e
            }
        };
        report.put(name, exp)?;
    }
    Ok(())
}

fn build_corpus(a: &AnalyzeArgs, s: &Session) -> Result<Vec<Sample>> {
    let enc = ChatEncoder::new(&s.tokenizer, s.model.images.clone())?;
    let mut out = Vec::new();
    if let Some(path) = &a.text_file {
        for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()).take(a.max_samples) {
            let mut ids = vec![enc.bos()];
            ids.extend(enc.text(line)?);
            ids.truncate(a.max_seq);
            let labels = vec![IGNORE; ids.len()];
            out.push(Sample { enc: Encoded { ids, labels, tiles: vec![] }, source: "text-file".into(), kind: "text".into() });
        }
        return Ok(out);
    }
    let splits: &[&str] = match a.corpus.as_str() {
        "train" => &["train"],
        "both" => &["eval", "train"],
        _ => &["eval"],
    };
    for split in splits {
        for row in prepared(&a.m.root, split)? {
            if out.len() >= a.max_samples {
                break;
            }
            let Ok(mut e) = enc.encode(&row.messages, &row.images, &a.m.root) else { continue };
            if e.tiles.is_empty() {
                e.ids.truncate(a.max_seq);
                e.labels.truncate(a.max_seq);
            } else if e.ids.len() > a.m.max_len {
                continue;
            }
            out.push(Sample { enc: e, source: row.source, kind: row.kind });
        }
    }
    Ok(out)
}
