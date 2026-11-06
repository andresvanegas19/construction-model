//! construction-model-train: LoRA fine-tuning and latent-geometry analysis of LiquidAI/LFM2.5-VL-1.6B in Candle.
mod config;
mod dataset;
mod jolt;
mod latent;
mod lora;
mod model;
mod train;
mod weights;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

use crate::lora::{LoraBuilder, LoraConfig, DEFAULT_TARGETS};
use crate::train::{ModelArgs, Session};

#[derive(Parser)]
#[command(name = "construction-model-train", about = "LFM2.5-VL LoRA SFT + JOLT + latent analysis (Candle, CPU)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build out/prepared/{train,eval}.jsonl from the datasets under --root/data.
    Prepare(PrepareArgs),
    /// LoRA SFT; writes out/runs/<run_id>/{metrics.jsonl,summary.json,adapter/}.
    Train(train::TrainArgs),
    /// Joint On-policy Learning and Teaching (JOLT / JOLT+) with supervision; writes out/runs/<run_id>/.
    Jolt(jolt::JoltArgs),
    /// Eval loss of the base model or of an adapter on the prepared eval split (JSON on stdout).
    Eval(EvalArgs),
    /// Parameter counts, layer types and the LoRA size for a rank, from the safetensors headers.
    Inspect(InspectArgs),
    /// Latent-space geometry (arXiv 2511.21594); writes out/analysis/<id>/report.json.
    Analyze(latent::AnalyzeArgs),
}

#[derive(Args)]
struct PrepareArgs {
    #[arg(long, default_value = env!("CARGO_MANIFEST_DIR"))]
    root: PathBuf,
    #[arg(long, default_value_t = 4)]
    image_eval_every: usize,
    #[arg(long)]
    max_rows: Option<usize>,
}

#[derive(Args)]
struct EvalArgs {
    #[command(flatten)]
    m: ModelArgs,
    #[arg(long)]
    adapter: Option<PathBuf>,
    #[arg(long, default_value_t = 64)]
    max: usize,
}

#[derive(Args)]
struct InspectArgs {
    #[arg(long, default_value = config::MODEL_ID)]
    model: String,
    #[arg(long, default_value_t = 16)]
    rank: usize,
    #[arg(long, value_delimiter = ',')]
    targets: Vec<String>,
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Prepare(a) => {
            let p = dataset::prepare::run(&dataset::prepare::PrepareOpts { root: a.root, image_eval_every: a.image_eval_every.max(2), max_rows: a.max_rows })?;
            println!("{}", serde_json::json!({"train": p.train, "eval": p.eval, "skipped": p.skipped}));
            Ok(())
        }
        Cmd::Train(a) => train::run(&a),
        Cmd::Jolt(a) => jolt::run(&a),
        Cmd::Eval(a) => eval(&a),
        Cmd::Inspect(a) => inspect(&a),
        Cmd::Analyze(a) => latent::run(&a),
    }
}

fn eval(a: &EvalArgs) -> Result<()> {
    let lora = match &a.adapter {
        Some(dir) => LoraBuilder::with_adapter(dir, &candle_core::Device::Cpu)?,
        None => LoraBuilder::new(None, 0),
    };
    let s = Session::open(&a.m, lora)?;
    s.model.set_lora(a.adapter.is_some());
    let (rows, dropped) = s.encode(&train::prepared(&a.m.root, "eval")?, &a.m.root, a.m.max_len)?;
    let loss = train::eval_loss(&s, &rows, a.max)?;
    println!("{}", serde_json::json!({"eval_loss": loss, "samples": rows.len().min(a.max), "dropped": dropped, "adapter": a.adapter}));
    Ok(())
}

fn inspect(a: &InspectArgs) -> Result<()> {
    let files = weights::fetch(&a.model)?;
    let (vl, _) = weights::read_configs(&files)?;
    let shapes = weights::shapes(&files)?;
    let mut groups: BTreeMap<&str, usize> = BTreeMap::new();
    for (name, shape) in &shapes {
        let group = if name.contains("vision_tower") {
            "vision_tower"
        } else if name.contains("multi_modal_projector") {
            "projector"
        } else {
            "language_model"
        };
        *groups.entry(group).or_default() += shape.iter().product::<usize>();
    }
    let total: usize = groups.values().sum();
    println!("tensors: {}  parameters: {total}", shapes.len());
    for (g, n) in &groups {
        println!("  {g:<16} {n:>13}  ({:.1}%)", 100.0 * *n as f64 / total.max(1) as f64);
    }
    let types: Vec<&str> = (0..vl.text_config.num_hidden_layers).map(|l| if vl.text_config.is_attention(l) { "attn" } else { "conv" }).collect();
    println!("text layers: {}", types.join(" "));
    let cfg = LoraConfig {
        rank: a.rank,
        alpha: 2.0 * a.rank as f64,
        dropout: 0.0,
        targets: if a.targets.is_empty() { DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect() } else { a.targets.clone() },
    };
    // LoRA size without loading the model: rank x (in + out) per targeted 2-D weight.
    let (mut modules, mut params) = (0usize, 0usize);
    for (name, shape) in &shapes {
        let Some(module) = name.strip_suffix(".weight") else { continue };
        if shape.len() == 2 && !module.contains("vision_tower") && !module.ends_with("embed_tokens") && cfg.targets_module(module) {
            modules += 1;
            params += a.rank * (shape[0] + shape[1]);
        }
    }
    println!("lora rank {} on [{}]: {modules} modules, {params} trainable parameters", a.rank, cfg.targets.join(","));
    Ok(())
}
