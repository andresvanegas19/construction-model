//! Fetching LFM2.5-VL from the Hugging Face Hub and opening its safetensors. The hub cache follows HF_HOME (the
//! runtime can point it at persistent storage), so the 3 GB checkpoint is downloaded once.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::config::{ProcessorConfig, VlConfig};

pub struct ModelFiles {
    pub config: PathBuf,
    pub processor: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub weights: Vec<PathBuf>,
}

/// `model` is a hub id or a local directory with the same files (useful when a sandbox already holds a snapshot).
pub fn fetch(model: &str) -> Result<ModelFiles> {
    let local = Path::new(model);
    if local.is_dir() {
        return from_dir(local);
    }
    let api = hf_hub::api::sync::ApiBuilder::from_env().with_progress(false).build().context("hf-hub client")?;
    let repo = api.model(model.to_string());
    let config = repo.get("config.json").context("config.json")?;
    let processor = repo.get("preprocessor_config.json").or_else(|_| repo.get("processor_config.json")).ok();
    let tokenizer = repo.get("tokenizer.json").context("tokenizer.json")?;
    let weights = match repo.get("model.safetensors") {
        Ok(p) => vec![p],
        Err(_) => {
            let index = repo.get("model.safetensors.index.json").context("no model.safetensors nor an index")?;
            shard_names(&index)?.iter().map(|n| repo.get(n).with_context(|| n.clone())).collect::<Result<_>>()?
        }
    };
    Ok(ModelFiles { config, processor, tokenizer, weights })
}

fn from_dir(dir: &Path) -> Result<ModelFiles> {
    let single = dir.join("model.safetensors");
    let weights = if single.is_file() {
        vec![single]
    } else {
        shard_names(&dir.join("model.safetensors.index.json"))?.iter().map(|n| dir.join(n)).collect()
    };
    let processor = ["preprocessor_config.json", "processor_config.json"].iter().map(|n| dir.join(n)).find(|p| p.is_file());
    Ok(ModelFiles { config: dir.join("config.json"), processor, tokenizer: dir.join("tokenizer.json"), weights })
}

fn shard_names(index: &Path) -> Result<Vec<String>> {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(index).with_context(|| index.display().to_string())?)?;
    let map = v["weight_map"].as_object().ok_or_else(|| anyhow!("index without weight_map"))?;
    let mut names: Vec<String> = map.values().filter_map(|s| s.as_str().map(String::from)).collect();
    names.sort();
    names.dedup();
    Ok(names)
}

pub fn read_configs(files: &ModelFiles) -> Result<(VlConfig, ProcessorConfig)> {
    let vl: VlConfig = serde_json::from_slice(&std::fs::read(&files.config)?).context("config.json")?;
    let processor = match &files.processor {
        Some(p) => serde_json::from_slice(&std::fs::read(p)?).context("processor config")?,
        None => ProcessorConfig::default(),
    };
    Ok((vl, processor))
}

/// f32 on CPU: candle's CPU backend has no bf16 matmul worth using for backprop, and LoRA gradients in bf16 lose the
/// small updates; 1.6B params x 4 bytes = about 6.4 GB for the base weights alone.
pub fn var_builder(files: &ModelFiles, device: &Device) -> Result<VarBuilder<'static>> {
    // SAFETY: the mmapped files are the hub cache's immutable blobs, not modified while the process runs.
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files.weights, DType::F32, device)? };
    Ok(vb)
}

/// Tensor name -> shape, from the safetensors headers only (no data is read), for `inspect` and width discovery.
pub fn shapes(files: &ModelFiles) -> Result<BTreeMap<String, Vec<usize>>> {
    // SAFETY: as in `var_builder`.
    let st = unsafe { MmapedSafetensors::multi(&files.weights)? };
    Ok(st.tensors().into_iter().map(|(name, view)| (name, view.shape().to_vec())).collect())
}

/// One tensor by name as f32 (the base weight next to an adapter, for `lora_delta`).
pub fn tensor(files: &ModelFiles, name: &str) -> Result<candle_core::Tensor> {
    // SAFETY: as in `var_builder`.
    let st = unsafe { MmapedSafetensors::multi(&files.weights)? };
    Ok(st.load(name, &Device::Cpu).with_context(|| format!("tensor {name}"))?.to_dtype(DType::F32)?)
}

pub fn tokenizer(files: &ModelFiles) -> Result<tokenizers::Tokenizer> {
    tokenizers::Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow!("tokenizer.json: {e}"))
}
