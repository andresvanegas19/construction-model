//! LoRA on frozen Linear layers. The base weight is a plain tensor loaded from the checkpoint (candle records no
//! graph for it, so backprop only reaches A and B); A and B are `Var`s. Adapters are written with PEFT's key names so
//! the Python side (`serve_openai.py`-style `PeftModel.from_pretrained` + merge) can load what this trainer produced.
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Text-model modules worth adapting on a hybrid conv/attention backbone; the latent `lora_delta` experiment ranks
/// them so later runs can narrow this list.
pub const DEFAULT_TARGETS: [&str; 8] = ["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3"];

/// Selects which adapter bank executes inside each `LoraLinear` forward pass, so base, student, and teacher share a
/// single in-memory copy of the frozen 6.4 GB backbone weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AdapterRole {
    Off = 0,
    Student = 1,
    Teacher = 2,
}

impl AdapterRole {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Off,
            2 => Self::Teacher,
            _ => Self::Student,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    pub dropout: f32,
    pub targets: Vec<String>,
}

impl LoraConfig {
    pub fn scale(&self) -> f64 {
        self.alpha / self.rank as f64
    }

    /// `name` is the full module path (`model.language_model.layers.3.self_attn.q_proj`); a target matches its last
    /// segment, or `projector` matches both projector layers.
    pub fn targets_module(&self, name: &str) -> bool {
        let last = name.rsplit('.').next().unwrap_or(name);
        self.targets.iter().any(|t| t == last || (t == "projector" && name.contains("multi_modal_projector")))
    }
}

pub struct Adapter {
    pub name: String,
    pub a: Var,
    pub b: Var,
}

/// Hands out LoRA-wrapped linears while the model is built and keeps the trainable pairs in build order.
pub struct LoraBuilder {
    pub cfg: Option<LoraConfig>,
    pub adapters: Vec<Adapter>,
    pub teacher_adapters: Vec<Adapter>,
    preset: HashMap<String, (Tensor, Tensor)>,
    teacher_preset: HashMap<String, (Tensor, Tensor)>,
    from_saved: bool,
    teacher_from_saved: bool,
    separate_teacher: bool,
    enabled: Arc<AtomicBool>,
    role: Arc<AtomicU8>,
    rng: StdRng,
}

impl LoraBuilder {
    pub fn new(cfg: Option<LoraConfig>, seed: u64) -> Self {
        Self {
            cfg,
            adapters: vec![],
            teacher_adapters: vec![],
            preset: HashMap::new(),
            teacher_preset: HashMap::new(),
            from_saved: false,
            teacher_from_saved: false,
            separate_teacher: false,
            enabled: Arc::new(AtomicBool::new(true)),
            role: Arc::new(AtomicU8::new(AdapterRole::Student as u8)),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Starts from saved A/B pairs (eval, analysis, or continuing a run) instead of fresh ones.
    pub fn with_adapter(dir: &Path, device: &Device) -> Result<Self> {
        let cfg = read_config(dir)?;
        let preset = load_preset(dir, device)?;
        let mut builder = Self::new(Some(cfg), 0);
        builder.preset = preset;
        builder.from_saved = true;
        Ok(builder)
    }

    /// Configures a second independent teacher LoRA bank on the same frozen linear layers when `separate_teacher` is
    /// true; when false, `AdapterRole::Teacher` shares the primary student adapter.
    pub fn with_teacher(mut self, separate_teacher: bool, teacher_preset: Option<HashMap<String, (Tensor, Tensor)>>) -> Self {
        self.separate_teacher = separate_teacher;
        if let Some(preset) = teacher_preset {
            self.teacher_preset = preset;
            self.teacher_from_saved = true;
        }
        self
    }

    /// Loads saved teacher A/B pairs from `dir` and enables a separate teacher adapter bank.
    pub fn load_teacher_adapter(&mut self, dir: &Path, device: &Device) -> Result<()> {
        self.teacher_preset = load_preset(dir, device)?;
        self.teacher_from_saved = true;
        self.separate_teacher = true;
        Ok(())
    }

    /// One switch shared by every adapted layer, so analysis can run base and tuned passes on one loaded model.
    pub fn switch(&self) -> Arc<AtomicBool> {
        self.enabled.clone()
    }

    /// Shared role selector (`Off`, `Student`, `Teacher`) read by every adapted layer during forward passes.
    pub fn role_switch(&self) -> Arc<AtomicU8> {
        self.role.clone()
    }

    pub fn has_separate_teacher(&self) -> bool {
        self.separate_teacher && !self.teacher_adapters.is_empty()
    }

    fn init_pair(&mut self, rank: usize, in_dim: usize, out_dim: usize, device: &Device) -> Result<(Var, Var)> {
        // PEFT's init: A ~ kaiming-uniform(a=sqrt 5) = U(-1/sqrt(in), 1/sqrt(in)), B = 0, so step 0 is the base.
        let bound = 1.0 / (in_dim as f64).sqrt();
        let values: Vec<f32> = (0..rank * in_dim).map(|_| self.rng.gen_range(-bound..bound) as f32).collect();
        let a = Tensor::from_vec(values, (rank, in_dim), device)?;
        Ok((Var::from_tensor(&a)?, Var::zeros((out_dim, rank), DType::F32, device)?))
    }

    pub fn linear(&mut self, vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<LoraLinear> {
        let name = vb.prefix();
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias { Some(vb.get(out_dim, "bias")?) } else { None };
        let mut layer = LoraLinear {
            weight,
            bias,
            lora: None,
            teacher_lora: None,
            enabled: self.enabled.clone(),
            role: self.role.clone(),
        };
        let Some(cfg) = self.cfg.clone() else { return Ok(layer) };
        let pair = if let Some((a, b)) = self.preset.remove(&name) {
            Some((Var::from_tensor(&a)?, Var::from_tensor(&b)?))
        } else if !self.from_saved && cfg.targets_module(&name) {
            Some(self.init_pair(cfg.rank, in_dim, out_dim, vb.device())?)
        } else {
            None
        };
        if let Some((a, b)) = pair {
            if self.separate_teacher {
                let t_pair = if let Some((ta, tb)) = self.teacher_preset.remove(&name) {
                    Some((Var::from_tensor(&ta)?, Var::from_tensor(&tb)?))
                } else if !self.teacher_from_saved {
                    if self.from_saved {
                        Some((Var::from_tensor(a.as_tensor())?, Var::from_tensor(b.as_tensor())?))
                    } else {
                        Some(self.init_pair(cfg.rank, in_dim, out_dim, vb.device())?)
                    }
                } else {
                    None
                };
                if let Some((ta, tb)) = t_pair {
                    self.teacher_adapters.push(Adapter { name: name.clone(), a: ta.clone(), b: tb.clone() });
                    layer.teacher_lora = Some(LoraPart { a: ta, b: tb, scale: cfg.scale(), dropout: cfg.dropout });
                }
            }
            self.adapters.push(Adapter { name, a: a.clone(), b: b.clone() });
            layer.lora = Some(LoraPart { a, b, scale: cfg.scale(), dropout: cfg.dropout });
        }
        Ok(layer)
    }

    pub fn vars(&self) -> Vec<Var> {
        self.adapters.iter().flat_map(|a| [a.a.clone(), a.b.clone()]).collect()
    }

    pub fn student_vars(&self) -> Vec<Var> {
        self.vars()
    }

    pub fn teacher_vars(&self) -> Vec<Var> {
        if self.teacher_adapters.is_empty() {
            self.student_vars()
        } else {
            self.teacher_adapters.iter().flat_map(|a| [a.a.clone(), a.b.clone()]).collect()
        }
    }

    pub fn trainable_params(&self) -> usize {
        self.adapters.iter().map(|a| a.a.elem_count() + a.b.elem_count()).sum()
    }
}

struct LoraPart {
    a: Var,
    b: Var,
    scale: f64,
    dropout: f32,
}

pub struct LoraLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    lora: Option<LoraPart>,
    teacher_lora: Option<LoraPart>,
    enabled: Arc<AtomicBool>,
    role: Arc<AtomicU8>,
}

impl LoraLinear {
    /// Plain frozen linear (vision tower): no builder, no adapter.
    pub fn frozen(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = if bias { Some(vb.get(out_dim, "bias")?) } else { None };
        Ok(Self {
            weight,
            bias,
            lora: None,
            teacher_lora: None,
            enabled: Arc::new(AtomicBool::new(false)),
            role: Arc::new(AtomicU8::new(AdapterRole::Off as u8)),
        })
    }

    pub fn forward(&self, x: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let mut y = x.broadcast_matmul(&self.weight.t()?)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        if !self.enabled.load(Ordering::Relaxed) {
            return Ok(y);
        }
        let active = match AdapterRole::from_u8(self.role.load(Ordering::Relaxed)) {
            AdapterRole::Off => None,
            AdapterRole::Student => self.lora.as_ref(),
            AdapterRole::Teacher => self.teacher_lora.as_ref().or(self.lora.as_ref()),
        };
        if let Some(l) = active {
            // candle's dropout draws from its own unseeded RNG; with the default 0.0 runs stay reproducible.
            let xin = if train && l.dropout > 0.0 { candle_nn::ops::dropout(x, l.dropout)? } else { x.clone() };
            let delta = xin.broadcast_matmul(&l.a.as_tensor().t()?)?.broadcast_matmul(&l.b.as_tensor().t()?)?;
            y = (y + delta.affine(l.scale, 0.0)?)?;
        }
        Ok(y)
    }
}

fn load_preset(dir: &Path, device: &Device) -> Result<HashMap<String, (Tensor, Tensor)>> {
    let tensors = candle_core::safetensors::load(dir.join("adapter_model.safetensors"), device)?;
    let mut preset = HashMap::new();
    for (key, a) in &tensors {
        if let Some(module) = key.strip_prefix("base_model.model.").and_then(|k| k.strip_suffix(".lora_A.weight")) {
            let b_key = format!("base_model.model.{module}.lora_B.weight");
            let b = tensors.get(&b_key).with_context(|| format!("adapter has {key} without {b_key}"))?;
            preset.insert(module.to_string(), (a.to_dtype(DType::F32)?, b.to_dtype(DType::F32)?));
        }
    }
    Ok(preset)
}

fn save_adapters(cfg: &LoraConfig, adapters: &[Adapter], dir: &Path, base_model: &str) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for a in adapters {
        tensors.insert(format!("base_model.model.{}.lora_A.weight", a.name), a.a.as_tensor().clone());
        tensors.insert(format!("base_model.model.{}.lora_B.weight", a.name), a.b.as_tensor().clone());
    }
    candle_core::safetensors::save(&tensors, dir.join("adapter_model.safetensors"))?;
    // PEFT reads target_modules as module-name suffixes; the projector shorthand expands to its two layers.
    let targets: Vec<String> = cfg
        .targets
        .iter()
        .flat_map(|t| if t == "projector" { vec!["linear_1".to_string(), "linear_2".to_string()] } else { vec![t.clone()] })
        .collect();
    let config = serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": cfg.rank,
        "lora_alpha": cfg.alpha,
        "lora_dropout": cfg.dropout,
        "target_modules": targets,
        "bias": "none",
        "fan_in_fan_out": false,
        "inference_mode": true,
        "modules_to_save": null,
        "render2d_targets": cfg.targets,
    });
    std::fs::write(dir.join("adapter_config.json"), serde_json::to_vec_pretty(&config)?)?;
    Ok(())
}

pub fn save(builder: &LoraBuilder, dir: &Path, base_model: &str) -> Result<()> {
    let cfg = builder.cfg.as_ref().context("no LoRA config to save")?;
    save_adapters(cfg, &builder.adapters, dir, base_model)
}

pub fn save_teacher(builder: &LoraBuilder, dir: &Path, base_model: &str) -> Result<()> {
    let cfg = builder.cfg.as_ref().context("no LoRA config to save")?;
    let adapters = if builder.teacher_adapters.is_empty() { &builder.adapters } else { &builder.teacher_adapters };
    save_adapters(cfg, adapters, dir, base_model)
}

pub fn read_config(dir: &Path) -> Result<LoraConfig> {
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("adapter_config.json")).with_context(|| format!("{}/adapter_config.json", dir.display()))?,
    )?;
    let targets = v["render2d_targets"]
        .as_array()
        .or_else(|| v["target_modules"].as_array())
        .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
        .unwrap_or_else(|| DEFAULT_TARGETS.iter().map(|s| s.to_string()).collect());
    Ok(LoraConfig {
        rank: v["r"].as_u64().unwrap_or(8) as usize,
        alpha: v["lora_alpha"].as_f64().unwrap_or(16.0),
        dropout: v["lora_dropout"].as_f64().unwrap_or(0.0) as f32,
        targets,
    })
}

/// Saved A/B pairs per module name, for the `lora_delta` experiment (no model needed).
pub fn read_pairs(dir: &Path) -> Result<BTreeMap<String, (Tensor, Tensor)>> {
    let tensors = candle_core::safetensors::load(dir.join("adapter_model.safetensors"), &Device::Cpu)?;
    let mut out = BTreeMap::new();
    for (key, a) in &tensors {
        if let Some(module) = key.strip_prefix("base_model.model.").and_then(|k| k.strip_suffix(".lora_A.weight")) {
            if let Some(b) = tensors.get(&format!("base_model.model.{module}.lora_B.weight")) {
                out.insert(module.to_string(), (a.to_dtype(DType::F32)?, b.to_dtype(DType::F32)?));
            }
        }
    }
    Ok(out)
}
