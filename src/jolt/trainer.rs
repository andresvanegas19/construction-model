//! Joint On-policy Learning and Teaching (JOLT / JOLT+) training loop.
//!
//! At each step, for every prompt $x$ in the micro-batch:
//! 1. Sample $G$ teacher rollouts $\tau_T \sim \pi_{\bar{\theta}_T}(\cdot \mid x, c)$ and $G$ student rollouts
//!    $\tau_S \sim \pi_{\bar{\theta}_S}(\cdot \mid x)$, verify rewards $R(\tau)$, and center advantages $\hat{A}_T, \hat{A}_S$.
//! 2. On each $\tau_T$, splice $\tau_T$ onto $(x, c)$ (Teacher with grad) and onto $x$ (Student detached) to optimize:
//!    $$\mathcal{L}_T(\theta_T) = -\text{sg}[\hat{A}_T(\tau_T)] \log \pi_{\theta_T}(\tau_T \mid x, c) + \beta_T \,\text{KL}(\pi_{\theta_T}(\cdot \mid s_t, c) \parallel \text{sg}[\pi_{\bar{\theta}_S}(\cdot \mid s_t)])$$
//! 3. On each $\tau_S$, splice $\tau_S$ onto $(x, c)$ (Teacher detached) and onto $x$ (Student with grad) to optimize:
//!    $$\mathcal{L}_S(\theta_S) = \beta_S \,\text{KL}(\pi_{\theta_S}(\cdot \mid s_t) \parallel \text{sg}[\pi_{\bar{\theta}_T}(\cdot \mid s_t, c)]) - \mathbb{I}[\text{JOLT+}] \cdot \text{sg}[\hat{A}_S(\tau_S)] \log \pi_{\theta_S}(\tau_S \mid x)$$
use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::Var;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Map};

use super::kl::{categorical_kl, mean_token_logprob};
use super::rollout::{prepare_prompts, sample_group, Trajectory};
use super::supervision::Supervisor;
use super::JoltArgs;
use crate::config::MODEL_ID;
use crate::dataset::chat::ChatEncoder;
use crate::dataset::Row;
use crate::lora::{self, AdapterRole, LoraBuilder};
use crate::train::metrics::{LoraSummary, RunLog};
use crate::train::{add_into, clip, eval_loss, lr_at, prepared, Session};

pub fn run(a: &JoltArgs) -> Result<()> {
    let dev = candle_core::Device::Cpu;
    let mut builder = match &a.train.resume {
        Some(dir) => LoraBuilder::with_adapter(dir, &dev)?,
        None => LoraBuilder::new(Some(a.train.lora()), a.train.seed),
    }
    .with_teacher(a.teacher_mode.is_separate(), None);
    if let Some(t_dir) = &a.teacher_init {
        builder.load_teacher_adapter(t_dir, &dev)?;
    }
    let cfg = builder.cfg.clone().context("LoRA config")?;
    let dir = a.train.m.root.join("out/runs").join(&a.train.run_id);
    let summary = LoraSummary { rank: cfg.rank, alpha: cfg.alpha, targets: cfg.targets.clone() };
    let sup_cfg = a.supervisor_config();
    let mut log = RunLog::start(&dir, &a.train.run_id, summary, 0, 0)?;
    log.summary.jolt = Some(a.summary_json(&sup_cfg));
    let supervisor = Supervisor::new(sup_cfg);
    let result = train_jolt(a, builder, &supervisor, &mut log);
    if let Err(e) = &result {
        eprintln!("jolt train failed: {e:#}");
    }
    log.finish(result.is_ok())?;
    result
}

#[derive(Default)]
struct StepStats {
    t_loss: f64,
    s_loss: f64,
    t_kl: f64,
    s_kl: f64,
    t_reward: f64,
    s_reward: f64,
    comp_tokens: u64,
    proc_tokens: u64,
    count: usize,
}

fn train_jolt(a: &JoltArgs, builder: LoraBuilder, supervisor: &Supervisor, log: &mut RunLog) -> Result<()> {
    let s = Session::open(&a.train.m, builder)?;
    let train_rows = prepared(&a.train.m.root, "train")?;
    let eval_rows = prepared(&a.train.m.root, "eval")?;
    let (eval, dropped_e) = s.encode(&eval_rows, &a.train.m.root, a.train.m.max_len)?;
    let chat_enc = ChatEncoder::new(&s.tokenizer, s.model.images.clone())?;

    let mut usable: Vec<Row> = Vec::with_capacity(train_rows.len());
    let mut dropped_t = 0usize;
    for row in &train_rows {
        match prepare_prompts(&chat_enc, row, &a.train.m.root, supervisor, a.train.m.max_len, a.max_new_tokens) {
            Ok(Some(_)) => usable.push(row.clone()),
            _ => dropped_t += 1,
        }
    }
    anyhow::ensure!(!usable.is_empty(), "no usable JOLT training rows: run `prepare` first");
    log.summary.train_samples = usable.len();
    log.summary.eval_samples = eval.len();

    let mut all_vars: Vec<Var> = s.lora.student_vars();
    if s.lora.has_separate_teacher() {
        all_vars.extend(s.lora.teacher_vars());
    }
    eprintln!(
        "jolt ({}) train {} (dropped {dropped_t}), eval {} (dropped {dropped_e}), mode={}, trainable vars={}",
        a.variant.as_str(),
        usable.len(),
        eval.len(),
        a.teacher_mode.as_str(),
        all_vars.len()
    );

    let params = ParamsAdamW { lr: a.train.lr, weight_decay: a.train.weight_decay, ..Default::default() };
    let mut opt = AdamW::new(all_vars.clone(), params)?;
    let mut rng = StdRng::seed_from_u64(a.train.seed);
    let mut order: Vec<usize> = vec![];
    let mut total_comp_tokens = 0u64;
    let mut total_proc_tokens = 0u64;
    let accum = a.train.grad_accum.max(1);
    let g = a.group_size.max(1);
    let scale = (accum * g) as f64;

    for step in 1..=a.train.steps {
        let lr = lr_at(step, a.train.steps, a.train.warmup, a.train.lr);
        opt.set_learning_rate(lr);
        let mut acc: Option<GradStore> = None;
        let mut stats = StepStats::default();

        for _ in 0..accum {
            if order.is_empty() {
                order = (0..usable.len()).collect();
                order.shuffle(&mut rng);
            }
            let row = &usable[order.pop().unwrap_or(0)];
            let Some((s_prompt, t_prompt)) =
                prepare_prompts(&chat_enc, row, &a.train.m.root, supervisor, a.train.m.max_len, a.max_new_tokens)?
            else {
                continue;
            };
            let group = sample_group(
                &s.model,
                &chat_enc,
                row,
                s_prompt,
                t_prompt,
                supervisor,
                g,
                a.max_new_tokens,
                a.temperature,
                &mut rng,
            )?;
            stats.t_reward += group.mean_teacher_reward();
            stats.s_reward += group.mean_student_reward();
            stats.comp_tokens += group.completion_tokens;
            stats.proc_tokens += group.processed_tokens;
            stats.count += 1;

            for traj in &group.teacher_trajs {
                let (loss_t, kl_val) = teacher_traj_loss(&s, &chat_enc, &group.student_prompt, &group.teacher_prompt, traj, a.beta_t)?;
                stats.t_loss += loss_t.to_scalar::<f32>()? as f64;
                stats.t_kl += kl_val;
                let grads = (loss_t / scale)?.backward()?;
                acc = Some(merge_grads(acc, grads, &all_vars)?);
            }

            for traj in &group.student_trajs {
                let (loss_s, kl_val) =
                    student_traj_loss(&s, &chat_enc, &group.student_prompt, &group.teacher_prompt, traj, a.beta_s, a.variant.student_rl())?;
                stats.s_loss += loss_s.to_scalar::<f32>()? as f64;
                stats.s_kl += kl_val;
                let grads = (loss_s / scale)?.backward()?;
                acc = Some(merge_grads(acc, grads, &all_vars)?);
            }
        }

        let mut store = acc.context("no gradients accumulated in JOLT step")?;
        clip(&mut store, &all_vars, a.train.clip)?;
        opt.step(&store)?;

        total_comp_tokens += stats.comp_tokens;
        total_proc_tokens += stats.proc_tokens;
        let denom = (stats.count * g).max(1) as f64;
        let group_denom = stats.count.max(1) as f64;
        let t_loss = stats.t_loss / denom;
        let s_loss = stats.s_loss / denom;
        let mut extra = Map::new();
        extra.insert("teacher_loss".into(), json!(t_loss));
        extra.insert("student_loss".into(), json!(s_loss));
        extra.insert("teacher_kl".into(), json!(stats.t_kl / denom));
        extra.insert("student_kl".into(), json!(stats.s_kl / denom));
        extra.insert("teacher_reward".into(), json!(stats.t_reward / group_denom));
        extra.insert("student_reward".into(), json!(stats.s_reward / group_denom));
        extra.insert("completion_tokens".into(), json!(total_comp_tokens));
        extra.insert("processed_tokens".into(), json!(total_proc_tokens));
        log.train_with_extra(step, t_loss + s_loss, lr, total_comp_tokens, Some(extra))?;

        if a.train.eval_every > 0 && step % a.train.eval_every == 0 && !eval.is_empty() {
            s.model.set_role(AdapterRole::Student);
            log.eval(step, eval_loss(&s, &eval, a.train.eval_max)?)?;
        }
        if a.train.save_every > 0 && step % a.train.save_every == 0 {
            save_both(&s.lora, &log.dir.join(format!("checkpoint-{step}")), &a.train.m.model)?;
        }
    }

    if !eval.is_empty() {
        s.model.set_role(AdapterRole::Student);
        log.eval(a.train.steps, eval_loss(&s, &eval, usize::MAX)?)?;
    }
    let base = if a.train.m.model.contains('/') { a.train.m.model.as_str() } else { MODEL_ID };
    lora::save(&s.lora, &log.dir.join("adapter"), base)?;
    if s.lora.has_separate_teacher() {
        lora::save_teacher(&s.lora, &log.dir.join("teacher_adapter"), base)?;
    }
    Ok(())
}

fn teacher_traj_loss(
    s: &Session,
    enc: &ChatEncoder,
    student_prompt: &crate::dataset::chat::Encoded,
    teacher_prompt: &crate::dataset::chat::Encoded,
    traj: &Trajectory,
    beta_t: f64,
) -> Result<(candle_core::Tensor, f64)> {
    let s_enc = enc.splice_completion(student_prompt, &traj.tokens);
    let t_enc = enc.splice_completion(teacher_prompt, &traj.tokens);
    s.model.set_role(AdapterRole::Student);
    let (s_logits, _) = s.model.completion_logits(&s_enc, false)?;
    let s_logits = s_logits.detach();

    s.model.set_role(AdapterRole::Teacher);
    let (t_logits, t_targets) = s.model.completion_logits(&t_enc, true)?;
    let kl = categorical_kl(&t_logits, &s_logits)?;
    let kl_val = kl.detach().to_scalar::<f32>()? as f64;
    let logp = mean_token_logprob(&t_logits, &t_targets)?;
    let pg = logp.affine(-traj.advantage, 0.0)?;
    let total = (pg + kl.affine(beta_t, 0.0)?)?;
    Ok((total, kl_val))
}

fn student_traj_loss(
    s: &Session,
    enc: &ChatEncoder,
    student_prompt: &crate::dataset::chat::Encoded,
    teacher_prompt: &crate::dataset::chat::Encoded,
    traj: &Trajectory,
    beta_s: f64,
    student_rl: bool,
) -> Result<(candle_core::Tensor, f64)> {
    let s_enc = enc.splice_completion(student_prompt, &traj.tokens);
    let t_enc = enc.splice_completion(teacher_prompt, &traj.tokens);
    s.model.set_role(AdapterRole::Teacher);
    let (t_logits, _) = s.model.completion_logits(&t_enc, false)?;
    let t_logits = t_logits.detach();

    s.model.set_role(AdapterRole::Student);
    let (s_logits, s_targets) = s.model.completion_logits(&s_enc, true)?;
    let kl = categorical_kl(&s_logits, &t_logits)?;
    let kl_val = kl.detach().to_scalar::<f32>()? as f64;
    let mut total = kl.affine(beta_s, 0.0)?;
    if student_rl && traj.advantage.abs() > 1e-12 {
        let logp = mean_token_logprob(&s_logits, &s_targets)?;
        total = (total + logp.affine(-traj.advantage, 0.0)?)?;
    }
    Ok((total, kl_val))
}

fn merge_grads(acc: Option<GradStore>, grads: GradStore, vars: &[Var]) -> Result<GradStore> {
    Ok(match acc {
        None => grads,
        Some(mut store) => {
            add_into(&mut store, &grads, vars)?;
            store
        }
    })
}

fn save_both(builder: &LoraBuilder, dir: &std::path::Path, model: &str) -> Result<()> {
    let base = if model.contains('/') { model } else { MODEL_ID };
    lora::save(builder, dir, base)?;
    if builder.has_separate_teacher() {
        lora::save_teacher(builder, &dir.join("teacher_adapter"), base)?;
    }
    Ok(())
}
