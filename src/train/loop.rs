//! The optimisation loop. candle's GradStore cannot be built empty, so the first micro-batch's store is the
//! accumulator and later ones are added into it var by var; clipping rewrites the same store before the step.
use anyhow::{Context, Result};
use candle_core::backprop::GradStore;
use candle_core::Var;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::metrics::{LoraSummary, RunLog};
use super::{lr_at, prepared, Session, TrainArgs};
use crate::config::MODEL_ID;
use crate::dataset::chat::Encoded;
use crate::dataset::Row;
use crate::lora::{self, LoraBuilder};

pub fn run(a: &TrainArgs) -> Result<()> {
    let builder = match &a.resume {
        Some(dir) => LoraBuilder::with_adapter(dir, &candle_core::Device::Cpu)?,
        None => LoraBuilder::new(Some(a.lora()), a.seed),
    };
    let cfg = builder.cfg.clone().context("LoRA config")?;
    let dir = a.m.root.join("out/runs").join(&a.run_id);
    let summary = LoraSummary { rank: cfg.rank, alpha: cfg.alpha, targets: cfg.targets.clone() };
    // The summary exists before the model loads, so a dashboard sees the run (and a failed load) at once.
    let mut log = RunLog::start(&dir, &a.run_id, summary, 0, 0)?;
    let result = train(a, builder, &mut log);
    if let Err(e) = &result {
        eprintln!("train failed: {e:#}");
    }
    log.finish(result.is_ok())?;
    result
}

fn train(a: &TrainArgs, builder: LoraBuilder, log: &mut RunLog) -> Result<()> {
    let s = Session::open(&a.m, builder)?;
    let (train, dropped_t) = s.encode(&prepared(&a.m.root, "train")?, &a.m.root, a.m.max_len)?;
    let (eval, dropped_e) = s.encode(&prepared(&a.m.root, "eval")?, &a.m.root, a.m.max_len)?;
    eprintln!(
        "train {} (dropped {dropped_t}), eval {} (dropped {dropped_e}), trainable params {}",
        train.len(),
        eval.len(),
        s.lora.trainable_params()
    );
    anyhow::ensure!(!train.is_empty(), "no trainable samples: run `prepare` first");
    log.summary.train_samples = train.len();
    log.summary.eval_samples = eval.len();

    let vars = s.lora.vars();
    let params = ParamsAdamW { lr: a.lr, weight_decay: a.weight_decay, ..Default::default() };
    let mut opt = AdamW::new(vars.clone(), params)?;
    let mut rng = StdRng::seed_from_u64(a.seed);
    let mut order: Vec<usize> = vec![];
    let mut tokens_seen = 0u64;
    for step in 1..=a.steps {
        let lr = lr_at(step, a.steps, a.warmup, a.lr);
        opt.set_learning_rate(lr);
        let mut acc: Option<GradStore> = None;
        let mut loss_sum = 0.0;
        for _ in 0..a.grad_accum.max(1) {
            if order.is_empty() {
                order = (0..train.len()).collect();
                order.shuffle(&mut rng);
            }
            let (_, enc) = &train[order.pop().unwrap_or(0)];
            let (loss, n) = s.model.loss(enc, true)?;
            loss_sum += loss.to_scalar::<f32>()? as f64;
            tokens_seen += n as u64;
            // Each sample's mean token loss weighs the same; dividing here makes the summed grads the batch mean.
            let grads = (loss / a.grad_accum.max(1) as f64)?.backward()?;
            acc = Some(match acc {
                None => grads,
                Some(mut store) => {
                    add_into(&mut store, &grads, &vars)?;
                    store
                }
            });
        }
        let mut store = acc.context("no gradients")?;
        clip(&mut store, &vars, a.clip)?;
        opt.step(&store)?;
        log.train(step, loss_sum / a.grad_accum.max(1) as f64, lr, tokens_seen)?;
        if a.eval_every > 0 && step % a.eval_every == 0 && !eval.is_empty() {
            log.eval(step, eval_loss(&s, &eval, a.eval_max)?)?;
        }
        if a.save_every > 0 && step % a.save_every == 0 {
            lora::save(&s.lora, &log.dir.join(format!("checkpoint-{step}")), &a.m.model)?;
        }
    }
    if !eval.is_empty() {
        log.eval(a.steps, eval_loss(&s, &eval, usize::MAX)?)?;
    }
    lora::save(&s.lora, &log.dir.join("adapter"), if a.m.model.contains('/') { a.m.model.as_str() } else { MODEL_ID })?;
    Ok(())
}

pub fn add_into(store: &mut GradStore, other: &GradStore, vars: &[Var]) -> Result<()> {
    for v in vars {
        if let Some(g) = other.get(v.as_tensor()) {
            let sum = match store.get(v.as_tensor()) {
                Some(prev) => (prev + g)?,
                None => g.clone(),
            };
            store.insert(v.as_tensor(), sum);
        }
    }
    Ok(())
}

/// Global-norm clipping over the adapter grads (an image sample's first steps can spike the projector's).
pub fn clip(store: &mut GradStore, vars: &[Var], max_norm: f64) -> Result<()> {
    if max_norm <= 0.0 {
        return Ok(());
    }
    let mut sq = 0.0f64;
    for v in vars {
        if let Some(g) = store.get(v.as_tensor()) {
            sq += g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        }
    }
    let norm = sq.sqrt();
    if norm.is_finite() && norm > max_norm {
        let k = max_norm / norm;
        for v in vars {
            if let Some(g) = store.get(v.as_tensor()) {
                let scaled = g.affine(k, 0.0)?;
                store.insert(v.as_tensor(), scaled);
            }
        }
    }
    Ok(())
}

/// Mean per-sample loss over the first `max` eval samples, without keeping any graph.
pub fn eval_loss(s: &Session, eval: &[(Row, Encoded)], max: usize) -> Result<f64> {
    let mut total = 0.0;
    let mut n = 0usize;
    for (_, enc) in eval.iter().take(max) {
        let (loss, _) = s.model.loss(enc, false)?;
        total += loss.detach().to_scalar::<f32>()? as f64;
        n += 1;
    }
    Ok(if n == 0 { f64::NAN } else { total / n as f64 })
}
