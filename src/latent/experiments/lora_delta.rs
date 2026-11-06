//! Which modules does fine-tuning actually move? Two views of an adapter: the weight change ‖scale·B·A‖_F relative to
//! ‖W‖_F per module, summed per module kind (attention q/k/v/out, conv in/out, MLP w1/w2/w3, projector), and the
//! hidden-state drift (1 - cosine, base vs tuned) at the operator output and the MLP output of every layer (the
//! paper's two separable sub-spaces, see attn_vs_mlp). `recommended_targets` = the kinds that carry 80 % of the
//! weight change, as PEFT suffixes, meant to replace `target_modules="all-linear"` in
//! backend/ai/finetune/train_lora.py: adapting fewer modules trains faster and generalises with less data.
use std::collections::BTreeMap;

use anyhow::Result;
use serde_json::json;

use super::layers_x;
use crate::latent::capture::capture;
use crate::latent::report::{Experiment, Report};
use crate::latent::stats::{cosine, mean};
use crate::latent::Ctx;
use crate::lora::{read_config, read_pairs};
use crate::model::Point;
use crate::weights;

/// Module kind = the last two path segments, so the attention and conv `out_proj` stay apart.
fn kind_of(module: &str) -> String {
    let parts: Vec<&str> = module.rsplit('.').take(2).collect();
    match parts.as_slice() {
        [last, parent] => format!("{parent}.{last}"),
        [last] => last.to_string(),
        _ => module.to_string(),
    }
}

fn layer_of(module: &str) -> Option<usize> {
    let mut it = module.split('.');
    while let Some(p) = it.next() {
        if p == "layers" {
            return it.next().and_then(|n| n.parse().ok());
        }
    }
    None
}

pub fn run(ctx: &mut Ctx, report: &Report) -> Result<Experiment> {
    let mut e = Experiment::default();
    let Some(dir) = ctx.args.adapter.clone() else {
        e.finding("no --adapter given: train an adapter first, then rerun analysis with --adapter <run>/adapter");
        return Ok(e);
    };
    let cfg = read_config(&dir)?;
    let scale = cfg.scale();
    let mut by_kind: BTreeMap<String, (f64, f64, usize)> = BTreeMap::new();
    let mut rows = Vec::new();
    for (module, (a, b)) in read_pairs(&dir)? {
        let delta = (b.matmul(&a)? * scale)?;
        let dn = delta.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let base = weights::tensor(&ctx.session.files, &format!("{module}.weight"))?;
        let bn = base.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt() as f64;
        let rel = if bn > 0.0 { dn / bn } else { f64::NAN };
        let k = kind_of(&module);
        let slot = by_kind.entry(k.clone()).or_default();
        slot.0 += dn;
        slot.1 += rel;
        slot.2 += 1;
        rows.push(json!({"module": module, "kind": k, "layer": layer_of(&module), "delta_fro": dn, "base_fro": bn, "relative": rel}));
    }
    if rows.is_empty() {
        e.finding(format!("{}: no lora_A/lora_B pairs found", dir.display()));
        return Ok(e);
    }
    let total: f64 = by_kind.values().map(|v| v.0).sum();
    let mut ranked: Vec<(String, f64, f64)> = by_kind.iter().map(|(k, v)| (k.clone(), v.0 / total.max(1e-30), v.1 / v.2 as f64)).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut recommended = Vec::new();
    let mut acc = 0.0;
    for (k, share, rel) in &ranked {
        e.metric(format!("share.{k}"), *share);
        e.metric(format!("mean_relative.{k}"), *rel);
        if acc < 0.8 {
            recommended.push(k.clone());
            acc += share;
        }
    }
    let (x, y): (Vec<f64>, Vec<f64>) = ranked.iter().enumerate().map(|(i, r)| (i as f64, r.1)).unzip();
    e.series(format!("delta_share_by_kind:{}", ranked.iter().map(|r| r.0.as_str()).collect::<Vec<_>>().join(",")), &x, &y);

    // Hidden-state drift: same corpus, same seed (rows line up), adapters off then on.
    let samples = ctx.samples();
    let points = [Point::OperatorOut, Point::MlpOut];
    ctx.session.model.set_lora(false);
    let base = capture(&ctx.session.model, &samples, &points, ctx.args.max_tokens, ctx.args.seed)?;
    ctx.session.model.set_lora(true);
    let tuned = capture(&ctx.session.model, &samples, &points, ctx.args.max_tokens, ctx.args.seed)?;
    ctx.default_lora();
    let mut drift = [vec![], vec![]];
    for l in 0..base.layers {
        for (j, p) in points.iter().enumerate() {
            let (Some(mb), Some(mt)) = (base.get(l, *p), tuned.get(l, *p)) else {
                drift[j].push(f64::NAN);
                continue;
            };
            let d: Vec<f64> = (0..base.rows()).filter(|&i| base.meta[i].pos > 0).map(|i| 1.0 - cosine(base.row(mb, i), tuned.row(mt, i))).collect();
            drift[j].push(mean(&d));
            e.metric(format!("L{l}.drift.{}", p.name()), mean(&d));
        }
    }
    let x = layers_x(base.layers);
    let text = &ctx.session.model.text;
    for (j, p) in points.iter().enumerate() {
        e.series(format!("drift:{}", p.name()), &x, &drift[j]);
        for ty in ["attn", "conv"] {
            let v: Vec<f64> = (0..base.layers).filter(|&l| text.layer_type(l) == ty).map(|l| drift[j][l]).collect();
            e.metric(format!("mean_drift.{}.{ty}", p.name()), mean(&v));
        }
    }
    let (op, mlp) = (mean(&drift[0]), mean(&drift[1]));
    e.metric("modules", rows.len() as f64);
    e.metric("rank", cfg.rank as f64);
    e.metric("alpha", cfg.alpha);
    e.finding(format!("recommended_targets={}", recommended.join(",")));
    e.finding(format!(
        "{} of {} module kinds carry 80% of the weight change; top: {} ({:.0}%)",
        recommended.len(),
        ranked.len(),
        ranked[0].0,
        100.0 * ranked[0].1
    ));
    e.finding(format!("mean hidden drift: operator output {op:.4}, MLP output {mlp:.4} (1 - cosine)"));
    report.side_file("lora_delta_modules.json", &rows)?;
    report.side_file(
        "recommended_targets.json",
        &json!({"recommended_targets": recommended, "ranked": ranked.iter().map(|r| json!({"kind": r.0, "share": r.1, "mean_relative": r.2})).collect::<Vec<_>>(), "adapter": dir}),
    )?;
    Ok(e)
}
