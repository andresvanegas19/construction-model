//! Paper method: linear probes on mean-pooled hidden states, layer by layer, with position 0 left out of the mean
//! because its outlier norm (first_position) dominates the average. Both variants are reported, which shows how much
//! the outlier costs. Labels: the prepared rows' `kind` (pipeline stage / image category) or `source`
//! (`--probe-label`). Accuracy far above the majority baseline at a layer means that layer linearly encodes the task
//! kind: the earliest such layer bounds where an adapter or a routing head needs to read.
use anyhow::Result;

use super::{probe_features, reduce};
use crate::latent::report::Experiment;
use crate::latent::stats::{encode_labels, logistic_probe, mean, mean_rows};
use crate::latent::Ctx;
use crate::model::Point;

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    ctx.ensure()?;
    let st = ctx.st();
    let names: Vec<String> =
        ctx.corpus.iter().map(|s| if ctx.args.probe_label == "source" { s.source.clone() } else { s.kind.clone() }).collect();
    let (labels, classes) = encode_labels(&names);
    let mut e = Experiment::default();
    e.metric("samples", names.len() as f64);
    e.metric("classes", classes.len() as f64);
    if classes.len() < 2 {
        e.finding(format!("only one {} label in the corpus ({:?}); use --corpus both or --probe-label", ctx.args.probe_label, classes));
        return Ok(e);
    }
    let samples = ctx.corpus.len();
    let mut best = (f64::NAN, 0usize);
    let mut baseline = f64::NAN;
    for (tag, skip0) in [("skip_pos0", true), ("with_pos0", false)] {
        let (mut x, mut acc) = (vec![], vec![]);
        for l in 0..st.layers {
            let Some(m) = st.get(l, Point::BlockOutput) else { continue };
            let mut pooled = Vec::with_capacity(samples * st.dim);
            for s in 0..samples {
                let rows: Vec<usize> = (0..st.rows()).filter(|&i| st.meta[i].sample == s && !(skip0 && st.meta[i].pos == 0)).collect();
                pooled.extend(if rows.is_empty() { vec![0.0; st.dim] } else { mean_rows(m, st.dim, rows) });
            }
            let r = reduce(&pooled, samples, st.dim, ctx.args.seed, false)?;
            let (feats, k) = probe_features(&r, samples, 16);
            let Some(p) = logistic_probe(&feats, samples, k, &labels, ctx.args.seed + l as u64) else { continue };
            baseline = p.majority_baseline;
            x.push(l as f64);
            acc.push(p.test_accuracy);
            e.metric(format!("L{l}.acc.{tag}"), p.test_accuracy);
            if skip0 && (best.0.is_nan() || p.test_accuracy > best.0) {
                best = (p.test_accuracy, l);
            }
            if skip0 && l + 1 == st.layers {
                e.points(&r.pca_xy, &names, ctx.args.seed);
            }
        }
        e.series(format!("probe_acc:{tag}"), &x, &acc);
        e.metric(format!("mean_acc.{tag}"), mean(&acc));
    }
    e.metric("majority_baseline", baseline);
    e.metric("best_layer", best.1 as f64);
    e.metric("best_acc", best.0);
    e.finding(format!(
        "{} probe over {} classes: best held-out accuracy {:.2} at layer {} (majority baseline {:.2})",
        ctx.args.probe_label,
        classes.len(),
        best.0,
        best.1,
        baseline
    ));
    let (a, b) = (e.summary.get("mean_acc.skip_pos0").copied().unwrap_or(f64::NAN), e.summary.get("mean_acc.with_pos0").copied().unwrap_or(f64::NAN));
    e.finding(format!("mean accuracy {a:.2} without position 0 vs {b:.2} with it"));
    Ok(e)
}
