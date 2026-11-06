//! Paper finding: attention outputs and MLP outputs occupy separable regions in intermediate layers. LFM2 is
//! hybrid, so the "operator" is attention on 6 layers and a gated short conv on 10: the per-layer curves are split by
//! operator type, which tells whether the conv blocks write into a subspace distinct from the MLPs as attention does
//! (new relative to the paper, which only studies attention-only Transformers). Position 0 is excluded: its outlier
//! norm would make the two clouds trivially separable along one direction.
use anyhow::Result;

use super::{probe_features, reduce, subsample};
use crate::latent::report::Experiment;
use crate::latent::stats::{cosine, logistic_probe, mean, mean_rows};
use crate::latent::Ctx;
use crate::model::Point;

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    ctx.ensure()?;
    let st = ctx.st();
    let seed = ctx.args.seed;
    let text = &ctx.session.model.text;
    let rows: Vec<usize> = (0..st.rows()).filter(|&i| st.meta[i].pos > 0).collect();
    let keep: Vec<usize> = subsample(rows.len(), 1000, seed).into_iter().map(|i| rows[i]).collect();
    let n = keep.len();
    let mut e = Experiment::default();
    let (mut xs, mut acc, mut cos) = ([vec![], vec![]], [vec![], vec![]], [vec![], vec![]]);
    let last_of = |t: &str| (0..st.layers).rev().find(|&l| text.layer_type(l) == t);
    let shown = [last_of("attn"), last_of("conv")];
    for l in 0..st.layers {
        let (Some(op), Some(mlp)) = (st.get(l, Point::OperatorOut), st.get(l, Point::MlpOut)) else { continue };
        let mut both = st.select(op, &keep);
        both.extend(st.select(mlp, &keep));
        let labels: Vec<usize> = (0..2 * n).map(|i| usize::from(i >= n)).collect();
        let show = shown.contains(&Some(l));
        let r = reduce(&both, 2 * n, st.dim, seed, show)?;
        let (feats, k) = probe_features(&r, 2 * n, 32);
        let a = logistic_probe(&feats, 2 * n, k, &labels, seed + l as u64).map_or(f64::NAN, |p| p.test_accuracy);
        let c = cosine(&mean_rows(&both, st.dim, 0..n), &mean_rows(&both, st.dim, n..2 * n));
        let ty = text.layer_type(l);
        let t = usize::from(ty == "attn");
        xs[t].push(l as f64);
        acc[t].push(a);
        cos[t].push(c);
        e.metric(format!("L{l}.{ty}.probe_acc"), a);
        e.metric(format!("L{l}.{ty}.centroid_cos"), c);
        if let (true, Some(xy)) = (show, &r.umap_xy) {
            // 1000 per shown layer, so both the attention and the conv layer fit under the 2000-point cap.
            let pick = subsample(2 * n, 1000, seed);
            let xy: Vec<[f32; 2]> = pick.iter().map(|&i| xy[i]).collect();
            let names: Vec<String> = pick.iter().map(|&i| format!("L{l} {}", if i < n { format!("{ty}_out") } else { "mlp_out".into() })).collect();
            e.points(&xy, &names, seed + l as u64);
        }
    }
    for (t, name) in [(0, "conv"), (1, "attn")] {
        e.series(format!("probe_acc:{name}"), &xs[t], &acc[t]);
        e.series(format!("centroid_cos:{name}"), &xs[t], &cos[t]);
        e.metric(format!("mean_probe_acc.{name}"), mean(&acc[t]));
        e.metric(format!("mean_centroid_cos.{name}"), mean(&cos[t]));
    }
    let (ca, aa) = (mean(&acc[0]), mean(&acc[1]));
    e.finding(format!(
        "operator vs MLP output, held-out linear-probe accuracy: conv layers {ca:.3}, attention layers {aa:.3} (0.5 = inseparable)"
    ));
    if ca.is_finite() && aa.is_finite() {
        e.finding(if (ca - aa).abs() < 0.05 {
            "conv and attention blocks separate from the MLP about equally: the paper's attention/MLP split carries over to the short-conv operator".to_string()
        } else if ca > aa {
            "conv outputs are more distinct from MLP outputs than attention outputs are".to_string()
        } else {
            "attention outputs are more distinct from MLP outputs than conv outputs are".to_string()
        });
    }
    Ok(e)
}
