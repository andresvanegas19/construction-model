//! The paper's base view: every (layer, point) reduced to 2-D with PCA and UMAP, plus how concentrated the variance
//! is (components to 90 %, PC1 share). Per-(layer, point) coordinates go to side files layers/L<l>_<point>.json; the
//! report keeps the curves and one UMAP (last layer's block output) as its points.
use anyhow::Result;
use serde_json::json;

use super::{layers_x, reduce, subsample};
use crate::latent::report::{Experiment, Report};
use crate::latent::Ctx;
use crate::model::Point;

pub fn run(ctx: &mut Ctx, report: &Report) -> Result<Experiment> {
    ctx.ensure()?;
    let st = ctx.st();
    let mut e = Experiment::default();
    let layers = st.layers;
    let keep = subsample(st.rows(), ctx.args.umap_points, ctx.args.seed);
    let labels: Vec<String> = keep
        .iter()
        .map(|&i| {
            let m = st.meta[i];
            if m.pos == 0 {
                "pos0".to_string()
            } else {
                ctx.corpus[m.sample].kind.clone()
            }
        })
        .collect();
    for point in Point::ALL {
        let (mut pcs90, mut pc1) = (vec![f64::NAN; layers], vec![f64::NAN; layers]);
        for l in 0..layers {
            let Some(m) = st.get(l, point) else { continue };
            let sub = st.select(m, &keep);
            let r = reduce(&sub, keep.len(), st.dim, ctx.args.seed, true)?;
            pcs90[l] = r.pca.components_for(0.9) as f64;
            pc1[l] = r.pca.explained_ratio().first().copied().unwrap_or(f64::NAN);
            e.metric(format!("L{l}.{}.pcs90", point.name()), pcs90[l]);
            report.side_file(
                &format!("layers/L{l:02}_{}.json", point.name()),
                &json!({
                    "layer": l,
                    "layer_type": ctx.session.model.text.layer_type(l),
                    "point": point.name(),
                    "explained_variance_ratio": r.pca.explained_ratio(),
                    "pca": r.pca_xy,
                    "umap": r.umap_xy,
                    "labels": labels,
                    "positions": keep.iter().map(|&i| st.meta[i].pos).collect::<Vec<_>>(),
                }),
            )?;
            if l + 1 == layers && point == Point::BlockOutput {
                if let Some(xy) = &r.umap_xy {
                    e.points(xy, &labels, ctx.args.seed);
                }
            }
        }
        e.series(format!("pcs90:{}", point.name()), &layers_x(layers), &pcs90);
        e.series(format!("pc1_ratio:{}", point.name()), &layers_x(layers), &pc1);
    }
    e.metric("layers", layers as f64);
    e.metric("tokens", st.rows() as f64);
    e.metric("samples", ctx.corpus.len() as f64);
    e.finding(format!(
        "{} layers x {} points reduced with PCA-50 then UMAP on {} tokens; per-layer coordinates in layers/*.json",
        layers,
        Point::ALL.len(),
        keep.len()
    ));
    Ok(e)
}
