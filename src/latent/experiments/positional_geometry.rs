//! Paper finding: models with learned absolute position embeddings (GPT-2) lay positions on a helix, while RoPE
//! models (LLaMA) show only a weak, low-dimensional positional pattern. LFM2 has RoPE on its attention layers but its
//! short convs are position-blind, so the question is answered from hidden states: tokens are binned by position
//! (position 0 excluded, see first_position), bin centroids average out content, and PCA over the centroids says how
//! many directions carry position (components to 90 %), how strongly the top PCs track position, and whether the
//! PC1/PC2 angle turns steadily with position (a helix / circle) or not.
use anyhow::Result;

use crate::latent::pca::Pca;
use crate::latent::report::Experiment;
use crate::latent::stats::{mean, pearson};
use crate::latent::Ctx;
use crate::model::Point;

const BINS: usize = 32;

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    ctx.ensure()?;
    let st = ctx.st();
    let rows: Vec<usize> = (0..st.rows()).filter(|&i| st.meta[i].pos > 0).collect();
    let max_pos = rows.iter().map(|&i| st.meta[i].pos).max().unwrap_or(1);
    let bins = BINS.min(max_pos).max(2);
    let bin_of = |pos: usize| ((pos - 1) * bins / max_pos).min(bins - 1);
    let mut e = Experiment::default();
    let (mut pcs90, mut r1, mut r2, mut share, mut turn) = (vec![], vec![], vec![], vec![], vec![]);
    for l in 0..st.layers {
        let Some(m) = st.get(l, Point::BlockOutput) else { continue };
        let mut sums = vec![0f64; bins * st.dim];
        let mut counts = vec![0usize; bins];
        for &i in &rows {
            let b = bin_of(st.meta[i].pos);
            counts[b] += 1;
            for (s, v) in sums[b * st.dim..(b + 1) * st.dim].iter_mut().zip(st.row(m, i)) {
                *s += *v as f64;
            }
        }
        let used: Vec<usize> = (0..bins).filter(|&b| counts[b] > 0).collect();
        if used.len() < 3 {
            continue;
        }
        let centroids: Vec<f32> =
            used.iter().flat_map(|&b| sums[b * st.dim..(b + 1) * st.dim].iter().map(move |s| (s / counts[b] as f64) as f32)).collect();
        let centers: Vec<f64> = used.iter().map(|&b| ((b as f64 + 0.5) * max_pos as f64 / bins as f64) + 1.0).collect();
        let pca = Pca::fit(&centroids, used.len(), st.dim, used.len(), ctx.args.seed)?;
        let scores = pca.transform(&centroids, used.len())?;
        let pc = |c: usize| -> Vec<f64> { (0..used.len()).map(|i| if c < pca.k { scores[i * pca.k + c] as f64 } else { 0.0 }).collect() };
        let (p1, p2) = (pc(0), pc(1));
        // Positional share: between-bin variance (count-weighted) over the total variance of the same tokens.
        let all_mean: Vec<f64> = (0..st.dim).map(|j| rows.iter().map(|&i| st.row(m, i)[j] as f64).sum::<f64>() / rows.len() as f64).collect();
        let total: f64 = rows.iter().map(|&i| st.row(m, i).iter().zip(&all_mean).map(|(v, mu)| (*v as f64 - mu).powi(2)).sum::<f64>()).sum();
        let between: f64 = used
            .iter()
            .enumerate()
            .map(|(k, &b)| counts[b] as f64 * (0..st.dim).map(|j| (centroids[k * st.dim + j] as f64 - all_mean[j]).powi(2)).sum::<f64>())
            .sum();
        let s = if total > 0.0 { between / total } else { f64::NAN };
        let tr = angle_trend(&p1, &p2, &centers);
        let c90 = pca.components_for(0.9) as f64;
        let (c1, c2) = (pearson(&p1, &centers).abs(), pearson(&p2, &centers).abs());
        e.metric(format!("L{l}.pcs90"), c90);
        e.metric(format!("L{l}.pc1_pos_corr"), c1);
        e.metric(format!("L{l}.positional_share"), s);
        pcs90.push((l, c90));
        r1.push((l, c1));
        r2.push((l, c2));
        share.push((l, s));
        turn.push((l, tr));
        if l + 1 == st.layers || l == st.layers / 2 {
            let xy: Vec<[f32; 2]> = p1.iter().zip(&p2).map(|(a, b)| [*a as f32, *b as f32]).collect();
            let labels: Vec<String> = centers.iter().map(|c| format!("L{l} pos~{}", c.round())).collect();
            e.points(&xy, &labels, ctx.args.seed);
        }
    }
    let split = |v: &[(usize, f64)]| -> (Vec<f64>, Vec<f64>) { v.iter().map(|(l, y)| (*l as f64, *y)).unzip() };
    for (name, v) in [("pcs90", &pcs90), ("pc1_pos_corr", &r1), ("pc2_pos_corr", &r2), ("positional_share", &share), ("pc12_angle_trend", &turn)] {
        let (x, y) = split(v);
        e.series(name, &x, &y);
        e.metric(format!("mean_{name}"), mean(&y));
    }
    let (c90, t, sh) = (mean(&split(&pcs90).1), mean(&split(&turn).1), mean(&split(&share).1));
    e.metric("bins", bins as f64);
    e.metric("max_position", max_pos as f64);
    e.finding(format!(
        "position-bin centroids need {c90:.1} PCs for 90% of positional variance; position explains {:.1}% of token variance",
        100.0 * sh
    ));
    e.finding(if t > 0.8 {
        format!("PC1/PC2 angle turns steadily with position (|r|={t:.2}): helix-like, as with learned position embeddings")
    } else {
        format!("no steady PC1/PC2 rotation with position (|r|={t:.2}): low-dimensional positional pattern, as the paper reports for RoPE models")
    });
    Ok(e)
}

/// |pearson| between position and the unwrapped PC1/PC2 angle: near 1 when the centroids wind around a circle in
/// position order (a helix seen from above), near 0 when the angle wanders.
fn angle_trend(p1: &[f64], p2: &[f64], pos: &[f64]) -> f64 {
    let mut angles = Vec::with_capacity(p1.len());
    let mut prev = 0.0f64;
    let mut offset = 0.0;
    for (i, (a, b)) in p1.iter().zip(p2).enumerate() {
        let raw = b.atan2(*a);
        if i > 0 {
            let d = raw - prev;
            if d > std::f64::consts::PI {
                offset -= 2.0 * std::f64::consts::PI;
            } else if d < -std::f64::consts::PI {
                offset += 2.0 * std::f64::consts::PI;
            }
        }
        prev = raw;
        angles.push(raw + offset);
    }
    pearson(&angles, pos).abs()
}
