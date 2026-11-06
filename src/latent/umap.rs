//! A small, deterministic UMAP (McInnes et al. 2018) for <= ~5k points: exact kNN, per-point smooth-kNN bandwidths,
//! fuzzy-union symmetrisation, and the negative-sampling SGD layout with the curve a = 1.577, b = 0.895 (the
//! min_dist = 0.1 defaults of umap-learn). Exact kNN instead of NN-descent and a fixed edge-sampling rule keep it
//! short and bit-reproducible for a seed, which matters more here than speed.
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub struct UmapOpts {
    pub neighbors: usize,
    pub epochs: usize,
    pub negatives: usize,
    pub seed: u64,
}

impl Default for UmapOpts {
    fn default() -> Self {
        Self { neighbors: 15, epochs: 200, negatives: 5, seed: 7 }
    }
}

const A: f32 = 1.577;
const B: f32 = 0.895;

/// `data` (n x p) row-major, typically PCA-50 scores; `init` (n x 2) is the starting layout (PCA 2-D).
pub fn embed(data: &[f32], n: usize, p: usize, init: &[[f32; 2]], o: &UmapOpts) -> Vec<[f32; 2]> {
    if n < 4 {
        return init.to_vec();
    }
    let k = o.neighbors.min(n - 1);
    let graph = fuzzy_graph(data, n, p, k);
    layout(n, &graph, init, o)
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn fuzzy_graph(data: &[f32], n: usize, p: usize, k: usize) -> Vec<(usize, usize, f32)> {
    let row = |i: usize| &data[i * p..(i + 1) * p];
    let target = (k as f32).log2();
    let mut directed: HashMap<(usize, usize), f32> = HashMap::new();
    for i in 0..n {
        let mut d: Vec<(f32, usize)> = (0..n).filter(|&j| j != i).map(|j| (dist2(row(i), row(j)).sqrt(), j)).collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        d.truncate(k);
        let rho = d.iter().map(|x| x.0).find(|&x| x > 0.0).unwrap_or(0.0);
        let mass = |sigma: f32| d.iter().map(|x| (-((x.0 - rho).max(0.0)) / sigma).exp()).sum::<f32>();
        let (mut lo, mut hi, mut sigma) = (0.0f32, f32::INFINITY, 1.0f32);
        for _ in 0..64 {
            let m = mass(sigma);
            if (m - target).abs() < 1e-5 {
                break;
            }
            if m > target {
                hi = sigma;
                sigma = (lo + hi) / 2.0;
            } else {
                lo = sigma;
                sigma = if hi.is_infinite() { sigma * 2.0 } else { (lo + hi) / 2.0 };
            }
        }
        for &(dist, j) in &d {
            directed.insert((i, j), (-((dist - rho).max(0.0)) / sigma.max(1e-6)).exp());
        }
    }
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    for (&(i, j), &w) in &directed {
        if i < j || !directed.contains_key(&(j, i)) {
            let back = directed.get(&(j, i)).copied().unwrap_or(0.0);
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            edges.push((a, b, w + back - w * back));
        }
    }
    // HashMap order is random per process; sorting restores determinism for the seeded SGD.
    edges.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    edges
}

fn layout(n: usize, edges: &[(usize, usize, f32)], init: &[[f32; 2]], o: &UmapOpts) -> Vec<[f32; 2]> {
    let mut y = scaled(init);
    let mut rng = StdRng::seed_from_u64(o.seed);
    let w_max = edges.iter().map(|e| e.2).fold(0.0f32, f32::max).max(1e-12);
    for epoch in 0..o.epochs {
        let alpha = 1.0 - epoch as f32 / o.epochs as f32;
        for &(i, j, w) in edges {
            // An edge of weight w is sampled in a fraction w / w_max of the epochs, like umap-learn's
            // epochs_per_sample, but drawn from the seeded RNG instead of a per-edge schedule.
            if rng.gen::<f32>() > w / w_max {
                continue;
            }
            let (yi, yj) = (y[i], y[j]);
            let d2 = (yi[0] - yj[0]).powi(2) + (yi[1] - yj[1]).powi(2);
            let coef = if d2 > 0.0 { -2.0 * A * B * d2.powf(B - 1.0) / (1.0 + A * d2.powf(B)) } else { 0.0 };
            for c in 0..2 {
                let g = (coef * (yi[c] - yj[c])).clamp(-4.0, 4.0) * alpha;
                y[i][c] += g;
                y[j][c] -= g;
            }
            for _ in 0..o.negatives {
                let m = rng.gen_range(0..n);
                if m == i {
                    continue;
                }
                let (yi, ym) = (y[i], y[m]);
                let d2 = (yi[0] - ym[0]).powi(2) + (yi[1] - ym[1]).powi(2);
                let coef = 2.0 * B / ((0.001 + d2) * (1.0 + A * d2.powf(B)));
                for c in 0..2 {
                    let g = if coef > 0.0 { (coef * (yi[c] - ym[c])).clamp(-4.0, 4.0) } else { 4.0 };
                    y[i][c] += g * alpha;
                }
            }
        }
    }
    y
}

/// The PCA start rescaled to a [-10, 10] box, the range umap-learn's spectral init is scaled to.
fn scaled(init: &[[f32; 2]]) -> Vec<[f32; 2]> {
    let mut out = init.to_vec();
    for c in 0..2 {
        let (lo, hi) = out.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| (lo.min(p[c]), hi.max(p[c])));
        let span = (hi - lo).max(1e-6);
        for p in &mut out {
            p[c] = (p[c] - lo) / span * 20.0 - 10.0;
        }
    }
    out
}
