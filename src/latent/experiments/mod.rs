//! The experiments, one module each, plus the reductions they share. Every experiment returns one
//! `report::Experiment`; `run` dispatches by name.
pub mod attn_vs_mlp;
pub mod cache_threshold;
pub mod first_position;
pub mod layerwise;
pub mod lora_delta;
pub mod positional_geometry;
pub mod probes;
pub mod repeated_token;
pub mod vision_separation;

use anyhow::{bail, Result};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::pca::{take_cols, to_xy, Pca};
use super::report::{Experiment, Report};
use super::umap::{self, UmapOpts};
use super::Ctx;

pub const NAMES: [&str; 9] = [
    "layerwise",
    "attn_vs_mlp",
    "first_position",
    "positional_geometry",
    "repeated_token",
    "probes",
    "lora_delta",
    "vision_separation",
    "cache_threshold",
];

pub fn run(name: &str, ctx: &mut Ctx, report: &Report) -> Result<Experiment> {
    match name {
        "layerwise" => layerwise::run(ctx, report),
        "attn_vs_mlp" => attn_vs_mlp::run(ctx),
        "first_position" => first_position::run(ctx),
        "positional_geometry" => positional_geometry::run(ctx),
        "repeated_token" => repeated_token::run(ctx),
        "probes" => probes::run(ctx),
        "lora_delta" => lora_delta::run(ctx, report),
        "vision_separation" => vision_separation::run(ctx),
        "cache_threshold" => cache_threshold::run(ctx),
        other => bail!("unknown experiment {other}; known: {}", NAMES.join(", ")),
    }
}

pub struct Reduced {
    pub pca: Pca,
    /// (n x k) PCA scores, k = pca.k.
    pub scores: Vec<f32>,
    pub pca_xy: Vec<[f32; 2]>,
    pub umap_xy: Option<Vec<[f32; 2]>>,
}

/// PCA to (up to) 50 components, its 2-D view, and optionally UMAP on the 50-D scores (the usual UMAP-after-PCA
/// recipe; the paper reduces the same way).
pub fn reduce(m: &[f32], n: usize, dim: usize, seed: u64, with_umap: bool) -> Result<Reduced> {
    let pca = Pca::fit(m, n, dim, 50, seed)?;
    let scores = pca.transform(m, n)?;
    let pca_xy = to_xy(&scores, n, pca.k);
    let umap_xy = with_umap.then(|| umap::embed(&scores, n, pca.k, &pca_xy, &UmapOpts { seed, ..Default::default() }));
    Ok(Reduced { pca, scores, pca_xy, umap_xy })
}

/// Probe features: the first `k` PCA scores (a 2048-D logistic regression on a few hundred rows would just memorise).
pub fn probe_features(r: &Reduced, n: usize, k: usize) -> (Vec<f32>, usize) {
    let k = k.min(r.pca.k);
    (take_cols(&r.scores, n, r.pca.k, k), k)
}

/// A seeded subset of 0..n of at most `max` indices, in increasing order.
pub fn subsample(n: usize, max: usize, seed: u64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    if n > max {
        idx.shuffle(&mut StdRng::seed_from_u64(seed));
        idx.truncate(max);
        idx.sort_unstable();
    }
    idx
}

pub fn layers_x(n: usize) -> Vec<f64> {
    (0..n).map(|l| l as f64).collect()
}
