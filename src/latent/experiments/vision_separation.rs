//! Does the vision tower itself tell floor plans from other images? Each image goes through the SigLIP2 tower
//! (its last tile: the thumbnail when tiled, else the single tile); every layer's block output is mean-pooled and a
//! held-out linear probe separates plan / other per layer. A tower that separates them early and near perfectly means
//! backend vision/check.rs could gate uploads on a cheap probe over these features instead of a model call plus a
//! Jev decision. Without real non-plan images (`--other-dir`, default data/images/nonplan) a synthetic control set is
//! drawn (gradients, noise, shapes): it only proves the pipeline runs, and the finding says so.
use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{DType, Tensor};
use image::{Rgb, RgbImage};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::{probe_features, reduce, subsample};
use crate::dataset::image::{load, process, Processed};
use crate::latent::report::Experiment;
use crate::latent::stats::{logistic_probe, mean};
use crate::latent::Ctx;
use crate::model::{Point, Probe, Tower};

struct Pool {
    layers: Vec<Vec<f32>>,
}

impl Probe for Pool {
    fn wants(&self, point: Point) -> bool {
        point == Point::BlockOutput
    }

    fn record(&mut self, tower: Tower, layer: usize, _point: Point, x: &Tensor) -> candle_core::Result<()> {
        if tower == Tower::Vision {
            if self.layers.len() <= layer {
                self.layers.resize(layer + 1, vec![]);
            }
            self.layers[layer] = x.to_dtype(DType::F32)?.mean(0)?.to_vec1::<f32>()?;
        }
        Ok(())
    }
}

fn list_images(dir: &Path, max: usize, seed: u64) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    files.retain(|p| {
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        // `_raw` files are the un-annotated twins of another image in the set; counting both would leak across the split.
        matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "webp") && !p.to_string_lossy().contains("_raw.")
    });
    files.sort();
    let keep = subsample(files.len(), max, seed);
    keep.into_iter().map(|i| files[i].clone()).collect()
}

/// Non-plan stand-ins: smooth colour fields with noise and filled discs, i.e. photo-like statistics without lines.
fn synthetic(n: usize, seed: u64) -> Vec<RgbImage> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let (w, h) = (rng.gen_range(384..768u32), rng.gen_range(384..768u32));
            let c0: [f32; 3] = [rng.gen(), rng.gen(), rng.gen()];
            let c1: [f32; 3] = [rng.gen(), rng.gen(), rng.gen()];
            let discs: Vec<(f32, f32, f32, [u8; 3])> =
                (0..rng.gen_range(2..8)).map(|_| (rng.gen::<f32>() * w as f32, rng.gen::<f32>() * h as f32, rng.gen_range(20.0..160.0), rng.gen())).collect();
            let mut img = RgbImage::new(w, h);
            for (x, y, px) in img.enumerate_pixels_mut() {
                let t = (x as f32 / w as f32 + y as f32 / h as f32) / 2.0;
                let mut c = [0u8; 3];
                for k in 0..3 {
                    let v = c0[k] * (1.0 - t) + c1[k] * t + (rng.gen::<f32>() - 0.5) * 0.15;
                    c[k] = (v.clamp(0.0, 1.0) * 255.0) as u8;
                }
                for (cx, cy, r, col) in &discs {
                    if (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2) < r * r {
                        c = *col;
                    }
                }
                *px = Rgb(c);
            }
            img
        })
        .collect()
}

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    let a = ctx.args;
    let root = &a.m.root;
    let plan_dir = a.plan_dir.clone().unwrap_or_else(|| root.join("data/images/floorplans/images"));
    let other_dir = a.other_dir.clone().unwrap_or_else(|| root.join("data/images/nonplan"));
    let settings = &ctx.session.model.images;
    let mut inputs: Vec<(Processed, String)> = Vec::new();
    for p in list_images(&plan_dir, a.max_images, a.seed) {
        match load(&p, settings) {
            Ok(pr) => inputs.push((pr, "plan".into())),
            Err(err) => eprintln!("skip {}: {err:#}", p.display()),
        }
    }
    let plans = inputs.len();
    let mut other: Vec<Processed> = list_images(&other_dir, a.max_images, a.seed).iter().filter_map(|p| load(p, settings).ok()).collect();
    let synthetic_used = other.is_empty();
    if synthetic_used {
        other = synthetic(plans.max(8), a.seed).iter().map(|img| process(img, settings)).collect();
    }
    inputs.extend(other.into_iter().map(|p| (p, "other".to_string())));
    let mut e = Experiment::default();
    e.metric("plan_images", plans as f64);
    e.metric("other_images", (inputs.len() - plans) as f64);
    if plans < 2 || inputs.len() - plans < 2 {
        e.finding(format!("need 2+ images per class (plans in {}, others in {})", plan_dir.display(), other_dir.display()));
        return Ok(e);
    }
    let mut pooled: Vec<Vec<Vec<f32>>> = Vec::new();
    for (pr, _) in &inputs {
        let Some(tile) = pr.tiles.last() else { continue };
        let mut pool = Pool { layers: vec![] };
        ctx.session.model.tile_states(tile, Some(&mut pool as &mut dyn Probe))?;
        pooled.push(pool.layers);
    }
    let labels: Vec<usize> = inputs.iter().map(|(_, l)| usize::from(l == "plan")).collect();
    let names: Vec<String> = inputs.iter().map(|(_, l)| l.clone()).collect();
    let n = pooled.len();
    let layers = pooled.iter().map(|v| v.len()).min().unwrap_or(0);
    let dim = pooled.first().and_then(|v| v.first()).map_or(0, |v| v.len());
    let (mut xs, mut acc) = (vec![], vec![]);
    let mut first_perfect = None;
    for l in 0..layers {
        let m: Vec<f32> = pooled.iter().flat_map(|v| v[l].iter().copied()).collect();
        let last = l + 1 == layers;
        let r = reduce(&m, n, dim, a.seed, last)?;
        let (feats, k) = probe_features(&r, n, 8);
        let Some(p) = logistic_probe(&feats, n, k, &labels, a.seed + l as u64) else { continue };
        xs.push(l as f64);
        acc.push(p.test_accuracy);
        e.metric(format!("L{l}.probe_acc"), p.test_accuracy);
        if first_perfect.is_none() && p.test_accuracy >= 0.999 {
            first_perfect = Some(l);
        }
        if last {
            e.metric("majority_baseline", p.majority_baseline);
            e.points(r.umap_xy.as_deref().unwrap_or(&r.pca_xy[..]), &names, a.seed);
        }
    }
    e.series("probe_acc", &xs, &acc);
    e.metric("mean_probe_acc", mean(&acc));
    e.metric("first_perfect_layer", first_perfect.map_or(f64::NAN, |l| l as f64));
    e.metric("synthetic_control", if synthetic_used { 1.0 } else { 0.0 });
    e.finding(match first_perfect {
        Some(l) => format!("plan vs other separates perfectly from vision layer {l} of {layers} (held-out linear probe)"),
        None => format!("no vision layer separates plan vs other perfectly; mean probe accuracy {:.2}", mean(&acc)),
    });
    if synthetic_used {
        e.finding(format!(
            "no real non-plan images in {}: the 'other' class is synthetic, so this only checks the pipeline; add photos / renders / documents there for a real answer",
            other_dir.display()
        ));
    }
    Ok(e)
}
