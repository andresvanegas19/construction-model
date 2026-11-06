//! Forward-hook capture of text hidden states. One `States` holds, for every (layer, point) asked for, the same
//! token rows in the same order, plus each row's sample / position / token id. Positions are subsampled per sample
//! to stay inside `--max-tokens` (position 0 is always kept: the first-position experiment needs it), and the subset
//! is chosen from a seed, so a base and a tuned capture of the same corpus line up row for row.
use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use candle_core::{Tensor, D};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::dataset::chat::Encoded;
use crate::model::vl::Lfm2Vl;
use crate::model::{Point, Probe, Tower};

#[derive(Debug, Clone, Copy)]
pub struct TokenMeta {
    pub sample: usize,
    pub pos: usize,
    pub token: u32,
}

#[derive(Default)]
pub struct States {
    pub dim: usize,
    pub layers: usize,
    pub meta: Vec<TokenMeta>,
    /// Row-major (rows x dim) per (layer, point).
    pub data: BTreeMap<(usize, Point), Vec<f32>>,
}

impl States {
    pub fn rows(&self) -> usize {
        self.meta.len()
    }

    pub fn get(&self, layer: usize, point: Point) -> Option<&[f32]> {
        self.data.get(&(layer, point)).map(|v| v.as_slice())
    }

    pub fn row<'a>(&self, m: &'a [f32], i: usize) -> &'a [f32] {
        &m[i * self.dim..(i + 1) * self.dim]
    }

    /// Rows selected by `keep`, as a new (n x dim) matrix.
    pub fn select(&self, m: &[f32], keep: &[usize]) -> Vec<f32> {
        keep.iter().flat_map(|&i| self.row(m, i).iter().copied()).collect()
    }
}

struct Hook<'a> {
    points: &'a BTreeSet<Point>,
    positions: Tensor,
    out: &'a mut States,
}

impl Probe for Hook<'_> {
    fn wants(&self, point: Point) -> bool {
        self.points.contains(&point)
    }

    fn record(&mut self, tower: Tower, layer: usize, point: Point, x: &Tensor) -> candle_core::Result<()> {
        if tower != Tower::Text {
            return Ok(());
        }
        let rows = x.squeeze(0)?.index_select(&self.positions, 0)?.detach();
        self.out.dim = rows.dim(D::Minus1)?;
        self.out.layers = self.out.layers.max(layer + 1);
        self.out.data.entry((layer, point)).or_default().extend(rows.flatten_all()?.to_vec1::<f32>()?);
        Ok(())
    }
}

/// Positions kept for one sample of `len` tokens under a per-sample `budget`.
pub fn pick_positions(len: usize, budget: usize, rng: &mut StdRng) -> Vec<usize> {
    if len <= budget {
        return (0..len).collect();
    }
    let mut rest: Vec<usize> = (1..len).collect();
    rest.shuffle(rng);
    rest.truncate(budget.saturating_sub(1));
    let mut keep = vec![0];
    keep.extend(rest);
    keep.sort_unstable();
    keep
}

pub fn capture(model: &Lfm2Vl, samples: &[&Encoded], points: &[Point], max_tokens: usize, seed: u64) -> Result<States> {
    let points: BTreeSet<Point> = points.iter().copied().collect();
    let mut rng = StdRng::seed_from_u64(seed);
    let budget = (max_tokens / samples.len().max(1)).max(2);
    let mut out = States::default();
    for (s, enc) in samples.iter().enumerate() {
        let keep = pick_positions(enc.ids.len(), budget, &mut rng);
        let positions = Tensor::new(keep.iter().map(|&p| p as u32).collect::<Vec<_>>().as_slice(), &model.device)?;
        let mut hook = Hook { points: &points, positions, out: &mut out };
        model.hidden(enc, Some(&mut hook as &mut dyn Probe))?;
        out.meta.extend(keep.iter().map(|&pos| TokenMeta { sample: s, pos, token: enc.ids[pos] }));
    }
    Ok(out)
}
