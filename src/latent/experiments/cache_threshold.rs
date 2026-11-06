//! The semantic LLM cache (platform llm/response_cache/semantic.rs) answers a prompt from a stored one when their
//! 384-dim embeddings reach cosine `LLM_CACHE_SEMANTIC_MIN` (0.98). That number was picked by hand; this experiment
//! sets it from labeled pairs: the lowest threshold whose precision (pairs judged same that are truly same) stays
//! >= 0.99, its recall (cache hits kept), and the same figures at the current setting. A false "same" returns another
//! request's answer, so precision is the constraint and recall the gain. Input JSONL rows: {"a":[..],"b":[..],"same":b}
//! or {"a_text":"..","b_text":"..","same":b}; text rows are embedded through the same TEI server the backend uses
//! (LLM_CACHE_EMBEDDING_BASE, "query: " prefix, normalized). README: how to export pairs.
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::json;

use super::reduce;
use crate::latent::report::Experiment;
use crate::latent::stats::{cosine, mean};
use crate::latent::Ctx;

const TARGET_PRECISION: f64 = 0.99;

#[derive(Deserialize)]
struct Pair {
    a: Option<Vec<f32>>,
    b: Option<Vec<f32>>,
    a_text: Option<String>,
    b_text: Option<String>,
    same: bool,
}

fn embed(base: &str, text: &str) -> Result<Vec<f32>> {
    let v: Vec<Vec<f32>> = ureq::post(&format!("{}/embed", base.trim_end_matches('/')))
        .send_json(json!({"inputs": format!("query: {text}"), "truncate": false, "normalize": true}))?
        .into_json()?;
    v.into_iter().next().context("empty embedding reply")
}

fn vectors(p: Pair, base: Option<&str>) -> Result<(Vec<f32>, Vec<f32>, bool)> {
    match (p.a, p.b, p.a_text, p.b_text) {
        (Some(a), Some(b), _, _) => Ok((a, b, p.same)),
        (_, _, Some(at), Some(bt)) => {
            let Some(base) = base else { bail!("text pairs need LLM_CACHE_EMBEDDING_BASE (the TEI embedder)") };
            Ok((embed(base, &at)?, embed(base, &bt)?, p.same))
        }
        _ => bail!("a pair needs a/b vectors or a_text/b_text"),
    }
}

/// (precision, recall) when cos >= t means "same".
fn at(cos: &[(f64, bool)], t: f64) -> (f64, f64) {
    let (mut tp, mut fp, mut pos) = (0usize, 0usize, 0usize);
    for &(c, same) in cos {
        pos += usize::from(same);
        if c >= t {
            if same {
                tp += 1;
            } else {
                fp += 1;
            }
        }
    }
    let precision = if tp + fp == 0 { 1.0 } else { tp as f64 / (tp + fp) as f64 };
    (precision, tp as f64 / pos.max(1) as f64)
}

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    let mut e = Experiment::default();
    let Some(path) = ctx.args.pairs.clone() else {
        e.finding("no --pairs file: export labeled pairs as described in training/README.md (Latent geometry, cache_threshold)");
        return Ok(e);
    };
    let base = std::env::var("LLM_CACHE_EMBEDDING_BASE").ok().filter(|s| !s.is_empty());
    let mut pairs = Vec::new();
    for (i, line) in std::fs::read_to_string(&path).with_context(|| path.display().to_string())?.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let p: Pair = serde_json::from_str(line).with_context(|| format!("{}:{}", path.display(), i + 1))?;
        match vectors(p, base.as_deref()) {
            Ok(v) if v.0.len() == v.1.len() && !v.0.is_empty() => pairs.push(v),
            Ok(_) => eprintln!("pair {}: dimension mismatch, skipped", i + 1),
            Err(err) => eprintln!("pair {}: {err:#}", i + 1),
        }
    }
    let cos: Vec<(f64, bool)> = pairs.iter().map(|(a, b, s)| (cosine(a, b), *s)).collect();
    let (same, diff): (Vec<f64>, Vec<f64>) = (
        cos.iter().filter(|c| c.1).map(|c| c.0).collect(),
        cos.iter().filter(|c| !c.1).map(|c| c.0).collect(),
    );
    e.metric("pairs", cos.len() as f64);
    e.metric("same_pairs", same.len() as f64);
    e.metric("different_pairs", diff.len() as f64);
    if same.is_empty() || diff.is_empty() {
        e.finding("need both same and different pairs");
        return Ok(e);
    }
    e.metric("mean_cos.same", mean(&same));
    e.metric("mean_cos.different", mean(&diff));

    let lo = cos.iter().map(|c| c.0).fold(1.0, f64::min).max(-1.0);
    let bins = 40;
    let width = ((1.0 - lo) / bins as f64).max(1e-9);
    let centers: Vec<f64> = (0..bins).map(|b| lo + (b as f64 + 0.5) * width).collect();
    let hist = |v: &[f64]| -> Vec<f64> {
        let mut h = vec![0f64; bins];
        for c in v {
            h[(((c - lo) / width) as usize).min(bins - 1)] += 1.0;
        }
        h
    };
    e.series("cos_histogram:same", &centers, &hist(&same));
    e.series("cos_histogram:different", &centers, &hist(&diff));

    // Candidate thresholds: every observed cosine (the curve only changes there) plus the current setting.
    let mut cands: Vec<f64> = cos.iter().map(|c| c.0).collect();
    cands.push(ctx.args.cache_min);
    cands.sort_by(f64::total_cmp);
    cands.dedup();
    let (mut ts, mut ps, mut rs) = (vec![], vec![], vec![]);
    let mut best: Option<(f64, f64, f64)> = None;
    for &t in &cands {
        let (p, r) = at(&cos, t);
        ts.push(t);
        ps.push(p);
        rs.push(r);
        // The lowest qualifying threshold has the highest recall; precision is not monotone, so keep scanning.
        if p >= TARGET_PRECISION && best.map_or(true, |b| r > b.2) {
            best = Some((t, p, r));
        }
    }
    e.series("precision_by_threshold", &ts, &ps);
    e.series("recall_by_threshold", &ts, &rs);
    let (cp, cr) = at(&cos, ctx.args.cache_min);
    e.metric("current_threshold", ctx.args.cache_min);
    e.metric("current_precision", cp);
    e.metric("current_recall", cr);
    match best {
        Some((t, p, r)) => {
            e.metric("recommended_threshold", t);
            e.metric("recommended_precision", p);
            e.metric("recommended_recall", r);
            e.finding(format!(
                "LLM_CACHE_SEMANTIC_MIN={t:.4} keeps precision {p:.3} with recall {r:.3}; the current {:.2} gives precision {cp:.3}, recall {cr:.3}",
                ctx.args.cache_min
            ));
        }
        None => e.finding(format!("no threshold reaches precision {TARGET_PRECISION}: the embedder cannot separate these pairs safely")),
    }
    if cos.len() < 200 {
        e.finding(format!("only {} pairs: the precision estimate is coarse (one false hit moves it by {:.3})", cos.len(), 1.0 / same.len() as f64));
    }

    let dim = pairs[0].0.len();
    let rows: Vec<&(Vec<f32>, Vec<f32>, bool)> = pairs.iter().filter(|p| p.0.len() == dim).collect();
    let diffs: Vec<f32> = rows.iter().flat_map(|(a, b, _)| a.iter().zip(b).map(|(x, y)| x - y)).collect();
    if rows.len() >= 3 {
        let r = reduce(&diffs, rows.len(), dim, ctx.args.seed, false)?;
        let labels: Vec<String> = rows.iter().map(|p| if p.2 { "same".to_string() } else { "different".to_string() }).collect();
        e.points(&r.pca_xy, &labels, ctx.args.seed);
    }
    Ok(e)
}
