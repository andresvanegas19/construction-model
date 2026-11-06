//! Paper finding: the first position carries hidden states with a far larger norm than every other position (an
//! attention sink). Measured per layer on the block output as norm(pos 0) / median norm(other positions), once with
//! the BOS token and once with the BOS dropped, which separates "the BOS token is special" from "whatever sits at
//! position 0 is special". Also counted: later tokens whose norm reaches half of position 0's (secondary sinks such
//! as `<|im_start|>` or image markers), since those also distort mean pooling and probe features.
use std::collections::BTreeMap;

use anyhow::Result;

use super::layers_x;
use crate::latent::capture::{capture, States};
use crate::latent::report::Experiment;
use crate::latent::stats::{mean, median, norm};
use crate::latent::Ctx;
use crate::model::Point;

struct Curves {
    ratio: Vec<f64>,
    pos0: Vec<f64>,
    others: Vec<f64>,
    high_frac: Vec<f64>,
    high_tokens: BTreeMap<u32, usize>,
}

fn curves(st: &States) -> Curves {
    let mut c = Curves { ratio: vec![], pos0: vec![], others: vec![], high_frac: vec![], high_tokens: BTreeMap::new() };
    for l in 0..st.layers {
        let Some(m) = st.get(l, Point::BlockOutput) else {
            for v in [&mut c.ratio, &mut c.pos0, &mut c.others, &mut c.high_frac] {
                v.push(f64::NAN);
            }
            continue;
        };
        let norms: Vec<f64> = (0..st.rows()).map(|i| norm(st.row(m, i))).collect();
        let mut first = BTreeMap::new();
        for (i, t) in st.meta.iter().enumerate() {
            if t.pos == 0 {
                first.insert(t.sample, norms[i]);
            }
        }
        let p0: Vec<f64> = first.values().copied().collect();
        let rest: Vec<f64> = (0..st.rows()).filter(|&i| st.meta[i].pos > 0).map(|i| norms[i]).collect();
        let (n0, nr) = (mean(&p0), median(&rest));
        c.pos0.push(n0);
        c.others.push(nr);
        c.ratio.push(n0 / nr);
        let mut high = 0usize;
        for i in (0..st.rows()).filter(|&i| st.meta[i].pos > 0) {
            if let Some(&f) = first.get(&st.meta[i].sample) {
                if norms[i] > 0.5 * f {
                    high += 1;
                    if l + 1 == st.layers / 2 {
                        *c.high_tokens.entry(st.meta[i].token).or_default() += 1;
                    }
                }
            }
        }
        c.high_frac.push(high as f64 / rest.len().max(1) as f64);
    }
    c
}

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    ctx.ensure()?;
    let with = curves(ctx.st());
    let stripped: Vec<_> = ctx.corpus.iter().filter(|s| s.enc.ids.len() > 2).map(|s| s.enc.without_bos()).collect();
    let refs: Vec<_> = stripped.iter().collect();
    let st = capture(&ctx.session.model, &refs, &[Point::BlockOutput], ctx.args.max_tokens, ctx.args.seed)?;
    let without = curves(&st);
    let layers = ctx.st().layers;
    let x = layers_x(layers);
    let mut e = Experiment::default();
    for (tag, c) in [("bos", &with), ("no_bos", &without)] {
        e.series(format!("pos0_ratio:{tag}"), &x, &c.ratio);
        e.series(format!("pos0_norm:{tag}"), &x, &c.pos0);
        e.series(format!("median_norm_others:{tag}"), &x, &c.others);
        e.series(format!("high_norm_fraction:{tag}"), &x, &c.high_frac);
        let peak = c.ratio.iter().copied().filter(|v| v.is_finite()).fold(f64::NAN, f64::max);
        let at = c.ratio.iter().position(|&v| v == peak).unwrap_or(0);
        e.metric(format!("max_ratio.{tag}"), peak);
        e.metric(format!("max_ratio_layer.{tag}"), at as f64);
        e.metric(format!("mean_ratio.{tag}"), mean(&c.ratio));
        e.metric(format!("mean_high_norm_fraction.{tag}"), mean(&c.high_frac));
        for (l, r) in c.ratio.iter().enumerate() {
            e.metric(format!("L{l}.ratio.{tag}"), *r);
        }
    }
    let (wb, nb) = (mean(&with.ratio), mean(&without.ratio));
    e.finding(format!("mean norm ratio pos0/median(others): {wb:.1}x with BOS, {nb:.1}x without BOS"));
    e.finding(if nb > 0.5 * wb {
        "the outlier follows position 0, not the BOS token: drop position 0 (not just BOS) before pooling or probing".to_string()
    } else {
        "the outlier is tied to the BOS token: dropping BOS removes it".to_string()
    });
    let mut top: Vec<(u32, usize)> = with.high_tokens.into_iter().collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));
    if !top.is_empty() {
        let names: Vec<String> = top
            .iter()
            .take(5)
            .map(|(t, n)| format!("{:?} x{n}", ctx.session.tokenizer.id_to_token(*t).unwrap_or_else(|| t.to_string())))
            .collect();
        e.finding(format!("tokens after position 0 with at least half its norm (middle layer): {}", names.join(", ")));
    }
    Ok(e)
}
