//! Paper probe: a sequence of one token repeated N times. With identical content at every position, any difference
//! between positions is the model's positional machinery and the first-position sink. Reported per layer: cosine of
//! each position's block output to position 0's, and to the mean BOS direction of the corpus (the sink state). Run
//! with and without a leading BOS: without it, a high cosine of the first repeated token to the BOS direction means
//! the sink forms at whatever token comes first.
use anyhow::{bail, Result};

use super::layers_x;
use crate::dataset::chat::Encoded;
use crate::latent::capture::{capture, States};
use crate::latent::report::Experiment;
use crate::latent::stats::{cosine, mean, mean_rows};
use crate::latent::Ctx;
use crate::model::vl::IGNORE;
use crate::model::Point;

pub fn run(ctx: &mut Ctx) -> Result<Experiment> {
    ctx.ensure()?;
    let (bos, tok) = {
        let enc = ctx.encoder()?;
        let ids = enc.text(&ctx.args.repeat_token)?;
        let Some(&tok) = ids.first() else { bail!("--repeat-token {:?} tokenizes to nothing", ctx.args.repeat_token) };
        (enc.bos(), tok)
    };
    let n = ctx.args.repeat_len.max(2);
    let mut with_bos = vec![bos];
    with_bos.extend(std::iter::repeat(tok).take(n));
    let no_bos = vec![tok; n];
    let model = &ctx.session.model;
    let run_one = |ids: Vec<u32>| -> Result<States> {
        let labels = vec![IGNORE; ids.len()];
        let len = ids.len();
        let enc = Encoded { ids, labels, tiles: vec![] };
        capture(model, &[&enc], &[Point::BlockOutput], len, 0)
    };
    let (sb, sn) = (run_one(with_bos)?, run_one(no_bos)?);
    let corpus = ctx.st();
    let layers = corpus.layers;
    let bos_dir: Vec<Option<Vec<f32>>> = (0..layers)
        .map(|l| {
            let rows: Vec<usize> = (0..corpus.rows()).filter(|&i| corpus.meta[i].pos == 0).collect();
            corpus.get(l, Point::BlockOutput).filter(|_| !rows.is_empty()).map(|m| mean_rows(m, corpus.dim, rows))
        })
        .collect();
    let mut e = Experiment::default();
    let token_name = ctx.session.tokenizer.id_to_token(tok).unwrap_or_else(|| tok.to_string());
    for (tag, st, first_rep) in [("bos", &sb, 1usize), ("no_bos", &sn, 0usize)] {
        let (mut to_first, mut to_bos, mut first_to_bos) = (vec![], vec![], vec![]);
        for l in 0..layers {
            let Some(m) = st.get(l, Point::BlockOutput) else { continue };
            let p0 = st.row(m, 0);
            let rest: Vec<f64> = (1..st.rows()).map(|i| cosine(st.row(m, i), p0)).collect();
            to_first.push(mean(&rest));
            let dir = bos_dir.get(l).and_then(|d| d.as_deref());
            let later: Vec<f64> = (first_rep + 1..st.rows()).map(|i| dir.map_or(f64::NAN, |d| cosine(st.row(m, i), d))).collect();
            to_bos.push(mean(&later));
            first_to_bos.push(dir.map_or(f64::NAN, |d| cosine(st.row(m, first_rep), d)));
            if l + 1 == layers {
                let x: Vec<f64> = (1..st.rows()).map(|p| p as f64).collect();
                e.series(format!("last_layer_cos_to_pos0_by_position:{tag}"), &x, &rest);
            }
        }
        let x = layers_x(to_first.len());
        e.series(format!("mean_cos_to_pos0:{tag}"), &x, &to_first);
        e.series(format!("mean_cos_to_bos_direction:{tag}"), &x, &to_bos);
        e.series(format!("first_repeat_cos_to_bos_direction:{tag}"), &x, &first_to_bos);
        e.metric(format!("mean_cos_to_pos0.{tag}"), mean(&to_first));
        e.metric(format!("mean_cos_to_bos_direction.{tag}"), mean(&to_bos));
        e.metric(format!("first_repeat_cos_to_bos_direction.{tag}"), mean(&first_to_bos));
    }
    e.metric("repeat_len", n as f64);
    e.metric("token_id", tok as f64);
    let (fb, fn_) = (
        e.summary.get("first_repeat_cos_to_bos_direction.bos").copied().unwrap_or(f64::NAN),
        e.summary.get("first_repeat_cos_to_bos_direction.no_bos").copied().unwrap_or(f64::NAN),
    );
    e.finding(format!("token {token_name:?} repeated {n}x: first repeated token vs BOS sink direction, cos {fb:.2} after BOS, {fn_:.2} with no BOS"));
    if fn_ > 0.5 && fn_ > fb + 0.2 {
        e.finding("without BOS the first repeated token takes on the sink state: position 0 is the sink, whatever the token");
    }
    let later = e.summary.get("mean_cos_to_pos0.bos").copied().unwrap_or(f64::NAN);
    e.finding(format!("with BOS, later repeats have mean cos {later:.2} to the BOS state (low = BOS is kept apart from content)"));
    Ok(e)
}
