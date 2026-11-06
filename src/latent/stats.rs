//! Statistics for the experiments: a multinomial logistic-regression probe with a stratified held-out split, and
//! the vector helpers (cosine, norms, means, correlation) the paper's measurements reduce to.
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

pub fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let d = norm(a) * norm(b);
    if d > 0.0 {
        dot(a, b) / d
    } else {
        0.0
    }
}

/// Column means of the listed rows of an (n x dim) matrix.
pub fn mean_rows(m: &[f32], dim: usize, rows: impl IntoIterator<Item = usize>) -> Vec<f32> {
    let mut acc = vec![0f64; dim];
    let mut count = 0usize;
    for i in rows {
        for (a, v) in acc.iter_mut().zip(&m[i * dim..(i + 1) * dim]) {
            *a += *v as f64;
        }
        count += 1;
    }
    acc.into_iter().map(|a| (a / count.max(1) as f64) as f32).collect()
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        f64::NAN
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

pub fn median(v: &[f64]) -> f64 {
    let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if s.is_empty() {
        return f64::NAN;
    }
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = s.len() / 2;
    if s.len() % 2 == 0 {
        (s[m - 1] + s[m]) / 2.0
    } else {
        s[m]
    }
}

pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx).powi(2);
        syy += (b - my).powi(2);
    }
    if sxx > 0.0 && syy > 0.0 {
        sxy / (sxx * syy).sqrt()
    } else {
        0.0
    }
}

/// Stratified split: about `test_frac` of every class goes to test, at least one per class when the class has 2+.
pub fn split(labels: &[usize], test_frac: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let classes = labels.iter().copied().max().map_or(0, |m| m + 1);
    let (mut train, mut test) = (vec![], vec![]);
    for c in 0..classes {
        let mut idx: Vec<usize> = (0..labels.len()).filter(|&i| labels[i] == c).collect();
        idx.shuffle(&mut rng);
        let t = if idx.len() >= 2 { ((idx.len() as f64 * test_frac).round() as usize).clamp(1, idx.len() - 1) } else { 0 };
        test.extend_from_slice(&idx[..t]);
        train.extend_from_slice(&idx[t..]);
    }
    (train, test)
}

pub struct ProbeResult {
    pub test_accuracy: f64,
    pub train_accuracy: f64,
    pub majority_baseline: f64,
    pub test_size: usize,
}

/// Softmax regression by full-batch gradient descent on standardized features (L2 1e-3, 300 steps). The probe is
/// deliberately weak: high held-out accuracy means the classes are linearly separable in that space, the paper's
/// criterion, not that a strong classifier could tell them apart.
pub fn logistic_probe(x: &[f32], n: usize, p: usize, labels: &[usize], seed: u64) -> Option<ProbeResult> {
    let classes = labels.iter().copied().max()? + 1;
    if classes < 2 || n < 4 {
        return None;
    }
    let (train, test) = split(labels, 0.3, seed);
    if test.is_empty() || train.is_empty() {
        return None;
    }
    let mu: Vec<f64> = (0..p).map(|j| train.iter().map(|&i| x[i * p + j] as f64).sum::<f64>() / train.len() as f64).collect();
    let sd: Vec<f64> = (0..p)
        .map(|j| (train.iter().map(|&i| (x[i * p + j] as f64 - mu[j]).powi(2)).sum::<f64>() / train.len() as f64).sqrt().max(1e-8))
        .collect();
    let feat = |i: usize, j: usize| (x[i * p + j] as f64 - mu[j]) / sd[j];
    let mut w = vec![0f64; classes * (p + 1)];
    let lr = 0.5;
    for _ in 0..300 {
        let mut grad = vec![0f64; w.len()];
        for &i in &train {
            let probs = softmax(&w, i, p, classes, &feat);
            for c in 0..classes {
                let err = probs[c] - if labels[i] == c { 1.0 } else { 0.0 };
                for j in 0..p {
                    grad[c * (p + 1) + j] += err * feat(i, j);
                }
                grad[c * (p + 1) + p] += err;
            }
        }
        for (wi, g) in w.iter_mut().zip(&grad) {
            *wi -= lr * (g / train.len() as f64 + 1e-3 * *wi);
        }
    }
    let acc = |set: &[usize]| {
        let ok = set
            .iter()
            .filter(|&&i| {
                let pr = softmax(&w, i, p, classes, &feat);
                let best = (0..classes).max_by(|&a, &b| pr[a].partial_cmp(&pr[b]).unwrap()).unwrap_or(0);
                best == labels[i]
            })
            .count();
        ok as f64 / set.len().max(1) as f64
    };
    let mut counts = vec![0usize; classes];
    for &i in &test {
        counts[labels[i]] += 1;
    }
    Some(ProbeResult {
        test_accuracy: acc(&test),
        train_accuracy: acc(&train),
        majority_baseline: *counts.iter().max().unwrap_or(&0) as f64 / test.len() as f64,
        test_size: test.len(),
    })
}

fn softmax(w: &[f64], i: usize, p: usize, classes: usize, feat: &impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let logits: Vec<f64> = (0..classes).map(|c| (0..p).map(|j| w[c * (p + 1) + j] * feat(i, j)).sum::<f64>() + w[c * (p + 1) + p]).collect();
    let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = logits.iter().map(|l| (l - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Maps string labels to dense ids (sorted names, so ids are stable across runs).
pub fn encode_labels(names: &[String]) -> (Vec<usize>, Vec<String>) {
    let mut uniq: Vec<String> = names.to_vec();
    uniq.sort();
    uniq.dedup();
    let ids = names.iter().map(|n| uniq.binary_search(n).unwrap_or(0)).collect();
    (ids, uniq)
}
