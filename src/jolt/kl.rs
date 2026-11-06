//! Token-level categorical KL divergence, sequence log-probabilities, and group advantage centering for JOLT / JOLT+.
//!
//! Both the teacher alignment penalty $\text{KL}(\pi_{\theta_T}(\cdot \mid s_t, c) \parallel \text{sg}[\pi_{\bar{\theta}_S}(\cdot \mid s_t)])$
//! and the student OPSD distillation objective $\text{KL}(\pi_{\theta_S}(\cdot \mid s_t) \parallel \text{sg}[\pi_{\bar{\theta}_T}(\cdot \mid s_t, c)])$
//! have the same functional form $\text{KL}(\pi_p \parallel \text{sg}[\pi_q])$ with gradients flowing only through $p$.
use anyhow::{bail, Result};
use candle_core::{DType, Tensor, D};

/// Numerically stable log-softmax along the vocabulary dimension (`D::Minus1`) in F32.
pub fn log_softmax_vocab(logits: &Tensor) -> Result<Tensor> {
    let logits = logits.to_dtype(DType::F32)?;
    Ok(candle_nn::ops::log_softmax(&logits, D::Minus1)?)
}

/// Token-averaged categorical KL divergence $\frac{1}{T} \sum_{t=1}^T \sum_v \pi_p(v \mid s_t) (\log \pi_p(v \mid s_t) - \log \text{sg}[\pi_q(v \mid s_t)])$
/// over completion positions $t \in [0, T)$. Gradients flow through `p_logits` only; `q_logits_detached` is detached.
pub fn categorical_kl(p_logits: &Tensor, q_logits_detached: &Tensor) -> Result<Tensor> {
    let t = p_logits.dim(0)?.min(q_logits_detached.dim(0)?);
    if t == 0 {
        bail!("cannot compute categorical_kl on empty completion slice");
    }
    let p = if p_logits.dim(0)? == t { p_logits.clone() } else { p_logits.narrow(0, 0, t)? };
    let q = if q_logits_detached.dim(0)? == t {
        q_logits_detached.detach()
    } else {
        q_logits_detached.narrow(0, 0, t)?.detach()
    };
    let log_p = log_softmax_vocab(&p)?;
    let log_q = log_softmax_vocab(&q)?.detach();
    let prob_p = log_p.exp()?;
    let token_kl = (prob_p * (&log_p - &log_q)?)?.sum(D::Minus1)?;
    Ok(token_kl.mean_all()?)
}

/// Mean token log-probability $\frac{1}{T} \sum_{t=1}^T \log \pi(a_t \mid s_t)$ at `targets` along `[T, V]` `logits`.
pub fn mean_token_logprob(logits: &Tensor, targets: &[u32]) -> Result<Tensor> {
    let t = logits.dim(0)?.min(targets.len());
    if t == 0 {
        bail!("cannot compute mean_token_logprob on empty completion slice");
    }
    let sliced = if logits.dim(0)? == t { logits.clone() } else { logits.narrow(0, 0, t)? };
    let log_probs = log_softmax_vocab(&sliced)?;
    let idx = Tensor::new(&targets[..t], logits.device())?.unsqueeze(1)?;
    let gathered = log_probs.gather(&idx, D::Minus1)?.squeeze(1)?;
    Ok(gathered.mean_all()?)
}

/// Group-centered advantages $\hat{A}_i = R_i - \frac{1}{G}\sum_{j=1}^G R_j$ (GRPO baseline).
pub fn center_advantages(rewards: &[f64]) -> Vec<f64> {
    if rewards.is_empty() {
        return vec![];
    }
    let mean = rewards.iter().copied().sum::<f64>() / rewards.len() as f64;
    rewards.iter().map(|&r| r - mean).collect()
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor, Var};

    use super::{categorical_kl, center_advantages, mean_token_logprob};

    #[test]
    fn kl_is_zero_on_identical_logits_and_positive_when_divergent() {
        let dev = Device::Cpu;
        let p_var = Var::from_slice(&[1.2f32, -0.4, 0.3, 0.1, 0.9, -1.1], (2, 3), &dev).unwrap();
        let q_same = p_var.as_tensor().clone();
        let kl_same = categorical_kl(p_var.as_tensor(), &q_same).unwrap();
        assert!(kl_same.to_scalar::<f32>().unwrap().abs() < 1e-6);

        let q_diff = Tensor::from_slice(&[-0.8f32, 1.5, 0.0, -0.5, -0.2, 2.0], (2, 3), &dev).unwrap();
        let kl_diff = categorical_kl(p_var.as_tensor(), &q_diff).unwrap();
        let val = kl_diff.to_scalar::<f32>().unwrap();
        assert!(val > 0.2, "expected positive KL, got {val}");

        let grads = kl_diff.backward().unwrap();
        let g = grads.get(p_var.as_tensor()).expect("gradient wrt p_logits");
        let g_norm = g.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(g_norm > 1e-4, "expected non-zero gradient wrt p_logits");
    }

    #[test]
    fn mean_logprob_and_centered_advantages_match_definitions() {
        let dev = Device::Cpu;
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0], (2, 2), &dev).unwrap();
        let lp = mean_token_logprob(&logits, &[0, 1]).unwrap().to_scalar::<f32>().unwrap();
        let expected = -(2.0f32.ln());
        assert!((lp - expected).abs() < 1e-6);

        let adv = center_advantages(&[0.2, 0.6, 1.0, 0.2]);
        let sum: f64 = adv.iter().sum();
        assert!(sum.abs() < 1e-12);
        assert!((adv[2] - 0.5).abs() < 1e-12);
        assert!((adv[0] + 0.3).abs() < 1e-12);
    }
}
