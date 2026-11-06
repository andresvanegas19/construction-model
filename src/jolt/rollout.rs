//! On-policy group rollout generation and cross-context prompt construction for JOLT / JOLT+.
//!
//! For each row $x$ with privileged supervision $c$:
//! 1. Builds the unprivileged student prompt $x$ and the privileged teacher prompt $(x, c)$.
//! 2. Samples $G$ teacher trajectories $\tau_T^{(i)} \sim \pi_{\bar{\theta}_T}(\cdot \mid x, c)$ and $G$ student
//!    trajectories $\tau_S^{(i)} \sim \pi_{\bar{\theta}_S}(\cdot \mid x)$.
//! 3. Verifies every trajectory via `Supervisor::score` and computes group-centered advantages $\hat{A}_T, \hat{A}_S$.
use std::path::Path;

use anyhow::Result;
use rand::rngs::StdRng;

use super::kl::center_advantages;
use super::supervision::{Supervisor, Verification};
use crate::dataset::chat::{ChatEncoder, Encoded};
use crate::dataset::Row;
use crate::lora::AdapterRole;
use crate::model::vl::Lfm2Vl;

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub tokens: Vec<u32>,
    pub text: String,
    pub verification: Verification,
    pub advantage: f64,
}

#[derive(Debug, Clone)]
pub struct GroupRollout {
    pub student_prompt: Encoded,
    pub teacher_prompt: Encoded,
    pub teacher_trajs: Vec<Trajectory>,
    pub student_trajs: Vec<Trajectory>,
    pub completion_tokens: u64,
    pub processed_tokens: u64,
}

impl GroupRollout {
    pub fn mean_teacher_reward(&self) -> f64 {
        if self.teacher_trajs.is_empty() {
            0.0
        } else {
            self.teacher_trajs.iter().map(|t| t.verification.reward).sum::<f64>() / self.teacher_trajs.len() as f64
        }
    }

    pub fn mean_student_reward(&self) -> f64 {
        if self.student_trajs.is_empty() {
            0.0
        } else {
            self.student_trajs.iter().map(|t| t.verification.reward).sum::<f64>() / self.student_trajs.len() as f64
        }
    }
}

/// Encodes both the unprivileged student prompt $x$ and privileged teacher prompt $(x, c)$, returning `None` if either
/// prompt plus `max_new_tokens` would exceed `max_len`.
pub fn prepare_prompts(
    enc: &ChatEncoder,
    row: &Row,
    root: &Path,
    supervisor: &Supervisor,
    max_len: usize,
    max_new_tokens: usize,
) -> Result<Option<(Encoded, Encoded)>> {
    let s_msgs = row.prompt_messages();
    if s_msgs.is_empty() {
        return Ok(None);
    }
    let hint = supervisor.privileged_hint(row);
    let t_msgs = row.teacher_prompt_messages(hint.as_deref());
    let s_prompt = enc.encode_prompt(&s_msgs, &row.images, root)?;
    let t_prompt = enc.encode_prompt(&t_msgs, &row.images, root)?;
    let budget = max_new_tokens.max(8) + 4;
    if s_prompt.ids.len() + budget > max_len || t_prompt.ids.len() + budget > max_len {
        return Ok(None);
    }
    Ok(Some((s_prompt, t_prompt)))
}

/// Generates $G$ teacher rollouts under `AdapterRole::Teacher` and $G$ student rollouts under `AdapterRole::Student`,
/// scores each trajectory, and computes group-centered advantages $\hat{A}_T$ and $\hat{A}_S$.
pub fn sample_group(
    model: &Lfm2Vl,
    enc: &ChatEncoder,
    row: &Row,
    student_prompt: Encoded,
    teacher_prompt: Encoded,
    supervisor: &Supervisor,
    group_size: usize,
    max_new_tokens: usize,
    temperature: f64,
    rng: &mut StdRng,
) -> Result<GroupRollout> {
    let g = group_size.max(1);
    let eos = enc.im_end();
    let mut completion_tokens = 0u64;
    let mut processed_tokens = 0u64;

    model.set_role(AdapterRole::Teacher);
    let mut teacher_trajs = Vec::with_capacity(g);
    for _ in 0..g {
        let mut tokens = model.generate(&teacher_prompt, max_new_tokens, temperature, eos, rng)?;
        if tokens.is_empty() {
            tokens.push(eos);
        }
        completion_tokens += tokens.len() as u64;
        processed_tokens += (teacher_prompt.ids.len() + tokens.len()) as u64;
        let text = enc.decode(&tokens);
        let verification = supervisor.score(row, &text);
        teacher_trajs.push(Trajectory { tokens, text, verification, advantage: 0.0 });
    }
    let t_rewards: Vec<f64> = teacher_trajs.iter().map(|t| t.verification.reward).collect();
    for (traj, adv) in teacher_trajs.iter_mut().zip(center_advantages(&t_rewards)) {
        traj.advantage = adv;
    }

    model.set_role(AdapterRole::Student);
    let mut student_trajs = Vec::with_capacity(g);
    for _ in 0..g {
        let mut tokens = model.generate(&student_prompt, max_new_tokens, temperature, eos, rng)?;
        if tokens.is_empty() {
            tokens.push(eos);
        }
        completion_tokens += tokens.len() as u64;
        processed_tokens += (student_prompt.ids.len() + tokens.len()) as u64;
        let text = enc.decode(&tokens);
        let verification = supervisor.score(row, &text);
        student_trajs.push(Trajectory { tokens, text, verification, advantage: 0.0 });
    }
    let s_rewards: Vec<f64> = student_trajs.iter().map(|t| t.verification.reward).collect();
    for (traj, adv) in student_trajs.iter_mut().zip(center_advantages(&s_rewards)) {
        traj.advantage = adv;
    }

    Ok(GroupRollout {
        student_prompt,
        teacher_prompt,
        teacher_trajs,
        student_trajs,
        completion_tokens,
        processed_tokens,
    })
}
