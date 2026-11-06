//! Joint On-policy Learning and Teaching (JOLT / JOLT+) with a supervision model.
//!
//! Couples a privileged teacher adapter $\pi_{\theta_T}(\cdot \mid x, c)$ and an unprivileged student adapter
//! $\pi_{\theta_S}(\cdot \mid x)$ over a single frozen `LFM2.5-VL-1.6B` backbone:
//! - Teacher objective $\mathcal{J}_T$: maximizes verifiable outcome rewards on teacher rollouts while staying
//!   distributionally close to the student via forward KL $\beta_T \,\text{KL}(\pi_{\theta_T} \parallel \text{sg}[\pi_{\bar{\theta}_S}])$.
//! - Student objective $\mathcal{J}_{\text{OPSD}} + \mathcal{J}_{S,\text{RL}}$: minimizes reverse KL to the privileged
//!   teacher on student rollouts $\beta_S \,\text{KL}(\pi_{\theta_S} \parallel \text{sg}[\pi_{\bar{\theta}_T}])$, and in
//!   **JOLT+** adds the centered student outcome reward $\hat{A}_S(\tau_S) \log \pi_{\theta_S}(a_t \mid s_t)$.
pub mod kl;
pub mod rollout;
pub mod supervision;
pub mod trainer;

use std::path::PathBuf;

use clap::{Args, ValueEnum};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::train::TrainArgs;
use supervision::SupervisorConfig;

pub use trainer::run;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum JoltVariant {
    /// Teacher RL + KL alignment, Student reverse-KL distillation only ($\mathcal{J}_T + \mathcal{J}_{\text{OPSD}}$).
    #[value(name = "jolt")]
    Jolt,
    /// Full JOLT+ objective adding direct student RL outcome bonus ($\mathcal{J}_T + \mathcal{J}_{\text{OPSD}} + \mathcal{J}_{S,\text{RL}}$).
    #[value(name = "jolt-plus")]
    JoltPlus,
}

impl JoltVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Jolt => "jolt",
            Self::JoltPlus => "jolt-plus",
        }
    }

    pub fn student_rl(self) -> bool {
        matches!(self, Self::JoltPlus)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum TeacherMode {
    /// Separate trainable LoRA banks $(A_S, B_S)$ and $(A_T, B_T)$ on the shared frozen backbone.
    #[value(name = "separate")]
    Separate,
    /// Shared LoRA parameters ($\theta_T = \theta_S$) conditioned on privileged vs. unprivileged prompts.
    #[value(name = "shared")]
    Shared,
}

impl TeacherMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Shared => "shared",
        }
    }

    pub fn is_separate(self) -> bool {
        matches!(self, Self::Separate)
    }
}

#[derive(Debug, Clone, Args)]
pub struct JoltArgs {
    #[command(flatten)]
    pub train: TrainArgs,
    /// Objective variant: `jolt-plus` (default, adds student RL reward bonus) or `jolt` (OPSD student only).
    #[arg(long, value_enum, default_value_t = JoltVariant::JoltPlus)]
    pub variant: JoltVariant,
    /// Weight $\beta_T$ on teacher-to-student forward KL alignment $\text{KL}(\pi_{\theta_T} \parallel \text{sg}[\pi_{\bar{\theta}_S}])$.
    #[arg(long, default_value_t = 0.1)]
    pub beta_t: f64,
    /// Weight $\beta_S$ on student-to-teacher reverse KL distillation $\text{KL}(\pi_{\theta_S} \parallel \text{sg}[\pi_{\bar{\theta}_T}])$.
    #[arg(long, default_value_t = 1.0)]
    pub beta_s: f64,
    /// Number of on-policy trajectories $G$ sampled per prompt for both teacher and student.
    #[arg(long, default_value_t = 4)]
    pub group_size: usize,
    /// Maximum generated completion tokens per on-policy rollout trajectory.
    #[arg(long, default_value_t = 192)]
    pub max_new_tokens: usize,
    /// Sampling temperature for teacher and student on-policy rollouts.
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,
    /// Whether the teacher uses a separate LoRA adapter bank (`separate`) or shares the student's (`shared`).
    #[arg(long, value_enum, default_value_t = TeacherMode::Separate)]
    pub teacher_mode: TeacherMode,
    /// Optional path to an initial LoRA adapter directory for the teacher policy.
    #[arg(long)]
    pub teacher_init: Option<PathBuf>,
    /// Optional OpenAI-compatible base URL for an external supervision model (`SUPERVISOR_BASE` / `OPENROUTER_BASE`).
    #[arg(long)]
    pub supervisor_url: Option<String>,
    /// Optional model ID for the external supervision model (`SUPERVISOR_MODEL` / `CRITIC_MODEL`).
    #[arg(long)]
    pub supervisor_model: Option<String>,
    /// Blend weight $w_{\text{sup}} \in [0, 1]$ for the external supervision model reward (default 0.3 when active).
    #[arg(long)]
    pub supervisor_weight: Option<f64>,
}

impl JoltArgs {
    pub fn supervisor_config(&self) -> SupervisorConfig {
        let mut cfg = SupervisorConfig::from_env();
        if let Some(u) = self.supervisor_url.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            cfg.url = Some(u.to_string());
        }
        if let Some(m) = self.supervisor_model.as_ref().map(|s| s.trim()).filter(|s| !s.is_empty()) {
            cfg.model = Some(m.to_string());
        }
        if cfg.url.is_none() && cfg.model.is_some() && cfg.api_key.is_some() {
            cfg.url = Some("https://openrouter.ai/api/v1".to_string());
        }
        if let Some(w) = self.supervisor_weight {
            cfg.weight = w.clamp(0.0, 1.0);
        } else if cfg.is_active() && cfg.weight == 0.0 {
            cfg.weight = 0.3;
        }
        cfg
    }

    pub fn summary_json(&self, sup: &SupervisorConfig) -> Value {
        json!({
            "variant": self.variant.as_str(),
            "beta_t": self.beta_t,
            "beta_s": self.beta_s,
            "group_size": self.group_size.max(1),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "teacher_mode": self.teacher_mode.as_str(),
            "supervisor_active": sup.is_active(),
            "supervisor_model": sup.model,
            "supervisor_weight": if sup.is_active() { sup.weight } else { 0.0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_flags_and_modes() {
        assert_eq!(JoltVariant::Jolt.as_str(), "jolt");
        assert!(!JoltVariant::Jolt.student_rl());
        assert_eq!(JoltVariant::JoltPlus.as_str(), "jolt-plus");
        assert!(JoltVariant::JoltPlus.student_rl());
        assert!(TeacherMode::Separate.is_separate());
        assert!(!TeacherMode::Shared.is_separate());
    }
}
