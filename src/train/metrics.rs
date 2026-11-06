//! The run files an admin dashboard reads (contract in training/README.md; keep the keys exact):
//! runs/<id>/metrics.jsonl (train lines {step, loss, lr, tokens, elapsed_s}; eval lines {step, eval_loss}) and
//! runs/<id>/summary.json (written as "running" at start, rewritten at the end).
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde::Serialize;
use serde_json::json;

use crate::config::MODEL_ID;

#[derive(Debug, Clone, Serialize)]
pub struct LoraSummary {
    pub rank: usize,
    pub alpha: f64,
    pub targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub status: String,
    pub model: String,
    pub run_id: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    pub steps: u64,
    pub final_loss: Option<f64>,
    pub best_eval_loss: Option<f64>,
    pub train_samples: usize,
    pub eval_samples: usize,
    pub lora: LoraSummary,
    pub device: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jolt: Option<serde_json::Value>,
}

pub struct RunLog {
    pub dir: PathBuf,
    metrics: std::fs::File,
    pub summary: Summary,
    started: Instant,
}

impl RunLog {
    pub fn start(dir: &Path, run_id: &str, lora: LoraSummary, train_samples: usize, eval_samples: usize) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let metrics = std::fs::OpenOptions::new().create(true).append(true).open(dir.join("metrics.jsonl"))?;
        let summary = Summary {
            status: "running".into(),
            model: MODEL_ID.into(),
            run_id: run_id.into(),
            started_at: now_iso(),
            finished_at: None,
            steps: 0,
            final_loss: None,
            best_eval_loss: None,
            train_samples,
            eval_samples,
            lora,
            device: "cpu".into(),
            jolt: None,
        };
        let log = Self { dir: dir.to_path_buf(), metrics, summary, started: Instant::now() };
        log.write_summary()?;
        Ok(log)
    }

    pub fn elapsed_s(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    pub fn train(&mut self, step: u64, loss: f64, lr: f64, tokens: u64) -> Result<()> {
        self.train_with_extra(step, loss, lr, tokens, None)
    }

    pub fn train_with_extra(
        &mut self,
        step: u64,
        loss: f64,
        lr: f64,
        tokens: u64,
        extra: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<()> {
        self.summary.steps = step;
        self.summary.final_loss = finite(loss);
        let mut line = json!({"step": step, "loss": loss, "lr": lr, "tokens": tokens, "elapsed_s": self.elapsed_s()});
        if let (Some(obj), Some(ext)) = (line.as_object_mut(), extra) {
            for (k, v) in ext {
                obj.insert(k, v);
            }
        }
        self.line(&line)?;
        // The summary's step count is refreshed on every log line so a dashboard polling a long run sees progress.
        self.write_summary()
    }

    pub fn eval(&mut self, step: u64, eval_loss: f64) -> Result<()> {
        if eval_loss.is_finite() && self.summary.best_eval_loss.map_or(true, |b| eval_loss < b) {
            self.summary.best_eval_loss = Some(eval_loss);
        }
        self.line(&json!({"step": step, "eval_loss": eval_loss}))?;
        self.write_summary()
    }

    pub fn finish(&mut self, ok: bool) -> Result<()> {
        self.summary.status = if ok { "completed" } else { "failed" }.into();
        self.summary.finished_at = Some(now_iso());
        self.write_summary()
    }

    fn line(&mut self, v: &serde_json::Value) -> Result<()> {
        writeln!(self.metrics, "{v}")?;
        Ok(self.metrics.flush()?)
    }

    fn write_summary(&self) -> Result<()> {
        // Written to a temp name and renamed, so a reader never sees half a file.
        let tmp = self.dir.join("summary.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self.summary)?)?;
        Ok(std::fs::rename(tmp, self.dir.join("summary.json"))?)
    }
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

/// UTC "YYYY-MM-DDTHH:MM:SSZ" without a date crate (Howard Hinnant's civil-from-days).
pub fn now_iso() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
    iso_from_unix(secs)
}

pub fn iso_from_unix(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    #[test]
    fn iso_dates() {
        assert_eq!(super::iso_from_unix(0), "1970-01-01T00:00:00Z");
        assert_eq!(super::iso_from_unix(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(super::iso_from_unix(1_700_000_000), "2023-11-14T22:13:20Z");
    }
}
