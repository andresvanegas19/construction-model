//! out/analysis/<id>/report.json. Contract (keep exact):
//! {"analysis_id","model","created_at","status","experiments":{"<name>":{"summary":{flat numbers},
//! "findings":[strings],"series":[{"name","x":[..],"y":[..]}],"points":[{"x","y","label"}]}}}.
//! Non-finite numbers are dropped from `summary` and turned into null nowhere else (JSON has no NaN).
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Serialize;

use crate::config::MODEL_ID;
use crate::train::metrics::now_iso;

pub const MAX_POINTS: usize = 2000;

#[derive(Debug, Clone, Serialize)]
pub struct Series {
    pub name: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Pt {
    pub x: f64,
    pub y: f64,
    pub label: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Experiment {
    pub summary: BTreeMap<String, f64>,
    pub findings: Vec<String>,
    pub series: Vec<Series>,
    pub points: Vec<Pt>,
}

impl Experiment {
    pub fn metric(&mut self, key: impl Into<String>, v: f64) {
        if v.is_finite() {
            self.summary.insert(key.into(), v);
        }
    }

    pub fn finding(&mut self, s: impl Into<String>) {
        self.findings.push(s.into());
    }

    /// Non-finite y values become the series' gaps: the pair is dropped so the arrays stay plain numbers.
    pub fn series(&mut self, name: impl Into<String>, x: &[f64], y: &[f64]) {
        let (x, y): (Vec<f64>, Vec<f64>) = x.iter().zip(y).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(a, b)| (*a, *b)).unzip();
        self.series.push(Series { name: name.into(), x, y });
    }

    /// Adds 2-D points, subsampled (seeded) so the experiment never exceeds MAX_POINTS in total.
    pub fn points(&mut self, xy: &[[f32; 2]], labels: &[String], seed: u64) {
        let room = MAX_POINTS.saturating_sub(self.points.len());
        let mut idx: Vec<usize> = (0..xy.len().min(labels.len())).collect();
        if idx.len() > room {
            idx.shuffle(&mut StdRng::seed_from_u64(seed));
            idx.truncate(room);
            idx.sort_unstable();
        }
        for i in idx {
            let [x, y] = xy[i];
            if x.is_finite() && y.is_finite() {
                self.points.push(Pt { x: x as f64, y: y as f64, label: labels[i].clone() });
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct Report {
    pub analysis_id: String,
    pub model: String,
    pub created_at: String,
    pub status: String,
    pub experiments: BTreeMap<String, Experiment>,
    #[serde(skip)]
    pub dir: PathBuf,
}

impl Report {
    pub fn new(root: &Path, id: &str) -> Result<Self> {
        let dir = root.join("out/analysis").join(id);
        std::fs::create_dir_all(&dir)?;
        let r = Self {
            analysis_id: id.into(),
            model: MODEL_ID.into(),
            created_at: now_iso(),
            status: "running".into(),
            experiments: BTreeMap::new(),
            dir,
        };
        r.save()?;
        Ok(r)
    }

    /// Saved after every experiment, so a long analysis shows partial results and a crash keeps what finished.
    pub fn put(&mut self, name: &str, e: Experiment) -> Result<()> {
        self.experiments.insert(name.into(), e);
        self.save()
    }

    pub fn finish(&mut self, ok: bool) -> Result<()> {
        self.status = if ok { "completed" } else { "failed" }.into();
        self.save()
    }

    pub fn save(&self) -> Result<()> {
        let tmp = self.dir.join("report.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        Ok(std::fs::rename(tmp, self.dir.join("report.json"))?)
    }

    /// Side files (per-layer coordinates, recommended targets) next to the report; not part of the contract.
    pub fn side_file(&self, rel: &str, v: &impl Serialize) -> Result<()> {
        let path = self.dir.join(rel);
        if let Some(p) = path.parent() {
            std::fs::create_dir_all(p)?;
        }
        Ok(std::fs::write(path, serde_json::to_vec(v)?)?)
    }
}
