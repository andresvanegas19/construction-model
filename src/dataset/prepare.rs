//! `prepare`: builds out/prepared/{train,eval}.jsonl from datasets under the selected data root.
//! (1) Text SFT rows (`{prompt:[..], completion:[..], stage, ...}`), split as
//! exported, enriched with privileged context and DPO rejected contrast when `dpo.jsonl` is present.
//! (2) Image rows from data/images/floorplans/cases.json: the case's expected floor / balcony /
//! room count become a short structured description, the target a vision model should read off the drawing; a case
//! without any expectation or without its image is skipped.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;

use super::chat::{Message, IMAGE_PLACEHOLDER};
use super::{read_jsonl, write_jsonl, Row};

pub struct PrepareOpts {
    pub root: PathBuf,
    /// Every n-th image case goes to eval (there are few, so a fixed stride keeps both splits non-empty).
    pub image_eval_every: usize,
    pub max_rows: Option<usize>,
}

pub struct Prepared {
    pub train: usize,
    pub eval: usize,
    pub skipped: usize,
}

#[derive(Debug, Clone)]
pub struct DpoEntry {
    pub rejected: String,
    pub chosen_score: Option<f64>,
    pub rejected_score: Option<f64>,
}

/// Indexes `dpo.jsonl` by `(house, stage)` and by user prompt content so `sketch` rows inherit their rejected pair.
#[derive(Debug, Default)]
pub struct DpoLookup {
    pub by_house: HashMap<(u64, String), DpoEntry>,
    pub by_prompt: HashMap<String, DpoEntry>,
}

impl DpoLookup {
    pub fn from_values(rows: &[Value]) -> Self {
        let mut out = Self::default();
        for v in rows {
            let Some(rejected) = assistant_content(&v["rejected"]) else {
                continue;
            };
            let entry = DpoEntry {
                rejected,
                chosen_score: v["chosen_score"].as_f64(),
                rejected_score: v["rejected_score"].as_f64(),
            };
            let stage = v["stage"].as_str().unwrap_or("sketch").to_string();
            if let Some(house) = v["house"].as_u64() {
                out.by_house.insert((house, stage.clone()), entry.clone());
            }
            if let Some(p) = prompt_key(&v["prompt"], &stage) {
                out.by_prompt.insert(p, entry);
            }
        }
        out
    }

    pub fn get(&self, v: &Value, stage: &str) -> Option<&DpoEntry> {
        if let Some(house) = v["house"].as_u64() {
            if let Some(e) = self.by_house.get(&(house, stage.to_string())) {
                return Some(e);
            }
        }
        prompt_key(&v["prompt"], stage).and_then(|k| self.by_prompt.get(&k))
    }
}

pub fn run(o: &PrepareOpts) -> Result<Prepared> {
    let (mut train, mut eval, mut skipped) = (vec![], vec![], 0);
    let datasets = o.root.join("data/text/datasets");
    if datasets.is_dir() {
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&datasets)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for dir in dirs {
            let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let dpo_path = dir.join("dpo.jsonl");
            let dpo = if dpo_path.is_file() {
                DpoLookup::from_values(&read_jsonl::<Value>(&dpo_path)?)
            } else {
                DpoLookup::default()
            };
            for (file, out) in [("train.jsonl", &mut train), ("eval.jsonl", &mut eval)] {
                let path = dir.join(file);
                if !path.is_file() {
                    continue;
                }
                for v in read_jsonl::<Value>(&path)? {
                    match sft_row(&v, &name, &dpo) {
                        Some(r) => out.push(r),
                        None => skipped += 1,
                    }
                }
            }
        }
    }
    let (it, ie, is) = image_rows(&o.root, o.image_eval_every.max(2))?;
    train.extend(it);
    eval.extend(ie);
    skipped += is;
    if let Some(m) = o.max_rows {
        train.truncate(m);
        eval.truncate((m / 5).max(1));
    }
    let out = o.root.join("out/prepared");
    write_jsonl(&out.join("train.jsonl"), &train)?;
    write_jsonl(&out.join("eval.jsonl"), &eval)?;
    Ok(Prepared { train: train.len(), eval: eval.len(), skipped })
}

/// Content is a string, or OpenAI-style parts of which the text ones are kept (these rows carry no images).
fn content(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")),
        _ => None,
    }
}

fn messages(v: &Value) -> Option<Vec<Message>> {
    v.as_array()?
        .iter()
        .map(|m| Some(Message { role: m["role"].as_str()?.to_string(), content: content(&m["content"])? }))
        .collect()
}

fn assistant_content(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => messages(v)?.into_iter().rev().find(|m| m.role == "assistant").map(|m| m.content).filter(|s| !s.is_empty()),
    }
}

fn prompt_key(prompt: &Value, stage: &str) -> Option<String> {
    let msgs = messages(prompt)?;
    let user = msgs.iter().rev().find(|m| m.role == "user").unwrap_or_else(|| msgs.last()?);
    Some(format!("{stage}:{}", user.content.trim()))
}

fn program_summary(program: &Value) -> Option<String> {
    let items: Vec<String> = program
        .as_array()?
        .iter()
        .filter_map(|r| {
            let id = r["id"].as_str()?;
            let kind = r["type"].as_str().unwrap_or("room");
            match r["target_area"].as_f64().or_else(|| r["target_sqft"].as_f64()) {
                Some(sqft) => Some(format!("{id}:{kind}:{sqft:.0}")),
                None => Some(format!("{id}:{kind}")),
            }
        })
        .collect();
    (!items.is_empty()).then(|| items.join(", "))
}

pub fn sft_row(v: &Value, dataset: &str, dpo: &DpoLookup) -> Option<Row> {
    let mut msgs = match (messages(&v["prompt"]), messages(&v["completion"])) {
        (Some(p), Some(c)) => [p, c].concat(),
        // Already in chat form (`{"messages": [...]}`), e.g. a hand-made extra file.
        _ => messages(&v["messages"])?,
    };
    msgs.retain(|m| !m.content.is_empty());
    let reference = msgs.iter().rev().find(|m| m.role == "assistant")?.content.clone();
    let stage = v["stage"].as_str().unwrap_or("chat").to_string();
    let dpo_entry = dpo.get(v, &stage);
    let score = v["score"].as_f64().or_else(|| dpo_entry.and_then(|d| d.chosen_score));
    let rejected = dpo_entry.map(|d| d.rejected.clone());

    let mut priv_lines = Vec::new();
    let mut header = format!("Stage: {stage}");
    if let Some(s) = score {
        header.push_str(&format!(" | Validator score: {s:.1}"));
    }
    if let Some(rs) = dpo_entry.and_then(|d| d.rejected_score) {
        header.push_str(&format!(" (rejected score: {rs:.1})"));
    }
    priv_lines.push(header);
    if let Some(prog) = program_summary(&v["program"]) {
        priv_lines.push(format!("Program (id:type:target_sqft): {prog}"));
    }
    priv_lines.push(format!("Verified reference completion: {reference}"));
    if let Some(rej) = &rejected {
        priv_lines.push(format!("Rejected layout (lower validator score): {rej}"));
    }

    Some(Row {
        messages: msgs,
        images: vec![],
        source: format!("sft:{dataset}"),
        kind: stage,
        reference: Some(reference),
        privileged: Some(priv_lines.join("\n")),
        rejected,
        score,
    })
}

const IMAGE_PROMPT: &str = "Read this floor plan. Say which floor it shows, whether it has a balcony, and how many rooms it has.";

fn image_rows(root: &Path, every: usize) -> Result<(Vec<Row>, Vec<Row>, usize)> {
    let base = root.join("data/images/floorplans");
    let cases = base.join("cases.json");
    let (mut train, mut eval, mut skipped) = (vec![], vec![], 0);
    if !cases.is_file() {
        return Ok((train, eval, skipped));
    }
    let v: Value = serde_json::from_slice(&std::fs::read(&cases)?).context("cases.json")?;
    for (i, case) in v.as_array().into_iter().flatten().enumerate() {
        let Some(rel) = case["image"].as_str() else {
            skipped += 1;
            continue;
        };
        let (Some(target), true) = (target_text(case), base.join(rel).is_file()) else {
            skipped += 1;
            continue;
        };
        let privileged = image_privileged(case, &target);
        let row = Row {
            messages: vec![
                Message { role: "user".into(), content: format!("{IMAGE_PLACEHOLDER}\n{IMAGE_PROMPT}") },
                Message { role: "assistant".into(), content: target.clone() },
            ],
            images: vec![format!("data/images/floorplans/{rel}")],
            source: "floorplans".into(),
            kind: case["category"].as_str().unwrap_or("floorplan").to_string(),
            reference: Some(target),
            privileged: Some(privileged),
            rejected: None,
            score: case["score"].as_f64(),
        };
        if i % every == every - 1 {
            eval.push(row);
        } else {
            train.push(row);
        }
    }
    Ok((train, eval, skipped))
}

fn image_privileged(case: &Value, target: &str) -> String {
    let mut meta = Vec::new();
    if let Some(cat) = case["category"].as_str() {
        meta.push(format!("category={cat}"));
    }
    if let Some(f) = case["expected_floor"].as_u64() {
        meta.push(format!("expected_floor={f}"));
    }
    if let Some(b) = case["expected_balcony"].as_bool() {
        meta.push(format!("expected_balcony={b}"));
    }
    if let Some(r) = case["expected_rooms"].as_u64() {
        meta.push(format!("expected_rooms={r}"));
    } else if let Some(r) = case["expected_min_rooms"].as_u64() {
        meta.push(format!("expected_min_rooms={r}"));
    }
    if let Some(notes) = case["notes"].as_str().map(str::trim).filter(|s| !s.is_empty()) {
        meta.push(format!("notes={notes}"));
    }
    format!("Floor-plan ground truth ({}): {target}", meta.join(", "))
}

fn target_text(case: &Value) -> Option<String> {
    let mut parts = vec![];
    if let Some(f) = case["expected_floor"].as_u64() {
        parts.push(format!("Floor: {f}."));
    }
    if let Some(b) = case["expected_balcony"].as_bool() {
        parts.push(format!("Balcony: {}.", if b { "yes" } else { "no" }));
    }
    if let Some(n) = case["expected_rooms"].as_u64() {
        parts.push(format!("Rooms: {n}."));
    } else if let Some(n) = case["expected_min_rooms"].as_u64() {
        parts.push(format!("Rooms: at least {n}."));
    }
    (!parts.is_empty()).then(|| parts.join(" "))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{sft_row, DpoLookup};
    use crate::dataset::chat::{Message, IMAGE_PLACEHOLDER};
    use crate::dataset::Row;

    #[test]
    fn teacher_prompt_preserves_image_placeholder_count_and_attaches_privileged_context() {
        let row = Row {
            messages: vec![
                Message { role: "user".into(), content: format!("{IMAGE_PLACEHOLDER}\nDescribe this plan.") },
                Message { role: "assistant".into(), content: "Floor: 2. Balcony: yes. Rooms: 6.".into() },
            ],
            images: vec!["data/images/floorplans/test.png".into()],
            source: "floorplans".into(),
            kind: "multi_floor_balcony".into(),
            reference: Some("Floor: 2. Balcony: yes. Rooms: 6.".into()),
            privileged: Some(format!("Ground truth with accidental {IMAGE_PLACEHOLDER}: Floor: 2. Balcony: yes. Rooms: 6.")),
            rejected: Some("Floor: 1. Balcony: no. Rooms: 4.".into()),
            score: Some(100.0),
        };
        let prompt = row.prompt_messages();
        assert_eq!(prompt.len(), 1);
        assert_eq!(row.reference_completion(), Some("Floor: 2. Balcony: yes. Rooms: 6."));

        let teacher = row.teacher_prompt_messages(Some("Check upper-floor terrace railing."));
        assert_eq!(teacher.len(), 1);
        let placeholder_count = teacher.iter().map(|m| m.content.matches(IMAGE_PLACEHOLDER).count()).sum::<usize>();
        assert_eq!(placeholder_count, row.images.len());
        assert!(teacher[0].content.contains("Floor: 2. Balcony: yes. Rooms: 6."));
        assert!(teacher[0].content.contains("Floor: 1. Balcony: no. Rooms: 4."));
        assert!(teacher[0].content.contains("Check upper-floor terrace railing."));
    }

    #[test]
    fn sft_row_populates_reference_privileged_and_dpo_rejected() {
        let dpo = DpoLookup::from_values(&[json!({
            "house": 9,
            "stage": "sketch",
            "prompt": [{"role": "user", "content": "front: living, kitchen\nback: bed1, bath1"}],
            "chosen": [{"role": "assistant", "content": "{\"front\":[[\"living\"],[\"kitchen\"]],\"back\":[[\"bed1\"],[\"bath1\"]]}"}],
            "rejected": [{"role": "assistant", "content": "{\"front\":[[\"bath1\"],[\"living\"]],\"back\":[[\"kitchen\"],[\"bed1\"]]}"}],
            "chosen_score": 100,
            "rejected_score": 64
        })]);
        let v = json!({
            "prompt": [
                {"role": "system", "content": "Order columns."},
                {"role": "user", "content": "front: living, kitchen\nback: bed1, bath1"}
            ],
            "completion": [
                {"role": "assistant", "content": "{\"front\":[[\"living\"],[\"kitchen\"]],\"back\":[[\"bed1\"],[\"bath1\"]]}"}
            ],
            "stage": "sketch",
            "score": 100,
            "house": 9,
            "program": [
                {"id": "living", "type": "living", "target_area": 300.0},
                {"id": "kitchen", "type": "kitchen", "target_area": 180.0},
                {"id": "bed1", "type": "bedroom", "target_area": 220.0},
                {"id": "bath1", "type": "full_bath", "target_area": 65.0}
            ]
        });
        let row = sft_row(&v, "pilot", &dpo).expect("sft_row");
        assert_eq!(row.reference.as_deref(), Some("{\"front\":[[\"living\"],[\"kitchen\"]],\"back\":[[\"bed1\"],[\"bath1\"]]}"));
        assert_eq!(row.score, Some(100.0));
        assert!(row.rejected.as_deref().unwrap_or_default().contains("\"bath1\""));
        let priv_text = row.privileged.as_deref().expect("privileged");
        assert!(priv_text.contains("Stage: sketch"));
        assert!(priv_text.contains("living:living:300"));
        assert!(priv_text.contains("Rejected layout (lower validator score):"));
    }
}
