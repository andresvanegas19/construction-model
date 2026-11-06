//! Supervision model client and deterministic stage/VL verifier for JOLT and JOLT+.
//!
//! JOLT+ grounds both teacher distillation and student RL in verifiable outcome rewards $R(\tau) \in [0, 1]$:
//! CoT stages (`brief`, `graph`, `sketch`) are checked against structural schema, room-ID conservation, adjacency
//! validity, band membership, and DPO rejected-candidate avoidance; vision-language floor-plan descriptions are
//! checked on extracted floor, balcony, and room-count facts. When an external supervisor LLM is configured, it
//! supplies privileged teaching hints for the teacher context $(x, c)$ and blends a semantic critique score with the
//! deterministic verifier (failing open to deterministic verification when offline).
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::dataset::Row;

const DEFAULT_OPENROUTER_URL: &str = "https://openrouter.ai/api/v1";
const VALID_RELATIONS: [&str; 4] = ["door", "open", "near", "avoid"];

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    pub url: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub weight: f64,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl SupervisorConfig {
    pub fn from_env() -> Self {
        let env_nonempty = |k: &str| std::env::var(k).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
        let model = env_nonempty("SUPERVISOR_MODEL").or_else(|| env_nonempty("CRITIC_MODEL"));
        let api_key = env_nonempty("SUPERVISOR_API_KEY").or_else(|| env_nonempty("OPENROUTER_API_KEY"));
        let url = env_nonempty("SUPERVISOR_BASE")
            .or_else(|| env_nonempty("OPENROUTER_BASE"))
            .or_else(|| (model.is_some() && api_key.is_some()).then(|| DEFAULT_OPENROUTER_URL.to_string()));
        let active = url.is_some() && model.is_some();
        let weight = env_nonempty("SUPERVISOR_WEIGHT").and_then(|s| s.parse::<f64>().ok()).unwrap_or(if active { 0.3 } else { 0.0 }).clamp(0.0, 1.0);
        Self { url, model, api_key, weight }
    }

    pub fn is_active(&self) -> bool {
        self.url.as_deref().is_some_and(|u| !u.trim().is_empty()) && self.model.as_deref().is_some_and(|m| !m.trim().is_empty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verification {
    pub reward: f64,
    pub det_reward: f64,
    pub sup_reward: Option<f64>,
    pub json_valid: bool,
    pub critique: Option<String>,
}

pub struct Supervisor {
    pub cfg: SupervisorConfig,
}

impl Supervisor {
    pub fn new(cfg: SupervisorConfig) -> Self { Self { cfg } }
    pub fn from_env() -> Self { Self::new(SupervisorConfig::from_env()) }

    /// Queries the external supervision model for a 2-3 sentence privileged teaching hint on `row` and its reference
    /// solution so the teacher policy $\pi_{\text{teach}}(\cdot \mid x, c)$ conditions on explicit domain rationale.
    pub fn privileged_hint(&self, row: &Row) -> Option<String> {
        if !self.cfg.is_active() { return None; }
        let prompt_summary = row.prompt_messages().into_iter().rev().find(|m| m.role == "user").map(|m| m.content).unwrap_or_default();
        let reference = row.reference_completion().unwrap_or_default();
        let sys = "You are a senior architectural supervisor training a smaller floor-plan reasoning model. \
                   In 2 to 3 concise sentences, explain the critical spatial, circulation, or visual constraints \
                   that make the reference answer succeed so a teacher model can guide distillation.";
        let user = format!("Stage/Kind: {} ({})\nTask prompt:\n{}\n\nVerified reference:\n{}", row.kind, row.source, prompt_summary, reference);
        let reply = self.chat(sys, &user, 220)?;
        let trimmed = reply.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    /// Verifies `completion` deterministically and, when the external supervisor is active with `weight > 0`, blends
    /// its semantic score into `reward`. Fails open to `det_reward` on network or parse errors.
    pub fn score(&self, row: &Row, completion: &str) -> Verification {
        let mut v = verify_deterministic(row, completion);
        let w = self.cfg.weight.clamp(0.0, 1.0);
        if !self.cfg.is_active() || w <= 0.0 { return v; }
        if let Some((sup_score, sup_critique)) = self.grade_completion(row, completion, &v) {
            let sup = sup_score.clamp(0.0, 1.0);
            v.sup_reward = Some(sup);
            v.reward = ((1.0 - w) * v.det_reward + w * sup).clamp(0.0, 1.0);
            if let Some(c) = sup_critique.filter(|s| !s.trim().is_empty()) {
                v.critique = Some(match v.critique.take() { Some(det_c) => format!("{det_c}; supervisor: {c}"), None => c });
            }
        }
        v
    }

    fn grade_completion(&self, row: &Row, completion: &str, det: &Verification) -> Option<(f64, Option<String>)> {
        let reference = row.reference_completion().unwrap_or_default();
        let sys = "You are grading a candidate floor-plan model completion against a verified reference. \
                   Return ONLY a compact JSON object: {\"score\": <0..100>, \"critique\": \"<short diagnostic>\"}.";
        let user = format!("Kind: {}\nDeterministic score: {:.2} ({})\nReference:\n{}\n\nCandidate completion:\n{}",
            row.kind, det.det_reward, det.critique.as_deref().unwrap_or("ok"), reference, completion);
        let parsed = extract_json_object(&self.chat(sys, &user, 180)?)?;
        let raw = parsed.get("score")?.as_f64()?;
        let score = if raw > 1.0 { raw / 100.0 } else { raw };
        let critique = parsed.get("critique").and_then(Value::as_str).map(|s| s.trim().to_string());
        Some((score.clamp(0.0, 1.0), critique))
    }

    fn chat(&self, system: &str, user: &str, max_tokens: u32) -> Option<String> {
        let base = self.cfg.url.as_deref()?.trim_end_matches('/');
        let model = self.cfg.model.as_deref()?;
        let mut req = ureq::post(&format!("{base}/chat/completions")).timeout(Duration::from_secs(20));
        if let Some(key) = self.cfg.api_key.as_deref().filter(|k| !k.is_empty()) {
            req = req.set("Authorization", &["Bearer", key].join(" "));
        }
        let body = json!({
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        });
        let resp: Value = req.send_json(body).ok()?.into_json().ok()?;
        resp["choices"].get(0)?["message"]["content"].as_str().map(String::from)
    }
}

/// Deterministic outcome verifier for both CoT JSON stages (`brief`, `graph`, `sketch`) and VL floorplan descriptions.
pub fn verify_deterministic(row: &Row, completion: &str) -> Verification {
    let ref_text = row.reference_completion().unwrap_or("").trim();
    let is_vl = row.source == "floorplans"
        || (!matches!(row.kind.as_str(), "brief" | "graph" | "sketch")
            && !ref_text.starts_with('{')
            && (looks_like_vl_facts(ref_text) || looks_like_vl_facts(completion)));
    if is_vl { verify_vl_floorplan(ref_text, completion) } else { verify_cot_json(row, ref_text, completion) }
}

fn verify_cot_json(row: &Row, ref_text: &str, completion: &str) -> Verification {
    let Some(cand) = extract_json_object(completion) else {
        return Verification { reward: 0.0, det_reward: 0.0, sup_reward: None, json_valid: false, critique: Some("invalid or missing JSON object in completion".into()) };
    };
    let Some(cand_obj) = cand.as_object() else {
        return Verification { reward: 0.0, det_reward: 0.0, sup_reward: None, json_valid: false, critique: Some("completion JSON root is not an object".into()) };
    };
    let ref_val = extract_json_object(ref_text);
    let ref_obj = ref_val.as_ref().and_then(Value::as_object);

    // 1. Top-level schema key coverage (0.25).
    let key_score = match ref_obj {
        Some(ro) if !ro.is_empty() => {
            let req_keys: Vec<&String> = ro.keys().filter(|k| k.as_str() != "confidence").collect();
            if req_keys.is_empty() { 1.0 } else { req_keys.iter().filter(|k| cand_obj.contains_key(**k)).count() as f64 / req_keys.len() as f64 }
        }
        _ => 1.0,
    };

    // 2. Non-empty reasoning string (0.10); compact sketch references without reasoning also pass.
    let has_reasoning = cand_obj.get("reasoning").and_then(Value::as_str).is_some_and(|s| s.trim().len() >= 8);
    let reasoning_score = if has_reasoning || !ref_obj.is_some_and(|ro| ro.contains_key("reasoning")) { 1.0 } else { 0.0 };

    // 3. Stage-specific structural fidelity (0.65).
    let stage = infer_stage(&row.kind, cand_obj, ref_obj);
    let (stage_score, mut notes) = match stage {
        "brief" => score_brief(cand_obj, ref_obj),
        "graph" => score_graph(cand_obj, ref_obj),
        "sketch" => score_sketch(cand_obj, ref_obj, row.rejected.as_deref()),
        _ => score_generic_json(cand_obj, ref_obj),
    };
    if key_score < 1.0 { notes.push(format!("schema key coverage {:.0}%", key_score * 100.0)); }
    if reasoning_score < 1.0 { notes.push("missing reasoning".into()); }

    let det = (0.25 * key_score + 0.10 * reasoning_score + 0.65 * stage_score).clamp(0.0, 1.0);
    let critique = Some(if notes.is_empty() { format!("{stage} verified ({:.0}%)", det * 100.0) } else { notes.join("; ") });
    Verification { reward: det, det_reward: det, sup_reward: None, json_valid: true, critique }
}

fn infer_stage<'a>(kind: &'a str, cand: &serde_json::Map<String, Value>, refer: Option<&serde_json::Map<String, Value>>) -> &'a str {
    if matches!(kind, "brief" | "graph" | "sketch") { return kind; }
    let has = |k: &str| cand.contains_key(k) || refer.is_some_and(|r| r.contains_key(k));
    if has("front") && has("back") { "sketch" } else if has("edges") { "graph" } else if has("rooms") { "brief" } else { kind }
}

fn score_brief(cand: &serde_json::Map<String, Value>, refer: Option<&serde_json::Map<String, Value>>) -> (f64, Vec<String>) {
    let mut notes = Vec::new();
    let cand_rooms = extract_brief_rooms(cand.get("rooms"));
    if cand_rooms.is_empty() { return (0.0, vec!["brief has no valid rooms map".into()]); }
    let Some(ref_rooms) = refer.map(|r| extract_brief_rooms(r.get("rooms"))).filter(|m| !m.is_empty()) else { return (1.0, notes); };
    let cand_ids: BTreeSet<&str> = cand_rooms.keys().map(String::as_str).collect();
    let ref_ids: BTreeSet<&str> = ref_rooms.keys().map(String::as_str).collect();
    let id_jaccard = jaccard(&cand_ids, &ref_ids);
    if id_jaccard < 1.0 { notes.push(format!("room ID Jaccard {:.0}%", id_jaccard * 100.0)); }
    let within_tol = ref_rooms.iter().filter(|(id, &ref_a)| cand_rooms.get(*id).is_some_and(|&ca| (ca - ref_a).abs() / ref_a.max(1.0) <= 0.20)).count();
    let area_score = within_tol as f64 / ref_rooms.len().max(1) as f64;
    if area_score < 1.0 { notes.push(format!("{within_tol}/{} rooms within ±20% area", ref_rooms.len())); }
    (0.55 * id_jaccard + 0.45 * area_score, notes)
}

fn extract_brief_rooms(v: Option<&Value>) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    match v {
        Some(Value::Object(map)) => {
            for (k, val) in map {
                let area = match val {
                    Value::Array(arr) => arr.first().and_then(Value::as_f64),
                    Value::Number(n) => n.as_f64(),
                    Value::Object(o) => o.get("target_area").or_else(|| o.get("area")).and_then(Value::as_f64),
                    _ => None,
                };
                if let Some(a) = area.filter(|&a| a > 0.0) { out.insert(k.clone(), a); }
            }
        }
        Some(Value::Array(arr)) => {
            for item in arr {
                if let Some(id) = item.get("id").and_then(Value::as_str) {
                    out.insert(id.to_string(), item.get("target_area").or_else(|| item.get("area")).and_then(Value::as_f64).unwrap_or(100.0));
                }
            }
        }
        _ => {}
    }
    out
}

#[derive(Debug, Clone)]
struct GraphEdge { pair: (String, String), relation: String }

fn score_graph(cand: &serde_json::Map<String, Value>, refer: Option<&serde_json::Map<String, Value>>) -> (f64, Vec<String>) {
    let mut notes = Vec::new();
    let Some(arr) = cand.get("edges").and_then(Value::as_array).filter(|a| !a.is_empty()) else {
        return (0.0, vec!["missing or empty edges array".into()]);
    };
    let cand_edges = parse_edges(arr);
    let valid_rel_count = cand_edges.iter().filter(|e| VALID_RELATIONS.contains(&e.relation.as_str()) && e.pair.0 != e.pair.1).count();
    let rel_validity = valid_rel_count as f64 / arr.len().max(1) as f64;
    if rel_validity < 1.0 { notes.push(format!("valid relations {valid_rel_count}/{}", arr.len())); }
    let Some(ref_edges) = refer.and_then(|r| r.get("edges")).and_then(Value::as_array).map(|a| parse_edges(a)).filter(|e| !e.is_empty()) else {
        return (rel_validity, notes);
    };
    let cand_rooms: BTreeSet<&str> = cand_edges.iter().flat_map(|e| [e.pair.0.as_str(), e.pair.1.as_str()]).collect();
    let ref_rooms: BTreeSet<&str> = ref_edges.iter().flat_map(|e| [e.pair.0.as_str(), e.pair.1.as_str()]).collect();
    let room_overlap = jaccard(&cand_rooms, &ref_rooms);
    let cand_pairs: BTreeSet<(&str, &str)> = cand_edges.iter().map(|e| (e.pair.0.as_str(), e.pair.1.as_str())).collect();
    let ref_pairs: BTreeSet<(&str, &str)> = ref_edges.iter().map(|e| (e.pair.0.as_str(), e.pair.1.as_str())).collect();
    let pair_jaccard = jaccard(&cand_pairs, &ref_pairs);
    if pair_jaccard < 0.5 { notes.push(format!("edge-pair Jaccard {:.0}%", pair_jaccard * 100.0)); }
    (0.30 * rel_validity + 0.35 * room_overlap + 0.35 * pair_jaccard, notes)
}

fn parse_edges(arr: &[Value]) -> Vec<GraphEdge> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let (a, b, rel) = match item {
            Value::Array(t) if t.len() >= 3 => (t[0].as_str().unwrap_or("").trim(), t[1].as_str().unwrap_or("").trim(), t[2].as_str().unwrap_or("").trim()),
            Value::Object(o) => (o.get("a").and_then(Value::as_str).unwrap_or("").trim(), o.get("b").and_then(Value::as_str).unwrap_or("").trim(), o.get("relation").and_then(Value::as_str).unwrap_or("").trim()),
            _ => continue,
        };
        if !a.is_empty() && !b.is_empty() {
            let pair = if a <= b { (a.to_string(), b.to_string()) } else { (b.to_string(), a.to_string()) };
            out.push(GraphEdge { pair, relation: rel.to_ascii_lowercase() });
        }
    }
    out
}

fn score_sketch(cand: &serde_json::Map<String, Value>, refer: Option<&serde_json::Map<String, Value>>, rejected: Option<&str>) -> (f64, Vec<String>) {
    let mut notes = Vec::new();
    let Some((c_front, c_back)) = parse_band_rooms(cand) else { return (0.0, vec!["sketch missing front/back room lists".into()]); };
    let all_cand: Vec<&str> = c_front.iter().chain(c_back.iter()).map(String::as_str).collect();
    if all_cand.is_empty() { return (0.0, vec!["sketch has no room IDs".into()]); }
    let unique_cand: BTreeSet<&str> = all_cand.iter().copied().collect();
    let dup_count = all_cand.len().saturating_sub(unique_cand.len());
    let no_dup_factor = unique_cand.len() as f64 / all_cand.len() as f64;
    if dup_count > 0 { notes.push(format!("{dup_count} duplicate room ID(s)")); }

    let mut score = if let Some((r_front, r_back)) = refer.and_then(parse_band_rooms) {
        let ref_front_set: BTreeSet<&str> = r_front.iter().map(String::as_str).collect();
        let ref_back_set: BTreeSet<&str> = r_back.iter().map(String::as_str).collect();
        let ref_all: BTreeSet<&str> = ref_front_set.iter().chain(ref_back_set.iter()).copied().collect();
        let id_conservation = jaccard(&unique_cand, &ref_all) * no_dup_factor;
        if id_conservation < 1.0 { notes.push(format!("room conservation {:.0}%", id_conservation * 100.0)); }
        let cand_front_set: BTreeSet<&str> = c_front.iter().map(String::as_str).collect();
        let cand_back_set: BTreeSet<&str> = c_back.iter().map(String::as_str).collect();
        let band_matches = ref_front_set.intersection(&cand_front_set).count() + ref_back_set.intersection(&cand_back_set).count();
        let band_score = band_matches as f64 / ref_all.len().max(1) as f64;
        if band_score < 1.0 { notes.push(format!("band match {band_matches}/{}", ref_all.len())); }
        let order_score = 0.5 * order_similarity(&c_front, &r_front) + 0.5 * order_similarity(&c_back, &r_back);
        0.45 * id_conservation + 0.35 * band_score + 0.20 * order_score
    } else {
        no_dup_factor
    };
    if let Some((rej_f, rej_b)) = rejected.and_then(extract_json_object).as_ref().and_then(Value::as_object).and_then(parse_band_rooms) {
        if c_front == rej_f && c_back == rej_b {
            score *= 0.35;
            notes.push("matches rejected lower-score layout".into());
        }
    }
    (score.clamp(0.0, 1.0), notes)
}

fn parse_band_rooms(obj: &serde_json::Map<String, Value>) -> Option<(Vec<String>, Vec<String>)> {
    Some((flatten_band(obj.get("front")?), flatten_band(obj.get("back")?)))
}

fn flatten_band(v: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for col in v.as_array().into_iter().flatten() {
        match col {
            Value::Array(items) => out.extend(items.iter().filter_map(Value::as_str).map(str::trim).filter(|s| !s.is_empty()).map(String::from)),
            Value::String(s) => out.extend(s.split('+').map(str::trim).filter(|s| !s.is_empty()).map(String::from)),
            _ => {}
        }
    }
    out
}

fn order_similarity(cand: &[String], refer: &[String]) -> f64 {
    if refer.len() <= 1 { return usize::from(if refer.is_empty() { cand.is_empty() } else { cand.first() == refer.first() }) as f64; }
    let adj = |xs: &[String]| -> BTreeSet<(&str, &str)> {
        xs.windows(2).map(|w| if w[0] <= w[1] { (w[0].as_str(), w[1].as_str()) } else { (w[1].as_str(), w[0].as_str()) }).collect()
    };
    jaccard(&adj(cand), &adj(refer))
}

fn score_generic_json(cand: &serde_json::Map<String, Value>, refer: Option<&serde_json::Map<String, Value>>) -> (f64, Vec<String>) {
    let Some(ro) = refer.filter(|r| !r.is_empty()) else { return (1.0, Vec::new()) };
    (ro.iter().filter(|(k, v)| cand.get(*k) == Some(*v)).count() as f64 / ro.len() as f64, Vec::new())
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct VlFacts { floor: Option<i64>, balcony: Option<bool>, rooms: Option<i64>, rooms_at_least: bool }

fn looks_like_vl_facts(s: &str) -> bool {
    let low = s.to_ascii_lowercase();
    low.contains("floor:") || low.contains("balcony:") || low.contains("rooms:")
}

fn parse_vl_facts(s: &str) -> VlFacts {
    let mut facts = VlFacts::default();
    for seg in s.to_ascii_lowercase().split(['.', '\n', ';', ',']).map(str::trim) {
        if let Some(rest) = seg.strip_prefix("floor:") {
            facts.floor = first_int(rest);
        } else if let Some(rest) = seg.strip_prefix("balcony:") {
            facts.balcony = match rest.split_whitespace().next().unwrap_or("") {
                "yes" | "true" => Some(true),
                "no" | "false" | "none" => Some(false),
                _ => None,
            };
        } else if let Some(rest) = seg.strip_prefix("rooms:").map(str::trim) {
            if let Some(after) = rest.strip_prefix("at least") { facts.rooms_at_least = true; facts.rooms = first_int(after); }
            else { facts.rooms = first_int(rest); }
        }
    }
    facts
}

fn first_int(s: &str) -> Option<i64> {
    s.chars().skip_while(|c| !c.is_ascii_digit() && *c != '-').take_while(|c| c.is_ascii_digit() || *c == '-').collect::<String>().parse::<i64>().ok()
}

fn verify_vl_floorplan(ref_text: &str, completion: &str) -> Verification {
    let (ref_facts, cand_facts) = (parse_vl_facts(ref_text), parse_vl_facts(completion));
    if cand_facts.floor.is_none() && cand_facts.balcony.is_none() && cand_facts.rooms.is_none() {
        return Verification { reward: 0.0, det_reward: 0.0, sup_reward: None, json_valid: false, critique: Some("missing structured Floor/Balcony/Rooms fields".into()) };
    }
    let (mut parts, mut notes) = (Vec::new(), Vec::new());
    if let Some(rf) = ref_facts.floor {
        let s = if cand_facts.floor == Some(rf) { 1.0 } else { notes.push(format!("floor {:?} != {rf}", cand_facts.floor)); 0.0 };
        parts.push(s);
    }
    if let Some(rb) = ref_facts.balcony {
        let s = if cand_facts.balcony == Some(rb) { 1.0 } else { notes.push(format!("balcony {:?} != {rb}", cand_facts.balcony)); 0.0 };
        parts.push(s);
    }
    if let Some(rr) = ref_facts.rooms {
        let s = match cand_facts.rooms {
            Some(cr) if ref_facts.rooms_at_least && cr >= rr => 1.0,
            Some(cr) => 1.0 / (1.0 + (cr - rr).unsigned_abs() as f64),
            None => 0.0,
        };
        if s < 1.0 { notes.push(format!("rooms {:?} vs {rr}", cand_facts.rooms)); }
        parts.push(s);
    }
    let det = if parts.is_empty() { 1.0 } else { (parts.iter().sum::<f64>() / parts.len() as f64).clamp(0.0, 1.0) };
    let critique = Some(if notes.is_empty() { "VL floorplan facts matched".into() } else { notes.join("; ") });
    Verification { reward: det, det_reward: det, sup_reward: None, json_valid: true, critique }
}

pub fn extract_json_object(s: &str) -> Option<Value> {
    let trimmed = s.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed).ok().filter(Value::is_object).ok_or(()) { return Some(v); }
    let (start, end) = (trimmed.find('{')?, trimmed.rfind('}')?);
    (end > start).then(|| serde_json::from_str::<Value>(&trimmed[start..=end]).ok()).flatten().filter(Value::is_object)
}

fn jaccard<T: Ord>(a: &BTreeSet<T>, b: &BTreeSet<T>) -> f64 {
    if a.is_empty() && b.is_empty() { 1.0 } else { a.intersection(b).count() as f64 / a.union(b).count().max(1) as f64 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::chat::Message;

    fn sample_row(kind: &str, source: &str, reference: &str, rejected: Option<&str>) -> Row {
        Row {
            messages: vec![Message { role: "user".into(), content: "Design prompt".into() }, Message { role: "assistant".into(), content: reference.into() }],
            images: vec![], source: source.into(), kind: kind.into(),
            reference: Some(reference.into()), privileged: None, rejected: rejected.map(String::from), score: Some(100.0),
        }
    }

    #[test]
    fn sketch_verifier_rewards_conservation_and_penalizes_rejected() {
        let chosen = r#"{"reasoning":"Public rooms on street side, bedrooms in rear band.","front":[["laundry"],["kitchen"],["dining"],["living"]],"back":[["bed1"],["bath1"],["bed2"]]}"#;
        let rejected = r#"{"reasoning":"Kitchen at far right.","front":[["laundry"],["dining"],["living"],["kitchen"]],"back":[["bed1"],["bath1"],["bed2"]]}"#;
        let row = sample_row("sketch", "dpo:pilot", chosen, Some(rejected));
        let exact = verify_deterministic(&row, chosen);
        assert!(exact.json_valid && (exact.det_reward - 1.0).abs() < 1e-6);
        let rej_v = verify_deterministic(&row, rejected);
        assert!(rej_v.json_valid && rej_v.det_reward < 0.65 && rej_v.critique.as_deref().unwrap_or("").contains("rejected"));
        let malformed = verify_deterministic(&row, "not valid json {front: [");
        assert!(!malformed.json_valid && malformed.det_reward == 0.0);
    }

    #[test]
    fn brief_and_graph_verifiers_score_structural_fidelity() {
        let brief_ref = r#"{"reasoning":"Two-bedroom compact plan with open living.","household":"couple","priorities":["open living"],"rooms":{"living":[300,"public",1],"kitchen":[180,"public",1],"bed1":[220,"private",1],"bath1":[65,"private",1]}}"#;
        let brief_cand = r#"{"reasoning":"Compact two-bedroom home with open living and dining.","household":"couple","priorities":["open living"],"rooms":{"living":[315,"public",1],"kitchen":[170,"public",1],"bed1":[210,"private",1],"bath1":[65,"private",1]}}"#;
        let bv = verify_deterministic(&sample_row("brief", "sft:pilot", brief_ref, None), brief_cand);
        assert!(bv.json_valid && bv.det_reward > 0.95);

        let graph_ref = r#"{"reasoning":"Open public cluster and private suite doors.","edges":[["kitchen","dining","open","flow"],["dining","living","open","flow"],["bath1","bed1","door","ensuite"]]}"#;
        let graph_cand = r#"{"reasoning":"Public rooms connect openly while bath serves bed1.","edges":[{"a":"kitchen","b":"dining","relation":"open","why":"flow"},{"a":"dining","b":"living","relation":"open","why":"flow"},{"a":"bed1","b":"bath1","relation":"door","why":"ensuite"}]}"#;
        let gv = verify_deterministic(&sample_row("graph", "sft:pilot", graph_ref, None), graph_cand);
        assert!(gv.json_valid && (gv.det_reward - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vl_floorplan_verifier_handles_exact_and_partial_matches() {
        let row = sample_row("floorplan", "floorplans", "Floor: 2. Balcony: yes. Rooms: 6.", None);
        assert!((verify_deterministic(&row, "Floor: 2. Balcony: yes. Rooms: 6.").det_reward - 1.0).abs() < 1e-6);
        assert!((verify_deterministic(&row, "floor: 2. balcony: yes. rooms: 7.").det_reward - (2.5 / 3.0)).abs() < 1e-6);
        assert!((verify_deterministic(&row, "Floor: 1. Balcony: no. Rooms: 3.").det_reward - (0.25 / 3.0)).abs() < 1e-6);
    }
}
