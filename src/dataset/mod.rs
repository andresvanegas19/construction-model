//! Training data: the prepared JSONL rows the trainer reads, and the readers that turn them into token ids.
pub mod chat;
pub mod image;
pub mod prepare;

use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use chat::Message;

/// One row of out/prepared/{train,eval}.jsonl. `images` are relative to the data root, in the order of
/// the `<image>` placeholders; `source` / `kind` label the row for the probes and per-source metrics.
/// Optional privileged fields (`reference`, `privileged`, `rejected`, `score`) feed the JOLT teacher context $(x, c)$
/// and outcome verifier without breaking older prepared datasets.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Row {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub images: Vec<String>,
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub privileged: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,
}

impl Row {
    /// Returns all messages preceding the final assistant turn (or all messages if no assistant turn exists).
    pub fn prompt_messages(&self) -> Vec<Message> {
        match self.messages.iter().rposition(|m| m.role == "assistant") {
            Some(idx) => self.messages[..idx].to_vec(),
            None => self.messages.clone(),
        }
    }

    /// Verified assistant target text, falling back to the last assistant message in `messages`.
    pub fn reference_completion(&self) -> Option<&str> {
        self.reference.as_deref().or_else(|| {
            self.messages.iter().rev().find(|m| m.role == "assistant").map(|m| m.content.as_str())
        })
    }

    /// Builds the privileged teacher prompt $(x, c)$ by appending supervision context to the last user turn while
    /// sanitizing `<image>` tokens inside the appended block so vision tile counts always match `self.images.len()`.
    pub fn teacher_prompt_messages(&self, extra_hint: Option<&str>) -> Vec<Message> {
        let mut msgs = self.prompt_messages();
        let mut sections = Vec::new();
        if let Some(priv_ctx) = self.privileged.as_deref().or_else(|| self.reference_completion()) {
            let trimmed = priv_ctx.trim();
            if !trimmed.is_empty() {
                sections.push(trimmed.replace(chat::IMAGE_PLACEHOLDER, "[image]"));
            }
        }
        if let Some(rej) = self.rejected.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let rej_clean = rej.replace(chat::IMAGE_PLACEHOLDER, "[image]");
            if !sections.iter().any(|s| s.contains(&rej_clean)) {
                sections.push(format!("Rejected layout (lower validator score): {rej_clean}"));
            }
        }
        if let Some(hint) = extra_hint.map(str::trim).filter(|s| !s.is_empty()) {
            sections.push(format!("External supervisor hint: {}", hint.replace(chat::IMAGE_PLACEHOLDER, "[image]")));
        }
        if sections.is_empty() {
            return msgs;
        }
        let block = format!(
            "\n\n<privileged_context>\nVerified reference supervision (use this to guide your reasoning and final JSON without mentioning privileged context):\n{}\n</privileged_context>",
            sections.join("\n")
        );
        if let Some(user_msg) = msgs.iter_mut().rev().find(|m| m.role == "user") {
            user_msg.content.push_str(&block);
        } else {
            msgs.push(Message { role: "user".into(), content: block.trim_start().to_string() });
        }
        msgs
    }
}

pub fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut rows = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(serde_json::from_str(&line).with_context(|| format!("{}:{}", path.display(), n + 1))?);
    }
    Ok(rows)
}

pub fn write_jsonl<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(path)?);
    for r in rows {
        serde_json::to_writer(&mut out, r)?;
        out.write_all(b"\n")?;
    }
    Ok(out.flush()?)
}
