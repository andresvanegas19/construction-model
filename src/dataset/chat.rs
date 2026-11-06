//! LFM2 chat template + image token expansion + label masking. The template is
//! `<|startoftext|><|im_start|>{role}\n{content}<|im_end|>\n...`; each `<image>` in a message's text becomes
//! `<|image_start|>`, the tile markers and image tokens the processor emits, then `<|image_end|>`. Text segments are
//! tokenized one by one without special tokens and the special ids are spliced in between: the HF processor tokenizes
//! the whole expanded string, which can merge a character across a segment boundary differently (rare; documented).
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::config::ImageSettings;
use crate::dataset::image::{self, Processed, Tile};
use crate::model::vl::IGNORE;

pub const IMAGE_PLACEHOLDER: &str = "<image>";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct Encoded {
    pub ids: Vec<u32>,
    /// Same length as `ids`; IGNORE everywhere but assistant content and its closing `<|im_end|>`.
    pub labels: Vec<i64>,
    pub tiles: Vec<Tile>,
}

impl Encoded {
    pub fn trained_tokens(&self) -> usize {
        self.labels.iter().filter(|&&l| l != IGNORE).count()
    }
}

struct Specials {
    bos: u32,
    im_start: u32,
    im_end: u32,
    image: u32,
    image_start: u32,
    image_end: u32,
    thumbnail: u32,
}

pub struct ChatEncoder<'a> {
    tok: &'a Tokenizer,
    specials: Specials,
    pub settings: ImageSettings,
}

fn id(tok: &Tokenizer, s: &str) -> Result<u32> {
    tok.token_to_id(s).ok_or_else(|| anyhow!("tokenizer has no {s}"))
}

impl<'a> ChatEncoder<'a> {
    pub fn new(tok: &'a Tokenizer, settings: ImageSettings) -> Result<Self> {
        let specials = Specials {
            bos: id(tok, "<|startoftext|>")?,
            im_start: id(tok, "<|im_start|>")?,
            im_end: id(tok, "<|im_end|>")?,
            image: id(tok, IMAGE_PLACEHOLDER)?,
            image_start: id(tok, "<|image_start|>")?,
            image_end: id(tok, "<|image_end|>")?,
            thumbnail: id(tok, "<|img_thumbnail|>")?,
        };
        Ok(Self { tok, specials, settings })
    }

    pub fn image_token(&self) -> u32 {
        self.specials.image
    }

    pub fn bos(&self) -> u32 {
        self.specials.bos
    }

    pub fn im_end(&self) -> u32 {
        self.specials.im_end
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids, true).unwrap_or_default()
    }

    pub fn text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(s, false).map_err(|e| anyhow!("tokenize: {e}"))?.get_ids().to_vec())
    }

    /// `images` are resolved against `root`, consumed in placeholder order. Fails when the counts differ, so a
    /// malformed row is skipped instead of training on misaligned features.
    pub fn encode(&self, messages: &[Message], images: &[String], root: &Path) -> Result<Encoded> {
        let mut out = Encoded { ids: vec![self.specials.bos], labels: vec![IGNORE], tiles: vec![] };
        let mut next_image = 0;
        for m in messages {
            self.push_turn(&mut out, m, m.role == "assistant", images, &mut next_image, root)?;
        }
        if next_image != images.len() {
            bail!("{} images but {next_image} <image> placeholders", images.len());
        }
        Ok(out)
    }

    /// Encodes the prompt turns before the assistant answer and appends `<|im_start|>assistant\n` with `IGNORE`
    /// labels, ready for autoregressive rollout (`Lfm2Vl::generate`) or cross-context splicing (`splice_completion`).
    pub fn encode_prompt(&self, messages: &[Message], images: &[String], root: &Path) -> Result<Encoded> {
        let mut out = Encoded { ids: vec![self.specials.bos], labels: vec![IGNORE], tiles: vec![] };
        let mut next_image = 0;
        for m in messages {
            self.push_turn(&mut out, m, false, images, &mut next_image, root)?;
        }
        if next_image != images.len() {
            bail!("{} images but {next_image} <image> placeholders", images.len());
        }
        out.push(&[self.specials.im_start], false);
        out.push(&self.text("assistant\n")?, false);
        Ok(out)
    }

    /// Splices `completion_ids` onto an already-encoded prompt (student $x$ or privileged teacher $(x, c)$) so
    /// `Lfm2Vl::completion_logits` extracts aligned token distributions over the exact same completion positions.
    pub fn splice_completion(&self, prompt: &Encoded, completion_ids: &[u32]) -> Encoded {
        let mut out = Encoded {
            ids: prompt.ids.clone(),
            labels: vec![IGNORE; prompt.ids.len()],
            tiles: prompt.tiles.clone(),
        };
        out.push(completion_ids, true);
        if completion_ids.last().copied() != Some(self.specials.im_end) {
            out.push(&[self.specials.im_end], true);
        }
        if let Ok(nl) = self.text("\n") {
            out.push(&nl, false);
        }
        out
    }

    fn push_turn(
        &self,
        out: &mut Encoded,
        m: &Message,
        train: bool,
        images: &[String],
        next_image: &mut usize,
        root: &Path,
    ) -> Result<()> {
        out.push(&[self.specials.im_start], false);
        out.push(&self.text(&format!("{}\n", m.role))?, false);
        for (k, part) in m.content.split(IMAGE_PLACEHOLDER).enumerate() {
            if k > 0 {
                let path = images.get(*next_image).with_context(|| "more <image> placeholders than images")?;
                *next_image += 1;
                let processed = image::load(&root.join(path), &self.settings)?;
                self.push_image(out, processed);
            }
            if !part.is_empty() {
                out.push(&self.text(part)?, train);
            }
        }
        out.push(&[self.specials.im_end], train);
        out.push(&self.text("\n")?, false);
        Ok(())
    }

    fn push_image(&self, out: &mut Encoded, p: Processed) {
        let s = &self.specials;
        let d = self.settings.downsample;
        out.push(&[s.image_start], false);
        match p.grid {
            Some((rows, cols)) => {
                for (k, tile) in p.tiles.iter().enumerate() {
                    if k < rows * cols {
                        let marker = format!("<|img_row_{}_col_{}|>", k / cols + 1, k % cols + 1);
                        out.push(&self.text(&marker).unwrap_or_default(), false);
                    } else {
                        out.push(&[s.thumbnail], false);
                    }
                    out.push(&vec![s.image; tile.tokens(d)], false);
                }
            }
            None => {
                let n = p.tiles[0].tokens(d);
                out.push(&vec![s.image; n], false);
            }
        }
        out.push(&[s.image_end], false);
        out.tiles.extend(p.tiles);
    }
}

impl Encoded {
    fn push(&mut self, ids: &[u32], train: bool) {
        self.ids.extend_from_slice(ids);
        self.labels.extend(ids.iter().map(|&t| if train { t as i64 } else { IGNORE }));
    }

    /// The same sample without its leading BOS (the `first_position` experiment's control run).
    pub fn without_bos(&self) -> Encoded {
        Encoded { ids: self.ids[1..].to_vec(), labels: self.labels[1..].to_vec(), tiles: self.tiles.clone() }
    }
}
