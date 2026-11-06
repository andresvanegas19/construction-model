//! The model's `config.json` and `preprocessor_config.json`. Every field has a default equal to the published
//! LFM2.5-VL-1.6B value, so a trimmed or renamed key in a future revision degrades to the known model instead of
//! failing a long-running training job at load time.
use serde::Deserialize;

pub const MODEL_ID: &str = "LiquidAI/LFM2.5-VL-1.6B";

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VlConfig {
    pub image_token_id: u32,
    pub downsample_factor: usize,
    pub encoder_patch_size: usize,
    pub tile_size: usize,
    pub min_image_tokens: usize,
    pub max_image_tokens: usize,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub use_thumbnail: bool,
    pub do_image_splitting: bool,
    pub projector_hidden_size: usize,
    pub projector_bias: bool,
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
}

impl Default for VlConfig {
    fn default() -> Self {
        Self {
            image_token_id: 396,
            downsample_factor: 2,
            encoder_patch_size: 16,
            tile_size: 512,
            min_image_tokens: 64,
            max_image_tokens: 256,
            min_tiles: 2,
            max_tiles: 10,
            use_thumbnail: true,
            do_image_splitting: true,
            projector_hidden_size: 2048,
            projector_bias: true,
            text_config: TextConfig::default(),
            vision_config: VisionConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    #[serde(alias = "conv_L_cache")]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub layer_types: Vec<String>,
}

impl Default for TextConfig {
    fn default() -> Self {
        // c = short conv, a = full attention: the published 16-layer pattern.
        let layer_types = "ccaccaccacacacac"
            .chars()
            .map(|c| if c == 'a' { "full_attention".to_string() } else { "conv".to_string() })
            .collect();
        Self {
            hidden_size: 2048,
            num_hidden_layers: 16,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            norm_eps: 1e-5,
            rope_theta: 1_000_000.0,
            vocab_size: 65536,
            conv_l_cache: 3,
            conv_bias: false,
            eos_token_id: 7,
            bos_token_id: 1,
            layer_types,
        }
    }
}

impl TextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        self.layer_types.get(layer).is_some_and(|t| t.contains("attention"))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_eps: f64,
    pub num_patches: usize,
    pub patch_size: usize,
    pub num_channels: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            layer_norm_eps: 1e-6,
            num_patches: 256,
            patch_size: 16,
            num_channels: 3,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ProcessorConfig {
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub rescale_factor: f32,
    pub max_num_patches: usize,
    /// The HF processor keeps an image as one tile up to this multiple of the patch budget before splitting it.
    pub max_pixels_tolerance: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            image_mean: vec![0.5; 3],
            image_std: vec![0.5; 3],
            rescale_factor: 1.0 / 255.0,
            max_num_patches: 1024,
            max_pixels_tolerance: 2.0,
        }
    }
}

/// Everything image preprocessing needs, merged from both files plus the CLI's `--max-tiles` cap (CPU memory: each
/// tile is 1024 patches through 27 vision layers and 256 more text tokens).
#[derive(Debug, Clone)]
pub struct ImageSettings {
    pub patch: usize,
    pub downsample: usize,
    pub tile: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub min_tiles: usize,
    pub max_tiles: usize,
    pub thumbnail: bool,
    pub split: bool,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub rescale: f32,
    pub tolerance: f32,
}

impl ImageSettings {
    pub fn new(vl: &VlConfig, p: &ProcessorConfig, max_tiles: Option<usize>) -> Self {
        let pick = |v: &[f32], i: usize| v.get(i).copied().unwrap_or(0.5);
        let max_tiles = max_tiles.map_or(vl.max_tiles, |m| m.min(vl.max_tiles));
        Self {
            patch: vl.encoder_patch_size,
            downsample: vl.downsample_factor,
            tile: vl.tile_size,
            min_tokens: vl.min_image_tokens,
            max_tokens: vl.max_image_tokens,
            min_tiles: vl.min_tiles.min(max_tiles),
            max_tiles,
            thumbnail: vl.use_thumbnail,
            split: vl.do_image_splitting && max_tiles >= 2,
            mean: [pick(&p.image_mean, 0), pick(&p.image_mean, 1), pick(&p.image_mean, 2)],
            std: [pick(&p.image_std, 0), pick(&p.image_std, 1), pick(&p.image_std, 2)],
            rescale: p.rescale_factor,
            tolerance: p.max_pixels_tolerance,
        }
    }

    /// The single-tile pixel budget: `max_image_tokens` tokens of `downsample²` patches each.
    pub fn max_pixels(&self) -> usize {
        self.max_tokens * (self.patch * self.downsample).pow(2)
    }

    pub fn min_pixels(&self) -> usize {
        self.min_tokens * (self.patch * self.downsample).pow(2)
    }
}
