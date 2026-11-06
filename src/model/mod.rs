//! LFM2.5-VL in Candle: a SigLIP2 NaFlex vision tower, a pixel-unshuffle projector and the LFM2 hybrid
//! (short-conv + attention) text backbone. Written for full-sequence training and analysis, so there is no KV or conv
//! cache: every forward sees the whole sequence and the conv is computed causally over it.
pub mod norm;
pub mod projector;
pub mod text;
pub mod vision;
pub mod vl;

use candle_core::{Result, Tensor};

/// Where in a block a hidden state was read. The paper (arXiv 2511.21594) taps the same six places inside a
/// pre-norm Transformer block; for LFM2 the "operator" is attention on some layers and the short conv on the others.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Point {
    BlockInput,
    OperatorNorm,
    OperatorOut,
    ResidualMid,
    FfnNorm,
    MlpOut,
    BlockOutput,
}

impl Point {
    pub const ALL: [Point; 7] = [
        Point::BlockInput,
        Point::OperatorNorm,
        Point::OperatorOut,
        Point::ResidualMid,
        Point::FfnNorm,
        Point::MlpOut,
        Point::BlockOutput,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Point::BlockInput => "block_input",
            Point::OperatorNorm => "operator_norm",
            Point::OperatorOut => "operator_out",
            Point::ResidualMid => "residual_mid",
            Point::FfnNorm => "ffn_norm",
            Point::MlpOut => "mlp_out",
            Point::BlockOutput => "block_output",
        }
    }
}

/// Receives the hidden states of one forward pass. `x` is (1, T, H) for text and (T, H) for vision; the tower
/// reports which one it is.
pub trait Probe {
    fn wants(&self, point: Point) -> bool;
    fn record(&mut self, tower: Tower, layer: usize, point: Point, x: &Tensor) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tower {
    Text,
    Vision,
}

/// Calls the probe only when one is attached and wants this point, so the training path pays nothing.
pub(crate) fn tap(probe: &mut Option<&mut dyn Probe>, tower: Tower, layer: usize, point: Point, x: &Tensor) -> Result<()> {
    if let Some(p) = probe.as_deref_mut() {
        if p.wants(point) {
            p.record(tower, layer, point, x)?;
        }
    }
    Ok(())
}
