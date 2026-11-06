# Construction Model Training

Standalone Rust training implementation using Candle for LiquidAI's
`LFM2.5-VL-1.6B`. This repository contains implementation code and method
references only; no datasets, model weights, prepared examples, or run outputs
were copied from ARQA's training repository.

The implementation provides:

- **LoRA supervised fine-tuning (SFT):** frozen base weights, configurable
  adapter targets, gradient accumulation, AdamW, warmup/cosine learning rate,
  global-norm clipping, evaluation, and resumable checkpoints.
- **JOLT / JOLT+:** separate or shared teacher/student LoRA adapters, grouped
  on-policy rollouts, privileged teacher context, outcome rewards, and KL-based
  teacher/student objectives. JOLT+ also applies student outcome-policy
  gradients.
- **Evaluation and inspection:** eval loss, parameter/layer summaries, and
  estimated trainable LoRA parameter counts.
- **Latent analysis:** hidden-state capture and PCA/UMAP-based experiments for
  layer structure, attention/MLP separation, positional geometry, probes,
  LoRA deltas, and vision separation.

## Important scope

The included model implementation is specific to the LFM2.5-VL architecture
(LFM2 text layers, SigLIP2 NaFlex vision encoder, and multimodal projector). It
is not a model-agnostic trainer. The code can fine-tune construction-related
vision-language examples, but an alternate base model requires its own model
loader, architecture, processor, and tokenizer integration.

Data and generated artifacts are intentionally external and ignored by Git:

- `data/text/datasets/<dataset>/{train,eval,dpo}.jsonl`
- `data/images/floorplans/cases.json` and referenced images (optional)
- `out/prepared/`, `out/runs/`, and `out/analysis/`

No example or private data is included. Supply only data you have the right to
use. The default root is this repository; use `--root <path>` to point the CLI
at a separate local data directory. Hugging Face model files are fetched by the
runtime into its cache and are not included here.

## Build and use

Requires Rust and network access to download model files from Hugging Face.
The model is approximately 1.6B parameters; this CPU-oriented implementation
can require substantial memory and time. No training or evaluation was run as
part of this source-only transfer.

```sh
cargo run --release -- prepare --root .
cargo run --release -- train --run-id construction-r01 --root .
cargo run --release -- jolt --run-id construction-j01 --root .
cargo run --release -- eval --root .
cargo run --release -- inspect
cargo run --release -- analyze --id analysis-r01 --root .
```

Use `cargo run --release -- --help` and each subcommand's `--help` for all
options. Training and analysis write under `out/`; input data is read from
`data/` unless another root is selected. Outputs and model weights are not
source-controlled.

## Research references

The source training repository contained third-party paper PDFs. Their files
are not redistributed here; this list provides bibliographic identifiers and
links to the original records instead.

- Zha et al., *How to Enable LLM with 3D Capacity? A Survey of Spatial
  Reasoning in LLM*, arXiv:2504.05786:
  <https://arxiv.org/abs/2504.05786>
- Co-layout research paper, arXiv:2511.12474:
  <https://arxiv.org/abs/2511.12474>
- *Visualizing LLM Latent Space Geometry Through Dimensionality Reduction*,
  arXiv:2511.21594:
  <https://arxiv.org/abs/2511.21594>
- Additional training/research references from the source repository:
  [arXiv:2605.06832](https://arxiv.org/abs/2605.06832) and
  [arXiv:2610.11370](https://arxiv.org/abs/2610.11370).

## License

The implementation in this repository is licensed under the MIT License; see
[`LICENSE`](LICENSE). Model weights and data are not included and remain under
their respective licenses and terms.
