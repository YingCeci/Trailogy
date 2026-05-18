# `mattmireles/gemma-tuner-multimodal` — investigation findings

## TLDR

Investigation of `mattmireles/gemma-tuner-multimodal` as a potential reference for quantization code. Conclusion: no quantization code in the repo (only false-positive matches on "quantitative" / "quantify" in profiler comments). Repo is a PEFT/LoRA tuner with Apple-Silicon MPS support, GCS/BigQuery streaming, and a live training visualizer — useful as UX inspiration only, not a source for quantization implementations. Decision: do not adopt.

## Investigation Summary

**No quantization code.** Skip for this thread.

Repo is a PEFT/LoRA tuner with Apple-Silicon (MPS) support, GCS/BigQuery
streaming, and a live training visualizer. Useful as inspiration for
UX (wizard, real-time loss/attention dashboard) but **not a source for
quantization implementations**.

## What was checked

Searched the codebase for `quant`, `gptq`, `qat`, `awq` patterns
across all Python sources under:

- `gemma_tuner/scripts/`
- `gemma_tuner/core/`
- `gemma_tuner/models/gemma/`

Two matches found — both **false positives**:

- `gemma_tuner/scripts/gemma_preflight.py:175` — "Quantified benefits
  of proper configuration for user guidance" (a comment about
  *quantitative* benefits, not quantization).
- `gemma_tuner/scripts/gemma_profiler.py:19,43,82,181` — variations
  of "provides quantitative data", "quantifies Apple Silicon MPS
  acceleration benefits". All English-meaning of "quantify", no
  quantization-of-weights code.

No imports of `auto-gptq`, `gptqmodel`, `bitsandbytes`, `optimum`,
`mlx_vlm` (the quant CLI), `awq`, or `quanto`.

## What the repo IS good for

If we end up needing any of the following post-deadline, this repo
is worth a closer look:

- **Apple Silicon (MPS) training** — they handle MPS-specific quirks
  (fp16 → bf16 promotion, MPS allocator behavior, etc.) that we have
  not addressed.
- **GCS / BigQuery streaming dataloader** — `gemma_tuner/core/runs.py`
  + `gemma_tuner/data/`. Useful if TreeOfLife-200M ever lands as a
  scaling target.
- **Live training visualizer** — better than wandb for some debugging
  use cases (real-time attention heatmaps).
- **PEFT LoRA wiring on multimodal Gemma** — comparable to our own
  `finetune/src/finetune.py`. May have caught traps we haven't.

## Decision

Mark as "investigated, no quantization signal." Do not add as a
dependency or template for the quantization branch.

If a teammate has free cycles post-deadline and is curious about
their MPS or streaming-data work, this is a reasonable repo to read.
