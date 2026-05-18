# Gemma 4 E2B — Quantization Experiments

## TL;DR

- This folder is the public index for Gemma 4 E2B deploy-time quantization: the goal is an SFT'd 4-bit checkpoint under 4 GB with minimal PlantNet accuracy loss.
- The docs compare three routes: 4-bit-aware SFT, a CUDA/HF post-training path, and an MLX-native path that can produce the iOS artifact.
- The current recommended artifact comes from the MLX-native route; the CUDA/HF route remains useful as an independent accuracy and size cross-check.
- Start with `00-quantization-report-pub.md` for headline results, then use the route-specific files for reproduction details and failure analysis.

## Read First

| Read | Why |
|---|---|
| [`00-quantization-report-pub.md`](00-quantization-report-pub.md) | Headline result and ship candidate. |
| [`05-mlx-vlm-design.md`](05-mlx-vlm-design.md) | Why deploy artifacts must stay in the MLX/VLM model tree. |
| [`02-methods-and-eval.md`](02-methods-and-eval.md) | How methods are compared and what the metrics mean. |

## Detail / Reproduction

- [`B2-sft-results.md`](B2-sft-results.md): MLX deploy-candidate sweep.
- [`B2-sft-r8a8-13k-results.md`](B2-sft-r8a8-13k-results.md): later deploy-candidate matrix.
- [`B1-sft-results.md`](B1-sft-results.md): HF/CUDA reference sweep.
- [`04-calibration-data-design.md`](04-calibration-data-design.md): calibration data design.

## Failure Analysis

- [`B1-bnb-nf4-vision-collapse.md`](B1-bnb-nf4-vision-collapse.md): why quantizing the vision tower breaks plant behavior.
- [`B1-torchao-vs-gptqmodel.md`](B1-torchao-vs-gptqmodel.md): CUDA hybrid tooling decomposition.
- [`02b-mlx-torch-convert.md`](02b-mlx-torch-convert.md): HF to MLX conversion caveats.

## Historical / Planning

- [`00-quantization-roadmap.md`](00-quantization-roadmap.md)
- [`01-baselines.md`](01-baselines.md)
- [`A-baseline2-qlora-progress.md`](A-baseline2-qlora-progress.md)
- [`B2-research-spec.md`](B2-research-spec.md)
