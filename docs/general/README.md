# general/ — cross-cutting docs

## TL;DR

- This directory collects cross-cutting documentation for architecture, runtime behavior, evaluation, package bugs, postmortems, and final model results.
- The reading-order table maps each writeup section to the supporting technical notes.
- Evaluation numbers across docs can drift because sample counts, datasets, prompts, and loaders changed during development.
- The docs are organized by architecture, iOS runtime patterns, development timelines, eval parity audits, debugging notes, and the final shipped-model evaluation.

## Read First

| If you want to understand... | Read |
|---|---|
| What the product does offline | [`02-architecture-ios-app.md`](02-architecture-ios-app.md) |
| How the model pipeline connects data, SFT, quantization, and iOS | [`01-architecture-model-pipeline.md`](01-architecture-model-pipeline.md) |
| Why memory management shaped the app | [`03-memory-management.md`](03-memory-management.md) |
| What broke silently | [`15-postmortems.md`](15-postmortems.md) |
| Which package/version bugs mattered | [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md) |
| How to interpret final model numbers | [`16-final-model-eval.md`](16-final-model-eval.md) and [`10-eval-setup.md`](10-eval-setup.md) |

## Appendix Docs

These are useful for reproduction or debugging, but not required for first-pass
review:

- iOS build/dependency mechanics: [`04-xcode-build-and-deps.md`](04-xcode-build-and-deps.md)
- RAG runtime details: [`05-rag-runtime.md`](05-rag-runtime.md)
- Background/Metal lifecycle: [`06-scenephase-metal-background.md`](06-scenephase-metal-background.md)
- iOS development timeline: [`09-dev-timeline-ios.md`](09-dev-timeline-ios.md)
- Cross-platform parity audits: [`11-cuda-vs-mlx-eval-parity.md`](11-cuda-vs-mlx-eval-parity.md), [`12-mlx-vlm-vs-hf-kv-sharing.md`](12-mlx-vlm-vs-hf-kv-sharing.md), [`13-mlx-vision-input-parity.md`](13-mlx-vision-input-parity.md)
- Audio / TTS rationale and streaming: [`17-audio-and-tts.md`](17-audio-and-tts.md)

## Important Caveat

Evaluation numbers across docs are not always apples-to-apples. Sample counts,
splits, loaders, and prompt shapes changed during development. Use
[`10-eval-setup.md`](10-eval-setup.md) before comparing numbers across phases.
