# Postmortems

## TL;DR

- This is the follow-up doc for the writeup's "Silent PEFT loading failure" challenge: training looked successful, but the saved adapter reloaded as a partial adapter and PlantNet eval stayed at base-model behavior.
- The root cause was a Gemma 4 model-layout mismatch across `transformers` versions; PEFT silently dropped LoRA tensors that no longer matched modules in the newer layout.
- The same pattern appeared elsewhere: loaders, eval paths, package versions, and runtime backends accepted plausible inputs while silently measuring or running the wrong thing.
- The fix is not just "upgrade packages"; it is to add tripwires for save/reload tensor inventories, train/eval parity, adapter application, generation settings, and backend/runtime compatibility.

## Why This Doc Exists

The dangerous failures in this project were not loud crashes. They were runs
that completed, wrote metrics, and looked plausible while measuring the wrong
thing. This document keeps only the reviewer-relevant parts: symptom, cause,
fix, and prevention.

## 1. Cross-Backend Evaluation Drift

**Symptom.** The same quantized artifact scored very differently across Mac MLX
and Linux/CUDA MLX runs.

**Root cause.** Backend wheels and kernels did not have identical bug-fix
coverage. Quantized matmul behavior differed enough to corrupt generation under
some inputs.

**Fix.** Use the authoritative deployment backend for reporting and use other
backends only for within-backend comparisons unless parity has been verified.

**Tripwire.** Evaluate a known-good artifact on every backend before trusting a
new sweep on that backend.

## 2. Silent PEFT Loading Failure

**Symptom.** Training appeared successful: in-memory loss was near zero on
small overfit sets. After saving and reloading the adapter, PlantNet evaluation
stayed at 0% and the model behaved like the base model.

**Root cause.** Gemma 4's module layout changed between `transformers` versions.
Adapters trained on the older layout included LoRA tensors for K/V projection
modules that no longer existed as standalone modules in the newer layout. PEFT
loaded the matching tensors and silently dropped the rest.

**Why it mattered.** This invalidated evaluation without throwing an error. The
pipeline could train a real adapter, save it, reload a partial adapter, and then
report numbers for the wrong model.

**Fix.** Use the corrected Gemma 4 layout consistently and retrain. Add
save/reload tripwires that compare in-memory PEFT state with the saved
`adapter_model.safetensors` tensor inventory.

**Tripwire.** Training fails if a saved adapter reload would lose tensors or if
tensor values differ unexpectedly.

## 3. Quantizing Through The Wrong MLX Model Tree

**Symptom.** Several MLX quantization methods produced NaNs, `<pad>` spam, or
artifacts that could not load through the iOS-compatible VLM path.

**Root cause.** `mlx_lm` and `mlx_vlm` do not construct identical Gemma 4 model
trees. The differences include projection layers, RMSNorm behavior,
KV-shared-layer weights, and audio handling. A quantized checkpoint produced by
the wrong tree is not a reliable proxy for the iOS runtime.

**Fix.** Treat `mlx_vlm` as the deploy substrate. If an `mlx_lm` quantization
core is reused, load and save through the `mlx_vlm` tree and apply the quant
core only to the compatible subtree.

**Tripwire.** Deploy artifacts must load through the VLM path and preserve
vision weights. See [`../quantization/05-mlx-vlm-design.md`](../quantization/05-mlx-vlm-design.md).

## 4. Eval Pipeline Ignored The Adapter

**Symptom.** Multiple sweep results matched the base model fingerprint even
though they claimed to evaluate different adapters.

**Root cause.** One loader path accepted an adapter argument but never applied
it. The output was structurally valid because the base model loaded correctly.

**Fix.** Apply PEFT explicitly after loading and assert that merged adapters do
not leave `lora_*` parameters behind.

**Tripwire.** Refuse to publish sweep summaries if all fresh evals match the
base model fingerprint.

## 5. Non-Deterministic Eval Generation

**Symptom.** Repeated evals of the same checkpoint produced avoidable jitter.

**Root cause.** The HF generation path inherited sampling defaults from
`generation_config.json` instead of forcing greedy decode.

**Fix.** Eval uses explicit greedy beam-1 generation and pins RNGs where the
backend allows it.

**Tripwire.** Eval code sets generation arguments directly rather than relying
on model config defaults.

## Common Pattern

Most failures followed this shape:

1. Caller supplied the right input.
2. Receiver ignored, transformed, or partially loaded it.
3. No error was raised.
4. Metrics were produced anyway.

The defense is not better memory. The defense is a code-level invariant at the
boundary where silent drift can enter: tensor inventories, loader parity,
dataset manifests, generation settings, and adapter-merge assertions.

## Related Docs

- Package/version notes: [`14-package-versions-and-known-bugs.md`](14-package-versions-and-known-bugs.md)
- Eval setup and caveats: [`10-eval-setup.md`](10-eval-setup.md)
- MLX/VLM design: [`../quantization/05-mlx-vlm-design.md`](../quantization/05-mlx-vlm-design.md)
