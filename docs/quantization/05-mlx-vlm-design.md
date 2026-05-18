# MLX stack design — mlx-vlm is the substrate; mlx-lm only contributes (buggy) quant cores

## TL;DR

- This doc gives the MLX mental model: deployable Gemma 4 artifacts must be in mlx-vlm format because that is what the iOS runtime matches.
- Do not use mlx-lm's Gemma 4 model class for forward passes; it diverges from the deploy model on RMSNorm, projection layers, KV-shared K/V, and audio handling.
- mlx-lm quantization cores can still be reused if the model is loaded and saved through mlx-vlm, preserving the correct model tree.
- Treat mlx-lm's GPTQ, AWQ, DWQ, and dynamic-quant methods as research paths because each has Gemma 4-specific bugs or validation gaps in this stack.

## The Rule

All deployable Gemma 4 artifacts must be produced or saved in the `mlx-vlm`
format. The iOS runtime follows the same VLM-side model tree. A checkpoint that
looks valid under `mlx-lm` can still be the wrong artifact for Trailogy.

## Why This Matters

During quantization debugging, several failures looked like quantization
algorithm failures. The deeper issue was model-tree mismatch. `mlx-lm` and
`mlx-vlm` do not represent Gemma 4 identically.

| Area | Why `mlx-lm` is risky here |
|---|---|
| RMSNorm | Uses different norm behavior from the VLM/iOS path. |
| Projection layers | Represents some projection layers differently. |
| KV-shared layers | Drops or expects different K/V tensors. |
| Audio handling | Config defaults can allocate unused audio paths. |

Those differences are enough to make a quantized artifact misleading even if it
loads somewhere.

## Clean Mental Model

| Use case | Library to trust |
|---|---|
| Model class / forward pass / load path | `mlx-vlm` |
| iOS runtime compatibility | `mlx-swift-lm` VLM path, aligned with `mlx-vlm` |
| Basic MLX quantized artifact | `mlx_vlm.convert -q` |
| Research quant cores | `mlx_lm.quant.*`, only when applied to a tree loaded by `mlx-vlm` |
| Gemma 4 validation with `mlx_lm.load` | Do not use for deploy claims. |

## Safe Paths

### Fast deploy path

```text
bf16 multimodal checkpoint
  -> mlx_vlm.convert -q
  -> MLX/VLM artifact
  -> iOS
```

Use this for baseline deployable artifacts.

### Hybrid research path

```text
bf16 multimodal checkpoint
  -> load with mlx_vlm
  -> apply selected mlx_lm quant core to compatible language subtree
  -> save with mlx_vlm utilities
  -> MLX/VLM artifact
```

This path lets us reuse quantization research code without accepting the wrong
Gemma 4 forward pass.

## Unsafe Path

```text
bf16 checkpoint
  -> mlx_lm.load / mlx_lm convert
  -> quantized output
  -> treat as iOS-compatible
```

Do not use this path for Trailogy deployment claims. It answers a different
question: how the model behaves under the `mlx-lm` Gemma 4 implementation.

## Practical Consequences

- A quant method is not a ship candidate unless the saved artifact loads through
  the VLM path.
- `mlx_lm` quantization cores may still be useful, but only as functions applied
  to a VLM-loaded model tree.
- Result tables must label whether a row is deployable or only a reference.
- Bugs that appear to be GPTQ/AWQ/DWQ failures may actually be model-tree or
  loader failures.

## Recommended Path By Goal

| Goal | Path |
|---|---|
| Fast iOS artifact | `mlx_vlm.convert -q`, then eval. |
| Better quality under size budget | MLX/VLM affine quant plus EoRA. |
| CUDA quality cross-check | HF/CUDA GPTQ route, clearly labeled non-iOS unless bridged. |
| Research quant algorithm comparison | Hybrid VLM-loaded path, not pure `mlx_lm.load`. |

## Related Docs

- Quantization overview: [`00-quantization-report-pub.md`](00-quantization-report-pub.md)
- Methods and eval: [`02-methods-and-eval.md`](02-methods-and-eval.md)
- Quantization postmortem: [`../general/15-postmortems.md`](../general/15-postmortems.md)
- Full MLX results: [`B2-sft-results.md`](B2-sft-results.md)
