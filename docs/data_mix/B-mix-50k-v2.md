# B — mix-50k v2 (current production canonical) + mix-100k sibling

## TL;DR

- The current 50K-row mix contains about 44% plant examples, 30% general image examples, 15% text-only chat, and 10% refusal examples.
- This version replaced the blocked general vision source with a dataset format that streams reliably.
- Text-only records are handled with `image: null`, so they no longer need fake placeholder images.
- Separate validation files and retained checkpoints make it easier to notice if one behavior improves while another gets worse.
- A 100K-row sibling config keeps the same bucket idea for larger training runs.

## What This Mix Is For

This is the production mixed corpus used to keep the plant fine-tune from
turning Gemma into a plant-only model. The mix gives the model enough PlantNet
signal to learn plant identification while repeatedly reminding it how to handle
general image questions, text chat, refusals, and offline limitations.

## Composition

Source config: `src/data_mix/configs/mix-50k.yaml`.

| Bucket | Train records | Val records | Purpose |
|---|---:|---:|---|
| Plant | 22,000 | 1,100 | Learn plant ID and descriptions. |
| LLaVA/general VQA | 15,000 | 750 | Preserve general visual reasoning. |
| smoltalk/text-only | 7,500 | 375 | Preserve text assistant behavior. |
| Negative/non-plant | 5,000 | 250 | Refuse plant ID on non-plant images. |
| offline_qa | 38 | 4 | Teach offline persona and live-info limits. |

Total training rows: `50,038`. The extra 38 rows are the offline persona corpus
and are intentionally outside the main percentage ratio.

## Why v2 Replaced v1

| Change | Reason |
|---|---|
| Replaced the original general-vision source with LLaVA-style data. | The previous source was blocked by streaming/tar issues; the new source streams reliably. |
| Added `image: null` text-only records. | Removes fake placeholder images and lets text batches skip the vision tower. |
| Split validation by bucket. | A single mean loss can hide catastrophic forgetting. |
| Kept checkpoints. | The best checkpoint may be mid-training, especially when buckets trade off. |

## Why Separate Validation Files Matter

The failure mode we care about is not just higher total loss. It is one behavior
improving while another gets worse. Separate validation files expose that:

```text
eval_plant_loss       -> plant identification
eval_nonplant_loss    -> general VQA + text chat
eval_negative_loss    -> non-plant refusal
eval_offline_qa_loss  -> offline persona
```

If `eval_plant_loss` falls while `eval_nonplant_loss` rises, the model is
becoming a better classifier and a worse assistant.

## How The Finetune Uses It

The finetune config points to `train.jsonl` plus the per-bucket validation
files. It also enables the camera-state prefixes:

```yaml
data:
  prompt_prefixes:
    camera_on:  "[camera=on] "
    camera_off: "[camera=off] "
```

The prefix is dispatched by image presence, not by bucket name. This keeps the
training contract aligned with the iOS runtime: the app always knows whether a
photo is present.

## mix-100k Sibling

`mix-100k.yaml` doubles the same idea for larger runs. It uses the same bucket
structure and keeps the offline persona corpus unchanged. The main difference
is the larger plant budget and looser per-class cap.

Use `mix-50k` for fast iteration and reviewer-facing reproducibility. Use
`mix-100k` only when the extra training budget is justified.

## Reproduction Notes

The build is driven by `src/data_mix/scripts/build_mix.sh` and
`src/data_mix/src/mix.py`. Storage roots are provided through environment
variables so cached datasets and generated images can live outside the repo.

The expected outputs are:

```text
train.jsonl
val_plant.jsonl
val_nonplant.jsonl
val_negative.jsonl
val_offline_qa.jsonl
build_report.json
```

## Related Docs

- Bucket contracts: [`02-bucket-design.md`](02-bucket-design.md)
- Prefix mechanism: [`01-data-prefix.md`](01-data-prefix.md)
- Build orchestrator: [`03-orchestrator-and-build.md`](03-orchestrator-and-build.md)
- Final SFT recipe: [`../finetune/03-anti-forgetting-and-final-recipe.md`](../finetune/03-anti-forgetting-and-final-recipe.md)
