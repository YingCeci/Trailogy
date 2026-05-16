#!/usr/bin/env bash
# Swap the bundled Gemma 4 E2B (stock mlx-community 4-bit) with the
# finetune at TimS-ml/gemma-4-E2B/mlx_vlm_g128_sft_aug_enwiki on
# HuggingFace. The folder name decodes as:
#   mlx_vlm   → MLX-format VLM checkpoint (drop-in for VLMModelFactory)
#   g128      → group_size=128 quantization
#   sft       → supervised finetune
#   aug_enwiki → augmented with enwiki training data
#
# The repo is gated, so an HF token with read access is required. Pass
# it via env var (NOT as an arg, so it doesn't end up in shell history
# or process listings):
#
#   HF_TOKEN=hf_xxx bash scripts/fetch-gemma-finetune.sh
#
# Behavior:
#   1. If Models/Gemma/ contains the stock model and Models/Gemma.stock/
#      doesn't exist, MOVE Models/Gemma → Models/Gemma.stock (one-shot
#      backup of the original).
#   2. Create a fresh Models/Gemma/ and pull every file from the
#      finetune subfolder into it (flattens the HF subfolder prefix).
#   3. Apply the same processor_config.json patch fetch-gemma.sh uses:
#      hoist size/mean/std/do_normalize from image_processor to top
#      level, force size to 960x672 for Gemma 4's trained pooler.
#
# To restore the stock model later:
#   rm -rf HikeCompanion/Resources/Models/Gemma
#   mv HikeCompanion/Resources/Models/Gemma.stock HikeCompanion/Resources/Models/Gemma
#   bash scripts/generate-project.sh && rebuild

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: set HF_TOKEN=hf_... in your environment first." >&2
  exit 1
fi

REPO="TimS-ml/gemma-4-E2B"
SUBFOLDER="mlx_vlm_g128_sft_aug_enwiki"
API="https://huggingface.co/api/models/${REPO}/tree/main/${SUBFOLDER}"
RESOLVE="https://huggingface.co/${REPO}/resolve/main/${SUBFOLDER}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/HikeCompanion/Resources/Models/Gemma"
BACKUP="$ROOT/HikeCompanion/Resources/Models/Gemma.stock"

# --- one-shot backup of the stock model ------------------------------
if [[ -d "$DEST" && ! -d "$BACKUP" ]]; then
  # Heuristic: if the existing Gemma/ has the stock config (3 GB
  # safetensors, not 3.4 GB), back it up. We don't actually verify the
  # hash — just confirm there's something there worth saving.
  if [[ -f "$DEST/config.json" ]]; then
    echo "==> Backing up current Gemma/ → Gemma.stock/"
    mv "$DEST" "$BACKUP"
  fi
fi

mkdir -p "$DEST"

# --- list files under the subfolder ----------------------------------
echo "==> Listing files at $REPO/$SUBFOLDER ..."
LIST=$(curl -fsSL -H "Authorization: Bearer $HF_TOKEN" "$API")
FILES=$(echo "$LIST" | python3 -c '
import json, sys
items = json.load(sys.stdin)
for it in items:
    if it.get("type") == "file":
        print(it["path"])
')

if [[ -z "$FILES" ]]; then
  echo "ERROR: HF API returned no files for $REPO/$SUBFOLDER" >&2
  exit 1
fi

echo "==> $(echo "$FILES" | wc -l | tr -d ' ') files to fetch"

# --- download ---------------------------------------------------------
# Files come back with the subfolder prefix in their `path` (e.g.
# "mlx_vlm_g128_sft_aug_enwiki/config.json"); strip the prefix so we
# land flat in Models/Gemma/.
while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  basename="${file#${SUBFOLDER}/}"

  # Skip eval artifacts — they're not needed at runtime and inflate
  # the bundle.
  case "$basename" in
    eval.json|eval_per_sample.json|README.md)
      echo "  skip $basename (eval/readme artifact)"
      continue
      ;;
  esac

  out="$DEST/$basename"
  if [[ -f "$out" ]]; then
    sz=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000 ]]; then
      echo "  skip $basename ($(du -h "$out" | cut -f1))"
      continue
    fi
  fi
  echo "==> $basename"
  curl -fL -H "Authorization: Bearer $HF_TOKEN" --progress-bar \
       "$RESOLVE/$basename" -o "$out"
done <<< "$FILES"

# --- apply processor_config patch ------------------------------------
# Identical patch to fetch-gemma.sh — see that script's comment block
# for the full rationale (mlx-swift-lm's Gemma4ProcessorConfiguration
# expects flat fields; 960x672 matches the trained pooler grid).
echo ""
echo "==> Patching processor_config.json (hoist + 960x672) ..."
TRAINED_SIZE='{"height": 960, "width": 672}'
PCFG="$DEST/processor_config.json"
if [[ -f "$PCFG" ]]; then
  python3 - "$PCFG" "$TRAINED_SIZE" <<'PY'
import json, sys
p = sys.argv[1]
trained_size = json.loads(sys.argv[2])
with open(p) as f:
    cfg = json.load(f)
ip = cfg.get("image_processor", {})
patch = {
    "do_normalize": ip.get("do_normalize", False),
    "image_mean":   ip.get("image_mean",   [0.0, 0.0, 0.0]),
    "image_std":    ip.get("image_std",    [1.0, 1.0, 1.0]),
    "size":         trained_size,
}
changed = []
for k, v in patch.items():
    if cfg.get(k) != v:
        cfg[k] = v
        changed.append(k)
if ip and ip.get("size") != trained_size:
    ip["size"] = trained_size
    changed.append("image_processor.size")
if changed:
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    print("   patched: " + ", ".join(changed))
else:
    print("   already patched (no-op)")
PY
fi

echo ""
echo "==> Done. Models/Gemma/ contents:"
ls -lh "$DEST" | grep -v '^total'
echo ""
echo "Total Gemma/ size: $(du -sh "$DEST" | cut -f1)"
echo ""
echo "Next:"
echo "  bash scripts/generate-project.sh   # so Xcode picks up changes"
echo "  (then rebuild + deploy in Xcode)"
echo ""
echo "Restore stock model later with:"
echo "  rm -rf HikeCompanion/Resources/Models/Gemma"
echo "  mv HikeCompanion/Resources/Models/Gemma.stock HikeCompanion/Resources/Models/Gemma"
