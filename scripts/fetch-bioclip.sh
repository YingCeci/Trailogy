#!/usr/bin/env bash
# Build BioCLIP-2 vision encoder MLX INT4 + precomputed species embeddings,
# and write the resulting 4 files directly into the iOS app's Models/BioCLIP/.
#
# Despite the name "fetch", this script does NOT download a prebuilt
# bundle — it runs the conversion pipeline. The upstream BioCLIP-2
# PyTorch checkpoint is downloaded transparently by open_clip's
# `create_model_from_pretrained("hf-hub:imageomics/bioclip-2")` into
# ~/.cache/huggingface/hub/ during step 1; you don't fetch it directly.
#
# Pipeline:
#   1. scripts/bioclip/convert_to_mlx.py        — extract ViT-L/14 vision
#                                                 tower, remap weights,
#                                                 MLX INT4 quantize.
#                                                 Writes config.json +
#                                                 model.safetensors.
#   2. scripts/bioclip/precompute_embeddings.py — encode hardcoded species
#                                                 list (101 species,
#                                                 4-template avg + L2).
#                                                 Writes species_embeddings.npz
#                                                 + species_list.json.
#
# Both Python scripts live in scripts/bioclip/. They are vendored from the
# finetune project (now on the `feature/finetune-unsloth` branch under
# finetune/src/) so hikeCompanion is self-contained — no sibling repo
# dependency at conversion time. If conversion logic changes upstream,
# re-vendor with:
#   git show feature/finetune-unsloth:finetune/src/convert_bioclip_mlx.py \
#     > scripts/bioclip/convert_to_mlx.py
#   git show feature/finetune-unsloth:finetune/src/precompute_embeddings.py \
#     > scripts/bioclip/precompute_embeddings.py
#
# Output (single location, directly bundled by Xcode):
#   HikeCompanion/Resources/Models/BioCLIP/
#     ├── config.json              ~301 B   ViT-L/14 + INT4 quant config
#     ├── model.safetensors        ~184 MB  ViT-L/14 INT4 weights
#     ├── species_embeddings.npz   ~170 KB  101 × 768-d species embeddings
#     └── species_list.json        ~15 KB   species metadata (row-aligned)
#
# Optional sibling: ../bioclip-2/
#   If this directory exists (a clone of imageomics/bioclip-2), the
#   conversion uses its vendored open_clip — no need to `pip install
#   open_clip_torch`. If it doesn't exist, you must have open_clip_torch
#   installed in your Python environment.
#
# Required Python packages (Mac):
#   torch torchvision mlx numpy
#   open_clip_torch         (only if ../bioclip-2/ is not present)
#
# Usage:
#   bash scripts/fetch-bioclip.sh              # full conversion + deploy
#   bash scripts/fetch-bioclip.sh --force      # redo even if files exist
#
#   PYTHON=/path/to/env/bin/python bash scripts/fetch-bioclip.sh
#       Run conversion with a specific Python interpreter (e.g. a conda
#       env that has torch + open_clip + mlx). Defaults to `python3`.
#
# Re-run safe: idempotent. If Models/BioCLIP/ already has all 4 non-empty
# files, the conversion is skipped automatically; pass --force to redo.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
BIOCLIP_REPO="${BIOCLIP_REPO:-$REPO_ROOT/../bioclip-2}"
DEST_DIR="$REPO_ROOT/HikeCompanion/Resources/Models/BioCLIP"
# Override with PYTHON=/path/to/env/bin/python if your default python3
# does not have torch/open_clip/mlx installed.
PYTHON="${PYTHON:-python3}"

REQUIRED_FILES=("config.json" "model.safetensors" "species_embeddings.npz" "species_list.json")

# ── Argument parsing ──

FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '$d'
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

# ── Helpers ──

deployment_complete() {
    for f in "${REQUIRED_FILES[@]}"; do
        [ -s "$DEST_DIR/$f" ] || return 1
    done
    return 0
}

# ── Cache short-circuit ──

if [ "$FORCE" = "0" ] && deployment_complete; then
    echo "==> Cache hit: $DEST_DIR already has all 4 non-empty files."
    echo "    Pass --force to redo conversion."
    ls -lh "$DEST_DIR" | grep -v '^total'
    exit 0
fi

# ── Validate Python script availability ──

CONVERT_PY="$SCRIPTS_DIR/bioclip/convert_to_mlx.py"
EMBED_PY="$SCRIPTS_DIR/bioclip/precompute_embeddings.py"
[ -f "$CONVERT_PY" ] || { echo "ERROR: $CONVERT_PY not found" >&2; exit 1; }
[ -f "$EMBED_PY"   ] || { echo "ERROR: $EMBED_PY not found" >&2; exit 1; }

# ── open_clip source: prefer local sibling if available ──

BIOCLIP_ARGS=""
if [ -d "$BIOCLIP_REPO/src" ]; then
    BIOCLIP_ARGS="--bioclip_repo $BIOCLIP_REPO"
    echo "==> Using local bioclip-2 repo at $BIOCLIP_REPO (vendored open_clip)"
else
    echo "==> No local bioclip-2 repo at $BIOCLIP_REPO — relying on pip-installed open_clip_torch"
fi

mkdir -p "$DEST_DIR"

# ── Step 1: vision encoder → MLX INT4 ──

echo ""
echo "============================================================"
echo "  Step 1: Vision encoder → MLX INT4"
echo "============================================================"
"$PYTHON" "$CONVERT_PY" \
    $BIOCLIP_ARGS \
    --output_dir "$DEST_DIR" \
    --q_bits 4 \
    --q_group_size 64

# ── Step 2: species text embeddings ──

echo ""
echo "============================================================"
echo "  Step 2: Species text embeddings"
echo "============================================================"
"$PYTHON" "$EMBED_PY" \
    $BIOCLIP_ARGS \
    --output_dir "$DEST_DIR"

# ── Verify ──

echo ""
echo "==> Verifying output at $DEST_DIR"
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -s "$DEST_DIR/$f" ]; then
        echo "ERROR: $DEST_DIR/$f missing or empty after conversion" >&2
        exit 1
    fi
    printf "    %-26s %s\n" "$f" "$(du -h "$DEST_DIR/$f" | cut -f1)"
done

echo ""
echo "==> Done. Total BioCLIP/ size: $(du -sh "$DEST_DIR" | cut -f1)"
echo ""
echo "Note: BioCLIP/ lives alongside Gemma/ and Kokoro at"
echo "      HikeCompanion/Resources/Models/. The 'type: folder' reference"
echo "      in project.yml will pick it up automatically on next build."
