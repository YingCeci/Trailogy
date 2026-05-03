#!/usr/bin/env bash
# Downloads Gemma 4 E2B (INT4 MLX format) from Hugging Face into the
# bundle resource directory. ~3.5 GB total.
#
# Primary repo:   mlx-community/gemma-4-e2b-it-4bit  (Apache 2.0)
# Backup repo:    unsloth/gemma-4-E2B-it-UD-MLX-4bit (if primary has the
#                 PLE quantization bug — pass --backup to use this)
#
# After this script finishes, re-run scripts/generate-project.sh and rebuild
# in Xcode so the new files get bundled.
#
# Usage:
#   bash scripts/fetch-gemma.sh             # primary repo
#   bash scripts/fetch-gemma.sh --backup    # unsloth fallback
#
# Re-run safe: skips files that already exist with non-trivial size.

set -euo pipefail

REPO="mlx-community/gemma-4-e2b-it-4bit"
if [[ "${1:-}" == "--backup" ]]; then
  REPO="unsloth/gemma-4-E2B-it-UD-MLX-4bit"
  echo "==> Using backup repo: $REPO"
fi

DEST="$(cd "$(dirname "$0")/.." && pwd)/HikeCompanion/Resources/Models/Gemma"
API="https://huggingface.co/api/models/${REPO}/tree/main"
RESOLVE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"

echo "==> Listing files at $REPO ..."
LIST=$(curl -fsSL "$API")
FILES=$(echo "$LIST" | python3 -c '
import json, sys
items = json.load(sys.stdin)
# Only top-level files; MLX models are flat (no nested dirs)
for it in items:
    if it.get("type") == "file":
        print(it["path"])
')

if [[ -z "$FILES" ]]; then
  echo "ERROR: HF API returned no files for $REPO" >&2
  exit 1
fi

echo "==> $(echo "$FILES" | wc -l | tr -d ' ') files to fetch"

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  out="$DEST/$file"
  if [[ -f "$out" ]]; then
    sz=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000 ]]; then
      echo "  skip $file ($(du -h "$out" | cut -f1))"
      continue
    fi
  fi
  echo "==> $file"
  mkdir -p "$(dirname "$out")"
  curl -fL --progress-bar "$RESOLVE/$file" -o "$out"
done <<< "$FILES"

echo ""
echo "==> Done. Models/Gemma/ contents:"
ls -lh "$DEST" | grep -v '^total'
echo ""
echo "Total Gemma/ size: $(du -sh "$DEST" | cut -f1)"
echo ""
echo "Next: bash scripts/generate-project.sh   (so Xcode picks up the new files)"
