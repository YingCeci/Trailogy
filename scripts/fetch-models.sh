#!/usr/bin/env bash
# Downloads all Core ML model files from huggingface.co/mattmireles/kokoro-coreml
# into HikeCompanion/Resources/Models/. The mlpackage files are bundled as
# folder references (see project.yml) and compiled to .mlmodelc on device at
# first launch by KokoroPipeline.
#
# Usage:  bash scripts/fetch-models.sh
# Re-run safe: skips files that already exist with the right size.

set -euo pipefail

REPO="mattmireles/kokoro-coreml"
DEST="$(cd "$(dirname "$0")/.." && pwd)/HikeCompanion/Resources/Models"
API="https://huggingface.co/api/models/${REPO}/tree/main"
RESOLVE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"

echo "==> Listing files at $REPO ..."
LIST=$(curl -fsSL "$API")

# A .mlpackage is a directory containing files like "Manifest.json", "Data/...",
# "Metadata.json", etc. The HF API returns flat file paths; we recursively walk
# every directory entry and download every leaf file under each .mlpackage tree.

list_recursive() {
  local prefix="$1"
  local url
  if [[ -z "$prefix" ]]; then
    url="$API"
  else
    url="${API}/${prefix}"
  fi
  curl -fsSL "$url" | python3 -c '
import json, sys
items = json.load(sys.stdin)
for it in items:
    print(it["type"], it["path"])
' || true
}

download_file() {
  local relpath="$1"
  local out="${DEST}/${relpath}"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" && -s "$out" ]]; then
    return 0
  fi
  echo "    fetching $relpath"
  curl -fsSL "${RESOLVE}/${relpath}" -o "$out"
}

walk() {
  local prefix="$1"
  local entries
  entries=$(list_recursive "$prefix")
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local kind path
    kind=$(echo "$line" | awk '{print $1}')
    path=$(echo "$line" | cut -d' ' -f2-)
    case "$kind" in
      file) download_file "$path" ;;
      directory) walk "$path" ;;
    esac
  done <<< "$entries"
}

echo "==> Downloading into $DEST ..."
walk ""

echo ""
echo "==> Done. Contents:"
ls -1 "$DEST" | head -40
echo ""
echo "Total size: $(du -sh "$DEST" | cut -f1)"
