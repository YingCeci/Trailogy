#!/usr/bin/env bash
# Build BioCLIP species prompt cards from the iOS shortlist enrichment
# data and deploy them into the iOS bundle. These cards are injected
# into Gemma's prompt as the [BioCLIP candidates ...] block so the
# model has source-grounded morphology + range info per candidate.
#
# Pipeline (assumes the gemma4_note sibling repo is checked out at
# ../gemma4_note/):
#
#   1. Build the iOS-shortlist input CSV from species_list.json
#      (deterministic projection — no API calls; uses the index field
#      as species_id so the cards JSON keys line up with what
#      BioCLIPService.swift already loads).
#
#   2. Run enrich_plantnet300k_species.py over that CSV with
#      --kingdom none (the iOS list is mixed-kingdom: plants + fungi +
#      birds + herps + insects). Produces a species-list-shaped
#      enriched CSV via GBIF + Wikipedia. Resumable; per-row flush.
#
#   3. Run build_prompt_cards.py to compress the enriched data into
#      ~60-token cards per species, with rock entries (no biological
#      taxonomy) overridden from the hand-authored ios_rocks_cards.json.
#
#   4. Copy species_prompt_cards.json into
#      HikeCompanion/Resources/Models/BioCLIP/. The xcodegen
#      blue-folder reference picks it up on the next build.
#
# Output:
#   HikeCompanion/Resources/Models/BioCLIP/
#     ├── species_prompt_cards.json   (NEW; ~30 KB for the 101-entry list)
#     └── (existing 4 files from fetch-bioclip.sh untouched)
#
# Usage:
#   bash scripts/fetch-prompt-cards.sh              # full rebuild
#   bash scripts/fetch-prompt-cards.sh --force      # ignore enrich-cache resume

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
DEST_DIR="$REPO_ROOT/HikeCompanion/Resources/Models/BioCLIP"
ENRICH_DIR="${ENRICH_DIR:-$REPO_ROOT/../gemma4_note/05b-data_plantnet300k-enrich}"
PYTHON="${PYTHON:-python3}"

FORCE_ARG=""
for arg in "$@"; do
    case "$arg" in
        --force) FORCE_ARG="--no-resume" ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '$d'
            exit 0
            ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

if [ ! -d "$ENRICH_DIR" ]; then
    echo "ERROR: enrichment scripts directory not found at $ENRICH_DIR" >&2
    echo "       Set ENRICH_DIR=/path/to/05b-data_plantnet300k-enrich and re-run." >&2
    exit 1
fi

cd "$ENRICH_DIR"

# Step 1: project species_list.json → iOS-shortlist CSV
echo ""
echo "============================================================"
echo "  Step 1: build iOS shortlist input CSV"
echo "============================================================"
"$PYTHON" build_ios_input_csv.py \
    --input "$REPO_ROOT/HikeCompanion/Resources/Models/BioCLIP/species_list.json" \
    --output ios_shortlist_input.csv

# Step 2: enrich (kingdom=none for mixed-taxa list)
echo ""
echo "============================================================"
echo "  Step 2: enrich iOS shortlist via GBIF + Wikipedia"
echo "============================================================"
"$PYTHON" enrich_plantnet300k_species.py \
    --input ios_shortlist_input.csv \
    --output ios_shortlist_enriched.csv \
    --docs ios_shortlist_docs.jsonl \
    --cache-dir .ios_shortlist_cache \
    --kingdom none \
    --sleep 0.25 \
    --user-agent "hikeCompanion-rag/0.1 (kaggle gemma4 hackathon)" \
    $FORCE_ARG

# Step 3: heuristic card builder
echo ""
echo "============================================================"
echo "  Step 3: build prompt cards"
echo "============================================================"
"$PYTHON" build_prompt_cards.py \
    --input ios_shortlist_enriched.csv \
    --output species_prompt_cards.json \
    --stats cards_stats.json \
    --rocks-json ios_rocks_cards.json

# Step 4: deploy into iOS bundle
echo ""
echo "============================================================"
echo "  Step 4: deploy into iOS bundle"
echo "============================================================"
mkdir -p "$DEST_DIR"
cp species_prompt_cards.json "$DEST_DIR/species_prompt_cards.json"
echo "wrote $DEST_DIR/species_prompt_cards.json"

# ── Summary ──
echo ""
echo "==> Done."
SIZE_BYTES="$(stat -f%z "$DEST_DIR/species_prompt_cards.json" 2>/dev/null || stat -c%s "$DEST_DIR/species_prompt_cards.json")"
echo "    cards JSON: $SIZE_BYTES bytes"
"$PYTHON" -c "
import json
cards = json.load(open('$DEST_DIR/species_prompt_cards.json'))
print(f'    {len(cards)} cards bundled')
print(f'    sample [0]: {cards[\"0\"][:120]}...')
"
echo ""
echo "Re-run 'bash scripts/generate-project.sh' if you renamed the file or"
echo "added more bundle paths, then rebuild in Xcode."
