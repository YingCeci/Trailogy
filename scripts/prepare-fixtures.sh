#!/usr/bin/env bash
# Generates JSON input fixtures + HNSF weights for the Swift validator by
# running the upstream's prepare_swift_bench_inputs.py.
#
# Output files (in HikeCompanion/Resources/Fixtures/):
#   3s.json, 7s.json, 15s.json, 30s.json   -- pre-tokenized BenchInput per upstream BAKEOFF_INPUTS
#   hnsf_config.json                       -- learned linear weights for hn-nsf source module
#
# Requires:
#   - python3
#   - `uv` (https://docs.astral.sh/uv/) — installed automatically if missing
#
# Usage: bash scripts/prepare-fixtures.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/external/kokoro-coreml"
DEST="${REPO_ROOT}/HikeCompanion/Resources/Fixtures"

mkdir -p "$DEST"

# 1) Ensure uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv (Python package manager) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  . "$HOME/.local/bin/env" 2>/dev/null || true
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv install failed. Install manually from https://docs.astral.sh/uv/ and re-run." >&2
  exit 1
fi

# 2) Run upstream prep script via uv (uses upstream's pyproject.toml)
echo "==> Running upstream prepare_swift_bench_inputs.py ..."
cd "$UPSTREAM"

# The upstream uses uv to manage deps. First sync to install the kokoro
# package and its deps (PyTorch, numpy, transformers, etc.).
uv sync --frozen 2>/dev/null || uv sync

# uv-created venvs don't include pip by default. The kokoro/misaki runtime
# shells out to `python -m pip install` to fetch the spaCy en_core_web_sm
# model on first use, which fails without pip. Install it after sync (must
# be after, because `uv sync` removes packages not in the lockfile).
uv pip install pip

# Pre-download the spaCy model so the runtime pip-fetch is a no-op (more
# robust than relying on it at synthesis time). Best-effort; if this fails
# the prep script will fall back to the runtime download.
uv run python -m spacy download en_core_web_sm 2>/dev/null || \
    echo "  (spaCy model pre-download skipped; will fetch at runtime)"

uv run python scripts/prepare_swift_bench_inputs.py

# 3) Copy outputs into our Resources/Fixtures
SRC="${UPSTREAM}/outputs/swift_bench_inputs"
if [[ ! -d "$SRC" ]]; then
  echo "ERROR: Upstream prep did not create $SRC" >&2
  exit 1
fi

echo "==> Copying fixtures to $DEST ..."
cp -v "$SRC"/*.json "$DEST/"

echo ""
echo "==> Done. Fixtures in place:"
ls -lh "$DEST"
