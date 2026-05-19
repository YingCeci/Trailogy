#!/usr/bin/env bash
# Build a data mix. All storage roots are env-driven with safe in-repo
# defaults; operator overrides via environment.
#
# Default CONFIG = NA-trees-backed mix-50k.yaml. For the PlantNet-only
# 1.0 recipe, set CONFIG=$DATA_MIX_DIR/configs/mix-50k-plantnet.yaml.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_MIX_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_ROOT="$(cd "${DATA_MIX_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${DATA_MIX_DIR}/configs/mix-50k.yaml}"

PYTHON_BIN="${PYTHON_BIN:-python}"

echo "== data_mix build =="
echo "HF_HOME              = ${HF_HOME:-<unset, using huggingface_hub default>}"
echo "DATA_MIX_IMAGE_ROOT  = ${DATA_MIX_IMAGE_ROOT:-<unset, will fall back to ${DATA_MIX_DIR}/_local/images>}"
echo "DATA_MIX_OUTPUT_ROOT = ${DATA_MIX_OUTPUT_ROOT:-<unset, will fall back to per-config default (mix.py)>}"
echo "PLANTNET_JSONL       = ${PLANTNET_JSONL:-<unset; required only when the chosen CONFIG has plant.train > 0>}"
echo "NA_TREES_TRAIN_JSONL = ${NA_TREES_TRAIN_JSONL:-<set via na_trees.train_jsonl in the CONFIG file>}"
echo "CONFIG               = ${CONFIG}"
echo "PYTHON_BIN           = ${PYTHON_BIN}"
echo

# Pre-flight: PlantNet JSONL is REQUIRED only when the chosen CONFIG
# actually uses the plant bucket (plant.train > 0 OR plant.val > 0).
# NA-trees-only mixes skip the plant bucket entirely.
#
# We grep the CONFIG for an integer plant.{train,val} > 0 to decide.
# Operators with a yaml that uses jinja / anchors should set the env
# var BUILD_MIX_SKIP_PLANTNET_PREFLIGHT=1 to bypass this check.
if [[ "${BUILD_MIX_SKIP_PLANTNET_PREFLIGHT:-0}" != "1" ]]; then
  PLANT_USED=0
  if grep -E '^[[:space:]]*plant:' "${CONFIG}" >/dev/null 2>&1; then
    # Capture the numeric train + val from the plant: block. Naive but
    # adequate for our flat yaml schema.
    PLANT_BLOCK="$(awk '
      /^plant:[[:space:]]*$/         { in_block=1; next }
      in_block && /^[^[:space:]]/    { in_block=0 }
      in_block                       { print }
    ' "${CONFIG}")"
    if echo "${PLANT_BLOCK}" | grep -E '^[[:space:]]*(train|val):[[:space:]]*[1-9]' >/dev/null 2>&1; then
      PLANT_USED=1
    fi
  fi
  if [[ "${PLANT_USED}" == "1" ]]; then
    RESOLVED_PLANTNET="${PLANTNET_JSONL:-${SRC_ROOT}/finetune/data/english-desc-v2/train.jsonl}"
    if [[ ! -f "${RESOLVED_PLANTNET}" ]]; then
      echo "ERROR: ${CONFIG} declares a non-zero plant bucket but" >&2
      echo "       PlantNet JSONL is not at ${RESOLVED_PLANTNET}" >&2
      echo "       Set PLANTNET_JSONL to override, or set" >&2
      echo "       BUILD_MIX_SKIP_PLANTNET_PREFLIGHT=1 to bypass." >&2
      exit 1
    fi
  fi
fi

# Run from the public ML module parent so `data_mix.src.mix` resolves.
cd "${SRC_ROOT}"
"${PYTHON_BIN}" -m data_mix.src.mix --config "${CONFIG}"
