# data_mix/src/env_paths.py
"""Resolve storage roots from env vars with safe in-repo defaults.

No per-machine absolute paths are baked into this module — operator
points the envs at whatever storage they have.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# data_mix/src/env_paths.py -> src/data_mix/ -> src/ -> repo root
DATA_MIX_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DATA_MIX_ROOT.parent
HIKECOMPANION_ROOT = SRC_ROOT.parent
DEFAULT_LOCAL_ROOT = DATA_MIX_ROOT / "_local"


@dataclass(frozen=True)
class Paths:
    hf_home: Path | None  # None = let huggingface_hub use its own default
    image_root: Path
    output_root: Path
    plantnet_jsonl: Path       # train pool source
    plantnet_val_jsonl: Path   # v2: separate val pool source (was: a random
                               #     slice of plantnet_jsonl). Default is
                               #     sibling val.jsonl of plantnet_jsonl.


def _default_output_root(config_path: Path | None) -> Path:
    """Derive the default output dir from the config filename stem.

    ``mix-50k.yaml`` -> ``src/finetune/data/mix-50k``.
    ``mix-50k-plantnet.yaml`` -> ``src/finetune/data/mix-50k-plantnet``.

    Falls back to ``mix-50k`` when no config_path is given (legacy
    callers that don't pass it through; tests that patch resolve_paths
    directly should also patch this).
    """
    stem = config_path.stem if config_path is not None else "mix-50k"
    return SRC_ROOT / "finetune" / "data" / stem


def resolve_paths(config_path: Path | None = None) -> Paths:
    """Resolve storage paths. ``config_path``, when given, drives the
    default ``output_root`` so each yaml config writes to its own
    sibling-of-config output dir (yaml-stem == output-dir-name)."""
    hf_home_env = os.environ.get("HF_HOME")
    hf_home = Path(hf_home_env).resolve() if hf_home_env else None

    image_root = Path(
        os.environ.get("DATA_MIX_IMAGE_ROOT")
        or (DEFAULT_LOCAL_ROOT / "images")
    ).resolve()

    output_root = Path(
        os.environ.get("DATA_MIX_OUTPUT_ROOT")
        or _default_output_root(config_path)
    ).resolve()

    plantnet_jsonl = Path(
        os.environ.get("PLANTNET_JSONL")
        or (
            HIKECOMPANION_ROOT
            / "src"
            / "finetune"
            / "data"
            / "english-desc-v2"
            / "train.jsonl"
        )
    ).resolve()

    # v2: PLANTNET_VAL_JSONL points at the per-species val output of
    # prepare_plantnet_50k.sh. Default = sibling val.jsonl of plantnet_jsonl.
    plantnet_val_jsonl = Path(
        os.environ.get("PLANTNET_VAL_JSONL")
        or (plantnet_jsonl.parent / "val.jsonl")
    ).resolve()

    return Paths(
        hf_home=hf_home,
        image_root=image_root,
        output_root=output_root,
        plantnet_jsonl=plantnet_jsonl,
        plantnet_val_jsonl=plantnet_val_jsonl,
    )
