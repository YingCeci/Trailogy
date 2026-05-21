"""Tests for ``sample_na_plantae_records``.

The sampler now supports a per-class re-weighting controlled by
``train_temperature``:

  * ``train_temperature = 1.0`` (default): natural frequency-proportional
    sampling. The output preserves the input class distribution and
    matches the legacy shuffle+repeat behaviour exactly (back-compat).
  * ``train_temperature < 1.0``: per-class probability is tempered to
    ``p(class) ∝ n_c ** temperature``. Used for long-tail multi-class
    classification — ``temperature = 0.5`` is the canonical
    square-root-tempered sampling from Mahajan et al. 2018.
  * ``train_temperature -> 0``: fully balanced sampling (all classes
    seen equally often, irrespective of pool size).

Val sampling is intentionally LEFT NATURAL regardless of temperature so
held-out eval still reflects the underlying pool distribution.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from data_mix.src.na_plantae_sampler import sample_na_plantae_records


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _row(slug: str, idx: int) -> dict:
    return {
        "image": f"/img/{slug}/{idx}.jpg",
        "slug": slug,
        "species": slug,
        "family": "F",
        "conversations": [{"role": "user", "content": "?"},
                          {"role": "assistant", "content": slug}],
    }


def _pool(slug_counts: dict[str, int]) -> list[dict]:
    out = []
    for slug, n in slug_counts.items():
        for i in range(n):
            out.append(_row(slug, i))
    return out


# ---------------------------------------------------------------------------
# Back-compat: default temperature == 1.0 keeps legacy behaviour
# ---------------------------------------------------------------------------

def test_default_temperature_preserves_class_proportions(tmp_path: Path) -> None:
    """At temperature=1 (default), the expected per-class share equals
    n_c / total — exactly the legacy shuffle+repeat distribution."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"head": 90, "tail": 10}))  # 9:1 pool
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10_000, n_val=0, seed=0,
    )
    counts = Counter(r["slug"] for r in train_out)
    # Expected: head 90%, tail 10% ± 1.5%.
    assert 8500 <= counts["head"] <= 9500
    assert 500 <= counts["tail"] <= 1500


def test_default_matches_legacy_full_pass_shape(tmp_path: Path) -> None:
    """When n_train is an exact multiple of pool size and
    temperature=1, every record appears exactly ``n_train/pool_size``
    times (the legacy full-pass + shuffle invariant)."""
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    pool = _pool({"a": 3, "b": 2})  # size 5
    _write(train_p, pool)
    _write(val_p, [_row("a", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=15, n_val=0, seed=0,
    )
    img_counts = Counter(r["image"] for r in train_out)
    assert set(img_counts.values()) == {3}


# ---------------------------------------------------------------------------
# Sqrt tempering: temperature < 1 boosts tail
# ---------------------------------------------------------------------------

def test_temperature_half_makes_distribution_proportional_to_sqrt(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    # 100:1 pool, sqrt -> 10:1.
    _write(train_p, _pool({"head": 100, "tail": 1}))
    _write(val_p, [_row("head", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=11_000, n_val=0, seed=0, train_temperature=0.5,
    )
    counts = Counter(r["slug"] for r in train_out)
    ratio = counts["head"] / max(counts["tail"], 1)
    # Expected ratio ≈ sqrt(100/1) = 10. Allow ±25% Monte-Carlo noise
    # on n=11000.
    assert 7.5 <= ratio <= 12.5


def test_temperature_zero_yields_balanced_sampling(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"big": 100, "small": 5}))
    _write(val_p, [_row("big", 0)])

    train_out, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10_000, n_val=0, seed=0, train_temperature=0.0,
    )
    counts = Counter(r["slug"] for r in train_out)
    # Balanced -> 50/50 ± 3%.
    assert 4500 <= counts["big"] <= 5500
    assert 4500 <= counts["small"] <= 5500


# ---------------------------------------------------------------------------
# Val stays natural regardless of temperature
# ---------------------------------------------------------------------------

def test_val_sampling_is_unaffected_by_train_temperature(
    tmp_path: Path,
) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, [_row("x", 0)])
    _write(val_p, _pool({"head": 90, "tail": 10}))

    _, val_out = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=0, n_val=10_000, seed=0, train_temperature=0.0,
    )
    counts = Counter(r["slug"] for r in val_out)
    # Even with extreme temperature on the train side, val stays
    # frequency-proportional (90:10).
    assert 8500 <= counts["head"] <= 9500
    assert 500 <= counts["tail"] <= 1500


# ---------------------------------------------------------------------------
# Determinism + source stamp
# ---------------------------------------------------------------------------

def test_temperature_sampling_is_deterministic(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 50, "b": 50}))
    _write(val_p, [_row("a", 0)])

    out1, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=200, n_val=0, seed=42, train_temperature=0.5,
    )
    out2, _ = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=200, n_val=0, seed=42, train_temperature=0.5,
    )
    assert [r["image"] for r in out1] == [r["image"] for r in out2]


def test_records_carry_source_stamp(tmp_path: Path) -> None:
    train_p = tmp_path / "train.jsonl"
    val_p = tmp_path / "val.jsonl"
    _write(train_p, _pool({"a": 5}))
    _write(val_p, _pool({"a": 5}))

    train_out, val_out = sample_na_plantae_records(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10, n_val=10, seed=0, train_temperature=0.5,
    )
    for rec in train_out + val_out:
        assert rec["source"] == "na_plantae"
