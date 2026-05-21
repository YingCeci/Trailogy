"""Sample NA Plantae records from the prepared na_plantae JSONL.

Since the NA Plantae corpus is long-tailed (per-class image counts span
roughly 1.5 orders of magnitude even after the prepare-step caps), the
sampler supports per-class re-weighting via ``train_temperature``:

  * ``train_temperature = 1.0`` (default) — natural shuffle-and-repeat.
    Each record is sampled with uniform weight, so per-class share in
    the output equals per-class share in the pool. Bit-compatible with
    the legacy implementation.
  * ``train_temperature < 1.0`` — temper the distribution toward
    balanced. The per-record weight becomes ``n_class ** (T - 1)``,
    making the expected per-class share proportional to ``n_class ** T``.
    The canonical square-root tempering of Mahajan et al. 2018 is
    ``T = 0.5``.
  * ``train_temperature -> 0`` — fully balanced, every class shows up
    equally often.

Val sampling is always natural (frequency-proportional) so eval loss
keeps measuring the underlying class distribution.

Records always get a ``source: "na_plantae"`` stamp.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _natural_oversample(
    pool: List[dict], n: int, rng: random.Random,
) -> List[dict]:
    """Legacy shuffle-and-repeat: full passes through a re-shuffled
    pool, plus a partial tail. Preserves natural class proportions."""
    if n <= 0 or not pool:
        return []
    out: List[dict] = []
    full_passes, remainder = divmod(n, len(pool))
    for _ in range(full_passes):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        out.extend(shuffled)
    if remainder:
        shuffled = list(pool)
        rng.shuffle(shuffled)
        out.extend(shuffled[:remainder])
    return out


def _tempered_sample(
    pool: List[dict],
    n: int,
    temperature: float,
    rng: random.Random,
) -> List[dict]:
    """Per-class tempered sampling with replacement.

    For each record, weight = ``n_class[record's slug] ** (T - 1)``.
    Aggregating across the class then gives expected per-class share
    proportional to ``n_class ** T``.
    """
    if n <= 0 or not pool:
        return []
    counts = Counter(rec.get("slug", "") for rec in pool)
    if not counts:
        return []
    exponent = temperature - 1.0
    weights = [counts[rec.get("slug", "")] ** exponent for rec in pool]
    # random.choices does weighted sampling with replacement — exactly
    # the behaviour we want (small classes get visited many times,
    # large classes get down-weighted).
    return rng.choices(pool, weights=weights, k=n)


def sample_na_plantae_records(
    train_jsonl: Path,
    val_jsonl: Path,
    n_train: int,
    n_val: int,
    seed: int,
    train_temperature: float = 1.0,
) -> Tuple[List[dict], List[dict]]:
    """Read na_plantae train/val JSONLs, stamp source, and sample to
    the requested counts.

    ``train_temperature`` controls per-class re-weighting on the train
    pool (see module docstring). Val is always sampled naturally.
    """
    train_raw = _read_jsonl(train_jsonl)
    val_raw = _read_jsonl(val_jsonl)

    if n_train > 0 and not train_raw:
        raise RuntimeError(f"na_plantae train JSONL is empty: {train_jsonl}")
    if n_val > 0 and not val_raw:
        raise RuntimeError(f"na_plantae val JSONL is empty: {val_jsonl}")

    # Stamp source on every record (in-place is fine; the lists are
    # local to this call).
    for rec in train_raw:
        rec["source"] = "na_plantae"
    for rec in val_raw:
        rec["source"] = "na_plantae"

    rng_t = random.Random(seed)
    rng_v = random.Random(seed + 1)

    if train_temperature == 1.0:
        # Legacy path. Preserve bit-identical shuffle order to the
        # pre-temperature implementation for back-compat — same RNG,
        # same upstream shuffle, same divmod tail.
        rng_t.shuffle(train_raw)
        train_out = _natural_oversample(train_raw, n_train, rng_t)
    else:
        train_out = _tempered_sample(
            train_raw, n_train, train_temperature, rng_t,
        )

    # Val: always natural. Preserves the legacy contract that
    # eval_<key>_loss reflects the pool's class distribution.
    rng_v.shuffle(val_raw)
    val_out = _natural_oversample(val_raw, n_val, rng_v)

    return train_out, val_out
