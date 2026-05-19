#!/usr/bin/env python3
"""Prepare the NA-trees supplement / primary dataset.

Produces ``{train,val,test}.jsonl`` files in the same schema as
``finetune/src/prepare_plantnet.py``:

    {
      "image": "<abs path>",
      "conversations": [
        {"role": "user", "content": "<question template>"},
        {"role": "assistant", "content": "<species description>"}
      ]
    }

Reads species descriptions from a YAML file (operator-supplied; one
record per slug with at minimum ``slug``, ``common_name``, ``species``,
``family``, ``answer`` fields — see ``assets/na_trees/descriptions.yaml``
for the canonical schema). Each species folder under ``--source_root``
is matched by slug.

Default per-species split: 40 train / 5 val / 5 test (50 images per
species → 11 × 50 = 550 records total). Raw images are typically
produced by ``src/data_mix/scripts/na_tree_fetch.py``.

Usage:

    python -m data_mix.src.prepare_na_trees \\
        --source_root  $NA_TREES_ROOT \\
        --descriptions assets/na_trees/descriptions.yaml \\
        --output_dir   src/finetune/data/na_trees \\
        --train_per_species 40 \\
        --val_per_species   5 \\
        --test_per_species  5 \\
        --resize_to 960x672

``--resize_to`` mirrors ``prepare_plantnet.py`` — pre-resize every image
to the iOS-runtime shape so training and deploy see the same visual
distribution. Set to ``none`` to disable.

Output layout::

    <output_dir>/
        train.jsonl
        val.jsonl
        test.jsonl
        images_resized/
            train/<slug>/*.jpg
            val/<slug>/*.jpg
            test/<slug>/*.jpg
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml  # type: ignore

log = logging.getLogger("prepare_na_trees")

QUESTION_TEMPLATES = [
    "What plant is this?",
    "Can you identify this species?",
    "What am I looking at?",
    "Describe this tree.",
    "Do you know what kind of tree this is?",
    "I found this on the trail — what is it?",
    "What species is this tree?",
    "Can you tell me about this tree I just spotted?",
    "I saw this growing near the trail. Any idea what it is?",
    "What's the name of this tree?",
    "Help me identify this — is it a common species?",
    "I'm curious about this tree. What can you tell me?",
]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
DEFAULT_RESIZE_HW = (960, 672)  # height, width — iOS runtime contract


def load_descriptions(path: Path) -> dict[str, dict[str, Any]]:
    """Return {slug: record} mapping."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    trees = raw.get("trees", [])
    out = {}
    for t in trees:
        slug = t["slug"]
        if slug in out:
            raise ValueError(f"duplicate slug in descriptions: {slug}")
        out[slug] = t
    return out


def discover_species_images(source_root: Path, slugs: list[str]) -> dict[str, list[Path]]:
    """Return {slug: sorted [Path, ...]} for each species."""
    out: dict[str, list[Path]] = {}
    for slug in slugs:
        d = source_root / slug
        if not d.is_dir():
            log.warning("species folder missing: %s", d)
            out[slug] = []
            continue
        files = sorted([p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS])
        out[slug] = files
        log.info("  %s: %d images", slug, len(files))
    return out


def maybe_resize_and_save(
    src: Path, dst: Path, resize_hw: tuple[int, int] | None
) -> None:
    """Copy or pre-resize an image to dst. Skips if dst already exists."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if resize_hw is None:
        # Hard-link if possible (zero-byte cost on APFS), else copy.
        try:
            dst.hardlink_to(src)
        except (OSError, AttributeError):
            import shutil
            shutil.copy2(src, dst)
        return
    # Resize. Pillow handles JPEG / PNG / WEBP fine.
    from PIL import Image  # type: ignore

    h, w = resize_hw
    with Image.open(src) as im:
        im = im.convert("RGB")
        im_resized = im.resize((w, h), Image.LANCZOS)
        im_resized.save(dst, "JPEG", quality=92)


def parse_resize(s: str) -> tuple[int, int] | None:
    if s.lower() in ("none", "off", "no", ""):
        return None
    parts = s.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--resize_to expects WxH or 'none'; got {s!r}")
    w, h = int(parts[0]), int(parts[1])
    return (h, w)  # store as (H, W)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source_root", type=Path, required=True,
                    help="Directory containing <slug>/<N>.jpg subfolders.")
    ap.add_argument("--descriptions", type=Path, required=True,
                    help="YAML with the species answer + metadata.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--train_per_species", type=int, default=40)
    ap.add_argument("--val_per_species",   type=int, default=5)
    ap.add_argument("--test_per_species",  type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resize_to",
        type=parse_resize,
        default=(DEFAULT_RESIZE_HW[0], DEFAULT_RESIZE_HW[1]),
        help="WxH to pre-resize images. Use 'none' to disable. Default: 960x672 (iOS runtime).",
    )
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    descs = load_descriptions(args.descriptions)
    log.info("Loaded %d species descriptions.", len(descs))
    slugs = sorted(descs.keys())

    species_imgs = discover_species_images(args.source_root, slugs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    summary: dict[str, dict[str, int]] = {}

    for slug in slugs:
        imgs = species_imgs[slug]
        if not imgs:
            log.warning("  %s: no images, skipping species.", slug)
            continue
        rng_local = random.Random(args.seed + hash(slug) % 100000)
        idxs = list(range(len(imgs)))
        rng_local.shuffle(idxs)

        n_tr = min(args.train_per_species, len(idxs))
        n_va = min(args.val_per_species, max(0, len(idxs) - n_tr))
        n_te = min(args.test_per_species, max(0, len(idxs) - n_tr - n_va))

        plan = (
            ("train", idxs[:n_tr]),
            ("val", idxs[n_tr:n_tr + n_va]),
            ("test", idxs[n_tr + n_va:n_tr + n_va + n_te]),
        )

        rec = descs[slug]
        answer = rec["answer"].strip()

        for split_name, sel in plan:
            for i in sel:
                src = imgs[i]
                # Save resized copy
                dst = args.output_dir / "images_resized" / split_name / slug / f"{src.stem}.jpg"
                maybe_resize_and_save(src, dst, args.resize_to)
                question = rng.choice(QUESTION_TEMPLATES)
                splits[split_name].append({
                    "image": str(dst.resolve()),
                    "slug": slug,
                    "species": rec["species"],
                    "family": rec["family"],
                    "conversations": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer},
                    ],
                })
        summary[slug] = {"train": n_tr, "val": n_va, "test": n_te, "total": n_tr + n_va + n_te}
        log.info("  %s: train=%d val=%d test=%d", slug, n_tr, n_va, n_te)

    # Shuffle train globally so species aren't clumped together.
    rng.shuffle(splits["train"])

    for split_name, rows in splits.items():
        path = args.output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        log.info("Wrote %s (%d rows).", path, len(rows))

    # Summary
    log.info("")
    log.info("=== Summary ===")
    tot_tr = sum(s["train"] for s in summary.values())
    tot_va = sum(s["val"] for s in summary.values())
    tot_te = sum(s["test"] for s in summary.values())
    log.info("Total: train=%d val=%d test=%d (across %d species)",
             tot_tr, tot_va, tot_te, len(summary))

    with open(args.output_dir / "filter_report.json", "w") as f:
        json.dump({
            "source_root": str(args.source_root),
            "descriptions": str(args.descriptions),
            "split_per_species": {
                "train": args.train_per_species,
                "val": args.val_per_species,
                "test": args.test_per_species,
            },
            "resize_to": [args.resize_to[1], args.resize_to[0]] if args.resize_to else None,
            "totals": {"train": tot_tr, "val": tot_va, "test": tot_te},
            "per_species": summary,
        }, f, indent=2)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
