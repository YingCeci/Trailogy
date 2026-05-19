#!/usr/bin/env python3
"""Fetch plant observations near a target location from iNaturalist.

**Geographically-grounded data collection.** The default lat/lng centres
on Pittsburgh, PA (40.4406, -79.9959) with an 80 km radius — this is
the region of Trailogy's three trails (Kildoo / McConnells Mill,
Jennings, Frick Park, all in western PA). Training data therefore
matches the species distribution users actually encounter on the
supported trails, which the plain PlantNet-50k corpus does not
represent.

The lat/lng is THE primary design choice. Every other request param
(``taxon_name``, ``quality_grade``, ``photos``, ``per_page``, ``page``,
``order_by``, ``order``) defaults to the upstream reference
``na_tree_fetch.py`` byte-for-byte so a default run pulls the same
image set as the reference script — i.e. all CC-vetted Plantae photos
within 80 km of Pittsburgh, paginated until empty with a 1 s
inter-page sleep.

What this script does on top of the reference:

  1. Saves the per-photo metadata to ``observations.jsonl`` (the
     reference holds it in memory and discards on exit; we persist it).
  2. **Downloads** every photo to a local cache. Output layout::

         <output_dir>/
             observations.jsonl       # one JSON line per photo
             fetch_report.json
             train/<slug>/<obs_id>_<photo_idx>.jpg
             val/<slug>/<obs_id>_<photo_idx>.jpg
             test/<slug>/<obs_id>_<photo_idx>.jpg

     The slug = sanitised ``preferred_common_name`` (falls back to the
     Latin name). Split assignment is on a per-OBSERVATION key (so all
     photos of the same observation always land in the same split — no
     train/test leakage from near-duplicate frames of one plant).
  3. The default ``--output-dir`` is ``./inaturalist_na_trees/``,
     relative to the working directory, **outside the repo**. The
     directory is created on first run.

Usage::

    # Default sweep — Pittsburgh, 80 km, all plants, downloaded into
    # ./inaturalist_na_trees/{train,val,test}/<slug>/...
    python src/data_mix/scripts/na_tree_fetch.py

    # Same sweep, metadata only (no image bytes on disk).
    python src/data_mix/scripts/na_tree_fetch.py --no-download

Network: paginated request loop. A radius-80 km Pittsburgh sweep
typically yields a few thousand research-grade plant observations.
iNaturalist rate-limits to ~60 req/min per IP; the 1 s inter-page
sleep stays safely under that.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import requests  # type: ignore
except ImportError:
    print(
        "ERROR: requests is required (pip install requests).",
        file=sys.stderr,
    )
    sys.exit(2)


BASE = "https://api.inaturalist.org/v1/observations"

# --- Default request params (MUST stay identical to the upstream
#     reference ``na_tree_fetch.py``). Changing any of these silently
#     drifts Trailogy's fetched corpus away from the reference. ---
DEFAULT_PARAMS: dict[str, Any] = {
    "taxon_name":    "Plantae",
    "quality_grade": "research",
    "photos":        "true",
    "lat":           40.4406,        # Pittsburgh example
    "lng":           -79.9959,
    "radius":        80,             # km
    "per_page":      200,
    "page":          1,
    "order_by":      "created_at",
    "order":         "desc",
}

# Default cache lives OUTSIDE the repo (relative to cwd, not script
# location). The user keeps the bytes wherever they invoke this script.
DEFAULT_OUTPUT_DIR = Path("./inaturalist_na_trees")

# Default 80/10/10 split, mirrors the per-species counts in the legacy
# prepare_na_trees recipe (40 train / 5 val / 5 test out of 50).
DEFAULT_SPLIT_RATIOS = (0.80, 0.10, 0.10)
SPLIT_NAMES = ("train", "val", "test")

HTTP_TIMEOUT_SEC = 30.0
INTER_PAGE_SLEEP_SEC = 1.0


def slugify(name: str) -> str:
    """Lowercase, collapse non-alnum to ``_``, trim. Empty -> 'unknown'."""
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s or "unknown"


def derive_slug(taxon: dict[str, Any]) -> str:
    """Prefer ``preferred_common_name`` (Red Maple -> ``red_maple``),
    fall back to the Latin ``name`` (Acer rubrum -> ``acer_rubrum``)."""
    common = (taxon or {}).get("preferred_common_name") or ""
    sci = (taxon or {}).get("name") or ""
    return slugify(common) or slugify(sci)


def stable_bucket(key: str, seed: int) -> float:
    """Map a string key + seed to a stable float in [0, 1).

    Uses sha256 so the result is reproducible across processes (Python's
    builtin ``hash()`` is salted per-interpreter and would split the
    same observation into different folders on re-runs).
    """
    h = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_split(
    key: str,
    seed: int,
    ratios: tuple[float, float, float],
) -> str:
    """Deterministic train/val/test assignment from a key + seed."""
    r_train, r_val, _ = ratios
    b = stable_bucket(key, seed)
    if b < r_train:
        return "train"
    if b < r_train + r_val:
        return "val"
    return "test"


def paginate_observations(
    base_params: dict[str, Any],
    *,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Walk the iNaturalist obs endpoint with the given filter. Default
    loop: ``while True``, break when a page returns 0 results — identical
    to the upstream reference. ``max_pages`` is an optional safety cap.
    """
    params = dict(base_params)  # mutate a local copy
    obs_all: list[dict[str, Any]] = []
    while True:
        r = requests.get(BASE, params=params, timeout=HTTP_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            break

        obs_all.extend(results)
        total = data.get("total_results")
        print(
            f"  page {params['page']}: +{len(results)} obs "
            f"(running total {len(obs_all)} / {total or '?'})",
            flush=True,
        )

        params["page"] += 1
        if max_pages is not None and params["page"] > max_pages:
            break
        time.sleep(INTER_PAGE_SLEEP_SEC)

    return obs_all


def flatten_photos(
    observations: list[dict[str, Any]],
    *,
    seed: int,
    split_ratios: tuple[float, float, float],
) -> list[dict[str, Any]]:
    """Explode obs[*].photos[*] into per-photo records.

    Matches the upstream reference's ``all_records`` schema
    (observation_id, scientific_name, common_name, rank, photo_url,
    license_code, observed_on, place_guess, lat, lng) and appends three
    fields used by the downloader: ``slug``, ``photo_idx``, ``split``.

    The split key is the ``observation_id`` so all photos of the same
    observation land in the same split (avoids train/test leakage from
    near-duplicate frames).
    """
    out: list[dict[str, Any]] = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        slug = derive_slug(taxon)
        obs_id = obs.get("id")
        split = assign_split(str(obs_id), seed, split_ratios)
        for photo_idx, photo in enumerate(obs.get("photos", [])):
            out.append({
                "observation_id":  obs_id,
                "scientific_name": taxon.get("name"),
                "common_name":     taxon.get("preferred_common_name"),
                "rank":            taxon.get("rank"),
                "photo_url":       photo.get("url"),
                "license_code":    photo.get("license_code"),
                "observed_on":     obs.get("observed_on"),
                "place_guess":     obs.get("place_guess"),
                "lat": obs.get("geojson", {}).get("coordinates", [None, None])[1],
                "lng": obs.get("geojson", {}).get("coordinates", [None, None])[0],
                # Trailogy additions for the downloader / downstream
                # consumers; these do not affect the upstream record
                # schema (first 10 keys above).
                "slug":      slug,
                "photo_idx": photo_idx,
                "split":     split,
            })
    return out


def _download_one(job: tuple[str, Path]) -> bool:
    url, path = job
    if path.exists():
        return True  # idempotent
    try:
        r = requests.get(url, timeout=15.0)
        if r.status_code == 200 and r.content:
            path.write_bytes(r.content)
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def parse_split_ratios(s: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--split-ratios expects 'train,val,test'; got {s!r}"
        )
    try:
        ratios = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--split-ratios entries must be floats: {e}"
        ) from None
    total = sum(ratios)
    if total <= 0:
        raise argparse.ArgumentTypeError(
            f"--split-ratios sum must be > 0; got {total}"
        )
    if not abs(total - 1.0) < 1e-6:
        # Auto-normalise so e.g. '40,5,5' works as a shorthand.
        ratios = tuple(r / total for r in ratios)
    return ratios  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Top-level output dir (kept OUTSIDE the repo). "
             f"Contains observations.jsonl, fetch_report.json, and "
             f"train/val/test/<slug>/<obs>_<idx>.jpg subtrees. "
             f"Default: {str(DEFAULT_OUTPUT_DIR)!r}.",
    )
    # --- Geographic + taxon filter (defaults MUST match
    #     DEFAULT_PARAMS / upstream reference). ---
    ap.add_argument(
        "--lat", type=float, default=DEFAULT_PARAMS["lat"],
        help=f"Latitude of the search centre. Default: "
             f"{DEFAULT_PARAMS['lat']} (Pittsburgh example).",
    )
    ap.add_argument(
        "--lng", type=float, default=DEFAULT_PARAMS["lng"],
        help=f"Longitude of the search centre. Default: "
             f"{DEFAULT_PARAMS['lng']} (Pittsburgh example).",
    )
    ap.add_argument(
        "--radius", type=float, default=DEFAULT_PARAMS["radius"],
        help=f"Search radius in km. Default: {DEFAULT_PARAMS['radius']}.",
    )
    ap.add_argument(
        "--taxon-name", default=DEFAULT_PARAMS["taxon_name"],
        help=f"iNaturalist taxon_name filter. Default: "
             f"{DEFAULT_PARAMS['taxon_name']!r}.",
    )
    ap.add_argument(
        "--quality-grade", default=DEFAULT_PARAMS["quality_grade"],
        help=f"iNaturalist quality_grade filter. Default: "
             f"{DEFAULT_PARAMS['quality_grade']!r}.",
    )
    ap.add_argument(
        "--per-page", type=int, default=DEFAULT_PARAMS["per_page"],
        help=f"Page size (iNaturalist max=200). Default: "
             f"{DEFAULT_PARAMS['per_page']}.",
    )
    ap.add_argument(
        "--order-by", default=DEFAULT_PARAMS["order_by"],
        help=f"iNaturalist order_by. Default: "
             f"{DEFAULT_PARAMS['order_by']!r}.",
    )
    ap.add_argument(
        "--order", default=DEFAULT_PARAMS["order"],
        help=f"iNaturalist order direction. Default: "
             f"{DEFAULT_PARAMS['order']!r}.",
    )
    ap.add_argument(
        "--start-page", type=int, default=DEFAULT_PARAMS["page"],
        help=f"First page index to request. Default: "
             f"{DEFAULT_PARAMS['page']}.",
    )
    # --- Split + download knobs (Trailogy extensions). ---
    ap.add_argument(
        "--split-ratios", type=parse_split_ratios,
        default=DEFAULT_SPLIT_RATIOS,
        help="Train/val/test ratios as 'train,val,test'. Auto-normalised "
             f"if sum != 1. Default: "
             f"{','.join(str(r) for r in DEFAULT_SPLIT_RATIOS)}.",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Seed for the per-observation split hash. Same seed + same "
             "observation_id -> same split. Default: 42.",
    )
    ap.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Skip image downloads; write only observations.jsonl and "
             "fetch_report.json. Default: download is on.",
    )
    ap.set_defaults(download=True)
    ap.add_argument(
        "--max-pages", type=int, default=None,
        help="Optional safety cap on pagination depth. Default: None "
             "(paginate until results empty, matching the upstream "
             "reference).",
    )
    ap.add_argument(
        "--max-workers", type=int, default=32,
        help="Thread pool size for parallel image downloads. Default: 32.",
    )
    args = ap.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_params: dict[str, Any] = {
        "taxon_name":    args.taxon_name,
        "quality_grade": args.quality_grade,
        "photos":        "true",
        "lat":           args.lat,
        "lng":           args.lng,
        "radius":        args.radius,
        "per_page":      args.per_page,
        "page":          args.start_page,
        "order_by":      args.order_by,
        "order":         args.order,
    }

    print(
        f"== iNaturalist geographic fetch ==\n"
        f"  centre : lat={args.lat} lng={args.lng} radius={args.radius} km\n"
        f"  taxon  : {args.taxon_name!r}  quality={args.quality_grade!r}\n"
        f"  paging : per_page={args.per_page} order_by={args.order_by} "
        f"order={args.order}  max_pages={args.max_pages}\n"
        f"  splits : train/val/test = "
        f"{tuple(round(r, 4) for r in args.split_ratios)}  seed={args.seed}\n"
        f"  output : {args.output_dir}  download={args.download}",
        flush=True,
    )

    observations = paginate_observations(base_params, max_pages=args.max_pages)
    photo_records = flatten_photos(
        observations,
        seed=args.seed,
        split_ratios=args.split_ratios,
    )
    print(
        f"Found {len(observations)} observations "
        f"-> {len(photo_records)} photo records.",
        flush=True,
    )

    # Materialise metadata BEFORE downloading so a Ctrl-C mid-download
    # still leaves a useful trace of what was planned.
    obs_jsonl = args.output_dir / "observations.jsonl"
    with obs_jsonl.open("w") as f:
        for rec in photo_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {obs_jsonl}", flush=True)

    # Pre-bucket by split so we can also write per-split JSONLs (handy
    # for downstream consumers that just want one split).
    per_split_records: dict[str, list[dict[str, Any]]] = {
        s: [] for s in SPLIT_NAMES
    }
    for rec in photo_records:
        per_split_records[rec["split"]].append(rec)
    for split_name, recs in per_split_records.items():
        split_jsonl = args.output_dir / f"{split_name}.jsonl"
        with split_jsonl.open("w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(
            f"Wrote {split_jsonl} ({len(recs)} records)",
            flush=True,
        )

    requested_by_split: dict[str, dict[str, int]] = {
        s: {} for s in SPLIT_NAMES
    }
    jobs: list[tuple[str, Path]] = []
    for rec in photo_records:
        url = rec["photo_url"]
        if not url:
            continue
        split = rec["split"]
        slug = rec["slug"]
        (args.output_dir / split / slug).mkdir(parents=True, exist_ok=True)
        path = (
            args.output_dir
            / split
            / slug
            / f"{rec['observation_id']}_{rec['photo_idx']}.jpg"
        )
        jobs.append((url, path))
        requested_by_split[split][slug] = (
            requested_by_split[split].get(slug, 0) + 1
        )

    if not args.download:
        print(
            "--no-download set: skipping image downloads. "
            "(observations.jsonl + per-split JSONLs already on disk.)",
            flush=True,
        )
        _write_report(
            args, observations, photo_records,
            requested=requested_by_split,
            delivered={s: {} for s in SPLIT_NAMES},
            total_bytes_by_split={s: 0 for s in SPLIT_NAMES},
            n_succ=0, n_jobs=len(jobs),
        )
        return 0

    print(
        f"Downloading {len(jobs)} images with {args.max_workers} workers...",
        flush=True,
    )
    n_succ = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for ok in ex.map(_download_one, jobs):
            if ok:
                n_succ += 1

    delivered_by_split: dict[str, dict[str, int]] = {s: {} for s in SPLIT_NAMES}
    total_bytes_by_split: dict[str, int] = {s: 0 for s in SPLIT_NAMES}
    for split_name in SPLIT_NAMES:
        split_dir = args.output_dir / split_name
        if not split_dir.is_dir():
            continue
        for slug_dir in sorted(split_dir.iterdir()):
            if not slug_dir.is_dir():
                continue
            files = list(slug_dir.glob("*.jpg"))
            delivered_by_split[split_name][slug_dir.name] = len(files)
            total_bytes_by_split[split_name] += sum(
                f.stat().st_size for f in files
            )

    print()
    print("=== Summary ===")
    for split_name in SPLIT_NAMES:
        req_total = sum(requested_by_split[split_name].values())
        del_total = sum(delivered_by_split[split_name].values())
        bytes_mb = total_bytes_by_split[split_name] / 1024 / 1024
        print(
            f"  {split_name:5s}  species={len(requested_by_split[split_name]):4d}  "
            f"requested={req_total:5d}  delivered={del_total:5d}  "
            f"size={bytes_mb:7.1f} MB"
        )
    print(
        f"Overall: {n_succ}/{len(jobs)} downloads succeeded "
        f"(idempotent: re-running skips files already on disk)."
    )

    _write_report(
        args, observations, photo_records,
        requested=requested_by_split,
        delivered=delivered_by_split,
        total_bytes_by_split=total_bytes_by_split,
        n_succ=n_succ, n_jobs=len(jobs),
    )
    return 0


def _write_report(
    args: argparse.Namespace,
    observations: list[dict[str, Any]],
    photo_records: list[dict[str, Any]],
    *,
    requested: dict[str, dict[str, int]],
    delivered: dict[str, dict[str, int]],
    total_bytes_by_split: dict[str, int],
    n_succ: int,
    n_jobs: int,
) -> None:
    report = args.output_dir / "fetch_report.json"
    report.write_text(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "request_params": {
            "taxon_name":    args.taxon_name,
            "quality_grade": args.quality_grade,
            "photos":        "true",
            "lat":           args.lat,
            "lng":           args.lng,
            "radius":        args.radius,
            "per_page":      args.per_page,
            "page":          args.start_page,
            "order_by":      args.order_by,
            "order":         args.order,
        },
        "max_pages":          args.max_pages,
        "download":           args.download,
        "split_ratios":       list(args.split_ratios),
        "split_seed":         args.seed,
        "n_observations":     len(observations),
        "n_photo_records":    len(photo_records),
        "n_photos_requested": n_jobs,
        "n_photos_delivered": n_succ,
        "requested_by_split":          requested,
        "delivered_by_split":          delivered,
        "total_bytes_by_split":        total_bytes_by_split,
    }, indent=2))
    print(f"Wrote: {report}")


if __name__ == "__main__":
    sys.exit(main())
