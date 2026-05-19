#!/usr/bin/env python3
"""Fetch plant observations near a target location from iNaturalist.

**Geographically-grounded data collection.** The default lat/lng centres
on Pittsburgh, PA (40.4406, -79.9959) with an 80 km radius — this is
the region of Trailogy's three trails (Kildoo / McConnells Mill,
Jennings, Frick Park, all in western PA). Training data therefore
matches the species distribution users actually encounter on the
supported trails, which the plain PlantNet-50k corpus does not
represent (it skews global and South-American-heavy, with almost no
NA street/forest trees).

The lat/lng is THE primary design choice of this script: change it to
target a different deployment region. Every other knob (taxon filter,
page size, ordering) defaults to match the upstream reference
``na_tree_fetch.py`` byte-for-byte.

Default request params (identical to the upstream reference):

    {
        "taxon_name":    "Plantae",
        "quality_grade": "research",
        "photos":        "true",
        "lat":           40.4406,
        "lng":           -79.9959,
        "radius":        80,
        "per_page":      200,
        "page":          1,        # advances 1..N until results empty
        "order_by":      "created_at",
        "order":         "desc",
    }

Default behaviour: paginate every page until ``results`` is empty,
sleep 1 s between pages, write one JSON line per photo to
``observations.jsonl`` in ``--output-dir``. **No images are downloaded
by default** — pass ``--download`` to additionally pull the photo
bytes into ``<output_dir>/<slug>/<obs_id>_<photo_idx>.jpg`` for
``prepare_na_trees.py``.

Per-record schema (matches the upstream reference, with two trailing
additions consumed by ``prepare_na_trees.py``):

    observation_id, scientific_name, common_name, rank,
    photo_url, license_code, observed_on, place_guess,
    lat, lng,
    slug, photo_idx              # <- added; slug = sanitised taxon name

Usage::

    # Default sweep — Pittsburgh, 80 km, all plants, metadata only.
    python src/data_mix/scripts/na_tree_fetch.py \\
        --output-dir $NA_TREES_ROOT

    # Same sweep, also download the photos.
    python src/data_mix/scripts/na_tree_fetch.py \\
        --output-dir $NA_TREES_ROOT --download

Network: paginated request loop. A radius-80 km Pittsburgh sweep
typically yields a few thousand research-grade plant observations.
iNaturalist rate-limits to ~60 req/min per IP; the 1 s inter-page
sleep keeps us safely under that.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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


def paginate_observations(
    base_params: dict[str, Any],
    *,
    max_pages: int | None = None,
) -> list[dict[str, Any]]:
    """Walk the iNaturalist obs endpoint with the given geographic +
    taxon filter. Default loop: ``while True``, break when a page
    returns 0 results — identical to the upstream reference.

    ``max_pages`` is an optional safety cap (None = no cap, matching
    the upstream reference).
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


def flatten_photos(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Explode obs[*].photos[*] into per-photo records. Matches the
    upstream reference's ``all_records`` schema (observation_id,
    scientific_name, common_name, rank, photo_url, license_code,
    observed_on, place_guess, lat, lng) and appends two fields used by
    the downstream downloader / ``prepare_na_trees.py`` (slug,
    photo_idx).
    """
    out: list[dict[str, Any]] = []
    for obs in observations:
        taxon = obs.get("taxon") or {}
        slug = derive_slug(taxon)
        for photo_idx, photo in enumerate(obs.get("photos", [])):
            out.append({
                "observation_id":  obs["id"],
                "scientific_name": taxon.get("name"),
                "common_name":     taxon.get("preferred_common_name"),
                "rank":            taxon.get("rank"),
                "photo_url":       photo.get("url"),
                "license_code":    photo.get("license_code"),
                "observed_on":     obs.get("observed_on"),
                "place_guess":     obs.get("place_guess"),
                "lat": obs.get("geojson", {}).get("coordinates", [None, None])[1],
                "lng": obs.get("geojson", {}).get("coordinates", [None, None])[0],
                # Trailogy additions (downstream-only; do not affect the
                # upstream record schema):
                "slug":      slug,
                "photo_idx": photo_idx,
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--output-dir", type=Path, default=Path("./na_trees"),
        help="Top-level output dir; observations.jsonl lands here, "
             "and ``--download`` writes images under <slug>/ subfolders. "
             "Default: ./na_trees",
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
    # --- Trailogy-specific extensions (default = no-op, so the
    #     default behaviour matches the upstream reference). ---
    ap.add_argument(
        "--max-pages", type=int, default=None,
        help="Optional safety cap on pagination depth. Default: None "
             "(paginate until results empty, matching the upstream "
             "reference).",
    )
    ap.add_argument(
        "--download", action="store_true",
        help="Also download each photo to "
             "<output_dir>/<slug>/<obs_id>_<photo_idx>.jpg. Default: "
             "off (matches the upstream reference, which only collects "
             "metadata).",
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
        f"  centre: lat={args.lat} lng={args.lng} radius={args.radius} km\n"
        f"  taxon : {args.taxon_name!r}  quality={args.quality_grade!r}\n"
        f"  paging: per_page={args.per_page} order_by={args.order_by} "
        f"order={args.order}  max_pages={args.max_pages}\n"
        f"  output: {args.output_dir}  download={args.download}",
        flush=True,
    )

    observations = paginate_observations(base_params, max_pages=args.max_pages)

    photo_records = flatten_photos(observations)
    print(
        f"Found {len(observations)} observations "
        f"-> {len(photo_records)} photo records.",
        flush=True,
    )

    obs_jsonl = args.output_dir / "observations.jsonl"
    with obs_jsonl.open("w") as f:
        for rec in photo_records:
            f.write(json.dumps(rec) + "\n")
    print(f"Wrote {obs_jsonl}", flush=True)

    if not args.download:
        print(
            "Metadata-only mode (matches upstream reference). "
            "Re-run with --download to also pull image bytes.",
            flush=True,
        )
        _write_report(args, observations, photo_records, requested={},
                      delivered={}, total_bytes=0, n_succ=0, n_jobs=0)
        return 0

    requested: dict[str, int] = {}
    jobs: list[tuple[str, Path]] = []
    for rec in photo_records:
        slug = rec["slug"]
        url = rec["photo_url"]
        if not url:
            continue
        (args.output_dir / slug).mkdir(exist_ok=True)
        path = (
            args.output_dir
            / slug
            / f"{rec['observation_id']}_{rec['photo_idx']}.jpg"
        )
        jobs.append((url, path))
        requested[slug] = requested.get(slug, 0) + 1

    print(
        f"Downloading {len(jobs)} images with {args.max_workers} workers...",
        flush=True,
    )
    n_succ = 0
    with cf.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for ok in ex.map(_download_one, jobs):
            if ok:
                n_succ += 1

    delivered: dict[str, int] = {}
    total_bytes = 0
    for slug in sorted(requested):
        files = list((args.output_dir / slug).glob("*.jpg"))
        delivered[slug] = len(files)
        total_bytes += sum(f.stat().st_size for f in files)

    print()
    print("=== Summary ===")
    for slug in sorted(delivered):
        u = requested.get(slug, 0)
        d = delivered[slug]
        print(f"  {slug:35s} requested={u:4d}  delivered={d:4d}")
    print(
        f"Total bytes on disk: {total_bytes / 1024 / 1024:.1f} MB "
        f"({n_succ}/{len(jobs)} downloads succeeded)"
    )

    _write_report(args, observations, photo_records, requested=requested,
                  delivered=delivered, total_bytes=total_bytes,
                  n_succ=n_succ, n_jobs=len(jobs))
    return 0


def _write_report(
    args: argparse.Namespace,
    observations: list[dict[str, Any]],
    photo_records: list[dict[str, Any]],
    *,
    requested: dict[str, int],
    delivered: dict[str, int],
    total_bytes: int,
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
        "n_observations":     len(observations),
        "n_photo_records":    len(photo_records),
        "n_photos_requested": n_jobs,
        "n_photos_delivered": n_succ,
        "requested":          requested,
        "delivered":          delivered,
        "total_bytes":        total_bytes,
    }, indent=2))
    print(f"Wrote: {report}")


if __name__ == "__main__":
    sys.exit(main())
