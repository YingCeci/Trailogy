#!/usr/bin/env python3
"""Strip vision_tower / multi_modal_projector weights from a Gemma 4 variant.

Why
---
With BioCLIP-2 handling species identification as a bypass classifier, a
future text-only app variant can drop Gemma's built-in vision tower
(SigLIP). Only use this for variants that will be loaded through MLXLLM
text mode. The current bioclip-bypass app still sends attached photos to
Gemma VLM, so a vision-stripped Gemma will not work with image asks until
the app is switched to text-only photo handling.

Stripping the vision tower saves ~700 MB on disk for text-only variants.
This script is designed to run AFTER strip-gemma-audio.py. If audio
weights are already stripped, this only removes vision. If neither has
been stripped, it removes both vision and audio in one pass.

How it works
------------
Same approach as strip-gemma-audio.py: read the safetensors header,
filter out unwanted keys, copy kept tensor bytes with sequential offsets.
Memory cost: 8 MB streaming chunk buffer. No torch dependency.

Usage
-----
    python3 scripts/strip-gemma-vision.py ../gemma-variants/no-vision

This script intentionally requires an explicit target path. Do not run it
against HikeCompanion/Resources/Models/Gemma unless you are deliberately
mutating the active bundled copy.

Restore
-------
    mv scripts/backups/model.safetensors.<variant>.vision.bak \\
       ../gemma-variants/<variant>/model.safetensors

After stripping, switch to the variant with scripts/switch-gemma.sh so the
active bundled copy is regenerated.
"""
import json
import os
import struct
import sys
from pathlib import Path

VISION_PREFIXES = (
    "vision_tower",
    "multi_modal_projector",
    "embed_vision",
    # Also catch any remaining audio keys in case strip-audio wasn't run
    "audio_tower",
    "embed_audio",
)

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
ACTIVE_MODEL_DIR = REPO_ROOT / "HikeCompanion" / "Resources" / "Models" / "Gemma"
DEFAULT_BACKUP_DIR = SCRIPTS_DIR / "backups"
COPY_CHUNK = 8 * 1024 * 1024  # 8 MB
HEADER_ALIGN = 8


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.0f} MB"
    return f"{n} B"


def read_header(fin) -> tuple[int, dict]:
    """Read the safetensors header. Returns (data_buffer_start, header_dict)."""
    header_len_bytes = fin.read(8)
    if len(header_len_bytes) != 8:
        raise ValueError("file too short to be a safetensors file")
    header_len = struct.unpack("<Q", header_len_bytes)[0]
    header_bytes = fin.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError("file truncated; couldn't read declared header")
    header = json.loads(header_bytes)
    return 8 + header_len, header


def main(model_dir: Path, backup_dir: Path = DEFAULT_BACKUP_DIR) -> None:
    safetensors_path = model_dir / "model.safetensors"
    index_path = model_dir / "model.safetensors.index.json"

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"model.safetensors.{model_dir.name}.vision.bak"

    if not safetensors_path.exists():
        sys.exit(
            f"ERROR: {safetensors_path} not found.\n"
            f"       Run scripts/fetch-gemma.sh first."
        )

    if backup_path.exists():
        print(
            f"NOTE: {backup_path.relative_to(REPO_ROOT)} already exists — original\n"
            f"      is preserved and the in-place file appears already stripped.\n"
            f"      Current size: {fmt_bytes(safetensors_path.stat().st_size)}\n"
            f"      To restore: mv {backup_path} {safetensors_path}"
        )
        return

    orig_size = safetensors_path.stat().st_size
    print(f"==> Reading {safetensors_path.name} ({fmt_bytes(orig_size)})")

    # ---- Pass 1: read header, plan new layout ----
    with safetensors_path.open("rb") as fin:
        data_start, header = read_header(fin)

    metadata = header.pop("__metadata__", None)
    strip_keys = [
        k for k in header
        if any(k.startswith(p) for p in VISION_PREFIXES)
    ]

    if not strip_keys:
        print("==> Nothing to strip — no vision_tower/multi_modal_projector keys found.")
        return

    # Categorize for reporting
    vision_keys = [k for k in strip_keys if k.startswith(("vision_tower", "multi_modal_projector", "embed_vision"))]
    audio_keys = [k for k in strip_keys if k.startswith(("audio_tower", "embed_audio"))]
    strip_set = set(strip_keys)

    print(f"==> Found {len(strip_keys)} key(s) to strip:")
    if vision_keys:
        print(f"    vision: {len(vision_keys)} tensor(s)")
    if audio_keys:
        print(f"    audio:  {len(audio_keys)} tensor(s) (bonus cleanup)")
    print(f"    keeping {len(header) - len(strip_keys)} tensor(s)")

    # Build new header with sequential offsets
    kept_plan: list[tuple[str, int, int, int, int]] = []
    new_header: dict = {}
    if metadata is not None:
        new_header["__metadata__"] = metadata
    new_offset = 0
    for key, meta in header.items():
        if key in strip_set:
            continue
        orig_start, orig_end = meta["data_offsets"]
        size = orig_end - orig_start
        kept_plan.append((key, orig_start, orig_end, new_offset, new_offset + size))
        new_header[key] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [new_offset, new_offset + size],
        }
        new_offset += size

    new_data_size = new_offset
    new_header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    pad_n = (-len(new_header_bytes)) % HEADER_ALIGN
    if pad_n:
        new_header_bytes = new_header_bytes + b" " * pad_n

    estimated_new_size = 8 + len(new_header_bytes) + new_data_size
    print(
        f"==> Estimated new file size: {fmt_bytes(estimated_new_size)} "
        f"(saves {fmt_bytes(orig_size - estimated_new_size)})"
    )

    # ---- Pass 2: write new file, then rotate ----
    print(f"==> Backing up to {backup_path.name}")
    os.rename(safetensors_path, backup_path)

    print(f"==> Writing stripped {safetensors_path.name}")
    try:
        with backup_path.open("rb") as fin, safetensors_path.open("wb") as fout:
            fout.write(struct.pack("<Q", len(new_header_bytes)))
            fout.write(new_header_bytes)
            written = 0
            total = len(kept_plan)
            for i, (key, orig_start, orig_end, _ns, _ne) in enumerate(kept_plan, 1):
                fin.seek(data_start + orig_start)
                remaining = orig_end - orig_start
                while remaining > 0:
                    chunk = fin.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"unexpected EOF while copying tensor {key}")
                    fout.write(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
                if i % 200 == 0 or i == total:
                    pct = (i / total) * 100
                    print(
                        f"    [{i:5}/{total}]  {pct:5.1f}%  "
                        f"{fmt_bytes(written)} written",
                        flush=True,
                    )
    except Exception:
        print("ERROR during write — rolling back from backup", file=sys.stderr)
        if safetensors_path.exists():
            safetensors_path.unlink()
        os.rename(backup_path, safetensors_path)
        raise

    new_size = safetensors_path.stat().st_size
    saved = orig_size - new_size
    print(
        f"==> Done. {fmt_bytes(new_size)}  "
        f"(saved {fmt_bytes(saved)}, {saved / orig_size * 100:.1f}%)"
    )

    # ---- Update sidecar index if present ----
    if index_path.exists():
        print(f"==> Updating {index_path.name}")
        with index_path.open() as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        removed = sum(1 for k in strip_keys if wm.pop(k, None) is not None)
        idx["weight_map"] = wm
        if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
            del idx["metadata"]["total_size"]
        with index_path.open("w") as f:
            json.dump(idx, f, indent=2)
        print(f"    removed {removed} key(s) from weight_map")

    print()
    print(f"Backup kept at:")
    print(f"  {backup_path}")
    print(f"  (outside .app bundle path — won't be shipped to device)")
    print()
    print("Next:  bash scripts/generate-project.sh   (so Xcode rebundles)")
    print(f"Restore:  mv {backup_path} {safetensors_path}")
    print(f"          or:  bash scripts/fetch-gemma.sh")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: python3 scripts/strip-gemma-vision.py ../gemma-variants/<variant>")
        print("Refusing to default to HikeCompanion/Resources/Models/Gemma; strip a source variant instead.")
        sys.exit(0 if len(sys.argv) == 2 else 2)

    target = Path(sys.argv[1]).expanduser().resolve()
    if target == ACTIVE_MODEL_DIR.resolve():
        sys.exit(
            "ERROR: refusing to strip the active bundled Gemma copy.\n"
            "       Create or copy a source variant under ../gemma-variants/ and strip that path instead."
        )
    main(target)
