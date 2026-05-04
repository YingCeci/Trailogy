#!/usr/bin/env python3
"""Strip audio_tower / embed_audio weights from the bundled Gemma 4 model.

Why
---
The Gemma 4 E2B INT4 checkpoint we bundle (mlx-community/gemma-4-e2b-it-4bit)
is the multimodal variant: it carries language_model + vision_tower +
audio_tower. mlx-swift-lm filters audio weights at sanitize() time in BOTH
its current text-only loader (MLXLLM/Models/Gemma4.swift, lines 72-74) AND
its multimodal loader (MLXVLM/Models/Gemma4.swift, line 1754). So the
audio tower is dead bytes — about 930 MB of disk space the iPhone bundle
can never use.

How it works
------------
The safetensors format is dead simple:
  • 8 bytes: little-endian uint64 = header length (N)
  • N bytes: JSON header — dict of tensor_name -> {dtype, shape,
    data_offsets: [start, end]} (offsets relative to data buffer start)
  • rest of file: contiguous tensor bytes

We never interpret tensor values, so the bfloat16 dtype that confuses
numpy doesn't matter — we just copy raw bytes for the tensors we keep
and rewrite the header with new sequential offsets.

Memory cost: an 8 MB streaming chunk buffer, full stop. No torch dep,
no big in-memory dict.

Usage
-----
    python3 scripts/strip-gemma-audio.py
    # or
    python3 scripts/strip-gemma-audio.py path/to/Models/Gemma

Idempotent: if the .audio.bak backup already exists, the script no-ops
and tells you the file's already been stripped.

Restore
-------
    mv scripts/backups/model.safetensors.audio.bak \
       HikeCompanion/Resources/Models/Gemma/model.safetensors
    # or just re-run scripts/fetch-gemma.sh

Run scripts/generate-project.sh after stripping so Xcode rebundles.

NOTE: backup files MUST NOT live inside HikeCompanion/Resources/Models/
because that path is bundled wholesale into the .app via xcodegen's
`type: folder` reference — anything in there ships with the app. We
keep the .bak in scripts/backups/ instead (gitignored), so it sits next
to the script that produced it but never reaches the device.
"""
import json
import os
import struct
import sys
from pathlib import Path

AUDIO_PREFIXES = ("audio_tower", "embed_audio")
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
DEFAULT_MODEL_DIR = REPO_ROOT / "HikeCompanion" / "Resources" / "Models" / "Gemma"
# CRITICAL: backups go OUTSIDE the bundle resource path. The folder
# `HikeCompanion/Resources/Models` is referenced by xcodegen as
# `type: folder`, which copies *everything* inside (recursively) into
# the .app bundle on every build. A 3 GB .bak file in that path would
# ship to the device — exactly what stripping was supposed to avoid.
DEFAULT_BACKUP_DIR = SCRIPTS_DIR / "backups"
COPY_CHUNK = 8 * 1024 * 1024  # 8 MB streaming chunks

# safetensors recommends header padded to 8-byte alignment; not required
# by readers but keeps things tidy.
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
    backup_path = backup_dir / "model.safetensors.audio.bak"

    # Migration: detect a stale backup left in the OLD location (next to
    # the safetensors). Move it out of the bundle path before doing
    # anything else — leaving it there bloats the .app by 3+ GB.
    legacy_backup = model_dir / "model.safetensors.audio.bak"
    if legacy_backup.exists():
        print(
            f"NOTE: found legacy backup inside bundle path:\n"
            f"      {legacy_backup}\n"
            f"      Moving to {backup_path} so it isn't bundled into the .app."
        )
        if backup_path.exists():
            print(f"      (existing {backup_path.name} kept; legacy copy removed)")
            legacy_backup.unlink()
        else:
            os.rename(legacy_backup, backup_path)

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
    audio_keys = [k for k in header if any(k.startswith(p) for p in AUDIO_PREFIXES)]

    if not audio_keys:
        print("==> Nothing to strip — no audio_tower/embed_audio keys found.")
        return

    print(f"==> Found {len(audio_keys)} audio key(s) to drop")
    print(f"    keeping {len(header) - len(audio_keys)} tensor(s)")

    # Build new header with sequential offsets, in the original key order
    # (skipping audio). Track each kept tensor's original byte range so
    # we can copy its bytes through.
    kept_plan: list[tuple[str, int, int, int, int]] = []
    # (key, orig_start, orig_end, new_start, new_end)
    new_header: dict = {}
    if metadata is not None:
        new_header["__metadata__"] = metadata
    new_offset = 0
    for key, meta in header.items():
        if key in audio_keys:
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

    # 8-byte align (pad with spaces — safetensors readers tolerate this)
    pad_n = (-len(new_header_bytes)) % HEADER_ALIGN
    if pad_n:
        new_header_bytes = new_header_bytes + b" " * pad_n

    estimated_new_size = 8 + len(new_header_bytes) + new_data_size
    print(f"==> Estimated new file size: {fmt_bytes(estimated_new_size)} "
          f"(saves {fmt_bytes(orig_size - estimated_new_size)})")

    # ---- Pass 2: write new file, then rotate ----
    print(f"==> Backing up original to {backup_path.name}")
    os.rename(safetensors_path, backup_path)

    print(f"==> Writing stripped {safetensors_path.name}")
    try:
        with backup_path.open("rb") as fin, safetensors_path.open("wb") as fout:
            # Header
            fout.write(struct.pack("<Q", len(new_header_bytes)))
            fout.write(new_header_bytes)
            # Tensor blobs in order, copied from the original data buffer
            written = 0
            total = len(kept_plan)
            for i, (key, orig_start, orig_end, _ns, _ne) in enumerate(kept_plan, 1):
                fin.seek(data_start + orig_start)
                remaining = orig_end - orig_start
                while remaining > 0:
                    chunk = fin.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(
                            f"unexpected EOF while copying tensor {key}"
                        )
                    fout.write(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
                if i % 200 == 0 or i == total:
                    pct = (i / total) * 100
                    print(f"    [{i:5}/{total}]  {pct:5.1f}%  "
                          f"{fmt_bytes(written)} written",
                          flush=True)
    except Exception:
        # Roll back: restore original from backup
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

    # ---- Update the sidecar index file ----
    if index_path.exists():
        print(f"==> Updating {index_path.name}")
        with index_path.open() as f:
            idx = json.load(f)
        wm = idx.get("weight_map", {})
        removed = sum(1 for k in audio_keys if wm.pop(k, None) is not None)
        idx["weight_map"] = wm
        if isinstance(idx.get("metadata"), dict) and "total_size" in idx["metadata"]:
            del idx["metadata"]["total_size"]
        with index_path.open("w") as f:
            json.dump(idx, f, indent=2)
        print(f"    removed {removed} key(s) from weight_map")

    print()
    print(f"Backup kept at:")
    print(f"  {backup_path}")
    print(f"  (outside the .app bundle path — won't be shipped to device)")
    print()
    print("Next:  bash scripts/generate-project.sh   (so Xcode rebundles)")
    print(f"Restore:  mv {backup_path} {safetensors_path}")
    print(f"          or:  bash scripts/fetch-gemma.sh")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    main(target.expanduser().resolve())
