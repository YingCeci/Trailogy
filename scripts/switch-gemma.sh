#!/bin/bash
# Switch the active Gemma model variant.
#
# This script is a thin SWITCHER, not a builder. It does NOT know how to
# fetch, strip, finetune, or otherwise produce variants — that work lives
# in fetch-gemma.sh, strip-gemma-audio.py, the `feature/finetune-unsloth`
# branch, or wherever else you produce model directories. This script only:
#
#   1. Lists whatever <repo>/gemma-variants/*/ directories exist.
#   2. Materializes <repo>/HikeCompanion/Resources/Models/Gemma/ as a
#      copy of one of them.
#
# CRITICAL: variants live OUTSIDE Resources/Models/.
#
# project.yml uses `- path: Resources/Models, type: folder`, which is an
# Xcode blue-folder reference. At build time, builtin-copy recursively
# copies the ENTIRE Models/ tree into the .app bundle — there is no
# excludes mechanism. So if variants lived under Models/Gemma-*/, every
# variant would be embedded in the .app, blowing the bundle to tens of
# GB. They must live somewhere Xcode never looks at.
#
# Storage layout:
#
#   ../gemma-variants/<name>/      real directory, source of truth for a
#                                  variant. Sibling of hikeCompanion/ by
#                                  default — overridable via
#                                  GEMMA_VARIANTS_DIR env var. Auto-detected
#                                  by glob — no hardcoded list. <name> can
#                                  be anything (dashes allowed), e.g.
#                                  "default", "no-audio",
#                                  "lora-plantnet-50k-r8-a8-lr2e4".
#                                  Required contents: at minimum
#                                  config.json and a *.safetensors.
#                                  Lives outside the repo so it stays out
#                                  of git AND out of the .app bundle.
#
#   HikeCompanion/Resources/Models/Gemma/
#                                  ACTIVE working copy. cp'd from one of
#                                  the variants. Contains a marker file
#                                  `.active-variant` recording which
#                                  variant this copy was made from. iOS
#                                  code-signing requires this to be a
#                                  real directory (no symlinks).
#
# Disk efficiency:
#   On APFS (default macOS filesystem), this script uses `cp -c` so the
#   active copy is a clonefile — instant, near-zero extra disk usage.
#   On non-APFS filesystems, it falls back to `cp -R` (slower, full copy).
#
# Usage:
#   bash scripts/switch-gemma.sh
#       Show the active variant and list all available variants.
#
#   bash scripts/switch-gemma.sh <name>
#       Switch to ../gemma-variants/<name>/. Exact match required. On miss,
#       prints close substring matches as suggestions and exits non-zero.
#
#   bash scripts/switch-gemma.sh --adopt <name>
#       "Rescue" mode: takes whatever is currently at Models/Gemma/
#       (an unmanaged real directory with no marker file, typically the
#       immediate output of fetch-gemma.sh) and adopts it as
#       ../gemma-variants/<name>/, then re-materializes Models/Gemma/ as a
#       managed copy of it. Use this once after a fresh fetch.
#
#   GEMMA_VARIANTS_DIR=/some/other/path bash scripts/switch-gemma.sh ...
#       Override the variants location. Useful if you keep variants on
#       an external drive or shared with other Mac users.
#
# Typical workflows:
#
#   # First-time setup after fetch-gemma.sh:
#   bash scripts/fetch-gemma.sh
#   bash scripts/switch-gemma.sh --adopt default
#
#   # Add a stripped variant:
#   cp -c -R ../gemma-variants/default ../gemma-variants/no-audio
#   python3 scripts/strip-gemma-audio.py ../gemma-variants/no-audio
#   bash scripts/switch-gemma.sh no-audio
#
#   # Drop in a finetuned variant exported from a finetune branch/repo:
#   cp -c -R /path/to/mlx_export ../gemma-variants/lora-plantnet-50k-r8-a8-lr2e4
#   bash scripts/switch-gemma.sh lora-plantnet-50k-r8-a8-lr2e4
#
#   # Clean up an unused variant:
#   rm -rf ../gemma-variants/<name>
#
# Safety:
#   Switch refuses to destroy an unmanaged Models/Gemma/ (one without a
#   marker file). You'll be prompted to either --adopt it first or rm it
#   yourself.

set -euo pipefail
cd "$(dirname "$0")/.."

# Variants live OUTSIDE the repo by default — sibling of hikeCompanion/,
# matching how external/, bioclip-2/, PlantNet-300K/ etc. are laid out.
# Override with GEMMA_VARIANTS_DIR env var if you want them somewhere else.
GEMMA_VARIANTS_DIR="${GEMMA_VARIANTS_DIR:-../gemma-variants}"
VARIANTS_DIR="$GEMMA_VARIANTS_DIR"
GEMMA_DIR="HikeCompanion/Resources/Models/Gemma"
MARKER_FILE=".active-variant"

# ── Helpers ──

err() { echo "ERROR: $*" >&2; exit 1; }

# Read marker file inside Models/Gemma/. Empty string if missing.
active_variant() {
    if [ -f "$GEMMA_DIR/$MARKER_FILE" ]; then
        cat "$GEMMA_DIR/$MARKER_FILE"
    fi
}

# Enumerate variants by globbing gemma-variants/*/.
list_variant_names() {
    [ -d "$VARIANTS_DIR" ] || return 0
    local d name
    for d in "$VARIANTS_DIR"/*/; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        echo "$name"
    done
}

# Sanity-check that a candidate variant directory looks like a real model.
variant_looks_valid() {
    local dir="$1"
    [ -f "$dir/config.json" ] || return 1
    compgen -G "$dir/*.safetensors" > /dev/null 2>&1 || return 1
    return 0
}

# Copy with APFS clonefile if possible (instant, no extra disk), else
# fall back to regular recursive copy.
copy_dir() {
    local src="$1" dst="$2"
    local tmp="${dst}.tmp.$$"

    rm -rf "$tmp"
    if cp -c -R "$src" "$tmp" 2>/dev/null || cp -R "$src" "$tmp"; then
        rm -rf "$dst"
        mv "$tmp" "$dst"
    else
        rm -rf "$tmp"
        return 1
    fi
}

show_status() {
    if [ -L "$GEMMA_DIR" ]; then
        echo "Active: $(basename "$(readlink "$GEMMA_DIR")") (LEGACY symlink — switch to any variant to convert to copy-based)"
    elif [ -d "$GEMMA_DIR" ]; then
        local v
        v=$(active_variant)
        if [ -n "$v" ]; then
            echo "Active: $v (managed copy)"
        else
            echo "Active: <unmanaged> (Models/Gemma/ is a real directory with no marker file)"
            echo "        Run: bash scripts/switch-gemma.sh --adopt <name>"
            echo "        to bring it under switch-gemma management."
        fi
    else
        echo "Active: (none — Models/Gemma/ does not exist)"
    fi
}

list_variants() {
    echo ""
    echo "Available variants (auto-detected from $VARIANTS_DIR/*/):"
    local current names
    current=""
    [ -d "$GEMMA_DIR" ] && current=$(active_variant || true)

    names=$(list_variant_names)
    if [ -z "$names" ]; then
        echo "  (none — create one with:"
        echo "    bash scripts/fetch-gemma.sh && bash scripts/switch-gemma.sh --adopt default"
        echo "  )"
        return
    fi

    while IFS= read -r name; do
        [ -n "$name" ] || continue
        local dir="$VARIANTS_DIR/$name"
        local size marker valid
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        marker=""
        [ "$name" = "$current" ] && marker=" ← active"
        valid=""
        variant_looks_valid "$dir" || valid="  ⚠ missing config.json or *.safetensors"
        printf "  %s  (%s)%s%s\n" "$name" "$size" "$marker" "$valid"
    done <<< "$names"
}

# Suggest variants matching a substring of <query>. Used on miss.
suggest_matches() {
    local query="$1"
    local names hit=0
    names=$(list_variant_names)
    [ -z "$names" ] && return 1

    while IFS= read -r name; do
        [ -n "$name" ] || continue
        case "$name" in
            *"$query"*)
                if [ "$hit" = "0" ]; then
                    echo "  Did you mean one of:" >&2
                    hit=1
                fi
                echo "    bash scripts/switch-gemma.sh $name" >&2
                ;;
        esac
    done <<< "$names"
    [ "$hit" = "1" ]
}

# Materialize Models/Gemma/ as a fresh copy of gemma-variants/<name>/.
materialize() {
    local name="$1"
    local src="$VARIANTS_DIR/$name"

    [ -d "$src" ] || {
        echo "ERROR: Variant '$name' not found at $src" >&2
        suggest_matches "$name" || true
        exit 1
    }
    variant_looks_valid "$src" || err "Variant '$name' is missing config.json or *.safetensors at $src"

    if [ -L "$GEMMA_DIR" ]; then
        echo "  Removing legacy symlink Gemma → $(readlink "$GEMMA_DIR")"
        rm "$GEMMA_DIR"
    elif [ -d "$GEMMA_DIR" ]; then
        if [ -f "$GEMMA_DIR/$MARKER_FILE" ]; then
            echo "  Removing existing managed Gemma/ (was: $(active_variant))"
            rm -rf "$GEMMA_DIR"
        else
            cat >&2 <<EOF
ERROR: Models/Gemma/ exists as an unmanaged directory (no $MARKER_FILE).
Refusing to destroy it. Either:

  1. Adopt it under a name first:
       bash scripts/switch-gemma.sh --adopt <some-name>
     Then switch to '$name' afterwards.

  2. If you intentionally want to discard it:
       rm -rf $GEMMA_DIR
     Then re-run this command.
EOF
            exit 1
        fi
    fi

    # Make sure parent dir exists (Resources/Models/) — variant-only setups
    # may not have created it yet on a fresh repo.
    mkdir -p "$(dirname "$GEMMA_DIR")"

    echo "  Copying $src → $GEMMA_DIR (APFS clone if possible) ..."
    copy_dir "$src" "$GEMMA_DIR"
    echo "$name" > "$GEMMA_DIR/$MARKER_FILE"
}

cmd_switch() {
    local name="$1"
    materialize "$name"
    echo "Switched to: $name"
    echo ""
    echo "Next: bash scripts/generate-project.sh && rebuild in Xcode"
    echo "      (Xcode does NOT auto-detect Models/ changes — clean build"
    echo "      or delete the .app on the device for the new weights to land.)"
}

cmd_adopt() {
    local name="$1"
    [ -n "$name" ] || err "--adopt requires a variant name"

    local dst="$VARIANTS_DIR/$name"
    [ -e "$dst" ] && err "$dst already exists. Pick a different name or remove it first."

    if [ -L "$GEMMA_DIR" ]; then
        err "Models/Gemma/ is a symlink (legacy state). Resolve it manually before adopting."
    fi
    [ -d "$GEMMA_DIR" ] || err "Models/Gemma/ does not exist — nothing to adopt."
    variant_looks_valid "$GEMMA_DIR" || err "Models/Gemma/ doesn't look like a Gemma model (missing config.json or *.safetensors)."

    if [ -f "$GEMMA_DIR/$MARKER_FILE" ]; then
        err "Models/Gemma/ is already managed (active variant: $(active_variant)). --adopt is for unmanaged state only."
    fi

    mkdir -p "$VARIANTS_DIR"

    echo "==> Moving Models/Gemma/ → $dst ..."
    mv "$GEMMA_DIR" "$dst"

    echo "==> Re-materializing Models/Gemma/ as a managed copy of $name (APFS clone if possible) ..."
    copy_dir "$dst" "$GEMMA_DIR"
    echo "$name" > "$GEMMA_DIR/$MARKER_FILE"

    echo "Adopted '$name'. Now active."
    echo ""
    echo "Disk usage:"
    echo "  $dst       $(du -sh "$dst" | cut -f1) (variant, source of truth)"
    echo "  $GEMMA_DIR  $(du -sh "$GEMMA_DIR" | cut -f1) (active, APFS clone of variant)"
    echo ""
    echo "Note: on APFS the active copy shares blocks with the variant via clonefile,"
    echo "so the apparent total is misleading — physical disk usage is closer to one copy."
}

# ── Main ──

case "${1:-}" in
    "")
        show_status
        list_variants
        ;;
    --adopt)
        cmd_adopt "${2:-}"
        ;;
    -h|--help)
        sed -n '2,/^set -euo pipefail/p' "$0" | sed 's/^# \?//' | sed '$d'
        ;;
    --*)
        err "Unknown flag: $1 (try --help)"
        ;;
    *)
        cmd_switch "$1"
        ;;
esac
