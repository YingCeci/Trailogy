#!/usr/bin/env bash
# Generates HikeCompanion.xcodeproj from project.yml using xcodegen.
#
# If xcodegen is not on PATH, falls back to a locally-built copy at
# /tmp/XcodeGen-src/.build/release/xcodegen (which scripts/bootstrap-tools.sh
# can produce).

set -euo pipefail
cd "$(dirname "$0")/.."

XCODEGEN_BIN=""
if command -v xcodegen >/dev/null 2>&1; then
  XCODEGEN_BIN="xcodegen"
elif [[ -x "/tmp/XcodeGen-src/.build/release/xcodegen" ]]; then
  XCODEGEN_BIN="/tmp/XcodeGen-src/.build/release/xcodegen"
else
  echo "xcodegen not found. Install via 'brew install xcodegen' or build from"
  echo "https://github.com/yonaskolb/XcodeGen and re-run." >&2
  exit 1
fi

echo "==> Generating Xcode project with $XCODEGEN_BIN ..."
"$XCODEGEN_BIN" generate
echo "==> Done. Open HikeCompanion.xcodeproj in Xcode."
