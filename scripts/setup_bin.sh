#!/usr/bin/env bash
# setup_bin.sh — download gdc-client binary into bin/ if missing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN_DIR="$PROJECT_ROOT/bin"
GDC_URL="https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip"

mkdir -p "$BIN_DIR"

if [[ -x "$BIN_DIR/gdc-client" ]]; then
  echo "gdc-client already present: $BIN_DIR/gdc-client"
  "$BIN_DIR/gdc-client" --version 2>/dev/null || true
  exit 0
fi

echo "Downloading gdc-client 2.3 (Ubuntu x64) ..."
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

curl -fSL -o "$TMPDIR/gdc.zip" "$GDC_URL"
cd "$TMPDIR" && unzip -qo gdc.zip
# nested zip inside the outer zip
if [[ -f gdc-client_2.3_Ubuntu_x64.zip ]]; then
  unzip -qo gdc-client_2.3_Ubuntu_x64.zip
fi
cp "$TMPDIR"/gdc-client "$BIN_DIR/gdc-client"
chmod +x "$BIN_DIR/gdc-client"

echo "Installed: $BIN_DIR/gdc-client"
"$BIN_DIR/gdc-client" --version 2>/dev/null || true
