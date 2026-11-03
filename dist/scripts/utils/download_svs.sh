#!/usr/bin/env bash
# =============================================================================
# download_svs.sh — download TCGA BLCA whole-slide images from the GDC
# =============================================================================
#
# The experiment needs the actual .svs files in image/.  This helper looks up
# each file name in scripts/utils/tcga_blca_slides.tsv and downloads it with
# the GDC Data Transfer Tool (gdc-client).
#
# Usage:
#   ./scripts/utils/download_svs.sh --demo
#       download the 5 slides used by this package's demo (3 MUT + 2 WT)
#
#   ./scripts/utils/download_svs.sh TCGA-....svs [TCGA-....svs ...]
#       download specific slides
#
#   echo TCGA-....svs | ./scripts/utils/download_svs.sh
#       read file names from stdin (one per line)
#
# Options:
#   -d, --outdir DIR        destination (default: <package>/image)
#   -c, --gdc-client PATH   gdc-client binary
#   -t, --tsv PATH          file-name -> file-id table
#   -h, --help
#
# gdc-client is looked up in <package>/bin/gdc-client, then on PATH.
# Install it from https://gdc.cancer.gov/access-data/gdc-data-transfer-tool
# or with conda:  conda install -c bioconda gdc-client
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TSV="$SCRIPT_DIR/tcga_blca_slides.tsv"
DEMO_LIST="$SCRIPT_DIR/demo_slides.txt"
OUTDIR="$ROOT/image"
GDC_CLIENT=""
USE_DEMO=0

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)       usage ;;
    --demo)          USE_DEMO=1; shift ;;
    -d|--outdir)     OUTDIR="$2"; shift 2 ;;
    -c|--gdc-client) GDC_CLIENT="$2"; shift 2 ;;
    -t|--tsv)        TSV="$2"; shift 2 ;;
    --)              shift; break ;;
    -*)              echo "unknown option: $1" >&2; exit 1 ;;
    *)               break ;;
  esac
done

# ---------------------------------------------------------------------------
# Locate gdc-client
# ---------------------------------------------------------------------------
if [ -z "$GDC_CLIENT" ]; then
  if [ -x "$ROOT/bin/gdc-client" ]; then
    GDC_CLIENT="$ROOT/bin/gdc-client"
  elif command -v gdc-client >/dev/null 2>&1; then
    GDC_CLIENT="$(command -v gdc-client)"
  fi
fi

if [ -z "$GDC_CLIENT" ] || [ ! -x "$GDC_CLIENT" ]; then
  cat >&2 <<EOF
ERROR: gdc-client not found.

Install the GDC Data Transfer Tool, then either put the binary at
  $ROOT/bin/gdc-client
or make it available on PATH:

  conda install -c bioconda gdc-client
  # or download from:
  # https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

You can also pass it explicitly:  $0 -c /path/to/gdc-client --demo
EOF
  exit 1
fi

if [ ! -f "$TSV" ]; then
  echo "ERROR: mapping table not found: $TSV" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Collect file names
# ---------------------------------------------------------------------------
FILES=()
if [ "$USE_DEMO" = "1" ]; then
  while IFS= read -r line; do
    line="$(printf '%s' "$line" | xargs)"
    [ -n "$line" ] && FILES+=("$line")
  done < "$DEMO_LIST"
elif [ $# -gt 0 ]; then
  FILES=("$@")
else
  if [ -t 0 ]; then
    echo "ERROR: no file names given. Use --demo, pass arguments, or pipe names on stdin." >&2
    exit 1
  fi
  while IFS= read -r line; do
    line="$(printf '%s' "$line" | xargs)"
    [ -n "$line" ] && FILES+=("$line")
  done
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "ERROR: no file names to download." >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# Map file names -> GDC file ids
# ---------------------------------------------------------------------------
IDS=()
for FILE in "${FILES[@]}"; do
  FILE="$(printf '%s' "$FILE" | xargs)"
  [ -z "$FILE" ] && continue

  if [ -f "$OUTDIR/$FILE" ]; then
    echo "already present: $FILE"
    continue
  fi

  ID="$(awk -F'\t' -v f="$FILE" '$4==f {print $3; exit}' "$TSV")"
  if [ -z "$ID" ]; then
    echo "WARNING: $FILE is not in $TSV — skipping" >&2
    continue
  fi
  echo "resolved: $FILE -> $ID"
  IDS+=("$ID")
done

if [ ${#IDS[@]} -eq 0 ]; then
  echo "Nothing to download (everything already present, or no known file names)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Download and flatten the UUID sub-folders gdc-client creates
# ---------------------------------------------------------------------------
echo "Downloading ${#IDS[@]} file(s) to $OUTDIR ..."
"$GDC_CLIENT" download "${IDS[@]}" -d "$OUTDIR"

echo "Flattening download folders..."
find "$OUTDIR" -mindepth 2 -type f -iname '*.svs' | while IFS= read -r f; do
  base="$(basename "$f")"
  if [ ! -e "$OUTDIR/$base" ]; then
    mv "$f" "$OUTDIR/$base"
  fi
done
find "$OUTDIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

echo
echo "Slides in $OUTDIR:"
find "$OUTDIR" -maxdepth 1 -type f -iname '*.svs' -printf '  %f\n' 2>/dev/null \
  || ls -1 "$OUTDIR"/*.svs 2>/dev/null || true