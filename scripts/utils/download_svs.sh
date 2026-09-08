#!/usr/bin/env bash
# download_svs.sh — download any TCGA SVS by file_name via GDC gdc-client
# Default image dir: /home/fahad/Documents/PROJECTS/Pathology_Image_Processing/image
# GDC client: ~/Downloads/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04/gdc-client_2.3_Ubuntu_x64/gdc-client
# TSV map: ~/Documents/PROJECTS/Pathology_Image_Processing/output/tcga_blca_slides.tsv (col4=file_name, col3=file_id)
#
# Usage:
#   ./scripts/utils/download_svs.sh TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs
#   ./scripts/utils/download_svs.sh -d /tmp/my_images TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs TCGA-FD-A3NA-01Z-00-DX1.2AD62CEE-0D76-4382-AE2E-9B7FCB1130D9.svs
#   ./scripts/utils/download_svs.sh --help
#   echo "TCGA-....svs" | ./scripts/utils/download_svs.sh          # stdin

set -euo pipefail

GDC_CLIENT="$HOME/Downloads/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04/gdc-client_2.3_Ubuntu_x64/gdc-client"
TSV="$HOME/Documents/PROJECTS/Pathology_Image_Processing/output/tcga_blca_slides.tsv"
OUTDIR="/home/fahad/Documents/PROJECTS/Pathology_Image_Processing/image"

# --- args ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $(basename "$0") [-d OUTDIR] [-c GDC_CLIENT] [-t TSV] FILE.svs [FILE2.svs ...]"
  echo "  Default OUTDIR: $OUTDIR"
  echo "  Default GDC_CLIENT: $GDC_CLIENT"
  echo "  Default TSV: $TSV"
  echo "  Reads file names from args or stdin (one per line)."
  exit 0
fi

while [[ $# -gt 0 && "$1" == -* ]]; do
  case "$1" in
    -d|--outdir) OUTDIR="$2"; shift 2;;
    -c|--gdc-client) GDC_CLIENT="$2"; shift 2;;
    -t|--tsv) TSV="$2"; shift 2;;
    --) shift; break;;
    -*) echo "Unknown option $1" >&2; exit 1;;
  esac
done

# collect file names from args or stdin
FILES=()
if [[ $# -gt 0 ]]; then
  FILES=("$@")
else
  # stdin
  if [[ -t 0 ]]; then
    echo "Error: no FILE.svs given. Pass as argument or via stdin." >&2
    echo "Example: $0 TCGA-2F-A9KQ-01Z-00-DX1.1C8CB2DD-5CC6-4E99-A0F9-32A0F598F5F9.svs" >&2
    exit 1
  fi
  mapfile -t FILES < /dev/stdin
fi

# --- checks ---
if [[ ! -x "$GDC_CLIENT" ]]; then
  echo "Error: gdc-client not found/executable: $GDC_CLIENT" >&2
  exit 1
fi
if [[ ! -f "$TSV" ]]; then
  echo "Error: TSV not found: $TSV" >&2
  exit 1
fi
mkdir -p "$OUTDIR"
command -v curl >/dev/null || { echo "Error: curl required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "Error: python3 required" >&2; exit 1; }

# --- build manifest for gdc-client: id filename md5 size state ---
MANIFEST=$(mktemp /tmp/gdc_manifest.XXXXXX.txt)
trap 'rm -f "$MANIFEST"' EXIT

echo -e "id\tfilename\tmd5\tsize\tstate" > "$MANIFEST"
ADDED=0
for FILE in "${FILES[@]}"; do
  FILE=$(echo "$FILE" | xargs) # trim
  [[ -z "$FILE" ]] && continue
  ID=$(awk -F'\t' -v f="$FILE" '$4==f{print $3}' "$TSV" | head -n1)
  if [[ -z "$ID" ]]; then
    echo "Warning: $FILE not found in $TSV — skipping" >&2
    continue
  fi
  echo "Resolving $FILE -> $ID ..." >&2
  # fetch md5/size/state from GDC API
  META=$(curl -s "https://api.gdc.cancer.gov/files/$ID?pretty=true")
  # python extracts fields; fails fast if ID invalid
  LINE=$(echo "$META" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)['data']
    print(f\"{d['file_id']}\t{d['file_name']}\t{d['md5sum']}\t{d['file_size']}\t{d['state']}\")
except Exception as e:
    print(f\"ERROR:{e}\", file=sys.stderr)
    sys.exit(1)
")
  if [[ "$LINE" == ERROR* || -z "$LINE" ]]; then
    echo "Warning: GDC lookup failed for $FILE ($ID) — skipping" >&2
    continue
  fi
  # avoid duplicate ids
  if grep -q "^$ID\t" "$MANIFEST"; then
    echo "Note: $FILE duplicate id $ID already in manifest — skipping duplicate" >&2
    continue
  fi
  echo -e "$LINE" >> "$MANIFEST"
  ADDED=$((ADDED+1))
done

if [[ $ADDED -eq 0 ]]; then
  echo "Error: no valid entries in manifest. Aborting." >&2
  cat "$MANIFEST" >&2
  exit 1
fi

echo "Manifest ready ($ADDED file(s)) -> $MANIFEST" >&2
cat "$MANIFEST" >&2
echo "Downloading to $OUTDIR via $GDC_CLIENT ..." >&2

# gdc-client will create subfolders by file id; we keep -d OUTDIR
"$GDC_CLIENT" download -m "$MANIFEST" -d "$OUTDIR"

echo "Done. Files in $OUTDIR :" >&2
ls -lh "$OUTDIR"/*"${FILES[0]:0:20}"* 2>/dev/null | head -n 20 >&2 || ls -lh "$OUTDIR" | tail -n 20 >&2
# flatten if gdc-client created UUID subdirs (optional): move *.svs up one level
# find "$OUTDIR" -type f -name "*.svs" -exec mv -n {} "$OUTDIR"/ \; 2>/dev/null || true
