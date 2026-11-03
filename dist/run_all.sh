#!/usr/bin/env bash
# =============================================================================
# FGFR3MUT — one-command reproduction of the full experiment
# =============================================================================
#
#   ./run_all.sh
#
# Steps (strictly sequential, no manual intervention):
#   1. create .venv + install requirements
#   2. discover the .svs slides in image/
#   3. download cBioPortal metadata for every slide
#   4. build results/labels.csv  (0 = WT, 1 = FGFR3 mutant)
#   5. tile every WSI (224 px @ ~1.0 um/px, Otsu tissue mask)
#   6. embed tiles with H-optimus-0 + train Chowder (patient-grouped CV)
#   7. out-of-fold inference + predictions/metrics/ROC curve
#
# Everything is written under ./data, ./results and ./output.  Re-running is
# cheap: finished downloads, tiles and feature caches are reused.
#
# See README.md for requirements, options and troubleshooting.
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------------------
# Defaults (override with environment variables or the flags below)
# ---------------------------------------------------------------------------
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv}"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_FETCH="${SKIP_FETCH:-0}"
SKIP_TILE="${SKIP_TILE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

FORCE_FETCH="${FORCE_FETCH:-0}"
FORCE_TILE="${FORCE_TILE:-0}"

DEVICE="${DEVICE:-auto}"
MAX_TILES="${MAX_TILES:-5000}"
TILE_SIZE="${TILE_SIZE:-224}"
TARGET_MPP="${TARGET_MPP:-1.0}"
MIN_TISSUE="${MIN_TISSUE:-0.60}"
SAVE_PNGS="${SAVE_PNGS:-0}"
EPOCHS="${EPOCHS:-30}"
SPLITS="${SPLITS:-5}"
REPEATS="${REPEATS:-1}"
SPLIT_UNIT="${SPLIT_UNIT:-patient}"
BATCH_SIZE="${BATCH_SIZE:-16}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"
TARGET_SENS="${TARGET_SENS:-0.92}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; BLUE=$'\033[1;34m'; GREEN=$'\033[1;32m'
  YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; RESET=$'\033[0m'
else
  BOLD=""; BLUE=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

log()  { printf '\n%s[%s]%s %s\n' "$BLUE" "$(date +%H:%M:%S)" "$RESET" "$*"; }
warn() { printf '%sWARNING:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%sERROR:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./run_all.sh [options]

Options:
  -h, --help            show this help and exit
  --skip-install        do not create .venv / install requirements
  --skip-fetch          do not contact cBioPortal (use existing ./data)
  --skip-tile           do not tile slides (use existing ./results/tiles_out)
  --skip-train          do not train (reuse ./results/models_trained/*.pt)
  --force-fetch         re-download metadata even if already present
  --force-tile          re-tile even if results/tiles_out/<slide>/ exists
  --max-tiles N         max tiles per slide (default: 5000, paper setting)
  --epochs N            Chowder training epochs (default: 30)
  --splits N            cross-validation folds (default: 5; auto-reduced)
  --repeats N           CV repeats (default: 1; paper recipe uses 25)
  --oof                  score each training slide only with the fold that held it out (honest CV)
  --no-oof               score with the ensemble average of all checkpoints
  --device DEV          auto | cpu | cuda:0 | cuda:1 (default: auto)
  --save-pngs           also write every kept tile as a PNG (large)
  --n-bootstrap N       bootstrap resamples for paper-style CIs (default: 1000)
  --target-sens X       operating-point sensitivity (default: 0.92)
  --batch-size N        training/inference batch size (default: 16)
  --split-unit U        patient (default, paper) | slide (leaky, for comparison)

Environment variables with the same names are also accepted, e.g.:
  SKIP_TRAIN=1 ./run_all.sh
  MAX_TILES=1000 EPOCHS=10 ./run_all.sh
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)          usage; exit 0 ;;
    --skip-install)     SKIP_INSTALL=1; shift ;;
    --skip-fetch)       SKIP_FETCH=1; shift ;;
    --skip-tile)        SKIP_TILE=1; shift ;;
    --skip-train)       SKIP_TRAIN=1; shift ;;
    --force-fetch)      FORCE_FETCH=1; shift ;;
    --force-tile)       FORCE_TILE=1; shift ;;
    --max-tiles)        MAX_TILES="$2"; shift 2 ;;
    --epochs)           EPOCHS="$2"; shift 2 ;;
    --splits)           SPLITS="$2"; shift 2 ;;
    --repeats)          REPEATS="$2"; shift 2 ;;
    --oof)              OOF=1; shift ;;
    --no-oof)           OOF=0; shift ;;
    --device)           DEVICE="$2"; shift 2 ;;
    --save-pngs)        SAVE_PNGS=1; shift ;;
    --n-bootstrap)      N_BOOTSTRAP="$2"; shift 2 ;;
    --target-sens)      TARGET_SENS="$2"; shift 2 ;;
    --batch-size)       BATCH_SIZE="$2"; shift 2 ;;
    --split-unit)       SPLIT_UNIT="$2"; shift 2 ;;
    *)                  die "unknown option: $1 (try --help)" ;;
  esac
done

# Default scoring mode: honest out-of-fold when this run trains the models,
# ensemble average when reusing checkpoints of unknown provenance.
if [ -z "${OOF:-}" ]; then
  if [ "$SKIP_TRAIN" = "1" ]; then OOF=0; else OOF=1; fi
fi

mkdir -p "$ROOT/results" "$ROOT/data" "$ROOT/image"

# ===========================================================================
# 1. Python environment
# ===========================================================================
log "${BOLD}STEP 1/7${RESET} — Python environment"

if [ ! -x "$VENV/bin/python" ] && [ "$SKIP_INSTALL" != "1" ]; then
  log "creating virtual environment: $VENV"
  "$PYTHON" -m venv "$VENV" || die \
    "could not create a virtual environment. On Debian/Ubuntu install python3-venv
       (sudo apt install python3-venv), or run with SKIP_INSTALL=1 PYTHON=/path/to/python."
fi

if [ -x "$VENV/bin/python" ]; then
  PY="$VENV/bin/python"
else
  PY="$PYTHON"
fi

"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' \
  || die "Python >= 3.9 is required (found $("$PY" --version 2>&1))."

if [ "$SKIP_INSTALL" != "1" ]; then
  log "installing Python dependencies (first run can take a few minutes)..."
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install -r "$ROOT/requirements.txt"
else
  log "SKIP_INSTALL=1 — using $PY as-is"
fi

"$PY" - <<'PY' || die "Python dependencies are incomplete. Re-run ./run_all.sh without --skip-install (or pip install -r requirements.txt into $PY)."
import importlib.util
import sys

required = [
    "numpy", "pandas", "sklearn", "scipy", "torch", "timm",
    "cv2", "skimage", "PIL", "openslide", "requests",
    "matplotlib", "loguru", "tqdm",
]
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print("missing Python modules: " + ", ".join(missing))
    sys.exit(1)

# find_spec() only proves the package exists; this actually loads the native
# libopenslide library (the most common system-level failure).
try:
    import openslide  # noqa: F401
except Exception as exc:  # pragma: no cover
    print(f"could not load OpenSlide: {exc}")
    print("On Debian/Ubuntu: sudo apt install libopenslide0")
    sys.exit(1)
PY

log "Python: $("$PY" --version 2>&1)"
"$PY" - <<'PY'
import torch
print(f"torch: {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
PY

# ===========================================================================
# 2. Input slides
# ===========================================================================
log "${BOLD}STEP 2/7${RESET} — Input slides"

SLIDES=()
for f in "$ROOT"/image/*.svs "$ROOT"/image/*.SVS; do
  [ -e "$f" ] || continue
  SLIDES+=("$f")
done

if [ ${#SLIDES[@]} -eq 0 ]; then
  cat >&2 <<EOF

${RED}ERROR:${RESET} no .svs files found in $ROOT/image/

Put the TCGA whole-slide images you want to process there, for example:

  cp /path/to/TCGA-*.svs $ROOT/image/

or download the 5 slides used by this package's demo:

  $ROOT/scripts/utils/download_svs.sh --demo

Training needs at least 2 patients per class (FGFR3 mutant = label 1 and
wild-type = label 0), because cross-validation is grouped by patient; the
demo set has 2 MUT + 2 WT patients (5 slides).
EOF
  exit 1
fi

log "found ${#SLIDES[@]} slide(s):"
for f in "${SLIDES[@]}"; do log "  - $(basename "$f")"; done

# ===========================================================================
# 3. cBioPortal metadata
# ===========================================================================
log "${BOLD}STEP 3/7${RESET} — cBioPortal metadata"

if [ "$SKIP_FETCH" = "1" ]; then
  log "SKIP_FETCH=1 — using existing metadata in $ROOT/data"
else
  need_fetch=0
  for f in "${SLIDES[@]}"; do
    stem="$(basename "$f" .svs)"
    if [ "$FORCE_FETCH" != "1" ] && compgen -G "$ROOT/data/$stem/cbioportal_*_summary.json" >/dev/null; then
      log "  cached: $stem"
    else
      need_fetch=1
    fi
  done

  if [ "$need_fetch" -eq 0 ] && [ "$FORCE_FETCH" != "1" ]; then
    log "all slide metadata already present — skipping download"
  else
    # Batch mode is much faster than one API call per slide: it fetches the
    # cohort table / expression / mutations once and writes one folder per
    # slide under data/<slide-stem>/.
    : > "$ROOT/results/slide_file_list.txt"
    for f in "${SLIDES[@]}"; do basename "$f" >> "$ROOT/results/slide_file_list.txt"; done
    "$PY" "$ROOT/scripts/analysis/fetch_cbioportal_data.py" \
      --file-list "$ROOT/results/slide_file_list.txt" \
      --outdir "$ROOT/data" \
      --batch-size 30
  fi
fi

# ===========================================================================
# 4. Ground-truth labels
# ===========================================================================
log "${BOLD}STEP 4/7${RESET} — Ground-truth labels"

# The activating-only table is the paper's positive-class definition (the 11
# activating FGFR3 mutations). Slides absent from it are labelled from the
# cBioPortal records downloaded above.
STATUS_CSV="$ROOT/reference/per_slide_table.csv"
if [ -f "$STATUS_CSV" ]; then
  STATUS_ARGS=(--status_csv "$STATUS_CSV")
else
  STATUS_ARGS=(--no_status_csv)
fi

"$PY" "$ROOT/scripts/analysis/make_labels_csv.py" \
  --tiles_root "$ROOT/results/tiles_out" \
  --slides_dir "$ROOT/image" \
  --data_dir "$ROOT/data" \
  --out_csv "$ROOT/results/labels.csv" \
  "${STATUS_ARGS[@]}"

"$PY" - "$ROOT/results/labels.csv" "$SKIP_TRAIN" <<'PY'
import collections
import csv
import re
import sys

path, skip_train = sys.argv[1], sys.argv[2] == "1"


def patient(slide: str) -> str:
    m = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", slide)
    return m.group(1) if m else slide.rsplit(".", 1)[0]


with open(path, newline="") as fh:
    rows = list(csv.DictReader(fh))
if not rows:
    sys.exit(f"ERROR: {path} is empty")

counts = collections.Counter(int(r["label"]) for r in rows)
patients = {
    lab: len({patient(r["slide"]) for r in rows if int(r["label"]) == lab})
    for lab in (0, 1)
}
n_mut, n_wt = counts.get(1, 0), counts.get(0, 0)
p_mut, p_wt = patients.get(1, 0), patients.get(0, 0)
print(f"  labelled slides: {len(rows)}  (MUT={n_mut}, WT={n_wt})")
print(f"  distinct patients: {p_mut + p_wt}  (MUT={p_mut}, WT={p_wt})")

if not skip_train and (n_mut < 1 or n_wt < 1 or p_mut < 2 or p_wt < 2):
    sys.exit(
        "\nERROR: training needs at least 2 patients per class (MUT and WT).\n"
        f"       Found {len(rows)} slide(s) = {p_mut + p_wt} patient(s): "
        f"MUT={n_mut} slide(s)/{p_mut} patient(s), WT={n_wt} slide(s)/{p_wt} patient(s).\n"
        "       Splitting is patient-grouped (the paper's unit), so two slides from\n"
        "       one patient cannot form two folds.\n"
        "       Add more .svs files to image/ (see scripts/utils/download_svs.sh --demo),\n"
        "       or run with SKIP_TRAIN=1 if checkpoints already exist in results/models_trained/."
    )
PY

# ===========================================================================
# 5. WSI tiling (preprocessing)
# ===========================================================================
log "${BOLD}STEP 5/7${RESET} — WSI tiling"

if [ "$SKIP_TILE" = "1" ]; then
  log "SKIP_TILE=1 — using existing tiles in $ROOT/results/tiles_out"
else
  TILE_EXTRA=()
  if [ "$SAVE_PNGS" = "1" ]; then TILE_EXTRA+=(--save_pngs); fi

  for f in "${SLIDES[@]}"; do
    stem="$(basename "$f" .svs)"
    if [ "$FORCE_TILE" != "1" ] && [ -f "$ROOT/results/tiles_out/$stem/metadata.json" ]; then
      log "  cached tiles: $stem"
      continue
    fi
    log "  tiling: $(basename "$f")"
    "$PY" "$ROOT/scripts/preprocessing/fgfr3mut_tile_wsi.py" \
      --svs_path "$f" \
      --out_dir "$ROOT/results/tiles_out" \
      --tissue_detector otsu \
      --device "$DEVICE" \
      --target_mpp "$TARGET_MPP" \
      --tile_size "$TILE_SIZE" \
      --min_tissue_fraction "$MIN_TISSUE" \
      --max_tiles "$MAX_TILES" \
      --seed 0 \
      ${TILE_EXTRA[@]+"${TILE_EXTRA[@]}"}
  done
fi

# Verify that every labelled slide actually produced tiles before paying the
# (expensive) H-optimus-0 embedding cost.
"$PY" - "$ROOT/results/tiles_out" "$ROOT/results/labels.csv" "$SKIP_TRAIN" <<'PY'
import collections
import csv
import re
import sys
from pathlib import Path

import numpy as np

tiles_root, labels_csv, skip_train = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1"
with open(labels_csv, newline="") as fh:
    rows = list(csv.DictReader(fh))


def patient(slide: str) -> str:
    m = re.match(r"(TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4})", slide)
    return m.group(1) if m else slide.rsplit(".", 1)[0]


counts = collections.Counter()
patients = {0: set(), 1: set()}
missing, empty = [], []
for r in rows:
    slide, label = r["slide"], int(r["label"])
    coords = tiles_root / slide / "coords.npy"
    if not coords.exists():
        missing.append(slide)
        continue
    n = int(np.load(coords, mmap_mode="r").shape[0])
    if n == 0:
        empty.append(slide)
    else:
        counts[label] += 1
        patients[label].add(patient(slide))

print(f"  slides with kept tiles: {sum(counts.values())}  "
      f"(MUT={counts.get(1, 0)}, WT={counts.get(0, 0)})")
print(f"  patients with kept tiles: {len(patients[1]) + len(patients[0])}  "
      f"(MUT={len(patients[1])}, WT={len(patients[0])})")
if missing:
    print("  WARNING: no tile output for: " + ", ".join(missing))
if empty:
    print("  WARNING: 0 tiles kept for: " + ", ".join(empty))

if not skip_train and (
    counts.get(1, 0) < 1
    or counts.get(0, 0) < 1
    or len(patients[1]) < 2
    or len(patients[0]) < 2
):
    sys.exit(
        "\nERROR: after tiling there are not enough usable cases to train.\n"
        "       Splitting is patient-grouped, so each class needs at least\n"
        "       2 patients (not just 2 slides). Check the tiling warnings\n"
        "       above, raise --max-tiles, or add more slides to image/."
    )
PY

# ===========================================================================
# 6. Feature extraction + training
# ===========================================================================
log "${BOLD}STEP 6/7${RESET} — H-optimus-0 features + Chowder training"

if [ "$SKIP_TRAIN" = "1" ]; then
  if ! find "$ROOT/results/models_trained" -name '*.pt' -print -quit 2>/dev/null | grep -q .; then
    die "SKIP_TRAIN=1 but no *.pt checkpoints found in $ROOT/results/models_trained/"
  fi
  log "SKIP_TRAIN=1 — reusing existing checkpoints"
else
  # Remove the whole checkpoint tree (including split_<repeat>/ subfolders and
  # the old manifest) so inference cannot ensemble models from a different run.
  rm -rf "$ROOT/results/models_trained"
  mkdir -p "$ROOT/results/models_trained"

  log "training (embedding is cached in results/features_cache after the first run)"
  "$PY" "$ROOT/scripts/fgfr3mut_training/train_fgfr3mut.py" \
    --tiles_root "$ROOT/results/tiles_out" \
    --slides_dir "$ROOT/image" \
    --labels_csv "$ROOT/results/labels.csv" \
    --feat_dir "$ROOT/results/features_cache" \
    --out_dir "$ROOT/results/models_trained" \
    --epochs "$EPOCHS" \
    --splits "$SPLITS" \
    --repeats "$REPEATS" \
    --split_unit "$SPLIT_UNIT" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE"
fi

# n_extreme is no longer passed here: train_fgfr3mut.py records the exact value
# per checkpoint in results/models_trained/train_manifest.json and inference
# reads it back automatically.

# ===========================================================================
# 7. Inference + metrics
# ===========================================================================
log "${BOLD}STEP 7/7${RESET} — Inference: predictions + metrics + ROC curve"

INFER_EXTRA=()
if [ "$OOF" = "1" ]; then
  log "scoring mode: out-of-fold (each slide scored only by the fold that held it out)"
  INFER_EXTRA+=(--oof)
else
  log "scoring mode: ensemble average over all checkpoints"
fi

"$PY" "$ROOT/scripts/fgfr3mut_training/infer_fgfr3mut.py" \
  --tiles_root "$ROOT/results/tiles_out" \
  --slides_dir "$ROOT/image" \
  --labels_csv "$ROOT/results/labels.csv" \
  --feat_dir "$ROOT/results/features_cache" \
  --checkpoints_dir "$ROOT/results/models_trained" \
  --out_csv "$ROOT/results/predictions.csv" \
  --metrics_out "$ROOT/results/metrics.json" \
  --roc_out "$ROOT/results/roc_curve.png" \
  --batch_size "$BATCH_SIZE" \
  --n_bootstrap "$N_BOOTSTRAP" \
  --target_sens "$TARGET_SENS" \
  --device "$DEVICE" \
  ${INFER_EXTRA[@]+"${INFER_EXTRA[@]}"}

# ===========================================================================
# Done
# ===========================================================================
log "${GREEN}${BOLD}DONE${RESET}"
cat <<EOF

Outputs
  labels:       $ROOT/results/labels.csv
  predictions:  $ROOT/results/predictions.csv
  metrics:      $ROOT/results/metrics.json
  ROC curve:    $ROOT/results/roc_curve.png
  tiles:        $ROOT/results/tiles_out/
  checkpoints:  $ROOT/results/models_trained/
  manifest:     $ROOT/results/models_trained/train_manifest.json
  embeddings:   $ROOT/results/features_cache/
  metadata:     $ROOT/data/

Re-run this script at any time; completed work is reused.
EOF