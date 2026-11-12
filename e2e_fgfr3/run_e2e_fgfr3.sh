#!/usr/bin/env bash
# =============================================================================
# FGFR3 mutation with E2E-ABMILX — one command, same data as the Chowder run
# =============================================================================
#
#   ./e2e_fgfr3/run_e2e_fgfr3.sh            (from anywhere; paths are repo-relative)
#
# Steps (each skips work that is already done):
#   1. create .venv-e2e + install E2E-ABMILX requirements
#   2. check the FGFR3 inputs (manifest, slides, tiles); tile missing slides
#   3. FGFR3 coords.npy -> CLAM patches/*.h5   (main tiles + 2x2 sub blocks)
#   4. CLAM/save_patches.py -> LMDBs           (main_224, main_256, sub_256)
#   5. fold CSVs = FGFR3 split_<REPEAT> folds  (test = Chowder held-out fold)
#   6. E2E training, 5 folds, + out-of-fold per-slide predictions
#
# The E2E code in third_party/E2E-WSI-ABMILX is run unmodified.
# See e2e_fgfr3/README.md for details and options.
# =============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
E2E="$ROOT/third_party/E2E-WSI-ABMILX"

# ---------------------------------------------------------------------------
# Settings (override with environment variables)
# ---------------------------------------------------------------------------
SLIDES_DIR="${SLIDES_DIR:-$ROOT/image}"
TILES_ROOT="${TILES_ROOT:-$ROOT/results/tiles_out}"
MANIFEST="${MANIFEST:-$ROOT/results/models_trained_125/train_manifest.json}"
FGFR3_TRAIN_SCRIPT="${FGFR3_TRAIN_SCRIPT:-$ROOT/scripts/fgfr3mut_training/train_fgfr3mut.py}"
FGFR3_TILER="${FGFR3_TILER:-$ROOT/scripts/preprocessing/fgfr3mut_tile_wsi.py}"
WORK_DIR="${WORK_DIR:-$ROOT/results/e2e_fgfr3}"

REPEAT="${REPEAT:-0}"
VAL_FRAC="${VAL_FRAC:-0.125}"
NUM_GPUS="${NUM_GPUS:-1}"
FOLD_START="${FOLD_START:-0}"
CONFIG="${CONFIG:-$HERE/config/r18_fgfr3.yaml}"
MODEL="${MODEL:-e2e_r18_abmilx}"
TITLE="${TITLE:-r18_abmilx_fgfr3_rep${REPEAT}}"

PYTHON="${PYTHON:-}"                 # interpreter used to create .venv-e2e
VENV="${VENV:-$ROOT/.venv-e2e}"
FGFR3_PYTHON="${FGFR3_PYTHON:-}"     # env that trained the FGFR3 models (folds + tiling)
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
LMDB_WORKERS="${LMDB_WORKERS:-}"

SKIP_INSTALL="${SKIP_INSTALL:-0}"
SKIP_TILE="${SKIP_TILE:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; BLUE=$'\033[1;34m'; GREEN=$'\033[1;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[1;31m'; RESET=$'\033[0m'
else
  BOLD=""; BLUE=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
log()  { printf '\n%s[%s]%s %s\n' "$BLUE" "$(date +%H:%M:%S)" "$RESET" "$*"; }
warn() { printf '%sWARNING:%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '\n%sERROR:%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: ./e2e_fgfr3/run_e2e_fgfr3.sh [-- extra main.py arguments]

Environment variables (defaults in brackets):
  SLIDES_DIR      .svs folder                    [$ROOT/image]
  TILES_ROOT      FGFR3 tiler output             [$ROOT/results/tiles_out]
  MANIFEST        FGFR3 train_manifest.json      [$ROOT/results/models_trained_125/train_manifest.json]
  WORK_DIR        everything this run writes     [$ROOT/results/e2e_fgfr3]
  REPEAT          FGFR3 CV repeat to reproduce   [0]
  VAL_FRAC        E2E val share of each train part [0.125]
  NUM_GPUS        1 = train.sh, >1 = torchrun    [1]
  FOLD_START      resume training at this fold   [0]
  CONFIG          E2E YAML config                [e2e_fgfr3/config/r18_fgfr3.yaml]
  FGFR3_PYTHON    Python of the FGFR3 run        [auto: .venv, dist/.venv, python3]
  TORCH_INDEX_URL pip index for a CUDA torch     [PyPI default]
  LMDB_WORKERS    save_patches.py workers        [6 on Linux, 0 on macOS]
  SKIP_INSTALL=1  use the existing .venv-e2e as is
  SKIP_TILE=1     never run the FGFR3 tiler (fail if tiles are missing)
  SKIP_TRAIN=1    stop after preparing LMDBs and folds

Anything after "--" is passed to E2E main.py and overrides the config,
e.g.  -- --same_psize=256 --p_batch_size=256   (less GPU memory)
EOF
}

EXTRA_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

# E2E derives the training LMDB path by replacing '224' with '256' in the whole
# path (main.py), so '224' may appear only in the LMDB file names.
case "$WORK_DIR" in *224*) die "WORK_DIR must not contain '224' (E2E rewrites it to '256'): $WORK_DIR" ;; esac

H5_DIR="$WORK_DIR/clam_h5"
LMDB_DIR="$WORK_DIR/lmdb"
FOLDS_DIR="$WORK_DIR/folds_rep${REPEAT}"
RUNS_DIR="$WORK_DIR/runs"
RUN_DIR="$RUNS_DIR/fgfr3/$TITLE"
OOF_CSV="$RUN_DIR/oof_predictions.csv"
mkdir -p "$WORK_DIR" "$LMDB_DIR" "$RUNS_DIR"

# ===========================================================================
# 1. E2E Python environment
# ===========================================================================
log "${BOLD}STEP 1/6${RESET} — E2E Python environment ($VENV)"

if [ -z "$PYTHON" ]; then
  for c in python3.11 python3.12 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PYTHON="$c"; break; fi
  done
fi

if [ ! -x "$VENV/bin/python" ]; then
  [ "$SKIP_INSTALL" = "1" ] && die "SKIP_INSTALL=1 but $VENV does not exist"
  "$PYTHON" -c 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' \
    || die "E2E needs Python 3.10-3.12 (torchaug does not install on 3.13); found $("$PYTHON" --version 2>&1). Set PYTHON=/path/to/python3.11"
  log "creating $VENV with $("$PYTHON" --version 2>&1)"
  "$PYTHON" -m venv "$VENV" || die "could not create a venv (Debian/Ubuntu: sudo apt install python3.11-venv)"
fi
PY="$VENV/bin/python"

if [ "$SKIP_INSTALL" != "1" ]; then
  "$PY" -m pip install --upgrade pip
  if [ -n "$TORCH_INDEX_URL" ]; then
    "$PY" -m pip install "torch>=2.5.1" torchvision --index-url "$TORCH_INDEX_URL"
  else
    "$PY" -m pip install "torch>=2.5.1" torchvision
  fi
  "$PY" -m pip install -r "$HERE/requirements-e2e.txt"
  "$PY" -m pip install torchaug==0.6.1 --no-deps
fi

"$PY" - <<'PY' || die "E2E environment is incomplete (see above). On Debian/Ubuntu: sudo apt install libturbojpeg openslide-tools"
import torch, timm, lmdb, h5py, openslide, turbojpeg, torchaug, fairscale, git, sklearn
from torchaug import transforms as t
assert all(hasattr(t, n) for n in ["RandomResizedCrop", "SequentialTransform"]), "torchaug too old"
turbojpeg.TurboJPEG()  # loads libturbojpeg
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | "
      f"GPUs: {torch.cuda.device_count()} | openslide {openslide.__library_version__}")
PY

# FGFR3 environment: builds the folds (same scikit-learn as the Chowder run) and,
# if needed, runs the FGFR3 tiler.
if [ -z "$FGFR3_PYTHON" ]; then
  for c in "$ROOT/.venv/bin/python" "$ROOT/dist/.venv/bin/python" "$(command -v python3 || true)"; do
    if [ -n "$c" ] && [ -x "$c" ] && "$c" -c 'import sklearn, numpy, pandas' >/dev/null 2>&1; then
      FGFR3_PYTHON="$c"; break
    fi
  done
fi
if [ -z "$FGFR3_PYTHON" ]; then
  warn "no FGFR3 Python with scikit-learn found; building folds with $PY. Set FGFR3_PYTHON to the env that trained the FGFR3 models so the split is guaranteed identical."
  FGFR3_PYTHON="$PY"
fi
log "FGFR3 Python: $FGFR3_PYTHON (scikit-learn $("$FGFR3_PYTHON" -c 'import sklearn; print(sklearn.__version__)'))"

# ===========================================================================
# 2. Inputs from the FGFR3 run
# ===========================================================================
log "${BOLD}STEP 2/6${RESET} — FGFR3 inputs"

[ -f "$MANIFEST" ] || die "FGFR3 manifest not found: $MANIFEST (set MANIFEST=...)"
[ -f "$FGFR3_TRAIN_SCRIPT" ] || die "FGFR3 training script not found: $FGFR3_TRAIN_SCRIPT"
[ -d "$SLIDES_DIR" ] || die "slide folder not found: $SLIDES_DIR (set SLIDES_DIR=...)"

MISSING_TILES="$WORK_DIR/.missing_tiles.txt"
"$PY" - "$MANIFEST" "$SLIDES_DIR" "$TILES_ROOT" "$MISSING_TILES" <<'PY'
import collections, json, sys
from pathlib import Path
manifest, slides_dir, tiles_root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
m = json.loads(Path(manifest).read_text())
slides, labels = m["slides"], m["labels"]
print(f"  manifest: {len(slides)} slides, {len(set(m['patients']))} patients, "
      f"labels {dict(collections.Counter(labels))}, {m['n_splits']}-fold x {m['repeats']} repeat(s), "
      f"split_unit={m['split_unit']}")
no_svs = [s for s in slides if not (slides_dir / s).exists()]
if no_svs:
    sys.exit(f"ERROR: {len(no_svs)} manifest slide(s) missing from {slides_dir}, e.g. {no_svs[0]}")
need = [s for s in slides if not (tiles_root / s / "coords.npy").exists()
        or not (tiles_root / s / "metadata.json").exists()]
out.write_text("".join(s + "\n" for s in need))
print(f"  slides found: {len(slides)}/{len(slides)} | tiled: {len(slides) - len(need)}/{len(slides)}")
PY

if [ -s "$MISSING_TILES" ]; then
  [ "$SKIP_TILE" = "1" ] && die "$(wc -l < "$MISSING_TILES") slide(s) have no FGFR3 tiles and SKIP_TILE=1 (list: $MISSING_TILES)"
  log "tiling $(wc -l < "$MISSING_TILES") slide(s) with the FGFR3 tiler (same settings as the Chowder run)"
  while IFS= read -r s; do
    "$FGFR3_PYTHON" "$FGFR3_TILER" --svs_path "$SLIDES_DIR/$s" --out_dir "$TILES_ROOT" \
      --tissue_detector otsu --device auto --target_mpp 1.0 --tile_size 224 \
      --min_tissue_fraction 0.60 --max_tiles 5000 --seed 0
  done < "$MISSING_TILES"
fi

# ===========================================================================
# 3. FGFR3 tiles -> CLAM .h5
# ===========================================================================
log "${BOLD}STEP 3/6${RESET} — FGFR3 coords -> CLAM patch files"
"$PY" "$HERE/tools/coords_to_clam_h5.py" --manifest "$MANIFEST" --tiles_root "$TILES_ROOT" --out_dir "$H5_DIR"

# ===========================================================================
# 4. LMDBs with E2E's own CLAM/save_patches.py
# ===========================================================================
log "${BOLD}STEP 4/6${RESET} — tile LMDBs (CLAM/save_patches.py)"

if [ -z "$LMDB_WORKERS" ]; then
  # macOS cannot pickle the OpenSlide handle into DataLoader workers.
  if [ "$(uname)" = "Darwin" ]; then LMDB_WORKERS=0; else LMDB_WORKERS=6; fi
fi
N_SLIDES="$("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["slides"]))' "$MANIFEST")"

lmdb_complete() {
  "$PY" - "$1" "$2" <<'PY'
import os, pickle, sys
import lmdb
path, n = sys.argv[1], int(sys.argv[2])
if not os.path.exists(path):
    sys.exit(1)
env = lmdb.open(path, subdir=False, readonly=True, lock=False)
with env.begin() as txn:
    raw = txn.get(b"__slide__")
sys.exit(0 if raw is not None and len(pickle.loads(raw)) == n else 1)
PY
}

# main_224: val/test input; main_256: training input; sub_256: training sub-scale.
# (main.py rewrites --dataset_sub_root ..._224 -> ..._256, so no sub_224 is needed.)
for spec in "main 224" "main 256" "sub 256"; do
  set -- $spec
  SCALE="$1"; SZ="$2"; NAME="${SCALE}_${SZ}"
  if lmdb_complete "$LMDB_DIR/$NAME.lmdb" "$N_SLIDES"; then
    log "  $NAME.lmdb complete ($N_SLIDES slides) — skipping"
    continue
  fi
  log "  building $NAME.lmdb (resumes from $LMDB_DIR/${NAME}_progress.txt if interrupted)"
  (cd "$E2E/CLAM" && "$PY" save_patches.py \
      --data_h5_dir "$H5_DIR/$SCALE" \
      --data_slide_dir "$SLIDES_DIR" \
      --csv_path "$H5_DIR/$SCALE/process_list.csv" \
      --patch_dir "$LMDB_DIR" \
      --slide_ext .svs --save_lmdb \
      --img_size "$SZ" --lmdb_name "$NAME" \
      --batch_size 256 --workers "$LMDB_WORKERS")
  lmdb_complete "$LMDB_DIR/$NAME.lmdb" "$N_SLIDES" || die "$NAME.lmdb is incomplete after save_patches.py"
done
set --

# ===========================================================================
# 5. Fold CSVs (= FGFR3 split)
# ===========================================================================
log "${BOLD}STEP 5/6${RESET} — fold CSVs from the FGFR3 split (repeat $REPEAT)"
"$FGFR3_PYTHON" "$HERE/tools/make_e2e_folds.py" --manifest "$MANIFEST" \
  --train_script "$FGFR3_TRAIN_SCRIPT" --repeat "$REPEAT" --val_frac "$VAL_FRAC" --out_dir "$FOLDS_DIR"
N_FOLDS="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["n_folds"])' "$FOLDS_DIR/folds_summary.json")"

if [ "$SKIP_TRAIN" = "1" ]; then
  log "SKIP_TRAIN=1 — data prepared in $WORK_DIR"
  exit 0
fi

# ===========================================================================
# 6. E2E training (+ out-of-fold predictions)
# ===========================================================================
log "${BOLD}STEP 6/6${RESET} — E2E training: $MODEL, $N_FOLDS folds (from fold $FOLD_START), $NUM_GPUS GPU(s)"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/train_$(date +%Y%m%d_%H%M%S).log"

MAIN_ARGS=(
  --datasets=bio_fgfr3                       # 'bio*' -> integer labels 0=WT, 1=MUT
  --project=fgfr3 --title="$TITLE"
  --csv_path="$FOLDS_DIR"                    # directory of fold_{k}.csv
  --cv_fold="$N_FOLDS" --fold_start="$FOLD_START"
  --dataset_root="$LMDB_DIR/main_224.lmdb"
  --dataset_sub_root="$LMDB_DIR/sub_224.lmdb"
  --model="$MODEL" -c="$CONFIG"
  --output_path="$RUNS_DIR" --lmdb
)
# Never pass --wandb: E2E would then `git commit -a` the repository it lives in.
export WANDB_MODE=disabled

WRAPPER=("$HERE/tools/train_with_oof_export.py" --e2e_root "$E2E" --oof_csv "$OOF_CSV"
         --expect_folds "$((N_FOLDS - FOLD_START))" --)
set +e
if [ "$NUM_GPUS" -gt 1 ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 \
    "$VENV/bin/torchrun" --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc_per_node="$NUM_GPUS" \
    "${WRAPPER[@]}" "${MAIN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
else
  # same environment as E2E's train.sh
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" "${WRAPPER[@]}" "${MAIN_ARGS[@]}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} 2>&1 | tee "$LOG_FILE"
fi
STATUS=${PIPESTATUS[0]}
set -e
[ "$STATUS" -eq 0 ] || die "training failed (exit $STATUS); log: $LOG_FILE. Resume with FOLD_START=<fold>."

log "${GREEN}${BOLD}DONE${RESET}"
cat <<EOF

Outputs
  checkpoints:     $RUN_DIR/fold_{k}_model_best.pt, fold_{k}_ema_model_best.pt
  OOF predictions: $OOF_CSV
  training log:    $LOG_FILE   (per-fold test metrics at the end)
  folds:           $FOLDS_DIR/fold_{k}.csv, folds_summary.json
  LMDBs:           $LMDB_DIR/{main_224,main_256,sub_256}.lmdb
EOF
