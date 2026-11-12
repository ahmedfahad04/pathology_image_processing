# Run guide: FGFR3 mutation with E2E-ABMILX (GPU PC)

Copy and run the commands in order. Each step says what you should see.
Background and design notes: [README.md](README.md).

---

## Step 0. Push the branch (on the Mac, once)

```bash
cd ~/Documents/research/pathology_image_processing
git push -u origin e2e-fgfr3
```

Everything below runs on the **GPU PC**.

---

## Step 1. Go to the repository

Use the clone where you ran the FGFR3 (Chowder) experiment:

```bash
cd /path/to/pathology_image_processing     # <- change this path
git status
```

`git status` should show no modified tracked files. If it does, save them first:

```bash
git stash
```

---

## Step 2. Get the branch

```bash
git fetch origin
git checkout e2e-fgfr3
git log --oneline -1
```

Expected: the last commit is on `e2e-fgfr3`, and `e2e_fgfr3/run_e2e_fgfr3.sh` exists.

---

## Step 3. Check the FGFR3 data

```bash
ls image/*.svs | wc -l                                   # expect 310 (or more)
ls -d results/tiles_out/*.svs | wc -l                    # expect 310
ls results/models_trained_125/train_manifest.json        # must exist
```

If tiles are missing for some slides, the script tiles them itself with the
original FGFR3 settings (slower, but correct).

If your files are somewhere else, set these before Step 7 (otherwise skip):

```bash
export SLIDES_DIR=/path/to/svs_folder
export TILES_ROOT=/path/to/tiles_out
export MANIFEST=/path/to/train_manifest.json
```

---

## Step 4. Check the Python that trained FGFR3

The fold split must be rebuilt with the same scikit-learn as the FGFR3 run.

```bash
.venv/bin/python -c "import sklearn; print('sklearn', sklearn.__version__)"
```

- If this prints a version, nothing to do (the script finds `.venv/` itself).
- If it fails, find the Python you used for FGFR3 (for example a conda env) and set it:

```bash
export FGFR3_PYTHON=/path/to/fgfr3/env/bin/python    # e.g. ~/miniconda3/envs/fgfr3/bin/python
$FGFR3_PYTHON -c "import sklearn; print('sklearn', sklearn.__version__)"
```

---

## Step 5. System packages and Python version

```bash
sudo apt install -y libturbojpeg python3-venv
python3 --version
```

- Python **3.10, 3.11 or 3.12**: nothing to do.
- Python **3.13**: install 3.11 and point the script at it:

```bash
sudo apt install -y python3.11 python3.11-venv
export PYTHON=python3.11
```

---

## Step 6. Pick the CUDA build of PyTorch

```bash
nvidia-smi
```

Read **CUDA Version** in the top-right corner and set the matching index:

| CUDA Version shown | Command |
|---|---|
| 12.4 or newer | `export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124` |
| 12.1 – 12.3 | `export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121` |
| 11.8 – 12.0 | `export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118` |

---

## Step 7. Start the run (inside tmux)

Training takes many hours, so run it inside `tmux` so it survives a closed
terminal or SSH session.

```bash
tmux new -s e2e
```

Inside tmux, `cd` to the repo again and re-export any variables you set in
Steps 3–6 (exports do not carry into a new tmux session):

```bash
cd /path/to/pathology_image_processing     # <- change this path
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121   # <- your value from Step 6
# export PYTHON=python3.11                 # only if Step 5 told you to
# export FGFR3_PYTHON=/path/to/python      # only if Step 4 told you to

./e2e_fgfr3/run_e2e_fgfr3.sh
```

Detach and leave it running: press `Ctrl-b`, then `d`.
Re-attach later:

```bash
tmux attach -t e2e
```

### What you should see

| Step printed | Expected |
|---|---|
| `STEP 1/6` environment | ends with `CUDA available: True`. **If it says `False`, stop (`Ctrl-c`) and redo Step 6.** |
| `STEP 2/6` inputs | `slides found: 310/310`, `tiled: 310/310` |
| `STEP 3/6` coords → h5 | `slides: 310` and tile counts |
| `STEP 4/6` LMDBs | builds `main_224`, `main_256`, `sub_256` (long, one time only) |
| `STEP 5/6` folds | 5 lines `fold k: train … | val … | test 62 slides …` |
| `STEP 6/6` training | `Epoch [n/75] … auc_value: …` lines for fold 0, then 1 … 4 |
| end | `[oof] 310 test predictions from fold(s) [0, 1, 2, 3, 4]` and `DONE` |

---

## Step 8. Watch progress (from any terminal)

```bash
cd /path/to/pathology_image_processing
tail -f results/e2e_fgfr3/runs/fgfr3/r18_abmilx_fgfr3_rep0/train_*.log
```

```bash
watch -n 5 nvidia-smi
```

Stop watching with `Ctrl-c` (this does not stop training).

---

## Step 9. Collect the results

```bash
ls results/e2e_fgfr3/runs/fgfr3/r18_abmilx_fgfr3_rep0/
head results/e2e_fgfr3/runs/fgfr3/r18_abmilx_fgfr3_rep0/oof_predictions.csv
grep "Cross validation" results/e2e_fgfr3/runs/fgfr3/r18_abmilx_fgfr3_rep0/train_*.log
```

| File | Content |
|---|---|
| `oof_predictions.csv` | one row per slide: `fold,slide,patient,label,prob_main,prob_ema` (each slide scored by the fold that held it out) |
| `fold_{k}_model_best.pt` / `fold_{k}_ema_model_best.pt` | best-on-validation checkpoints |
| `train_*.log` | full log; `Cross validation …` lines at the end (first main model, then EMA) |
| `../../../folds_rep0/` | the fold CSVs and `folds_summary.json` (with the scikit-learn version used) |

---

## If something goes wrong

**GPU out of memory** — rerun with fewer patches per step (everything after `--`
goes to E2E's `main.py`):

```bash
SKIP_INSTALL=1 ./e2e_fgfr3/run_e2e_fgfr3.sh -- --same_psize=256 --p_batch_size=256
```

**Stopped or crashed during fold *k*** — resume from that fold (LMDBs and folds
are reused; predictions of finished folds are kept):

```bash
SKIP_INSTALL=1 FOLD_START=2 ./e2e_fgfr3/run_e2e_fgfr3.sh      # <- replace 2 with the fold to redo
```

**Interrupted while building LMDBs** — just run Step 7 again with `SKIP_INSTALL=1`; it resumes.

**`turbojpeg` / `libturbojpeg` error** — `sudo apt install -y libturbojpeg`, then rerun with `SKIP_INSTALL=1`.

**Python 3.13 error in STEP 1** — see Step 5 (`export PYTHON=python3.11`), then delete the
half-made env and rerun:

```bash
rm -rf .venv-e2e
./e2e_fgfr3/run_e2e_fgfr3.sh
```

**Prepare data only, no training** (to check Steps 1–5 of the script):

```bash
SKIP_TRAIN=1 ./e2e_fgfr3/run_e2e_fgfr3.sh
```

**All options:**

```bash
./e2e_fgfr3/run_e2e_fgfr3.sh --help
```

---

## Do not

- add `--wandb` (the E2E code would run `git commit -a` in this repository);
- set `val2test` in the config (it would mix validation slides into the test folds);
- delete `results/models_trained_125/train_manifest.json` (it defines the slides, labels and folds).
