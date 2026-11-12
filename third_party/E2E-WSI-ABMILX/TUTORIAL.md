# ABMILX: run the full pipeline

Raw slides → patch coordinates → LMDB → end-to-end training → checkpoints.

Part 1–3 are **copy-paste commands**. They run the entire pipeline on the two slides bundled in this repo (`CLAM/heatmaps/demo/slides/*.svs`, one LUAD and one LUSC). Part 4 swaps in your own data and a GPU.

> **Verified:** every command in Parts 1–3 was run in order on macOS (Apple Silicon, CPU, Python 3.11) from a fresh virtual environment. Part 4 (GPU, multi-GPU, your data, feature-based training) follows the same code paths but was **not** run.
>
> Run everything from the **repository root**, in one shell session (variables carry over between blocks).

---

## Part 1. Setup

### 1.1 System packages

| Package | macOS | Ubuntu/Debian |
|---|---|---|
| Python **3.10–3.12** | `brew install python@3.11` | `sudo apt install python3.11 python3.11-venv` |
| OpenSlide (reads `.svs`/`.tif`) | `brew install openslide` | `sudo apt install openslide-tools` |
| libjpeg-turbo (`PyTurboJPEG`) | `brew install jpeg-turbo` | `sudo apt install libturbojpeg` |

Python must be **≤ 3.12**: `torchaug ≥ 0.4` (needed for `RandomResizedCrop`/`SequentialTransform`) does not install on 3.13.

### 1.2 Python environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip

# torch>=2.5.1 is required (modules/vit_mil.py imports torch.nn.attention)
pip install "torch>=2.5.1" torchvision

pip install timm einops lmdb opencv-python "pandas<3" scikit-learn torchmetrics \
            h5py pillow numpy jpeg4py PyTurboJPEG scikit-survival wandb \
            GitPython future fairscale \
            openslide-python matplotlib scipy tqdm PyYAML tensorboardX

# torchaug pins an old torch/torchvision; install it last, without dependencies
pip install torchaug==0.6.1 --no-deps
```

Why this differs from `requirements.txt`: that file lacks `GitPython` and `fairscale` (hard imports), doesn't pin `torchaug`, and allows `pandas 3.x`, which breaks `datasets/dataset_img.py`.
(When I tested, I installed `torchaug` normally, which downgraded torch to 2.2.1, then re-upgraded torch to 2.15. `--no-deps` is the equivalent clean order; the rest was run exactly as above.)

### 1.3 Check the install

```bash
python - <<'EOF'
import torch, timm, lmdb, openslide, turbojpeg, torchaug, fairscale, git
from torchaug import transforms as t
assert all(hasattr(t, n) for n in ["RandomResizedCrop", "SequentialTransform"])
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), "| openslide", openslide.__library_version__)
EOF
```

First training run also downloads ImageNet ResNet weights from the Hugging Face hub (`resnet18.tv_in1k`), so you need internet access once.

---

## Part 2. Preprocess the demo slides

```bash
export SLIDES=$PWD/CLAM/heatmaps/demo/slides
export DEMO=$PWD/demo_run
mkdir -p $DEMO
```

### 2.1 Segment tissue and extract patch coordinates

Two scales are produced, because every shipped e2e config uses multi-scale input (`mul_scale_input: true`): a main scale (256 px patches at level 1) and a coarser sub scale (512 px patches at level 1).

> **Must `cd CLAM`.** `create_patches_fp.py` resolves `--preset` as `presets/<name>` relative to the current directory. The README's `--preset CLAM/presets/...` fails with `FileNotFoundError`.

```bash
cd CLAM

# main scale: 256 px patches, level 1
python create_patches_fp.py --source $SLIDES --save_dir $DEMO/clam_main \
    --patch_size 256 --step_size 256 --patch_level 1 --preset tcga.csv --seg --patch

# sub scale: 512 px patches, level 1
python create_patches_fp.py --source $SLIDES --save_dir $DEMO/clam_sub \
    --patch_size 512 --step_size 512 --patch_level 1 --preset tcga.csv --seg --patch

cd ..
```

Expected: `Extracted 18 coordinates` / `26 coordinates` (main) and `9` / `6` (sub), and files `$DEMO/clam_main/patches/*.h5`, `$DEMO/clam_main/process_list_autogen.csv`.

### 2.2 Crop patches into LMDBs

Training reads **256 px** LMDBs and val/test read **224 px** LMDBs, so each scale is saved twice. The file names matter (see 2.3).

> `--workers 0` is required on **macOS** (spawned DataLoader workers cannot pickle the OpenSlide handle). On Linux you can raise it (e.g. `--workers 6`).

```bash
cd CLAM
for SCALE in main sub; do
  for SZ in 224 256; do
    python save_patches.py \
        --data_h5_dir $DEMO/clam_$SCALE \
        --data_slide_dir $SLIDES \
        --csv_path $DEMO/clam_$SCALE/process_list_autogen.csv \
        --patch_dir $DEMO/lmdb \
        --slide_ext .svs --save_lmdb \
        --img_size $SZ --lmdb_name ${SCALE}_${SZ} \
        --batch_size 64 --workers 0
  done
done
cd ..
ls $DEMO/lmdb/*.lmdb    # main_224 main_256 sub_224 sub_256
```

(`save_patches.py` originally imported an unused `get_encoder`, which needs a missing `models/vae_warpper.py` and `diffusers`. That import was removed in this repo.)

### 2.3 The `224` / `256` naming rule

When `img_size` is 256 (all shipped e2e configs), `main.py` does `dataset_root.replace('224','256')` (and the same for `--dataset_sub_root`) to find the training LMDBs. So you always pass the **224** paths, and a same-named **256** file must sit beside each. Keep `224` out of any other part of the path.

---

## Part 3. Train

### 3.1 Labels / splits

`--csv_path` needs columns `case,slide,label,fold`. `slide` must equal the slide ids stored in the LMDB. The two demo slides are LUAD (`C3L-03262-22`) and LUSC (`C3L-01663-21`). With only two slides, each is used for both `train` and `test`, which is fine for a pipeline check but not for real evaluation.

```bash
cat > $DEMO/labels.csv <<'EOF'
case,slide,label,fold
C3L-03262-22,C3L-03262-22,LUAD,train
C3L-01663-21,C3L-01663-21,LUSC,train
C3L-03262-22,C3L-03262-22,LUAD,test
C3L-01663-21,C3L-01663-21,LUSC,test
EOF
```

### 3.2 Run end-to-end training

`--output_path` must already exist (`main.py` uses `os.mkdir`, not `makedirs`).

```bash
mkdir -p $DEMO/results

WANDB_MODE=disabled python main.py \
  --datasets=nsclc --project=demo --title=demo \
  --dataset_root=$DEMO/lmdb/main_224.lmdb \
  --dataset_sub_root=$DEMO/lmdb/sub_224.lmdb \
  --csv_path=$DEMO/labels.csv \
  --model=e2e_r18_abmilx -c=config/e2e/r18_brca_subtyping.yaml \
  --output_path=$DEMO/results --lmdb --val_ratio=0 \
  --num_epoch=2 --max_epoch=2 --num_workers=2 \
  --batch_size=2 --same_psize=16 --p_batch_size=16
```

Flags that differ from the full-scale configs, and why:

| Flag | Demo value | Config value | Reason |
|---|---|---|---|
| `--datasets` | `nsclc` | | maps labels `LUAD→0`, `LUSC→1` |
| `--val_ratio` | `0` | 0.125 | only two slides |
| `--num_epoch/--max_epoch` | 2 | 75/50 | quick check |
| `--batch_size` | 2 | 3 | only two training slides |
| `--same_psize` | 16 | 512 | patches/slide per step (the demo slides have only 6–26 patches) |
| `--p_batch_size` | 16 | 512 | encoder chunk size |
| `--num_workers` | 2 | 6 | laptop |

### 3.3 Expected result (about 1 minute on CPU)

The run prints per-epoch lines like

```
Train: 0 [   0/1 (100%)]  Loss: 1.655 ...
 Epoch [1/2] train loss: ..., test loss: ..., accuracy: ..., auc_value: ...
...
Cross validation accuracy mean: ...
```

exit code 0, and checkpoints:

```bash
ls $DEMO/results/demo/demo
# fold_0_model_best.pt  fold_0_ema_model_best.pt
```

Metrics are meaningless here (2 slides, 2 epochs); what matters is that every stage ran.

Harmless warnings you will see: `amp_autocast(device_type='cuda', ...)` on CPU, `timm.models.layers` deprecation, `weights_only`/`warnings.warn`.

### 3.4 Clean up

```bash
rm -rf $DEMO        # demo_run/ is just scratch data
```

---

## Part 4. Real data and GPUs (not run here)

### 4.1 Get the data

- **Preprocessed LMDBs (skip Part 2):** https://huggingface.co/datasets/Dearcat/CPath_Image or Baidu (https://pan.baidu.com/s/1H0P024gRIPgi3Xeu-xPLUg?pwd=t179, password `t179`).
- **Your own slides:** repeat Part 2 with your `--source`, choosing size/level for your magnification:

  | Dataset / magnification | `--patch_size` / `--step_size` | `--patch_level` |
  |---|---|---|
  | PANDA ×40 | 256 | 0 |
  | PANDA ×20 | 512 | 0 |
  | other ×10 | 256 | 1 |
  | other ×5 | 512 | 1 |

  In `save_patches.py`, `--data_h5_dir` is the folder *containing* `patches/`, `--csv_path` must have a `slide_id` column including the extension (as `process_list_autogen.csv` does), and `--slide_ext` must match your files. On Linux raise `--workers`. The 256 vs 224 LMDB pair and the high-resolution main / lower-resolution sub pair work as in Part 2.

  With the LMDB map size set to ~2 TB in `save_patches.py`, use a filesystem that supports sparse files (ext4/APFS/XFS are fine).

### 4.2 Labels

Use `labels/panda.csv`, `labels/subtyping_tcga_brca_resample.csv` or `labels/subtyping_tcga_brca.csv` as examples. The shipped configs have `cv_fold: 1`, so the `fold` column (`train`/`val`/`test`) defines the split. Passing `--cv_fold=N` (N>1) instead builds stratified k-folds from the CSV and ignores `fold`. Survival uses a **directory** of `fold_{k}.csv` files (`labels/survival_tcga_brca_3fold/`).

Label strings per `--datasets`: `brca` → `IDC`/`ILC`; `nsclc` → `LUAD`/`LUSC`; `call` → `normal`/`tumor`; `panda` → integers (ISUP grade); `surv_*` → survival columns.

### 4.3 Train on GPU(s)

Single GPU:

```bash
mkdir -p results
bash train.sh --datasets=brca --project=MyProject --title=r18_brca_e2e_abmilx \
  --dataset_root=/data/lmdb/brca_224_x10.lmdb --dataset_sub_root=/data/lmdb/brca_224_x5.lmdb \
  --csv_path=labels/subtyping_tcga_brca_resample.csv \
  --model=e2e_r18_abmilx -c=config/e2e/r18_brca_subtyping.yaml \
  --output_path=results/brca_subtyping_e2e_r18_abmilx --lmdb
```

Multi-GPU (`torchrun`, e.g. 2 GPUs):

```bash
bash train_dist.sh 2 --datasets=panda --project=MyProject --title=r18_panda_e2e_abmilx \
  --dataset_root=/data/lmdb/panda_224_x20.lmdb --dataset_sub_root=/data/lmdb/panda_224_x5.lmdb \
  --csv_path=labels/panda.csv --model=e2e_r18_abmilx -c=config/e2e/r18_panda.yaml \
  --output_path=results/panda_e2e_r18_abmilx --lmdb
```

Notes:
- Configs: `config/e2e/{r18_panda, r50_panda, r18_brca_subtyping, r50_brca_subtyping, r18_brca_survival}.yaml`. CLI flags override YAML. Model names: `e2e_r18_abmilx`, `e2e_r50_abmilx`, ...
- Datasets: `--datasets={panda, brca, nsclc, call, surv_brca}`.
- Reduce `--same_psize`, `--p_batch_size` and `--batch_size` if you run out of GPU memory.
- Only one scale? set `mul_scale_input: false` in a copy of the config (not tested).
- Results land in `<output_path>/<project>/<title>/` (`fold_{k}_model_best.pt`, `fold_{k}_ema_model_best.pt`).
- **`--wandb` runs `git add` for untracked files and `git commit -a` in this repo** (`utils.check_and_commit_changes`). Omit it unless you want that; `WANDB_MODE=disabled` silences W&B.

### 4.4 Two-stage (pre-extracted features) training

Features: https://huggingface.co/datasets/Dearcat/CPathPatchFeature, or extract your own (TRIDENT, or `CLAM/extract_features_fp.py`; untested here). `--dataset_root` must contain a `pt_files/` folder with one `.pt` per slide.

```bash
bash train.sh --datasets=panda --project=MyProject --title=panda_feat_abmil \
  --dataset_root=/data/panda_features --csv_path=labels/panda.csv \
  --model=abmil -c=config/feat/fm_panda.yaml --output_path=results/panda_feat_abmil
```

Configs: `config/feat/{fm_panda, fm_subtyping, fm_survival}.yaml`. Use ABMILX as a head in your own code:

```python
from modules.abmilx import DAttentionX
head = DAttentionX(input_dim=1024, n_classes=2, mil_bias=True, attn_bias=True, n_heads=8)
# input (1, N, D) bag of features -> logits (1, n_classes)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: presets/CLAM/presets/tcga.csv` | `cd CLAM` and pass `--preset tcga.csv` (Part 2.1). |
| `ValueError: ctypes objects containing pointers cannot be pickled` | macOS: use `--workers 0` in `save_patches.py`. |
| `ModuleNotFoundError: models.vae_warpper` / `diffusers` | Old `save_patches.py` with the unused `get_encoder` import; use the version in this repo. |
| `ModuleNotFoundError: git` / `fairscale` / `future` | Install `GitPython` / `fairscale` / `future` (Part 1.2). |
| `No module named 'torch.nn.attention'` | torch too old (often downgraded by `torchaug`); reinstall `torch>=2.5.1`. |
| `torchaug.transforms has no attribute 'RandomResizedCrop'` | Python 3.13, or wrong `torchaug`; use Python ≤3.12 and `torchaug>=0.4`. |
| `TypeError: Invalid value ... for dtype 'str'` | pandas 3.x; `pip install "pandas<3"`. |
| `'NoneType' object has no attribute 'begin'` | `mul_scale_input: true` without `--dataset_sub_root`. |
| `lmdb.Error: ... already open in this process` | `--num_workers 0`, or root and sub-root are the same file. |
| `FileNotFoundError` for the project folder | `--output_path` must exist beforehand. |
| `lmdb.Error` on a `*256*` path | The 256 twin of the 224 LMDB is missing (Part 2.3). |
| Runs on CPU with a GPU installed | Install a CUDA build of torch; `torch.cuda.is_available()` must be `True`. |
