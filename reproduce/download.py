"""Pilot download for FGFR3MUT — 5 slides only, not the full 200GB.

Why this file exists:
- Original `download.py` in PABannier/fgfr3mut pulls the ENTIRE TCGA-BLCA
  feature cache (>200GB) + 50MB weights. Good for full reproduction,
  bad for a pilot on my laptop (I have ~70GB free).
- This script pulls the SAME files but filtered to my 5 local slides
  in `image/`, plus the small shared files (weights, labels, filters).
  Expected result: <1GB instead of >200GB.

What I get in --out_dir (same laIt as full download, so
`fgfr3mut/run_inference.py --data_dir` works unchanged):
    <out_dir>/
      models/split_0/*.pt ... split_4/*.pt   # 125 Chowder checkpoints (~50MB)
      features/<slidename>/features.npy      # only for my 5 slides (~150-350MB)
      filtered_slides_tcga.xlsx              # MIBC vs NMIBC filter
      mutations_blca_tcga_pancancer_atlas_cbioportal.txt  # ground-truth labels
      loeffler_tcga.xlsx                     # Loeffler et al. subset

Usage:
    python reproduce/download.py --out_dir ./data_fgfr3_mini
    python reproduce/download.py --out_dir ./data_fgfr3_mini --slides TCGA-2F-A9KQ TCGA-FJ-A871
    python reproduce/download.py --out_dir ./data_fgfr3_full --full   # fallback: everything (>200GB)

Tweak here for future requirements:
- PILOT_SLIDES: change IDs to add/remove slides.
- ALWAYS_PATTERNS: metadata + weights I always need.
- build_allow_patterns(): fnmatch patterns passed to snapshot_download.
"""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "PABannier/fgfr3mut"

# my 5 pilot slides (patient IDs = first 12 chars; matching is by substring,
# so TCGA-CF-A5U8 covers BOTH slides of that patient: DX1 + TSA).
PILOT_SLIDES = [
    "TCGA-2F-A9KQ",  # image/TCGA-2F-A9KQ-01Z-00-DX1...svs (1.5GB)
    "TCGA-4Z-AA7S",  # image/TCGA-4Z-AA7S-01Z-00-DX1...svs (1.1GB)
    "TCGA-CF-A5U8",  # image/TCGA-CF-A5U8-01Z-00-DX1... + ...TSA... (1006MB + 227MB)
    "TCGA-FJ-A871",  # image/TCGA-FJ-A871-01Z-00-DX5...svs (1.1GB)
]

# Small files every run needs (weights + labels + filters).
ALWAYS_PATTERNS = [
    "models/*",  # 125 Chowder .pt files, ~50MB
    "*.xlsx",  # filtered_slides_tcga.xlsx, loeffler_tcga.xlsx
    "*.txt",  # mutations_blca_tcga_pancancer_atlas_cbioportal.txt
]


def build_allow_patterns(slides):
    """Build fnmatch patterns for snapshot_download.

    Each slide ID becomes `*<ID>*` so it matches regardless of whether
    HuggingFace stores files as `features/<full_slidename>/features.npy`
    or any other nesting.
    """
    patterns = list(ALWAYS_PATTERNS)
    for s in slides:
        s = s.strip()
        if not s:
            continue
        # Match full slidenames and short patient IDs alike.
        patterns.append(f"*{s}*")
    return patterns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Where to put the pilot data, e.g. ./data_fgfr3_mini")
    parser.add_argument("--slides", nargs="*", default=PILOT_SLIDES,
                        help="Slide substrings to keep (default: my 5 pilot slides)")
    parser.add_argument("--full", action="store_true",
                        help="Ignore --slides and download everything (>200GB)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.full:
        print("FULL mode: downloading everything (>200GB). Make sure I have disk space.")
        snapshot_download(REPO_ID, local_dir=str(out_dir), repo_type="dataset")
    else:
        patterns = build_allow_patterns(args.slides)
        print(f"PILOT mode: {len(args.slides)} slide filter(s): {args.slides}")
        print("allow_patterns:")
        for p in patterns:
            print(f"  - {p}")
        snapshot_download(
            REPO_ID,
            local_dir=str(out_dir),
            repo_type="dataset",
            allow_patterns=patterns,
        )

    print("\nDone. Verify:")
    print(f"  ls {out_dir}/models          # expect split_0..split_4")
    print(f"  ls {out_dir}/features         # expect only my pilot slides")
    print(f"  ls {out_dir}/*.xlsx {out_dir}/*.txt")
    print("\nNext: run inference with --data_dir pointing here, e.g.")
    print(f"  python fgfr3mut/run_inference.py --data_dir {out_dir} --device cpu --n_tiles 5000 --keep_tcga_cases MIBC")


if __name__ == "__main__":
    main()
