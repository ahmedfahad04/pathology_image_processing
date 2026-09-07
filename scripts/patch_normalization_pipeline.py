"""
Patch Normalization Pipeline
Alternative lightweight approach: extract patches -> Macenko normalize -> save raw + normalized separately.

Logic (as requested):
  1. Extract patches from WSI using TileExtractor (no bg/noise removal)
  2. Save raw patch to raw_dir
  3. Pass same patch through MacenkoNormalizer (122_normalizing_HnE_images.py logic)
  4. Save normalized patch to normalized_dir

Features:
  - Separate folders for raw and normalized (configurable)
  - Initially processes first N patches (default 10) for quick verification
  - Option to process full WSI via --all or --max-tiles None / large number
  - Metadata JSON with coordinates linked to filenames

Usage:
  # First 10 patches (default, quick test):
  python patch_normalization_pipeline.py --config config.yaml

  # First 10 explicitly:
  python patch_normalization_pipeline.py --max-tiles 10

  # Whole WSI:
  python patch_normalization_pipeline.py --all
  python patch_normalization_pipeline.py --max-tiles 0   # 0 means all

  # Custom output:
  python patch_normalization_pipeline.py --output-dir ../output/macenko_test --max-tiles 50

  # With ROI:
  python patch_normalization_pipeline.py --roi 0 0 5000 5000 --max-tiles 20
"""

import os
import json
import yaml
import cv2
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, List

from tqdm import tqdm

from tile_extractor import TileExtractor
from macenko_normalizer import MacenkoNormalizer
from tissue_detector import TissueDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PatchNormalizationPipeline:
    """
    Simplified pipeline: extract -> save raw -> normalize (Macenko) -> save normalized.
    Keeps raw and normalized in separate folders for easy downstream use.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Patch extraction params
        patch_cfg = self.config.get('patch_extraction', {})
        self.patch_size = patch_cfg.get('patch_size', 1024)
        self.overlap = patch_cfg.get('overlap', 128)

        self.tile_extractor = TileExtractor(patch_size=self.patch_size, overlap=self.overlap)

        # Macenko params - support both old config and new section
        macenko_cfg = self.config.get('macenko_normalization', {})
        # fallback to color_normalization if needed
        self.normalizer = MacenkoNormalizer(
            Io=macenko_cfg.get('Io', 240),
            alpha=macenko_cfg.get('alpha', 1),
            beta=macenko_cfg.get('beta', 0.15),
        )

        # Tissue filter - only keep tissue patches, skip background/glass
        # Uses same thresholds as preprocess_pipeline for consistency
        tissue_cfg = self.config.get('tissue_detection', {})
        self.tissue_detector = TissueDetector(
            min_tissue_percent=tissue_cfg.get('min_tissue_percent', 5.0),
            focus_method=tissue_cfg.get('focus_method', 'laplacian'),
            focus_threshold=tissue_cfg.get('focus_threshold', 50.0),
            check_focus=tissue_cfg.get('check_focus', True),
        )
        # Allow macenko-specific override
        self.min_tissue_percent = macenko_cfg.get('min_tissue_percent', tissue_cfg.get('min_tissue_percent', 5.0))
        self.filter_tissue = macenko_cfg.get('filter_tissue', True)  # default ON
        logger.info(f"Tissue filter: enabled={self.filter_tissue}, min_tissue={self.min_tissue_percent}%, check_focus={tissue_cfg.get('check_focus', True)}")

        # Output dirs - new section preferred, fallback to legacy
        output_cfg = self.config.get('output', {})
        # base_dir for this pipeline: allow override via macenko_output or output.base_dir
        # If macenko_output defined use it, else reuse output.base_dir but under subfolders
        macenko_out = self.config.get('macenko_output', None)
        if macenko_out:
            base_dir = Path(macenko_out.get('base_dir', output_cfg.get('base_dir', '../output/macenko_patches')))
            raw_subdir = macenko_out.get('raw_dir', 'raw')
            norm_subdir = macenko_out.get('normalized_dir', 'normalized')
        else:
            # default: ../output/patches  with raw/ normalized
            base_dir = Path(output_cfg.get('base_dir', '../output/preprocessed')).parent / "patches"
            # if user hasn't configured macenko_output, use ../output/patches
            # Check if output base is ../output/preprocessed -> patches sibling
            # Keep explicit if config has patch_output?
            raw_subdir = "raw"
            norm_subdir = "normalized"

        # Allow explicit output override via CLI later, but init here
        self.base_dir = base_dir
        self.raw_dir = self.base_dir / raw_subdir
        self.normalized_dir = self.base_dir / norm_subdir
        self.metadata_dir = self.base_dir / "metadata"

        self.setup_output_dirs()
        logger.info(f"PatchNormalizationPipeline: patch={self.patch_size}, overlap={self.overlap}, Io={self.normalizer.Io}")
        logger.info(f"Output: raw={self.raw_dir}, normalized={self.normalized_dir}")

    def setup_output_dirs(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def set_output_base(self, base_dir: str):
        """Override base output dir (e.g., from CLI)."""
        self.base_dir = Path(base_dir)
        # keep same subfolder names
        self.raw_dir = self.base_dir / "raw"
        self.normalized_dir = self.base_dir / "normalized"
        self.metadata_dir = self.base_dir / "metadata"
        self.setup_output_dirs()

    def process(
        self,
        svs_path: Optional[str] = None,
        level: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        max_tiles: Optional[int] = 10,
        save_metadata: bool = True,
        filter_tissue: Optional[bool] = None,
        min_tissue_percent: Optional[float] = None,
    ) -> Dict:
        """
        Run extraction + normalization.

        Args:
            svs_path: override SVS path (else from config)
            level: pyramid level (else from config)
            roi: optional ROI (x, y, w, h)
            max_tiles: number of tiles to process. None or 0 means all tiles. Default 10.
            save_metadata: whether to write tiles.json

        Returns:
            stats dict
        """
        start_time = datetime.now()

        if svs_path is None:
            svs_path = self.config['input']['svs_path']
        if level is None:
            level = self.config['input'].get('level', 0)

        # Resolve relative paths w.r.t config location (scripts/)
        # config svs_path is like ../image/... so resolve from scripts dir
        svs_path_resolved = str(Path(svs_path).resolve() if Path(svs_path).is_absolute() else (Path(__file__).parent / svs_path).resolve())
        # fallback to original if not found
        if not Path(svs_path_resolved).exists():
            # try as-is relative to cwd
            if Path(svs_path).exists():
                svs_path_resolved = str(Path(svs_path).resolve())
            else:
                logger.warning(f"SVS path not found: {svs_path}, trying as given: {svs_path_resolved}")
                svs_path_resolved = svs_path

        # Resolve filter settings (CLI overrides config)
        do_filter = self.filter_tissue if filter_tissue is None else filter_tissue
        tissue_thresh = self.min_tissue_percent if min_tissue_percent is None else min_tissue_percent
        # Temporarily update detector threshold if CLI overrides
        if min_tissue_percent is not None:
            self.tissue_detector.min_tissue_percent = tissue_thresh

        logger.info(f"SVS: {svs_path_resolved} (level {level}), roi={roi}, max_tiles={max_tiles}, filter_tissue={do_filter}, min_tissue={tissue_thresh}%")

        width, height = self.tile_extractor.get_level_dimensions(svs_path_resolved, level)
        tile_coords = self.tile_extractor.generate_tile_coordinates(width, height, roi)

        total_available = len(tile_coords)

        # Interpret max_tiles: None or 0 => all tissue patches (or all grid if no filter)
        if max_tiles is None or max_tiles == 0:
            logger.info(f"Processing FULL WSI: {total_available} grid positions (will save only tissue patches)" if do_filter else f"Processing FULL WSI: {total_available} tiles")
        else:
            if do_filter:
                logger.info(f"Will scan up to {total_available} positions to collect {max_tiles} tissue patches (filter ON)")
            else:
                tile_coords = tile_coords[:max_tiles]
                logger.info(f"Processing first {len(tile_coords)}/{total_available} tiles (max_tiles={max_tiles}, filter OFF)")

        stats = {
            'total_available': total_available,
            'total_requested': max_tiles if max_tiles else total_available,
            'scanned': 0,
            'processed': 0,  # saved tissue patches
            'skipped_background': 0,
            'failed': 0,
            'filter_tissue': do_filter,
            'min_tissue_percent': tissue_thresh,
            'svs_path': svs_path_resolved,
            'level': level,
            'roi': roi,
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'raw_dir': str(self.raw_dir),
            'normalized_dir': str(self.normalized_dir),
            'start_time': start_time.isoformat(),
        }

        all_metadata: List[Dict] = []
        saved_idx = 0  # sequential id for saved tissue patches

        # If filtering, we need to iterate until we collect max_tiles tissue patches
        # If not filtering, iterate over sliced tile_coords
        iter_coords = tile_coords if do_filter else tile_coords
        # For tqdm total, use total_available if filtering with limit, else len(iter_coords)
        pbar_total = total_available if (do_filter and max_tiles) else len(iter_coords)

        for tile_idx, coords in enumerate(tqdm(iter_coords, desc="Extract+Normalize (tissue-only)" if do_filter else "Extract+Normalize", total=pbar_total)):
            # If filtering and we already have enough saved patches, stop
            if do_filter and max_tiles is not None and max_tiles != 0 and saved_idx >= max_tiles:
                break
            # If not filtering but we sliced already, no extra break needed
            # Track scanned for stats
            stats['scanned'] += 1
            x_start, y_start, x_end, y_end = coords
            try:
                # Extract raw tile (RGB)
                tile = self.tile_extractor.extract_tile(
                    svs_path_resolved, level, x_start, y_start, x_end, y_end
                )

                # Tissue filter: skip background/glass before saving/normalizing
                if do_filter:
                    is_valid, det_info = self.tissue_detector.detect_tissue(tile)
                    if not is_valid:
                        stats['skipped_background'] += 1
                        # Optional: debug log every 500 skips
                        if stats['skipped_background'] % 500 == 0:
                            logger.info(f"Skipped {stats['skipped_background']} background tiles so far (scanned {stats['scanned']})")
                        continue
                    tissue_pct = det_info.get('tissue_percentage', 0)
                    focus_score = det_info.get('focus_score', 0)
                else:
                    tissue_pct = None
                    focus_score = None
                    det_info = {}

                # Save raw (tissue only)
                raw_filename = f"patch_{saved_idx:06d}.png"
                raw_path = self.raw_dir / raw_filename
                cv2.imwrite(str(raw_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))

                # Normalize via Macenko (now guaranteed tissue, so no beta warnings)
                normalized = self.normalizer.normalize(tile)

                # Save normalized
                norm_filename = f"patch_{saved_idx:06d}.png"  # same name, different folder for easy pairing
                norm_path = self.normalized_dir / norm_filename
                cv2.imwrite(str(norm_path), cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))

                # Metadata - keep both saved_idx and original tile_idx for traceability
                meta = {
                    'patch_id': saved_idx,
                    'tile_idx': tile_idx,
                    'x': x_start,
                    'y': y_start,
                    'width': x_end - x_start,
                    'height': y_end - y_start,
                    'tissue_percentage': float(tissue_pct) if tissue_pct is not None else None,
                    'focus_score': float(focus_score) if focus_score is not None else None,
                    'raw_filename': raw_filename,
                    'normalized_filename': norm_filename,
                    'raw_path': str(raw_path),
                    'normalized_path': str(norm_path),
                }
                all_metadata.append(meta)
                saved_idx += 1
                stats['processed'] += 1

                if saved_idx % 50 == 0:
                    logger.info(f"Saved {saved_idx} tissue patches (scanned {stats['scanned']}/{total_available})")

            except Exception as e:
                logger.error(f"Failed tile {tile_idx} coords {coords}: {e}", exc_info=True)
                stats['failed'] += 1
                continue
            # Safety: if not filtering and we sliced, tqdm will end naturally

        # Save metadata
        if save_metadata:
            meta_path = self.metadata_dir / "tiles.json"
            with open(meta_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            logger.info(f"Metadata saved: {meta_path} ({len(all_metadata)} entries)")

        end_time = datetime.now()
        stats['end_time'] = end_time.isoformat()
        stats['duration_seconds'] = (end_time - start_time).total_seconds()
        stats['metadata_path'] = str(self.metadata_dir / "tiles.json")

        # Save stats
        stats_path = self.metadata_dir / "stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Done: {stats['processed']} patches in {stats['duration_seconds']:.1f}s "
                    f"(failed {stats['failed']}) -> raw:{self.raw_dir}, norm:{self.normalized_dir}")
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Patch Normalization Pipeline: extract raw patches and Macenko-normalize them into separate folders (tissue-only by default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--svs", help="Override SVS path")
    parser.add_argument("--level", type=int, help="Override pyramid level")
    parser.add_argument("--max-tiles", type=int, default=10,
                        help="Max tissue patches to save (default 10). Use 0 or --all for full WSI tissue patches. With filter ON, scans until N tissue patches found.")
    parser.add_argument("--all", action="store_true", help="Process full WSI (saves all tissue patches, skips background)")
    parser.add_argument("--output-dir", help="Override base output dir (will create raw/ normalized/ metadata inside)")
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="ROI x y w h")
    parser.add_argument("--min-tissue", type=float, default=None, help="Override min tissue %% (default from config tissue_detection.min_tissue_percent=5.0)")
    parser.add_argument("--no-filter", action="store_true", help="Disable tissue filter: save every grid tile (including background) - reproduces old behavior with warnings")
    parser.add_argument("--include-background", action="store_true", help="Alias for --no-filter")
    args = parser.parse_args()

    pipeline = PatchNormalizationPipeline(config_path=args.config)

    if args.output_dir:
        pipeline.set_output_base(args.output_dir)
        # re-log
        logger.info(f"Overridden output base: {pipeline.base_dir}")

    max_tiles = None if args.all else args.max_tiles
    # If --all not set and max_tiles == 0 => also means all
    if max_tiles == 0:
        max_tiles = None

    roi = tuple(args.roi) if args.roi else None
    do_filter = not (args.no_filter or args.include_background)

    stats = pipeline.process(
        svs_path=args.svs,
        level=args.level,
        roi=roi,
        max_tiles=max_tiles,
        filter_tissue=do_filter,
        min_tissue_percent=args.min_tissue,
    )

    print("\n" + "="*60)
    print("PATCH NORMALIZATION COMPLETE (tissue-only)" if stats.get('filter_tissue') else "PATCH NORMALIZATION COMPLETE (no filter)")
    print("="*60)
    print(f"Grid positions available: {stats['total_available']}")
    print(f"Requested tissue patches: {stats['total_requested'] if stats['total_requested'] else 'ALL'}")
    print(f"Scanned: {stats.get('scanned', 0)}, Saved (tissue): {stats['processed']}, Skipped background: {stats.get('skipped_background',0)}, Failed: {stats['failed']}")
    print(f"Min tissue: {stats.get('min_tissue_percent')}% | Filter: {stats.get('filter_tissue')}")
    print(f"Duration: {stats['duration_seconds']:.1f}s")
    print(f"Raw dir: {stats['raw_dir']}")
    print(f"Normalized dir: {stats['normalized_dir']}")
    print(f"Metadata: {stats['metadata_path']}")
    print("="*60)
    print("\nTip: For full WSI tissue patches: python patch_normalization_pipeline.py --all")
    print("     For 100 tissue patches: python patch_normalization_pipeline.py --max-tiles 100")
    print("     To include background (old, warns): python patch_normalization_pipeline.py --no-filter --max-tiles 10")


if __name__ == "__main__":
    main()
