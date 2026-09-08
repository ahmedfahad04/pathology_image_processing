"""
Main Preprocessing Pipeline
Orchestrates all preprocessing steps for pathology image tiles.

Logic:
- Combines all modules into a single pipeline
- Handles I/O, progress tracking, and error handling
- Saves preprocessed tiles and metadata
- Generates summary report
- Designed for parallel processing (each tile independent)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import yaml
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import logging
import argparse
from tqdm import tqdm

# Import local modules
from tile_extractor import TileExtractor
from noise_remover import NoiseRemover
from background_remover import BackgroundRemover
from tissue_detector import TissueDetector
from color_normalizer import ColorNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline for pathology images.
    Tissue-only: only patches containing tissue are processed and saved; background/glass is skipped early.
    
    Pipeline stages (per tile, early-exit on non-tissue):
    1. Extract overlapping tiles from SVS
    2. Detect tissue (percentage + focus + staining) — if not valid, skip immediately (no noise/bg/normalize)
    3. Remove noise (Gaussian + Median + Morphological) — tissue only
    4. Remove background (Otsu + Morphological) — tissue only
    5. Normalize color (percentile/Macenko/Reinhard) — tissue only
    6. Save processed tiles and metadata — tissue only; rejected optionally saved if configured
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.yaml")
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize modules
        self.tile_extractor = TileExtractor(
            patch_size=self.config['patch_extraction']['patch_size'],
            overlap=self.config['patch_extraction']['overlap']
        )
        
        self.noise_remover = NoiseRemover(**self.config['noise_reduction'])
        
        self.background_remover = BackgroundRemover(**self.config['background_removal'])
        
        self.tissue_detector = TissueDetector(**self.config['tissue_detection'])
        
        self.color_normalizer = ColorNormalizer(**self.config['color_normalization'])
        
        # Create output directories
        self.setup_output_dirs()
        
        logger.info("PreprocessingPipeline initialized")
    
    def setup_output_dirs(self):
        """Create output directory structure."""
        base_dir = Path(self.config['output']['base_dir'])
        
        self.tiles_dir = base_dir / self.config['output']['tiles_dir']
        self.masks_dir = base_dir / self.config['output']['masks_dir']
        self.metadata_dir = base_dir / self.config['output']['metadata_dir']
        self.rejected_dir = base_dir / "rejected"
        
        # Create directories
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config['quality_control']['save_rejected']:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
    
    def process_single_tile(
        self, 
        tile: np.ndarray, 
        coords: Tuple[int, int, int, int],
        tile_idx: int
    ) -> Optional[Dict]:
        """
        Process a single tile through the full pipeline.
        
        Args:
            tile: RGB input tile
            coords: (x_start, y_start, x_end, y_end)
            tile_idx: Index of the tile
            
        Returns:
            Dictionary with tile metadata if valid, None if rejected
        """
        x_start, y_start, x_end, y_end = coords
        
        # Step 1: Tissue detection (on original tile for accurate focus assessment)
        is_valid, detection_info = self.tissue_detector.detect_tissue(tile)
        
        if not is_valid:
            # Save rejected tile if configured
            if self.config['quality_control']['save_rejected']:
                rejected_path = self.rejected_dir / f"tile_{tile_idx:06d}.png"
                cv2.imwrite(str(rejected_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
            
            return None
        
        # Step 2: Noise reduction (tissue only — already validated)
        denoised = self.noise_remover.remove_noise(tile)
        
        # Step 3: Background removal (tissue only)
        masked_tile, tissue_mask, tissue_pct = self.background_remover.remove_background(denoised)
        
        # Step 4: Color normalization
        normalized = self.color_normalizer.normalize(masked_tile)
        
        # Step 5: Save processed tile
        tile_filename = f"tile_{tile_idx:06d}.png"
        tile_path = self.tiles_dir / tile_filename
        cv2.imwrite(str(tile_path), cv2.cvtColor(normalized, cv2.COLOR_RGB2BGR))
        
        # Save tissue mask
        mask_filename = f"mask_{tile_idx:06d}.png"
        mask_path = self.masks_dir / mask_filename
        cv2.imwrite(str(mask_path), tissue_mask)
        
        # Prepare metadata
        metadata = {
            'tile_id': tile_idx,
            'x': x_start,
            'y': y_start,
            'width': x_end - x_start,
            'height': y_end - y_start,
            'tissue_percentage': tissue_pct,
            'is_focused': detection_info.get('is_focused', True),
            'focus_score': detection_info.get('focus_score', 0.0),
            'is_stained': detection_info.get('is_stained', True),
            'filename': tile_filename,
            'mask_filename': mask_filename
        }
        
        return metadata
    
    def run(
        self, 
        roi: Optional[Tuple[int, int, int, int]] = None,
        max_tiles: Optional[int] = None,
        resume_from: Optional[int] = None
    ) -> Dict:
        """
        Run the complete preprocessing pipeline.
        Tissue-only: only patches with tissue are processed/saved; background is counted as rejected and skipped.
        
        Args:
            roi: Optional region of interest (x, y, width, height)
            max_tiles: Maximum number of *tissue* patches to save (not grid positions). None= all tissue patches. Scans until N accepted.
            resume_from: Resume from tile index (skip previous grid positions)
            
        Returns:
            Dictionary with processing statistics
        """
        start_time = datetime.now()
        
        svs_path = self.config['input']['svs_path']
        level = self.config['input']['level']
        tissue_thresh = self.config['tissue_detection']['min_tissue_percent']
        check_focus = self.config['tissue_detection'].get('check_focus', True)
        
        logger.info(f"Starting preprocessing pipeline (tissue-only, min_tissue={tissue_thresh}%, check_focus={check_focus})")
        logger.info(f"SVS: {svs_path}")
        logger.info(f"Level: {level}")
        logger.info(f"ROI: {roi} | max_tissue_tiles={max_tiles} | resume_from={resume_from}")
        
        # Get image dimensions
        width, height = self.tile_extractor.get_level_dimensions(svs_path, level)
        
        # Generate tile coordinates (full grid)
        tile_coords = self.tile_extractor.generate_tile_coordinates(width, height, roi)
        total_available = len(tile_coords)
        
        # Apply resume offset (skip first N grid positions)
        start_idx = 0
        if resume_from:
            start_idx = resume_from
            tile_coords = tile_coords[resume_from:]
            logger.info(f"Resuming from grid index {resume_from}, {len(tile_coords)} positions remaining")
        
        # Statistics - tissue-only semantics
        stats = {
            'total_available': total_available,  # total grid positions
            'total_tiles': total_available,
            'total_requested': max_tiles if max_tiles else total_available,
            'scanned': 0,       # grid positions examined
            'processed': 0,     # alias for scanned for backward compat
            'accepted': 0,      # tissue patches saved
            'rejected': 0,      # background / unfocused / poorly stained
            'start_time': start_time.isoformat(),
            'svs_path': svs_path,
            'level': level,
            'roi': roi,
            'filter_tissue': True,
            'min_tissue_percent': tissue_thresh,
        }
        
        # Process tiles - scan until we collect max_tiles accepted (or exhaust grid)
        all_metadata = []
        
        if max_tiles:
            logger.info(f"Will scan up to {len(tile_coords)} positions to collect {max_tiles} tissue patches")
        else:
            logger.info(f"Processing full WSI: {len(tile_coords)} grid positions, saving only tissue patches...")
        
        for idx, coords in enumerate(tqdm(tile_coords, desc="Processing tiles (tissue-only)", total=len(tile_coords))):
            # Early stop if we already have enough tissue patches
            if max_tiles is not None and stats['accepted'] >= max_tiles:
                logger.info(f"Reached requested {max_tiles} tissue patches, stopping scan")
                break
            tile_idx = start_idx + idx
            
            try:
                # Extract tile
                tile = self.tile_extractor.extract_tile(
                    svs_path, level, 
                    coords[0], coords[1], coords[2], coords[3]
                )
                
                stats['scanned'] += 1
                stats['processed'] += 1
                
                # Process tile - returns None if background/unfocused (rejected)
                metadata = self.process_single_tile(tile, coords, tile_idx)
                
                if metadata:
                    all_metadata.append(metadata)
                    stats['accepted'] += 1
                else:
                    stats['rejected'] += 1
                
                # Progress logging
                if stats['scanned'] % 100 == 0:
                    logger.info(f"Progress: scanned {stats['scanned']}/{len(tile_coords)} | accepted {stats['accepted']} | rejected {stats['rejected']}")
                if stats['accepted'] > 0 and stats['accepted'] % 50 == 0 and metadata:
                    logger.info(f"Saved {stats['accepted']} tissue patches so far")
                    
            except Exception as e:
                logger.error(f"Error processing tile {tile_idx}: {e}")
                stats['rejected'] += 1
                stats['scanned'] += 1
                continue
        
        # Save metadata
        metadata_path = self.metadata_dir / "tiles.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        # Calculate final statistics
        end_time = datetime.now()
        stats['end_time'] = end_time.isoformat()
        stats['duration_seconds'] = (end_time - start_time).total_seconds()
        
        if stats['accepted'] > 0:
            tissue_pcts = [m['tissue_percentage'] for m in all_metadata]
            stats['mean_tissue_percentage'] = np.mean(tissue_pcts)
            stats['std_tissue_percentage'] = np.std(tissue_pcts)
        
        # Save statistics
        stats_path = self.metadata_dir / "processing_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Pipeline completed in {stats['duration_seconds']:.1f} seconds")
        logger.info(f"Results: {stats['accepted']}/{stats['processed']} tiles accepted")
        
        return stats
    
    def generate_report(self, stats: Dict) -> str:
        """
        Generate HTML report of preprocessing results.
        
        Args:
            stats: Processing statistics
            
        Returns:
            Path to generated report
        """
        if not self.config['quality_control']['generate_report']:
            return ""
        
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Preprocessing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        .stat-box {{ background: #f5f5f5; padding: 20px; margin: 10px 0; border-radius: 5px; }}
        .stat-label {{ font-weight: bold; color: #555; }}
        .stat-value {{ font-size: 24px; color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Pathology Image Preprocessing Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="stat-box">
        <div class="stat-label">Input File</div>
        <div class="stat-value">{stats.get('svs_path', 'N/A')}</div>
    </div>
    
    <div class="stat-box">
        <div class="stat-label">Pyramid Level</div>
        <div class="stat-value">{stats.get('level', 'N/A')}</div>
    </div>
    
    <h2>Processing Summary</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Total Tiles</td><td>{stats.get('total_tiles', 0)}</td></tr>
        <tr><td>Processed</td><td>{stats.get('processed', 0)}</td></tr>
        <tr><td>Accepted</td><td>{stats.get('accepted', 0)}</td></tr>
        <tr><td>Rejected</td><td>{stats.get('rejected', 0)}</td></tr>
        <tr><td>Acceptance Rate</td><td>{(stats.get('accepted', 0) / max(stats.get('processed', 1), 1) * 100):.1f}%</td></tr>
    </table>
    
    <h2>Timing</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Duration</td><td>{stats.get('duration_seconds', 0):.1f} seconds</td></tr>
        <tr><td>Start Time</td><td>{stats.get('start_time', 'N/A')}</td></tr>
        <tr><td>End Time</td><td>{stats.get('end_time', 'N/A')}</td></tr>
    </table>
    
    <h2>Tissue Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Mean Tissue %</td><td>{stats.get('mean_tissue_percentage', 0):.1f}%</td></tr>
        <tr><td>Std Tissue %</td><td>{stats.get('std_tissue_percentage', 0):.1f}%</td></tr>
    </table>
    
    <h2>Output Directories</h2>
    <ul>
        <li>Tiles: {self.tiles_dir}</li>
        <li>Masks: {self.masks_dir}</li>
        <li>Metadata: {self.metadata_dir}</li>
    </ul>
</body>
</html>
"""
        
        report_path = self.metadata_dir / "report.html"
        with open(report_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)


def main():
    """Main entry point for command-line execution. Tissue-only: only tissue patches are saved."""
    parser = argparse.ArgumentParser(
        description="Pathology Image Preprocessing Pipeline (tissue-only: only patches with tissue are processed/saved)"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to configuration file (default: scripts/preprocessing/config.yaml)"
    )
    parser.add_argument(
        "--svs",
        help="Override SVS file path from config"
    )
    parser.add_argument(
        "--level",
        type=int,
        help="Override pyramid level from config"
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Region of interest (x, y, width, height)"
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        help="Maximum number of *tissue* patches to save (scans until N accepted). Omit or 0 for all tissue patches."
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        help="Resume from grid index (skip first N grid positions, not accepted count)"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory from config"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PreprocessingPipeline(args.config)
    
    # Apply command-line overrides
    if args.svs:
        pipeline.config['input']['svs_path'] = args.svs
    if args.level:
        pipeline.config['input']['level'] = args.level
    if args.output_dir:
        pipeline.config['output']['base_dir'] = args.output_dir
        pipeline.setup_output_dirs()
    
    # Run pipeline
    stats = pipeline.run(
        roi=tuple(args.roi) if args.roi else None,
        max_tiles=args.max_tiles,
        resume_from=args.resume_from
    )
    
    # Generate report
    report_path = pipeline.generate_report(stats)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE (tissue-only)")
    print("="*60)
    print(f"Grid positions available: {stats.get('total_available', stats['total_tiles'])}")
    print(f"Scanned: {stats.get('scanned', stats['processed'])}")
    print(f"Accepted (tissue saved): {stats['accepted']}")
    print(f"Rejected (background/unfocused): {stats['rejected']}")
    print(f"Duration: {stats['duration_seconds']:.1f} seconds")
    if report_path:
        print(f"Report: {report_path}")
    print(f"Tiles: {pipeline.tiles_dir} | Masks: {pipeline.masks_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
