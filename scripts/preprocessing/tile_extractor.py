"""
Tile Extractor Module
Extracts overlapping patches from Whole Slide Images (WSI) using OpenSlide.

Logic:
- Opens SVS/TIFF files using OpenSlide (optimized for tiled reading)
- Never loads the full image into memory (critical for 38GB+ images)
- Generates a grid of tile coordinates with specified overlap
- Handles edge tiles by padding with zeros
- Returns tiles as numpy arrays with their coordinates
"""

import numpy as np
try:
    import openslide
except ImportError:
    raise ImportError(
        "openslide-python is required. Install with: pip install openslide-python openslide-bin"
    )
from typing import List, Tuple, Generator, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TileExtractor:
    """
    Extracts overlapping tiles from Whole Slide Images.
    
    Why overlapping patches?
    - Prevents boundary artifacts when processing tiles independently
    - Nuclei at tile edges get full context from neighboring tiles
    - Enables better stitching during post-processing
    - Overlap region is used for NMS (Non-Maximum Suppression) later
    """
    
    def __init__(self, patch_size: int = 1024, overlap: int = 128):
        """
        Initialize tile extractor.
        
        Args:
            patch_size: Size of each square patch (e.g., 1024x1024)
            overlap: Overlap in pixels between adjacent patches
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        
        logger.info(f"TileExtractor initialized: patch={patch_size}, overlap={overlap}, stride={self.stride}")
    
    def get_level_dimensions(self, svs_path: str, level: int = 0) -> Tuple[int, int]:
        """
        Get image dimensions at specified pyramid level.
        
        Args:
            svs_path: Path to SVS/TIFF file
            level: Pyramid level (0 = highest resolution)
            
        Returns:
            Tuple of (width, height) in pixels
        """
        slide = openslide.OpenSlide(svs_path)
        width, height = slide.level_dimensions[level]
        slide.close()
        
        logger.info(f"Level {level} dimensions: {width}x{height}")
        return width, height
    
    def generate_tile_coordinates(
        self, 
        width: int, 
        height: int, 
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Generate grid of tile coordinates with overlap.
        
        Why this approach?
        - Creates a regular grid covering the entire image (or ROI)
        - Stride = patch_size - overlap ensures tiles overlap by specified amount
        - Edge tiles may be smaller (handled by padding in extract_tile)
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            roi: Optional region of interest (x, y, width, height)
            
        Returns:
            List of (x_start, y_start, x_end, y_end) tuples
        """
        if roi:
            x_offset, y_offset, roi_w, roi_h = roi
        else:
            x_offset, y_offset, roi_w, roi_h = 0, 0, width, height
        
        tiles = []
        for y in range(y_offset, y_offset + roi_h, self.stride):
            for x in range(x_offset, x_offset + roi_w, self.stride):
                x_end = min(x + self.patch_size, x_offset + roi_w)
                y_end = min(y + self.patch_size, y_offset + roi_h)
                tiles.append((x, y, x_end, y_end))
        
        logger.info(f"Generated {len(tiles)} tile coordinates for {roi_w}x{roi_h} region")
        return tiles
    
    def extract_tile(
        self, 
        svs_path: str, 
        level: int, 
        x_start: int, 
        y_start: int, 
        x_end: int, 
        y_end: int
    ) -> np.ndarray:
        """
        Extract a single tile from the SVS file.
        
        Why tiled reading?
        - SVS files store images as JPEG-compressed tiles (240x240)
        - OpenSlide reads only the needed tiles, not the full image
        - Memory usage: ~patch_size^2 * 3 bytes, not 38GB
        
        Args:
            svs_path: Path to SVS file
            level: Pyramid level
            x_start, y_start: Top-left corner
            x_end, y_end: Bottom-right corner
            
        Returns:
            numpy array of shape (height, width, 3) in RGB
        """
        tile_w = x_end - x_start
        tile_h = y_end - y_start
        
        slide = openslide.OpenSlide(svs_path)
        # OpenSlide.read_region takes (x, y, level, size)
        # Returns RGBA PIL Image
        tile_rgba = slide.read_region((x_start, y_start), level, (tile_w, tile_h))
        slide.close()
        # Convert RGBA to RGB
        tile = np.array(tile_rgba)[:, :, :3]
        
        # Handle grayscale (unlikely for H&E but just in case)
        if len(tile.shape) == 2:
            tile = np.stack([tile, tile, tile], axis=-1)
        
        # Pad if edge tile is smaller than patch_size
        if tile.shape[0] < self.patch_size or tile.shape[1] < self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size, 3), dtype=tile.dtype)
            padded[:tile.shape[0], :tile.shape[1], :] = tile
            tile = padded
        
        return tile
    
    def extract_tiles(
        self, 
        svs_path: str, 
        level: int = 0,
        roi: Optional[Tuple[int, int, int, int]] = None,
        max_tiles: Optional[int] = None
    ) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
        """
        Generator that extracts tiles one by one.
        
        Why a generator?
        - Memory efficient: only one tile in memory at a time
        - Can be interrupted/resumed
        - Enables streaming to disk without holding all tiles
        
        Args:
            svs_path: Path to SVS file
            level: Pyramid level
            roi: Optional region of interest
            max_tiles: Maximum number of tiles to extract (None = all)
            
        Yields:
            Tuple of (tile_array, (x_start, y_start, x_end, y_end))
        """
        width, height = self.get_level_dimensions(svs_path, level)
        tile_coords = self.generate_tile_coordinates(width, height, roi)
        
        if max_tiles:
            tile_coords = tile_coords[:max_tiles]
        
        for idx, (x_start, y_start, x_end, y_end) in enumerate(tile_coords):
            tile = self.extract_tile(svs_path, level, x_start, y_start, x_end, y_end)
            yield tile, (x_start, y_start, x_end, y_end)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Extracted {idx + 1}/{len(tile_coords)} tiles")
    
    def extract_all_tiles(
        self, 
        svs_path: str, 
        level: int = 0,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract all tiles and return as list.
        
        Warning: Uses more memory than generator. Use only for small ROIs.
        
        Args:
            svs_path: Path to SVS file
            level: Pyramid level
            roi: Optional region of interest
            
        Returns:
            List of (tile_array, coordinates) tuples
        """
        tiles = list(self.extract_tiles(svs_path, level, roi))
        logger.info(f"Extracted {len(tiles)} tiles total")
        return tiles


# Convenience function for simple usage
def extract_tiles_from_svs(
    svs_path: str,
    patch_size: int = 1024,
    overlap: int = 128,
    level: int = 0,
    roi: Optional[Tuple[int, int, int, int]] = None
) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
    """
    Convenience function to extract tiles from SVS file.
    
    Usage:
        for tile, coords in extract_tiles_from_svs("image.svs"):
            process(tile)
    """
    extractor = TileExtractor(patch_size=patch_size, overlap=overlap)
    yield from extractor.extract_tiles(svs_path, level, roi)


if __name__ == "__main__":
    import yaml
    
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    svs_path = config["input"]["svs_path"]
    level = config["input"]["level"]
    patch_size = config["patch_extraction"]["patch_size"]
    overlap = config["patch_extraction"]["overlap"]
    
    extractor = TileExtractor(patch_size=patch_size, overlap=overlap)
    
    # Test: extract first 5 tiles
    for tile, coords in extractor.extract_tiles(svs_path, level, max_tiles=5):
        print(f"Tile shape: {tile.shape}, coords: {coords}")
