"""
Background Removal Module
Removes glass/background from pathology image tiles, keeping only tissue.

Logic:
- H&E stained tissue appears pink/purple on white/light background
- Background removal creates a binary mask separating tissue from glass
- Multiple methods available:
  1. Otsu's method: automatic thresholding (works well for H&E)
  2. Adaptive thresholding: handles uneven illumination
  3. Color-based: uses HSV color space for better separation
- Post-processing:
  - Morphological closing fills holes in tissue mask
  - Remove small objects (debris, artifacts)
  - Apply mask to original image
"""

import cv2
import numpy as np
from skimage import morphology, measure
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    Removes background from pathology image tiles.
    
    Why background removal?
    - Reduces processing time (skip glass regions)
    - Improves segmentation accuracy (no false positives on glass)
    - Enables tissue percentage calculation
    - Standard preprocessing step in digital pathology
    """
    
    def __init__(
        self,
        method: str = "otsu",
        otsu_threshold: int = 0,
        morphological_disk: int = 5,
        min_area_percent: float = 1.0,
        invert: bool = True
    ):
        """
        Initialize background remover.
        
        Args:
            method: Thresholding method ("otsu", "adaptive", "color_based")
            otsu_threshold: Manual threshold for Otsu (0 = auto)
            morphological_disk: Radius for closing holes
            min_area_percent: Minimum object size as % of tile
            invert: If True, tissue=white, background=black
        """
        self.method = method
        self.otsu_threshold = otsu_threshold
        self.morphological_disk = morphological_disk
        self.min_area_percent = min_area_percent
        self.invert = invert
        
        # Pre-create morphological kernel
        self.kernel = morphology.disk(self.morphological_disk)
        
        logger.info(f"BackgroundRemover: method={method}, morpho_disk={morphological_disk}")
    
    def grayscale_threshold(self, tile: np.ndarray) -> np.ndarray:
        """
        Convert to grayscale and apply Otsu's thresholding.
        
        Why Otsu's?
        - Automatically finds optimal threshold between bimodal distributions
        - H&E images have clear tissue/background separation
        - Works well when tissue is darker than background
        
        Args:
            tile: RGB input image
            
        Returns:
            Binary mask (tissue=255, background=0)
        """
        # Convert to grayscale
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile.copy()
        
        # Apply Otsu's thresholding
        if self.otsu_threshold == 0:
            _, binary = cv2.threshold(
                gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, binary = cv2.threshold(
                gray, self.otsu_threshold, 255, 
                cv2.THRESH_BINARY
            )
        
        return binary
    
    def adaptive_threshold(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding for uneven illumination.
        
        Why adaptive?
        - Handles varying illumination across the slide
        - Local threshold adapts to local contrast
        - Good for slides with uneven staining
        
        Args:
            tile: RGB input image
            
        Returns:
            Binary mask
        """
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile.copy()
        
        # Adaptive threshold (Gaussian method)
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        return binary
    
    def color_based_threshold(self, tile: np.ndarray) -> np.ndarray:
        """
        Use HSV color space for tissue detection.
        
        Why HSV?
        - Better separation of tissue (stained) vs background (unstained)
        - Hue channel captures stain color
        - Saturation channel captures stain intensity
        - More robust to illumination changes than RGB
        
        Args:
            tile: RGB input image
            
        Returns:
            Binary mask
        """
        # Convert to HSV
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        
        # H&E tissue ranges (approximate)
        # Low saturation = background (glass)
        # High saturation = tissue (stained)
        sat_channel = hsv[:, :, 1]
        
        # Threshold on saturation (tissue has higher saturation)
        _, binary = cv2.threshold(sat_channel, 30, 255, cv2.THRESH_BINARY)
        
        return binary
    
    def get_tissue_mask(self, tile: np.ndarray) -> np.ndarray:
        """
        Get binary tissue mask using configured method.
        
        Args:
            tile: RGB input image
            
        Returns:
            Binary mask (tissue=255, background=0)
        """
        if self.method == "otsu":
            mask = self.grayscale_threshold(tile)
        elif self.method == "adaptive":
            mask = self.adaptive_threshold(tile)
        elif self.method == "color_based":
            mask = self.color_based_threshold(tile)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Invert if needed (make tissue white)
        if self.invert:
            mask = cv2.bitwise_not(mask)
        
        return mask
    
    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Post-process mask to remove artifacts and fill holes.
        
        Steps:
        1. Morphological closing: fills small holes in tissue
        2. Remove small objects: removes debris and artifacts
        
        Args:
            mask: Binary mask
            
        Returns:
            Cleaned binary mask
        """
        # Morphological closing (fill holes)
        closed = morphology.closing(mask, self.kernel)
        
        # Remove small objects
        min_area = int(mask.shape[0] * mask.shape[1] * self.min_area_percent / 100)
        cleaned = morphology.remove_small_objects(
            closed.astype(bool), 
            max_size=min_area
        )
        
        return cleaned.astype(np.uint8) * 255
    
    def apply_mask(self, tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply binary mask to original image.
        
        Args:
            tile: Original RGB image
            mask: Binary mask (255=tissue, 0=background)
            
        Returns:
            Masked image (background pixels set to 0)
        """
        # Expand mask to 3 channels if needed
        if len(tile.shape) == 3 and len(mask.shape) == 2:
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
        else:
            mask_3ch = mask
        
        # Apply mask
        masked = cv2.bitwise_and(tile, mask_3ch)
        
        return masked
    
    def remove_background(self, tile: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Full background removal pipeline.
        
        Args:
            tile: RGB input image
            
        Returns:
            Tuple of (masked_image, tissue_mask, tissue_percentage)
        """
        # Get tissue mask
        mask = self.get_tissue_mask(tile)
        
        # Post-process mask
        mask = self.postprocess_mask(mask)
        
        # Calculate tissue percentage
        tissue_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        tissue_percentage = (tissue_pixels / total_pixels) * 100
        
        # Apply mask to image
        masked_tile = self.apply_mask(tile, mask)
        
        return masked_tile, mask, tissue_percentage


# Convenience function
def remove_background_from_tile(
    tile: np.ndarray,
    method: str = "otsu",
    morphological_disk: int = 5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience function for background removal.
    
    Usage:
        masked, mask, pct = remove_background_from_tile(tile)
    """
    remover = BackgroundRemover(
        method=method,
        morphological_disk=morphological_disk
    )
    return remover.remove_background(tile)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    bg_config = config["background_removal"]
    remover = BackgroundRemover(**bg_config)
    
    # Test with synthetic image
    test_tile = np.random.randint(100, 255, (1024, 1024, 3), dtype=np.uint8)
    # Add a dark region (simulated tissue)
    test_tile[200:800, 200:800, :] = np.random.randint(0, 100, (600, 600, 3), dtype=np.uint8)
    
    masked, mask, pct = remover.remove_background(test_tile)
    print(f"Tissue percentage: {pct:.1f}%")
    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask)}")
