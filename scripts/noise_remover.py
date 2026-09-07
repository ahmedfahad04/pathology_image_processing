"""
Noise Removal Module
Applies various filters to reduce noise in pathology image tiles.

Logic:
- H&E stained slides have specific noise patterns:
  1. Salt-and-pepper noise from scanner artifacts
  2. Gaussian noise from electronic sensor
  3. Small debris/artifacts on slide
- Different filters target different noise types:
  - Gaussian blur: smooths Gaussian noise (preserves edges better)
  - Median filter: removes salt-and-pepper noise (preserves edges)
  - Morphological opening: removes small bright artifacts
- Pipeline: Gaussian → Median → Morphological (progressive refinement)
"""

import cv2
import numpy as np
from skimage import morphology
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NoiseRemover:
    """
    Removes noise from pathology image tiles using multi-stage filtering.
    
    Why multiple filters?
    - Single filter can't handle all noise types
    - Gaussian + Median combination is standard for medical images
    - Morphological opening removes small artifacts without affecting tissue
    """
    
    def __init__(
        self,
        gaussian_kernel: int = 3,
        median_kernel: int = 3,
        morphological_disk: int = 2,
        enabled: bool = True
    ):
        """
        Initialize noise remover.
        
        Args:
            gaussian_kernel: Kernel size for Gaussian blur (must be odd)
            median_kernel: Kernel size for median filter (must be odd)
            morphological_disk: Radius for morphological disk kernel
            enabled: Whether to apply noise reduction
        """
        # Ensure odd kernel sizes
        self.gaussian_kernel = gaussian_kernel if gaussian_kernel % 2 == 1 else gaussian_kernel + 1
        self.median_kernel = median_kernel if median_kernel % 2 == 1 else median_kernel + 1
        self.morphological_disk = morphological_disk
        self.enabled = enabled
        
        # Create morphological kernel once (reusable)
        self.kernel = morphology.disk(self.morphological_disk)
        
        logger.info(f"NoiseRemover: gaussian={self.gaussian_kernel}, "
                    f"median={self.median_kernel}, morpho_disk={self.morphological_disk}")
    
    def apply_gaussian_blur(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to reduce Gaussian noise.
        
        Why Gaussian?
        - Smooths random variations in pixel intensity
        - Preserves edges better than simple averaging
        - Kernel size 3x3 is standard for medical images (minimal blur)
        
        Args:
            tile: Input image (RGB or grayscale)
            
        Returns:
            Blurred image
        """
        return cv2.GaussianBlur(
            tile, 
            (self.gaussian_kernel, self.gaussian_kernel), 
            0  # sigmaX = 0 (auto-calculated from kernel)
        )
    
    def apply_median_filter(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply median filter to remove salt-and-pepper noise.
        
        Why Median?
        - Excellent for impulse noise (bright/dark spots)
        - Preserves edges better than mean filter
        - Common in scanner artifacts and dust
        
        Args:
            tile: Input image (RGB or grayscale)
            
        Returns:
            Filtered image
        """
        return cv2.medianBlur(tile, self.median_kernel)
    
    def apply_morphological_opening(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening to remove small bright artifacts.
        
        Why Opening?
        - Opening = erosion + dilation
        - Removes small bright objects (debris, dust)
        - Preserves larger structures (tissue, nuclei)
        - Disk kernel is isotropic (no directional bias)
        
        Args:
            tile: Input image (RGB or grayscale)
            
        Returns:
            Cleaned image
        """
        if len(tile.shape) == 3:
            # Apply to each channel separately for color images
            result = np.zeros_like(tile)
            for i in range(tile.shape[2]):
                result[:, :, i] = morphology.opening(tile[:, :, i], self.kernel)
            return result
        else:
            return morphology.opening(tile, self.kernel)
    
    def remove_noise(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply full noise removal pipeline.
        
        Pipeline logic:
        1. Gaussian blur: smooth random noise
        2. Median filter: remove impulse noise
        3. Morphological opening: remove small artifacts
        
        Args:
            tile: Input image (RGB, uint8)
            
        Returns:
            Denoised image
        """
        if not self.enabled:
            return tile
        
        # Ensure uint8
        if tile.dtype != np.uint8:
            tile = (tile * 255).astype(np.uint8) if tile.max() <= 1.0 else tile.astype(np.uint8)
        
        # Step 1: Gaussian blur (mild)
        denoised = self.apply_gaussian_blur(tile)
        
        # Step 2: Median filter (remove impulse noise)
        denoised = self.apply_median_filter(denoised)
        
        # Step 3: Morphological opening (remove small artifacts)
        denoised = self.apply_morphological_opening(denoised)
        
        return denoised
    
    def remove_noise_selective(self, tile: np.ndarray, method: str = "full") -> np.ndarray:
        """
        Apply specific noise removal method.
        
        Args:
            tile: Input image
            method: "gaussian", "median", "morphological", or "full"
            
        Returns:
            Filtered image
        """
        if method == "gaussian":
            return self.apply_gaussian_blur(tile)
        elif method == "median":
            return self.apply_median_filter(tile)
        elif method == "morphological":
            return self.apply_morphological_opening(tile)
        elif method == "full":
            return self.remove_noise(tile)
        else:
            raise ValueError(f"Unknown method: {method}")


# Convenience function
def remove_noise_from_tile(
    tile: np.ndarray,
    gaussian_kernel: int = 3,
    median_kernel: int = 3,
    morphological_disk: int = 2
) -> np.ndarray:
    """
    Convenience function for noise removal.
    
    Usage:
        clean_tile = remove_noise_from_tile(tile)
    """
    remover = NoiseRemover(
        gaussian_kernel=gaussian_kernel,
        median_kernel=median_kernel,
        morphological_disk=morphological_disk
    )
    return remover.remove_noise(tile)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    noise_config = config["noise_reduction"]
    remover = NoiseRemover(**noise_config)
    
    # Test with synthetic noise
    test_tile = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    clean = remover.remove_noise(test_tile)
    print(f"Input dtype: {test_tile.dtype}, Output dtype: {clean.dtype}")
    print(f"Input range: {test_tile.min()}-{test_tile.max()}, Output range: {clean.min()}-{clean.max()}")
