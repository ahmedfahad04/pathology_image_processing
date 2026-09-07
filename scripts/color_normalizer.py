"""
Color Normalization Module
Standardizes stain colors across pathology image tiles.

Logic:
- H&E stains vary across slides due to:
  1. Different staining protocols/labs
  2. Stain degradation over time
  3. Scanner color calibration differences
  4. Uneven staining within slide
- Color normalization ensures consistent appearance:
  - Critical for machine learning models (trained on normalized data)
  - Improves segmentation accuracy
  - Enables fair comparison across slides
- Methods:
  1. Percentile normalization: simple, fast, effective
  2. Macenko method: stain vector decomposition (gold standard)
  3. Reinhard method: color transfer (alternative)
"""

import cv2
import numpy as np
from skimage.color import rgb2hed, hed2rgb
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ColorNormalizer:
    """
    Normalizes stain colors in H&E pathology images.
    
    Why color normalization?
    - Machine learning models are sensitive to color variation
    - Normalized input improves generalization
    - Reduces batch effects across slides
    - Standard preprocessing in digital pathology pipelines
    """
    
    def __init__(
        self,
        method: str = "percentile",
        percentile_low: int = 1,
        percentile_high: int = 99,
        target_mean: float = 0.5,
        target_std: float = 0.25,
        enabled: bool = True
    ):
        """
        Initialize color normalizer.
        
        Args:
            method: Normalization method ("percentile", "macenko", "reinhard")
            percentile_low: Lower percentile for percentile normalization
            percentile_high: Upper percentile for percentile normalization
            target_mean: Target mean for reinhard method
            target_std: Target std for reinhard method
            enabled: Whether to apply color normalization
        """
        self.method = method
        self.percentile_low = percentile_low
        self.enabled = enabled
        self.percentile_high = percentile_high
        self.target_mean = target_mean
        self.target_std = target_std
        
        logger.info(f"ColorNormalizer: method={method}")
    
    def normalize_percentile(self, tile: np.ndarray) -> np.ndarray:
        """
        Normalize using percentile-based scaling.
        
        Why percentile?
        - Robust to outliers (ignore extreme values)
        - Simple and fast
        - Effective for H&E stains
        - Maps percentile range to 0-255
        
        Args:
            tile: RGB input image
            
        Returns:
            Normalized RGB image
        """
        # Ensure float for calculation
        tile_float = tile.astype(np.float64)
        
        # Normalize each channel independently
        normalized = np.zeros_like(tile_float)
        
        for i in range(3):  # R, G, B channels
            channel = tile_float[:, :, i]
            
            # Calculate percentiles
            p_low = np.percentile(channel, self.percentile_low)
            p_high = np.percentile(channel, self.percentile_high)
            
            # Scale to 0-255
            if p_high - p_low > 0:
                normalized[:, :, i] = np.clip(
                    (channel - p_low) / (p_high - p_low) * 255,
                    0, 255
                )
            else:
                normalized[:, :, i] = channel
        
        return normalized.astype(np.uint8)
    
    def normalize_macenko(self, tile: np.ndarray, stain_matrix: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize using Macenko's stain vector method.
        
        Why Macenko?
        - Decomposes image into hematoxylin and eosin channels
        - Normalizes stain concentrations independently
        - Gold standard for H&E normalization
        - More robust to varying tissue types
        
        Args:
            tile: RGB input image
            stain_matrix: Optional pre-computed stain matrix
            
        Returns:
            Normalized RGB image
        """
        # Convert to HED
        hed = rgb2hed(tile / 255.0 if tile.max() > 1 else tile)
        
        # Extract channels
        h_channel = hed[:, :, 0]  # Hematoxylin
        e_channel = hed[:, :, 1]  # Eosin
        
        # Normalize hematoxylin channel
        h_min, h_max = np.percentile(h_channel, [self.percentile_low, self.percentile_high])
        if h_max - h_min > 0:
            h_normalized = np.clip((h_channel - h_min) / (h_max - h_min), 0, 1)
        else:
            h_normalized = h_channel
        
        # Normalize eosin channel
        e_min, e_max = np.percentile(e_channel, [self.percentile_low, self.percentile_high])
        if e_max - e_min > 0:
            e_normalized = np.clip((e_channel - e_min) / (e_max - e_min), 0, 1)
        else:
            e_normalized = e_channel
        
        # Reconstruct HED
        hed_normalized = np.stack([h_normalized, e_normalized, hed[:, :, 2]], axis=2)
        
        # Convert back to RGB
        normalized_rgb = hed2rgb(hed_normalized)
        
        # Scale to 0-255
        normalized_rgb = (normalized_rgb * 255).astype(np.uint8)
        
        return normalized_rgb
    
    def normalize_reinhard(self, tile: np.ndarray, target_stats: Optional[dict] = None) -> np.ndarray:
        """
        Normalize using Reinhard's color transfer method.
        
        Why Reinhard?
        - Transfers color statistics from target to source
        - Works in LAB color space (perceptually uniform)
        - Good for matching different scanners/protocols
        
        Args:
            tile: RGB input image
            target_stats: Target statistics (mean, std in LAB space)
            
        Returns:
            Normalized RGB image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB)
        
        # Calculate source statistics
        src_mean = np.mean(lab, axis=(0, 1))
        src_std = np.std(lab, axis=(0, 1))
        
        # Use default target stats if not provided
        if target_stats is None:
            # Default: normalize each channel to target_mean, target_std
            target_mean = np.array([self.target_mean * 255, 128, 128])  # L, a, b
            target_std = np.array([self.target_std * 255, 50, 50])
        else:
            target_mean = target_stats["mean"]
            target_std = target_stats["std"]
        
        # Normalize each channel
        normalized = np.zeros_like(lab, dtype=np.float64)
        
        for i in range(3):
            if src_std[i] > 0:
                normalized[:, :, i] = (
                    (lab[:, :, i].astype(np.float64) - src_mean[i]) / src_std[i] * target_std[i] + target_mean[i]
                )
            else:
                normalized[:, :, i] = lab[:, :, i]
        
        # Clip to valid range
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        normalized_rgb = cv2.cvtColor(normalized, cv2.COLOR_LAB2RGB)
        
        return normalized_rgb
    
    def normalize_stain_concentration(self, tile: np.ndarray, target_concentration: Optional[dict] = None) -> np.ndarray:
        """
        Normalize stain concentrations directly.
        
        Why concentration normalization?
        - Directly controls stain intensity
        - Useful when you know target stain concentration
        - More intuitive than color transfer
        
        Args:
            tile: RGB input image
            target_concentration: Target stain concentrations
            
        Returns:
            Normalized RGB image
        """
        # Convert to float
        tile_float = tile.astype(np.float64) / 255.0
        
        # Simple channel-wise normalization
        # For H&E, we can normalize based on channel statistics
        
        # Calculate mean intensity per channel
        src_mean = np.mean(tile_float, axis=(0, 1))
        
        # Target means (default: 0.5 for each channel)
        if target_concentration is None:
            target_mean = np.array([0.45, 0.35, 0.45])  # Slightly adjusted for H&E
        else:
            target_mean = np.array([target_concentration.get("r", 0.5),
                                   target_concentration.get("g", 0.5),
                                   target_concentration.get("b", 0.5)])
        
        # Normalize
        normalized = np.zeros_like(tile_float)
        for i in range(3):
            if src_mean[i] > 0:
                normalized[:, :, i] = tile_float[:, :, i] * (target_mean[i] / src_mean[i])
            else:
                normalized[:, :, i] = tile_float[:, :, i]
        
        # Clip and convert back
        normalized = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        
        return normalized
    
    def normalize(self, tile: np.ndarray) -> np.ndarray:
        """
        Apply configured normalization method.
        
        Args:
            tile: RGB input image
            
        Returns:
            Normalized RGB image
        """
        if not self.enabled:
            return tile
        
        if self.method == "percentile":
            return self.normalize_percentile(tile)
        elif self.method == "macenko":
            return self.normalize_macenko(tile)
        elif self.method == "reinhard":
            return self.normalize_reinhard(tile)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def normalize_batch(self, tiles: list) -> list:
        """
        Normalize a batch of tiles.
        
        Args:
            tiles: List of RGB images
            
        Returns:
            List of normalized RGB images
        """
        return [self.normalize(tile) for tile in tiles]


# Convenience function
def normalize_stain_color(
    tile: np.ndarray,
    method: str = "percentile",
    percentile_low: int = 1,
    percentile_high: int = 99
) -> np.ndarray:
    """
    Convenience function for color normalization.
    
    Usage:
        normalized = normalize_stain_color(tile)
    """
    normalizer = ColorNormalizer(
        method=method,
        percentile_low=percentile_low,
        percentile_high=percentile_high
    )
    return normalizer.normalize(tile)


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    color_config = config["color_normalization"]
    normalizer = ColorNormalizer(**color_config)
    
    # Test with synthetic image
    test_tile = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    normalized = normalizer.normalize(test_tile)
    
    print(f"Input range: {test_tile.min()}-{test_tile.max()}")
    print(f"Output range: {normalized.min()}-{normalized.max()}")
    print(f"Input mean per channel: {np.mean(test_tile, axis=(0, 1))}")
    print(f"Output mean per channel: {np.mean(normalized, axis=(0, 1))}")
