"""
Tissue Detection Module
Identifies whether a tile contains valid tissue for processing.

Logic:
- Not all extracted tiles contain tissue (some are pure glass/background)
- Tissue tiles must meet criteria:
  1. Minimum tissue percentage (e.g., >10% tissue)
  2. Focus quality (sharp enough for analysis)
  3. Staining quality (properly stained, not faded)
- Filtering out non-tissue tiles:
  - Reduces processing time (skip empty tiles)
  - Prevents false positives in downstream analysis
  - Improves overall pipeline efficiency
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TissueDetector:
    """
    Detects whether a tile contains valid tissue.
    
    Why tissue detection?
    - SVS images have large background areas (glass slides)
    - Extracting/processing empty tiles wastes computation
    - Focus check ensures image quality for analysis
    - Enables statistics about tissue coverage
    """
    
    def __init__(
        self,
        min_tissue_percent: float = 10.0,
        focus_method: str = "laplacian",
        focus_threshold: float = 100.0,
        check_focus: bool = True
    ):
        """
        Initialize tissue detector.
        
        Args:
            min_tissue_percent: Minimum tissue % to consider tile valid
            focus_method: Method for focus assessment ("laplacian" or "variance")
            focus_threshold: Minimum focus score
            check_focus: Whether to check focus quality
        """
        self.min_tissue_percent = min_tissue_percent
        self.focus_method = focus_method
        self.focus_threshold = focus_threshold
        self.check_focus = check_focus
        
        logger.info(f"TissueDetector: min_tissue={min_tissue_percent}%, "
                    f"focus_method={focus_method}, threshold={focus_threshold}")
    
    def calculate_tissue_percentage(self, tile: np.ndarray) -> float:
        """
        Calculate percentage of tissue in tile.
        
        Method: Convert to grayscale, threshold to separate tissue from background.
        
        Args:
            tile: RGB input image
            
        Returns:
            Tissue percentage (0-100)
        """
        # Convert to grayscale
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile.copy()
        
        # Otsu threshold (automatic)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Tissue is darker (lower values in grayscale)
        # After Otsu, tissue should be the foreground
        tissue_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]
        
        tissue_percentage = (tissue_pixels / total_pixels) * 100
        
        return tissue_percentage
    
    def calculate_focus_score_laplacian(self, tile: np.ndarray) -> float:
        """
        Calculate focus score using Laplacian variance.
        
        Why Laplacian?
        - Laplacian detects edges (second derivative)
        - Variance of Laplacian indicates edge strength
        - High variance = many sharp edges = focused
        - Low variance = blurry = unfocused
        
        Args:
            tile: RGB input image
            
        Returns:
            Focus score (higher = more focused)
        """
        # Convert to grayscale
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile.copy()
        
        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # Return variance
        return laplacian.var()
    
    def calculate_focus_score_variance(self, tile: np.ndarray) -> float:
        """
        Calculate focus score using intensity variance.
        
        Why variance?
        - Focused images have higher contrast
        - Higher contrast = higher variance in intensity
        - Simple but effective for binary focus assessment
        
        Args:
            tile: RGB input image
            
        Returns:
            Focus score (higher = more focused)
        """
        # Convert to grayscale
        if len(tile.shape) == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile.copy()
        
        return gray.var()
    
    def assess_focus(self, tile: np.ndarray) -> Tuple[bool, float]:
        """
        Assess image focus quality.
        
        Args:
            tile: RGB input image
            
        Returns:
            Tuple of (is_focused, focus_score)
        """
        if not self.check_focus:
            return True, 0.0
        
        if self.focus_method == "laplacian":
            score = self.calculate_focus_score_laplacian(tile)
        elif self.focus_method == "variance":
            score = self.calculate_focus_score_variance(tile)
        else:
            raise ValueError(f"Unknown focus method: {self.focus_method}")
        
        is_focused = score >= self.focus_threshold
        
        return is_focused, score
    
    def assess_staining(self, tile: np.ndarray) -> Tuple[bool, dict]:
        """
        Assess staining quality (basic check).
        
        Why check staining?
        - Faded stains reduce segmentation accuracy
        - Over-saturation can obscure details
        - Basic check ensures stain is within acceptable range
        
        Args:
            tile: RGB input image
            
        Returns:
            Tuple of (is_acceptable, staining_stats)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        
        # H&E stain characteristics
        # Hue: ~120-180 (purple/blue for hematoxylin)
        # Saturation: moderate to high
        
        mean_hue = np.mean(hsv[:, :, 0])
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        stats = {
            "mean_hue": mean_hue,
            "mean_saturation": mean_saturation,
            "mean_value": mean_value
        }
        
        # Basic checks (adjust based on your stain protocol)
        # For H&E: saturation should be moderate, value should be in valid range
        # Relaxed thresholds to handle various staining intensities
        is_acceptable = (
            10 < mean_saturation < 220 and  # Not too pale, not oversaturated
            30 < mean_value < 240            # Not too dark, not too bright
        )
        
        return is_acceptable, stats
    
    def detect_tissue(self, tile: np.ndarray) -> Tuple[bool, dict]:
        """
        Full tissue detection pipeline.
        
        Args:
            tile: RGB input image
            
        Returns:
            Tuple of (is_valid, assessment_dict)
        """
        assessment = {}
        
        # 1. Calculate tissue percentage
        tissue_pct = float(self.calculate_tissue_percentage(tile))
        assessment["tissue_percentage"] = tissue_pct
        assessment["has_tissue"] = bool(tissue_pct >= self.min_tissue_percent)
        
        # 2. Assess focus
        is_focused, focus_score = self.assess_focus(tile)
        assessment["is_focused"] = bool(is_focused)
        assessment["focus_score"] = float(focus_score)
        
        # 3. Assess staining (basic)
        is_stained, stain_stats = self.assess_staining(tile)
        assessment["is_stained"] = bool(is_stained)
        assessment["staining"] = {k: float(v) for k, v in stain_stats.items()}
        
        # Overall decision
        is_valid = (
            assessment["has_tissue"] and
            assessment["is_focused"] and
            assessment["is_stained"]
        )
        
        assessment["is_valid"] = bool(is_valid)
        
        if not is_valid:
            reasons = []
            if not assessment["has_tissue"]:
                reasons.append(f"low tissue ({tissue_pct:.1f}%)")
            if not assessment["is_focused"]:
                reasons.append(f"unfocused ({focus_score:.1f})")
            if not assessment["is_stained"]:
                reasons.append("poor staining")
            logger.debug(f"Tile rejected: {', '.join(reasons)}")
        
        return is_valid, assessment


# Convenience function
def detect_tissue_in_tile(
    tile: np.ndarray,
    min_tissue_percent: float = 10.0,
    focus_threshold: float = 100.0
) -> Tuple[bool, float]:
    """
    Convenience function for tissue detection.
    
    Usage:
        is_valid, tissue_pct = detect_tissue_in_tile(tile)
    """
    detector = TissueDetector(
        min_tissue_percent=min_tissue_percent,
        focus_threshold=focus_threshold
    )
    is_valid, assessment = detector.detect_tissue(tile)
    return is_valid, assessment["tissue_percentage"]


if __name__ == "__main__":
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    tissue_config = config["tissue_detection"]
    detector = TissueDetector(**tissue_config)
    
    # Test with synthetic images
    # High tissue tile
    tissue_tile = np.random.randint(0, 100, (1024, 1024, 3), dtype=np.uint8)
    is_valid, assessment = detector.detect_tissue(tissue_tile)
    print(f"Tissue tile: valid={is_valid}, tissue%={assessment['tissue_percentage']:.1f}")
    
    # High background tile
    bg_tile = np.random.randint(200, 255, (1024, 1024, 3), dtype=np.uint8)
    is_valid, assessment = detector.detect_tissue(bg_tile)
    print(f"Background tile: valid={is_valid}, tissue%={assessment['tissue_percentage']:.1f}")
