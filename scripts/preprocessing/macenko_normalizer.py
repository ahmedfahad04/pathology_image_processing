"""
Macenko H&E Stain Normalization Module
Refactored from 122_normalizing_HnE_images.py (M. Macenko et al., ISBI 2009)

Workflow:
  Input: RGB image (uint8, H&E stained patch)
  Step 1: Convert RGB to OD (Optical Density)  OD = -log10((I+1)/Io)
  Step 2: Remove transparent pixels (OD < beta)
  Step 3: SVD on OD tuples -> covariance eigen-decomposition
  Step 4: Plane from two largest eigenvectors, project & normalize
  Step 5-6: Angle w.r.t first SVD direction
  Step 7: Robust extremes (alpha-th and 100-alpha-th percentiles)
  Step 8: Stain vectors HE, stain concentrations C, normalize by maxCRef
  Output: Normalized RGB image (uint8)
"""

import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

# Default reference values from Macenko paper / 122 script
HERef = np.array([[0.5626, 0.2159],
                  [0.7201, 0.8012],
                  [0.4062, 0.5581]])

maxCRef = np.array([1.9705, 1.0308])


class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E images.

    Reference:
        M. Macenko et al., "A method for normalizing histology slides for
        quantitative analysis", ISBI 2009.

    Args:
        Io: Transmitted light intensity (default 240)
        alpha: Tolerance for pseudo-min/max percentiles (default 1)
        beta: OD threshold for transparent pixels (default 0.15)
        HERef: Reference H&E OD matrix (3x2)
        maxCRef: Reference max stain concentrations (2,)
    """

    def __init__(
        self,
        Io: int = 240,
        alpha: int = 1,
        beta: float = 0.15,
        HERef: np.ndarray = None,
        maxCRef: np.ndarray = None,
    ):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = HERef if HERef is not None else np.array([[0.5626, 0.2159],
                                                               [0.7201, 0.8012],
                                                               [0.4062, 0.5581]])
        self.maxCRef = maxCRef if maxCRef is not None else np.array([1.9705, 1.0308])
        logger.info(f"MacenkoNormalizer: Io={Io}, alpha={alpha}, beta={beta}")

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize a single RGB tile using Macenko method.

        Args:
            img: RGB image, uint8, shape (H, W, 3), values 0-255

        Returns:
            Normalized RGB image, uint8, same shape

        Handles edge cases:
            - Grayscale input -> stacked to RGB
            - All-white / low tissue content -> returns original (with warning)
        """
        if img is None or img.size == 0:
            raise ValueError("Empty image provided to MacenkoNormalizer")

        # Ensure RGB 3-channel uint8
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            # If float in [0,1], scale
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        h, w, c = img.shape

        # Reshape to (N,3)
        img_flat = img.reshape((-1, 3)).astype(np.float64)

        # Step 1: Convert RGB to OD
        #  OD = -log10((I+1)/Io)  -- +1 to avoid log(0)
        OD = -np.log10((img_flat + 1) / self.Io)

        # Step 2: Remove transparent pixels
        ODhat = OD[~np.any(OD < self.beta, axis=1)]

        if ODhat.shape[0] < 10:
            logger.warning(f"Macenko: too few OD pixels above beta ({ODhat.shape[0]}), returning original tile {h}x{w}")
            return img

        # Step 3: SVD via eigen-decomposition of covariance
        try:
            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        except np.linalg.LinAlgError as e:
            logger.warning(f"Macenko: cov/eigh failed ({e}), returning original")
            return img

        # Step 4: Plane from two largest eigenvectors
        # eigvecs sorted ascending, so take last two columns
        eigvecs = eigvecs[:, [1, 2]] if eigvecs.shape[1] >= 3 else eigvecs[:, -2:]
        That = ODhat.dot(eigvecs)

        # Step 5 & 6: Angle w.r.t first SVD direction
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)

        vMin = eigvecs.dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs.dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

        # Heuristic: hematoxylin first
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T

        # Step 7-8: Concentrations and normalization
        Y = OD.T  # (3, N)

        try:
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]  # (2, N)
        except np.linalg.LinAlgError as e:
            logger.warning(f"Macenko: lstsq failed ({e}), returning original")
            return img

        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        # Avoid division by zero
        maxC = np.maximum(maxC, 1e-6)
        tmp = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])

        # Recreate normalized image
        Inorm = np.multiply(self.Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        return Inorm

    def normalize_with_he(self, img: np.ndarray):
        """
        Normalize and also return H and E separated components.

        Returns:
            Tuple (Inorm, H, E) all uint8 RGB
        """
        # Reuse logic but compute H/E
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        h, w, c = img.shape
        img_flat = img.reshape((-1, 3)).astype(np.float64)
        OD = -np.log10((img_flat + 1) / self.Io)
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        if ODhat.shape[0] < 10:
            return img, img, img
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        eigvecs = eigvecs[:, [1, 2]] if eigvecs.shape[1] >= 3 else eigvecs[:, -2:]
        That = ODhat.dot(eigvecs)
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        vMin = eigvecs.dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs.dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T
        Y = OD.T
        C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
        maxC = np.maximum(maxC, 1e-6)
        tmp = np.divide(maxC, self.maxCRef)
        C2 = np.divide(C, tmp[:, np.newaxis])
        Inorm = np.multiply(self.Io, np.exp(-self.HERef.dot(C2)))
        Inorm[Inorm > 255] = 254
        Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

        H = np.multiply(self.Io, np.exp(np.expand_dims(-self.HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))))
        H[H > 255] = 254
        H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

        E = np.multiply(self.Io, np.exp(np.expand_dims(-self.HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))))
        E[E > 255] = 254
        E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

        return Inorm, H, E

    def normalize_batch(self, tiles):
        """Normalize a list of tiles."""
        return [self.normalize(t) for t in tiles]


# Convenience function matching old script style
def normalizeStaining(img: np.ndarray, Io: int = 240, alpha: int = 1, beta: float = 0.15,
                      HERef=None, maxCRef=None) -> np.ndarray:
    """
    Functional interface, equivalent to the procedural code in 122_normalizing_HnE_images.py.
    Input: RGB uint8 image. Output: normalized RGB uint8.
    """
    normalizer = MacenkoNormalizer(Io=Io, alpha=alpha, beta=beta, HERef=HERef, maxCRef=maxCRef)
    return normalizer.normalize(img)


if __name__ == "__main__":
    import sys
    # Simple CLI test: python macenko_normalizer.py input.jpg output.jpg
    if len(sys.argv) >= 3:
        in_path, out_path = sys.argv[1], sys.argv[2]
        bgr = cv2.imread(in_path, 1)
        if bgr is None:
            print(f"Cannot read {in_path}")
            sys.exit(1)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = normalizeStaining(rgb)
        cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"Saved normalized to {out_path}")
    else:
        # Synthetic test
        test = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        out = normalizeStaining(test)
        print(f"Synthetic test: {test.shape} -> {out.shape}, range {out.min()}-{out.max()}")
