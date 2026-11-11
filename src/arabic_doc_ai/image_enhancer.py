from __future__ import annotations

import numpy as np


def enhance_document_image(image_rgb: np.ndarray) -> np.ndarray:
    """Apply conservative local contrast and mild sharpening to an RGB image.

    No binarization is performed: this avoids erasing Arabic dots and diacritics
    in textured or low-contrast documents.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        raise RuntimeError("OpenCV is required for image enhancement.") from exc

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an uint8 RGB image.")

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

    # Mild unsharp masking. The low amount is intentional: aggressive sharpening
    # can create false dots around Arabic letters and punctuation.
    blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=0.8)
    sharpened = cv2.addWeighted(enhanced, 1.25, blurred, -0.25, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)
