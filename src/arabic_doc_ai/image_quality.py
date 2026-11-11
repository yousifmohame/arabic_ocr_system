from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    brightness_mean: float
    contrast_std: float
    dynamic_range_p95_p5: float
    blur_laplacian_variance: float
    illumination_variation: float
    edge_density: float
    estimated_skew_degrees: float
    skew_line_support: int
    document_roi_used: bool

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    @property
    def short_side(self) -> int:
        return min(self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["long_side"] = self.long_side
        result["short_side"] = self.short_side
        return {key: round(value, 4) if isinstance(value, float) else value
                for key, value in result.items()}


def analyze_image_quality(
    image_rgb: np.ndarray, *, region_points: np.ndarray | None = None
) -> ImageQuality:
    """Measure inexpensive visual signals used to propose preprocessing candidates.

    These values describe image characteristics; they are not estimates of OCR
    accuracy. The image is downsampled before analysis to bound CPU and memory.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("OpenCV is required for image quality analysis.") from exc

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an uint8 RGB image.")

    height, width = image_rgb.shape[:2]
    scale = min(1.0, 900.0 / max(height, width))
    if scale < 1.0:
        sample = cv2.resize(
            image_rgb,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = image_rgb

    gray = cv2.cvtColor(sample, cv2.COLOR_RGB2GRAY)
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    document_roi_used = False
    if region_points is not None:
        points = np.asarray(region_points, dtype=np.float32).reshape(4, 2) * scale
        polygon = np.round(points).astype(np.int32)
        candidate_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(candidate_mask, polygon, 255)
        if np.count_nonzero(candidate_mask) >= int(gray.size * 0.15):
            mask = candidate_mask
            document_roi_used = True

    valid = mask > 0
    selected_gray = gray[valid]
    brightness = float(np.mean(selected_gray))
    contrast = float(np.std(selected_gray))
    p5, p95 = np.percentile(selected_gray, [5, 95])

    # Laplacian variance is a relative sharpness indicator. It is texture- and
    # resolution-dependent, so the decision engine uses it as a weak signal.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blur_score = float(laplacian[valid].var())

    # Estimate broad illumination variation after suppressing fine text edges.
    blur_sigma = max(3.0, min(gray.shape[:2]) / 18.0)
    low_frequency = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)
    illumination_variation = float(np.std(low_frequency[valid]))

    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.count_nonzero(edges[valid])) / float(np.count_nonzero(valid))
    skew_angle, skew_support = _estimate_skew(edges, mask=mask)

    return ImageQuality(
        width=int(width),
        height=int(height),
        brightness_mean=brightness,
        contrast_std=contrast,
        dynamic_range_p95_p5=float(p95 - p5),
        blur_laplacian_variance=blur_score,
        illumination_variation=illumination_variation,
        edge_density=edge_density,
        estimated_skew_degrees=skew_angle,
        skew_line_support=skew_support,
        document_roi_used=document_roi_used,
    )


def _estimate_skew(
    edges: np.ndarray, *, mask: np.ndarray | None = None
) -> tuple[float, int]:
    """Estimate dominant near-horizontal line angle from long Hough segments."""
    import cv2

    height, width = edges.shape[:2]
    min_length = max(40, int(round(max(height, width) * 0.28)))
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=max(25, int(round(min(height, width) * 0.07))),
        minLineLength=min_length,
        maxLineGap=max(8, int(round(min(height, width) * 0.025))),
    )
    if lines is None:
        return 0.0, 0

    angles: list[float] = []
    weights: list[float] = []
    for entry in lines[:, 0, :]:
        x1, y1, x2, y2 = map(float, entry)
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_length:
            continue
        angle = float(np.degrees(np.arctan2(dy, dx)))
        # Keep near-horizontal long edges; vertical page boundaries are ignored.
        if abs(angle) <= 20.0:
            angles.append(angle)
            weights.append(length)
    if not angles:
        return 0.0, 0

    # Weighted median is robust to multiple short text/decoration segments.
    order = np.argsort(np.asarray(angles))
    sorted_angles = np.asarray(angles)[order]
    sorted_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(sorted_weights)
    median_index = int(np.searchsorted(cumulative, cumulative[-1] / 2.0))
    return float(sorted_angles[min(median_index, len(sorted_angles) - 1)]), len(angles)
