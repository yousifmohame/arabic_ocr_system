from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DocumentQuad:
    """Detected document corners in source-image pixel coordinates."""

    points: np.ndarray  # float32, ordered TL, TR, BR, BL
    score: float
    area_ratio: float


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """Return four points in top-left, top-right, bottom-right, bottom-left order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)

    ordered = np.empty((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]       # top-left
    ordered[2] = points[np.argmax(sums)]       # bottom-right
    ordered[1] = points[np.argmin(differences)]  # top-right
    ordered[3] = points[np.argmax(differences)]  # bottom-left
    return ordered


def detect_document_quad(image_rgb: np.ndarray) -> DocumentQuad | None:
    """Conservatively detect a four-corner document in a photo.

    The detector intentionally returns None when the evidence is weak. In that
    case the original image is kept rather than risking an incorrect crop.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        raise RuntimeError("OpenCV is required for document-boundary detection.") from exc

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")

    original_height, original_width = image_rgb.shape[:2]
    if min(original_height, original_width) < 160:
        return None

    # Keep contour detection inexpensive on high-resolution phone photographs.
    detection_scale = min(1.0, 1000.0 / max(original_height, original_width))
    if detection_scale < 1.0:
        detect_width = max(1, int(round(original_width * detection_scale)))
        detect_height = max(1, int(round(original_height * detection_scale)))
        detection_rgb = cv2.resize(
            image_rgb, (detect_width, detect_height), interpolation=cv2.INTER_AREA
        )
    else:
        detection_rgb = image_rgb

    gray = cv2.cvtColor(detection_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    height, width = gray.shape[:2]
    image_area = float(height * width)
    min_margin = max(3.0, min(height, width) * 0.008)

    candidates: list[tuple[float, DocumentQuad]] = []
    # A small threshold sweep makes the boundary detector less sensitive to
    # the particular color and exposure of the photographed document.
    for low, high in ((35, 110), (55, 155), (75, 200)):
        edges = cv2.Canny(gray, low, high)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:35]:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            contour_area = float(cv2.contourArea(contour))
            if contour_area / image_area < 0.12:
                break

            for epsilon_ratio in (0.015, 0.02, 0.03, 0.04):
                polygon = cv2.approxPolyDP(
                    contour, epsilon_ratio * perimeter, True
                )
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue

                small_points = polygon.reshape(4, 2).astype(np.float32)
                ordered_small = order_quad_points(small_points)
                if not _has_safe_margin(ordered_small, width, height, min_margin):
                    continue

                area_ratio = abs(float(cv2.contourArea(ordered_small))) / image_area
                if area_ratio < 0.12 or area_ratio > 0.97:
                    continue

                ordered_original = ordered_small / detection_scale
                geometry = _quad_geometry_score(ordered_small)
                if geometry < 0.35:
                    continue

                # Prefer large, convex, roughly rectangular contours while
                # penalizing extreme perspective or implausible geometry.
                score = area_ratio * geometry
                quad = DocumentQuad(
                    points=ordered_original.astype(np.float32),
                    score=float(score),
                    area_ratio=float(area_ratio),
                )
                candidates.append((score, quad))
                break

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = candidates[0][1]
    if best.score < 0.22:
        return None
    return best


def _has_safe_margin(points: np.ndarray, width: int, height: int, margin: float) -> bool:
    for x, y in points:
        if x < margin or y < margin or x > width - 1 - margin or y > height - 1 - margin:
            return False
    return True


def _quad_geometry_score(points: np.ndarray) -> float:
    """Score quadrilateral angles and edge lengths; 1.0 is near-rectangular."""
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.min(lengths) < 30:
        return 0.0

    angles: list[float] = []
    for index in range(4):
        current = points[index]
        previous = points[(index - 1) % 4] - current
        following = points[(index + 1) % 4] - current
        denom = float(np.linalg.norm(previous) * np.linalg.norm(following))
        if denom <= 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(previous, following) / denom, -1.0, 1.0))
        angles.append(float(np.degrees(np.arccos(cosine))))

    angle_error = float(np.mean([abs(angle - 90.0) for angle in angles]))
    angle_score = max(0.0, 1.0 - angle_error / 75.0)

    top_bottom = (lengths[0] + lengths[2]) / 2.0
    left_right = (lengths[1] + lengths[3]) / 2.0
    ratio = max(top_bottom, left_right) / max(1.0, min(top_bottom, left_right))
    if ratio > 2.8:
        return 0.0
    ratio_score = 1.0 if ratio <= 2.2 else (2.8 - ratio) / 0.6
    return angle_score * ratio_score


def perspective_transform_for_quad(
    points: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return source-to-rectified homography and (width, height)."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for perspective correction.") from exc

    tl, tr, br, bl = order_quad_points(points)
    width_top = float(np.linalg.norm(tr - tl))
    width_bottom = float(np.linalg.norm(br - bl))
    height_right = float(np.linalg.norm(br - tr))
    height_left = float(np.linalg.norm(bl - tl))
    target_width = max(2, int(round(max(width_top, width_bottom))))
    target_height = max(2, int(round(max(height_left, height_right))))

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(
        np.asarray([tl, tr, br, bl], dtype=np.float32), destination
    )
    return transform, (target_width, target_height)
