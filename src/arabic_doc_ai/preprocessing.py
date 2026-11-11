from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import OCRConfig
from .image_enhancer import enhance_document_image
from .perspective import DocumentQuad, detect_document_quad, perspective_transform_for_quad


@dataclass
class PreprocessResult:
    image: np.ndarray
    # Maps coordinates in the processed image back to the input page image.
    inverse_transform: np.ndarray
    metadata: dict[str, Any]


def preprocess_variant(
    image_rgb: np.ndarray,
    config: OCRConfig,
    operations: Iterable[str],
    *,
    allow_perspective_correction: bool,
    detected_quad: DocumentQuad | None = None,
    deskew_angle_degrees: float | None = None,
) -> PreprocessResult:
    """Apply one explicit preprocessing candidate to an RGB image.

    Operations are intentionally independent in v0.3 (perspective, deskew,
    enhance, upscale). This supports controlled candidate comparison and makes
    it easy to learn which transform helps a particular document.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - OpenCV is a project dependency
        raise RuntimeError("OpenCV is required for preprocessing.") from exc

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3).")
    if image_rgb.dtype != np.uint8:
        raise ValueError("Expected an uint8 RGB image.")

    allowed = {"perspective", "deskew", "enhance", "upscale"}
    operations = tuple(dict.fromkeys(operations))
    unknown = set(operations) - allowed
    if unknown:
        raise ValueError(f"Unknown preprocessing operations: {sorted(unknown)}")

    source_height, source_width = image_rgb.shape[:2]
    processed = image_rgb.copy()
    source_to_processed = np.eye(3, dtype=np.float64)
    metadata: dict[str, Any] = {
        "enabled": bool(operations),
        "operations": list(operations),
        "perspective_corrected": False,
        "deskewed": False,
        "deskew_angle_degrees": 0.0,
        "document_boundary_score": None,
        "document_area_ratio": None,
        "enhancement": False,
        "upscaled": False,
        "upscale_factor": 1.0,
        "source_size": [int(source_width), int(source_height)],
    }

    if "perspective" in operations and allow_perspective_correction:
        quad = detected_quad or detect_document_quad(processed)
        if quad is not None:
            transform, (warped_width, warped_height) = perspective_transform_for_quad(
                quad.points
            )
            longest = max(warped_width, warped_height)
            if longest > config.preprocess_max_side:
                scale = config.preprocess_max_side / float(longest)
                warped_width = max(2, int(round(warped_width * scale)))
                warped_height = max(2, int(round(warped_height * scale)))
                scale_matrix = np.array(
                    [[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64
                )
                transform = scale_matrix @ transform

            processed = cv2.warpPerspective(
                processed,
                transform,
                (warped_width, warped_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            source_to_processed = transform @ source_to_processed
            metadata.update({
                "perspective_corrected": True,
                "document_boundary_score": round(quad.score, 4),
                "document_area_ratio": round(quad.area_ratio, 4),
            })

    if "deskew" in operations:
        # Re-estimate on the current image so this remains valid if operations
        # are later composed (for example, perspective followed by deskew).
        from .image_quality import analyze_image_quality

        quality = None
        if deskew_angle_degrees is None:
            quality = analyze_image_quality(processed)
            angle = quality.estimated_skew_degrees
            line_support = quality.skew_line_support
        else:
            angle = float(deskew_angle_degrees)
            line_support = config.deskew_min_line_support
        if (
            abs(angle) >= config.deskew_trigger_degrees
            and line_support >= config.deskew_min_line_support
        ):
            height, width = processed.shape[:2]
            center = (width / 2.0, height / 2.0)
            matrix_2d = cv2.getRotationMatrix2D(center, angle, 1.0)
            cos_value = abs(float(matrix_2d[0, 0]))
            sin_value = abs(float(matrix_2d[0, 1]))
            new_width = max(2, int(round(height * sin_value + width * cos_value)))
            new_height = max(2, int(round(height * cos_value + width * sin_value)))
            matrix_2d[0, 2] += (new_width / 2.0) - center[0]
            matrix_2d[1, 2] += (new_height / 2.0) - center[1]
            deskew_matrix = np.vstack([matrix_2d, [0.0, 0.0, 1.0]])
            processed = cv2.warpAffine(
                processed,
                matrix_2d,
                (new_width, new_height),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            source_to_processed = deskew_matrix @ source_to_processed
            metadata["deskewed"] = True
            metadata["deskew_angle_degrees"] = round(float(angle), 3)
        else:
            metadata["deskewed"] = False
            metadata["deskew_angle_degrees"] = round(float(angle), 3)

    if "upscale" in operations:
        height, width = processed.shape[:2]
        longest = max(height, width)
        if longest < config.preprocess_upscale_threshold:
            scale = min(
                config.preprocess_max_upscale,
                config.preprocess_target_long_side / float(longest),
            )
            if scale > 1.05:
                new_width = max(2, int(round(width * scale)))
                new_height = max(2, int(round(height * scale)))
                processed = cv2.resize(
                    processed,
                    (new_width, new_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                scale_matrix = np.array(
                    [[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64
                )
                source_to_processed = scale_matrix @ source_to_processed
                metadata.update({
                    "upscaled": True,
                    "upscale_factor": round(float(scale), 3),
                })

    if "enhance" in operations:
        processed = enhance_document_image(processed)
        metadata["enhancement"] = "clahe_plus_mild_unsharp_mask"

    metadata["processed_size"] = [int(processed.shape[1]), int(processed.shape[0])]
    inverse_transform = np.linalg.inv(source_to_processed)
    return PreprocessResult(
        image=processed,
        inverse_transform=inverse_transform,
        metadata=metadata,
    )


def preprocess_page(
    image_rgb: np.ndarray,
    config: OCRConfig,
    *,
    allow_perspective_correction: bool,
) -> PreprocessResult:
    """Backward-compatible combined preprocessing helper used by older tests."""
    operations: list[str] = []
    if config.correct_perspective and allow_perspective_correction:
        operations.append("perspective")
    if config.correct_skew:
        operations.append("deskew")
    if config.upscale_small_images:
        operations.append("upscale")
    if config.enhance_images:
        operations.append("enhance")
    return preprocess_variant(
        image_rgb,
        config,
        operations,
        allow_perspective_correction=allow_perspective_correction,
    )


def map_bbox_to_source(
    bbox: list[int] | tuple[int, int, int, int] | None,
    inverse_transform: np.ndarray,
) -> list[int] | None:
    """Map a processed-image axis-aligned box back to input-page coordinates."""
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = (float(value) for value in bbox[:4])
    corners = np.array(
        [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32
    )
    try:
        import cv2

        mapped = cv2.perspectiveTransform(
            corners, inverse_transform.astype(np.float64)
        )[0]
    except ImportError:  # pragma: no cover
        homogeneous = np.concatenate(
            [corners[0].astype(np.float64), np.ones((4, 1), dtype=np.float64)], axis=1
        )
        projected = (inverse_transform @ homogeneous.T).T
        mapped = projected[:, :2] / projected[:, 2:3]

    return [
        int(round(float(mapped[:, 0].min()))),
        int(round(float(mapped[:, 1].min()))),
        int(round(float(mapped[:, 0].max()))),
        int(round(float(mapped[:, 1].max()))),
    ]
