from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .config import (
    ARABIC_RECOGNITION_MODEL_DIR,
    DETECTION_MODEL_DIR,
    OCRConfig,
)


class PaddleArabicOCREngine:
    """Lazy-loaded PaddleOCR general pipeline with Arabic PP-OCRv5 recognition."""

    def __init__(self, config: OCRConfig) -> None:
        config.validate()
        self.config = config
        self._ocr: Any | None = None

    def _load(self) -> Any:
        if self._ocr is not None:
            return self._ocr

        # Avoid unnecessary CPU oversubscription on low-memory machines.
        os.environ["OMP_NUM_THREADS"] = str(self.config.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(self.config.threads)
        # All model weights are local. This disables PaddleX's optional source
        # availability probe so offline inference does not make a network call.
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"

        try:
            import paddle
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR dependencies are missing. Follow the installation steps in README.md."
            ) from exc

        try:
            paddle.set_device("cpu")
            paddle.set_num_threads(self.config.threads)
        except Exception:
            # Device is still forced to CPU in PaddleOCR below. Thread tuning can
            # differ between PaddlePaddle builds, so do not block startup on it.
            pass

        missing_models = [
            path for path in (DETECTION_MODEL_DIR, ARABIC_RECOGNITION_MODEL_DIR)
            if not (path / "inference.yml").is_file()
            or not (path / "inference.pdiparams").is_file()
        ]
        if missing_models:
            missing = ", ".join(str(path) for path in missing_models)
            raise RuntimeError(
                "Local OCR model files are missing. Run "
                "`python scripts/prepare_models.py` while online first. "
                f"Missing: {missing}"
            )

        # The Arabic PP-OCRv5 recognizer is loaded from its local exported
        # inference directory. Keep the explicit model names aligned with the
        # Global.model_name values embedded in the downloaded inference.yml files.
        # Both models are local, so inference does not need model-host access.
        self._ocr = PaddleOCR(
            device="cpu",
            cpu_threads=self.config.threads,
            # These names must match Global.model_name in each inference.yml.
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_detection_model_dir=str(DETECTION_MODEL_DIR),
            text_recognition_model_name="arabic_PP-OCRv5_mobile_rec",
            text_recognition_model_dir=str(ARABIC_RECOGNITION_MODEL_DIR),
            use_doc_orientation_classify=self.config.use_document_orientation,
            use_doc_unwarping=self.config.use_document_unwarping,
            use_textline_orientation=self.config.use_textline_orientation,
            text_recognition_batch_size=self.config.recognition_batch_size,
        )
        return self._ocr

    def recognize_page(self, image_rgb: np.ndarray) -> list[dict[str, Any]]:
        """Run recognition on one page and return line text, score and box."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape (height, width, 3).")

        # PaddleOCR's examples use image paths; its prediction interface also
        # accepts image arrays. Convert RGB to BGR to match OpenCV conventions.
        try:
            import cv2

            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            image_bgr = image_rgb[:, :, ::-1].copy()

        predictions = self._load().predict(image_bgr)
        lines: list[dict[str, Any]] = []
        for prediction in predictions:
            payload = _prediction_to_dict(prediction)
            result = payload.get("res", payload)
            texts = result.get("rec_texts")
            scores = result.get("rec_scores")
            boxes = result.get("rec_boxes")
            polygons = result.get("rec_polys")
            if polygons is None or len(polygons) == 0:
                polygons = result.get("dt_polys")
            texts = [] if texts is None else texts
            scores = [] if scores is None else scores

            for index, text in enumerate(texts):
                text = str(text).strip()
                if not text:
                    continue
                score = float(scores[index]) if index < len(scores) else None
                box = _get_box(boxes, polygons, index)
                lines.append({
                    "text": text,
                    "confidence": score,
                    "bbox": box,
                })

        return sort_reading_order(lines)


def _prediction_to_dict(prediction: Any) -> dict[str, Any]:
    raw = getattr(prediction, "json", None)
    if callable(raw):
        raw = raw()
    if raw is None and isinstance(prediction, dict):
        raw = prediction
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise RuntimeError(f"Unexpected PaddleOCR result format: {type(raw).__name__}")
    return raw


def _get_box(boxes: Any, polygons: Any, index: int) -> list[int] | None:
    if boxes is not None and len(boxes) > index:
        coords = np.asarray(boxes[index]).reshape(-1).tolist()
        if len(coords) >= 4:
            return [int(round(float(v))) for v in coords[:4]]
    if polygons is not None and len(polygons) > index:
        points = np.asarray(polygons[index]).reshape(-1, 2)
        if len(points):
            return [
                int(round(float(points[:, 0].min()))),
                int(round(float(points[:, 1].min()))),
                int(round(float(points[:, 0].max()))),
                int(round(float(points[:, 1].max()))),
            ]
    return None


def sort_reading_order(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Approximate Arabic reading order: top-to-bottom, right-to-left per row.

    Lines whose vertical centers are close are grouped into the same row before
    sorting horizontally. This remains a heuristic; multi-column layouts need
    a dedicated layout analysis stage.
    """
    positioned: list[tuple[float, float, float, dict[str, Any]]] = []
    unpositioned: list[dict[str, Any]] = []
    heights: list[float] = []
    for line in lines:
        box = line.get("bbox")
        if not box or len(box) < 4:
            unpositioned.append(line)
            continue
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        center_y = (y1 + y2) / 2.0
        height = max(1.0, y2 - y1)
        heights.append(height)
        positioned.append((center_y, -x2, height, line))

    if not positioned:
        return unpositioned

    heights.sort()
    median_height = heights[len(heights) // 2]
    row_tolerance = max(6.0, median_height * 0.65)
    positioned.sort(key=lambda item: item[0])

    rows: list[dict[str, Any]] = []
    for center_y, neg_x2, height, line in positioned:
        if not rows or abs(center_y - rows[-1]["center_y"]) > row_tolerance:
            rows.append({"center_y": center_y, "items": [(neg_x2, line)]})
        else:
            row = rows[-1]
            count = len(row["items"])
            row["center_y"] = (row["center_y"] * count + center_y) / (count + 1)
            row["items"].append((neg_x2, line))

    ordered: list[dict[str, Any]] = []
    for row in rows:
        ordered.extend(line for _, line in sorted(row["items"], key=lambda item: item[0]))
    ordered.extend(unpositioned)
    return ordered
