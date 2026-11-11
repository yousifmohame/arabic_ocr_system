from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
import unicodedata
from typing import Any, Iterable

import numpy as np

from .config import OCRConfig
from .image_quality import ImageQuality, analyze_image_quality
from .perspective import DocumentQuad, detect_document_quad


@dataclass(frozen=True)
class PreprocessingCandidate:
    name: str
    operations: tuple[str, ...]
    reasons: tuple[str, ...]


@dataclass
class CandidateEvaluation:
    name: str
    operations: list[str]
    raw_score: float
    adjusted_score: float
    metrics: dict[str, Any]
    text: str = field(default="", repr=False)
    selected: bool = False
    decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operations": self.operations,
            "raw_score": round(self.raw_score, 6),
            "adjusted_score": round(self.adjusted_score, 6),
            "metrics": self.metrics,
            "selected": self.selected,
            "decision": self.decision,
        }


def build_adaptive_plan(
    image_rgb: np.ndarray,
    config: OCRConfig,
    *,
    allow_perspective_correction: bool = True,
) -> tuple[ImageQuality, DocumentQuad | None, list[PreprocessingCandidate], list[str]]:
    """Analyze a page and propose a small, bounded set of independent variants."""
    quad = (
        detect_document_quad(image_rgb)
        if config.correct_perspective and allow_perspective_correction
        else None
    )
    quality = analyze_image_quality(
        image_rgb, region_points=quad.points if quad is not None else None
    )
    candidates: list[PreprocessingCandidate] = []
    diagnostics: list[str] = []

    if config.correct_perspective and allow_perspective_correction and quad is not None:
        severity = _perspective_severity(quad.points)
        # A boundary must be reasonably strong and large enough. Rectification
        # is only a candidate; OCR evidence decides whether to keep it.
        if quad.score >= config.perspective_min_boundary_score and (
            severity >= config.perspective_min_severity
        ) and quad.area_ratio <= config.perspective_max_area_ratio:
            candidates.append(PreprocessingCandidate(
                name="perspective",
                operations=("perspective",),
                reasons=(
                    f"document boundary score {quad.score:.3f}",
                    f"estimated perspective severity {severity:.3f}",
                ),
            ))
        else:
            diagnostics.append(
                "Perspective correction skipped: boundary/geometry did not pass conservative thresholds."
            )
    elif config.correct_perspective and allow_perspective_correction:
        diagnostics.append("No reliable four-corner document boundary detected.")

    if config.correct_skew:
        skew = quality.estimated_skew_degrees
        if (
            abs(skew) >= config.deskew_trigger_degrees
            and quality.skew_line_support >= config.deskew_min_line_support
        ):
            candidates.append(PreprocessingCandidate(
                name="deskew",
                operations=("deskew",),
                reasons=(
                    f"estimated skew {skew:.2f} degrees",
                    f"long-line support {quality.skew_line_support}",
                ),
            ))
        elif quality.skew_line_support:
            diagnostics.append(
                f"Deskew skipped: estimated angle {skew:.2f} degrees with "
                f"{quality.skew_line_support} supporting lines."
            )

    if config.enhance_images:
        reasons: list[str] = []
        if (
            quality.dynamic_range_p95_p5 < config.enhance_dynamic_range_threshold
            and quality.contrast_std < config.enhance_contrast_threshold
        ):
            reasons.append(
                f"low dynamic range ({quality.dynamic_range_p95_p5:.1f})"
            )
        if quality.brightness_mean < config.dark_brightness_threshold:
            reasons.append(f"underexposure ({quality.brightness_mean:.1f})")
        elif quality.brightness_mean > config.bright_brightness_threshold:
            reasons.append(f"high brightness ({quality.brightness_mean:.1f})")
        if quality.blur_laplacian_variance < config.blur_variance_threshold:
            reasons.append(
                f"low sharpness indicator ({quality.blur_laplacian_variance:.1f})"
            )
        if quality.illumination_variation > config.illumination_variation_threshold:
            reasons.append(
                f"uneven broad illumination ({quality.illumination_variation:.1f})"
            )
        if reasons:
            candidates.append(PreprocessingCandidate(
                name="enhancement",
                operations=("enhance",),
                reasons=tuple(reasons),
            ))

    if config.upscale_small_images and quality.long_side < config.upscale_trigger_long_side:
        candidates.append(PreprocessingCandidate(
            name="upscale",
            operations=("upscale",),
            reasons=(
                f"small source image ({quality.width}x{quality.height})",
            ),
        ))

    # Limit expensive extra OCR passes. Preserve deterministic priority order:
    # perspective, enhancement, then upscale. The original is always evaluated.
    candidates = candidates[: config.max_preprocessing_candidates]
    return quality, quad, candidates, diagnostics


def evaluate_ocr_candidate(
    name: str,
    operations: Iterable[str],
    lines: list[dict[str, Any]],
    *,
    baseline_text: str,
) -> CandidateEvaluation:
    """Compute a runtime heuristic score, not a claim of OCR correctness.

    Baseline agreement and line-count similarity are retained as diagnostics
    only. They are intentionally excluded from the ranking score: a corrected
    candidate must not be penalized merely because it differs from the original.
    """
    operations_list = list(operations)
    texts = [str(line.get("text", "")).strip() for line in lines]
    texts = [text for text in texts if text]
    combined = "\n".join(texts)

    scores = [
        float(np.clip(float(line["confidence"]), 0.0, 1.0))
        for line in lines
        if isinstance(line.get("confidence"), (int, float))
        and np.isfinite(float(line["confidence"]))
    ]
    mean_confidence = float(np.mean(scores)) if scores else 0.0
    median_confidence = float(np.median(scores)) if scores else 0.0
    confidence_score = 0.6 * mean_confidence + 0.4 * median_confidence

    non_space_chars = [char for char in combined if not char.isspace()]
    if non_space_chars:
        valid_chars = 0
        for char in non_space_chars:
            category = unicodedata.category(char)
            # Accept letters, numbers, combining marks, all Unicode punctuation,
            # and common OCR-relevant symbols. This includes Arabic comma/colon.
            if category[0] in {"L", "N", "M", "P"} or char in "$%+_=\\/":
                valid_chars += 1
        text_integrity = valid_chars / len(non_space_chars)
    else:
        text_integrity = 0.0

    if lines:
        meaningful_line_ratio = sum(
            1 for text in texts if len(re.sub(r"\s+", "", text)) >= 3
        ) / max(1, len(lines))
        low_conf_short_ratio = sum(
            1 for line in lines
            if len(re.sub(r"\s+", "", str(line.get("text", "")))) <= 2
            and isinstance(line.get("confidence"), (int, float))
            and float(line["confidence"]) < 0.70
        ) / len(lines)
    else:
        meaningful_line_ratio = 0.0
        low_conf_short_ratio = 1.0
    line_quality = max(0.0, meaningful_line_ratio - 0.35 * low_conf_short_ratio)

    positioned = [line for line in lines if _valid_box(line.get("bbox"))]
    bbox_coverage = len(positioned) / max(1, len(lines))
    overlap_penalty = _overlap_ratio(positioned)
    layout_quality = max(0.0, bbox_coverage - 0.35 * overlap_penalty)

    # Keep agreement and line-count similarity for diagnostics, never as rewards.
    agreement = _text_similarity(baseline_text, combined)
    baseline_line_count = baseline_text.count("\n") + (1 if baseline_text.strip() else 0)
    if baseline_line_count == 0 and len(lines) == 0:
        line_count_similarity = 1.0
    elif baseline_line_count == 0 or len(lines) == 0:
        line_count_similarity = 0.0
    else:
        line_count_similarity = min(baseline_line_count, len(lines)) / max(
            baseline_line_count, len(lines)
        )

    # A transparent heuristic. Confidence is uncalibrated and this score is not
    # used as ground-truth accuracy. Benchmark metrics remain separate.
    score_components = {
        "confidence": 0.30 * confidence_score,
        "text_integrity": 0.30 * text_integrity,
        "line_quality": 0.20 * line_quality,
        "layout_quality": 0.20 * layout_quality,
    }
    raw_score = sum(score_components.values())
    transform_penalty = 0.003 * len(operations_list)
    adjusted_score = raw_score - transform_penalty

    metrics = {
        "line_count": len(lines),
        "character_count": len(non_space_chars),
        "mean_confidence": round(mean_confidence, 6),
        "median_confidence": round(median_confidence, 6),
        "confidence_score": round(confidence_score, 6),
        "text_integrity": round(text_integrity, 6),
        "meaningful_line_ratio": round(meaningful_line_ratio, 6),
        "low_confidence_short_line_ratio": round(low_conf_short_ratio, 6),
        "bbox_coverage": round(bbox_coverage, 6),
        "bbox_overlap_ratio": round(overlap_penalty, 6),
        "layout_quality": round(layout_quality, 6),
        "baseline_text_similarity_diagnostic": round(agreement, 6),
        "line_count_similarity_diagnostic": round(line_count_similarity, 6),
        "score_components": {key: round(value, 6) for key, value in score_components.items()},
        "transform_penalty": round(transform_penalty, 6),
        "ranking_note": "Baseline agreement and line-count similarity are diagnostics only.",
    }
    return CandidateEvaluation(
        name=name,
        operations=operations_list,
        raw_score=raw_score,
        adjusted_score=adjusted_score,
        metrics=metrics,
        text=combined,
    )


def choose_candidate(
    evaluations: list[CandidateEvaluation],
    *,
    minimum_improvement: float,
) -> CandidateEvaluation:
    """Choose a candidate only when it clears a margin over the original."""
    if not evaluations:
        raise ValueError("At least the original OCR evaluation is required.")
    baseline = next((item for item in evaluations if item.name == "original"), None)
    if baseline is None:
        raise ValueError("The candidate list must contain the original result.")

    best = max(evaluations, key=lambda item: item.adjusted_score)
    if best.name != "original" and (
        best.adjusted_score < baseline.adjusted_score + minimum_improvement
    ):
        best = baseline

    baseline.selected = False
    for item in evaluations:
        item.selected = item is best
        if item is best:
            if item.name == "original":
                item.decision = "Selected original: no candidate beat it by the required margin."
            else:
                item.decision = (
                    "Selected: heuristic score exceeded original by "
                    f"{item.adjusted_score - baseline.adjusted_score:.4f}."
                )
        else:
            item.decision = (
                f"Not selected: adjusted score {item.adjusted_score:.4f}; "
                f"selected score {best.adjusted_score:.4f}."
            )
    return best


def _perspective_severity(points: np.ndarray) -> float:
    p = np.asarray(points, dtype=np.float32).reshape(4, 2)
    edge_lengths = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    horizontal_delta = abs(float(edge_lengths[0] - edge_lengths[2])) / max(
        1.0, float(max(edge_lengths[0], edge_lengths[2]))
    )
    vertical_delta = abs(float(edge_lengths[1] - edge_lengths[3])) / max(
        1.0, float(max(edge_lengths[1], edge_lengths[3]))
    )
    angles: list[float] = []
    for index in range(4):
        current = p[index]
        before = p[(index - 1) % 4] - current
        after = p[(index + 1) % 4] - current
        denominator = float(np.linalg.norm(before) * np.linalg.norm(after))
        if denominator <= 1e-6:
            continue
        angle = float(np.degrees(np.arccos(np.clip(np.dot(before, after) / denominator, -1, 1))))
        angles.append(abs(angle - 90.0))
    angle_delta = (float(np.mean(angles)) / 45.0) if angles else 0.0
    return float(min(1.0, 0.40 * horizontal_delta + 0.40 * vertical_delta + 0.20 * angle_delta))


def _valid_box(box: Any) -> bool:
    return (
        isinstance(box, (list, tuple)) and len(box) >= 4
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def _overlap_ratio(lines: list[dict[str, Any]]) -> float:
    boxes = [line["bbox"] for line in lines if _valid_box(line.get("bbox"))]
    # Avoid quadratic work on pathological pages with a very large number of
    # detector fragments. This diagnostic is intentionally approximate.
    boxes = boxes[:300]
    if len(boxes) < 2:
        return 0.0
    overlaps = 0
    pairs = 0
    for index, first in enumerate(boxes):
        ax1, ay1, ax2, ay2 = map(float, first[:4])
        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        for second in boxes[index + 1:]:
            bx1, by1, bx2, by2 = map(float, second[:4])
            intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
                0.0, min(ay2, by2) - max(ay1, by1)
            )
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            if intersection / min(area_a, area_b) > 0.35:
                overlaps += 1
            pairs += 1
    return overlaps / max(1, pairs)


def _text_similarity(first: str, second: str) -> float:
    first = re.sub(r"\s+", " ", first or "").strip()[:8000]
    second = re.sub(r"\s+", " ", second or "").strip()[:8000]
    if not first and not second:
        return 1.0
    return SequenceMatcher(None, first, second).ratio()
