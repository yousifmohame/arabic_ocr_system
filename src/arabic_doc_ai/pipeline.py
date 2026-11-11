from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import OCRConfig
from .decision_engine import (
    CandidateEvaluation,
    build_adaptive_plan,
    choose_candidate,
    evaluate_ocr_candidate,
)
from .engine import PaddleArabicOCREngine
from .image_quality import analyze_image_quality
from .evaluation import cer, wer, normalized_cer, normalized_wer
from .ingestion import iter_document_pages
from .normalization import normalize_arabic_for_matching
from .preprocessing import map_bbox_to_source, preprocess_variant


def _attach_ground_truth_metrics(evaluation: CandidateEvaluation, reference: str) -> None:
    """Attach benchmark-only metrics; these values never enter candidate selection."""
    hypothesis = evaluation.text
    evaluation.metrics.update({
        "cer": round(cer(reference, hypothesis), 6),
        "wer": round(wer(reference, hypothesis), 6),
        "normalized_cer": round(normalized_cer(reference, hypothesis), 6),
        "normalized_wer": round(normalized_wer(reference, hypothesis), 6),
    })



class ArabicDocumentOCR:
    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self.engine = PaddleArabicOCREngine(self.config)

    def extract(
        self, file_path: str | Path, *, ground_truth_text: str | None = None
    ) -> dict[str, Any]:
        source = Path(file_path).expanduser().resolve()
        pages: list[dict[str, Any]] = []
        is_raster_image = source.suffix.lower() != ".pdf"
        ground_truth_pages = (
            [part.strip() for part in ground_truth_text.split("\f")]
            if ground_truth_text is not None else []
        )

        for page_number, image in iter_document_pages(
            source,
            dpi=self.config.dpi,
            max_side=self.config.max_side,
            max_pages=self.config.max_pages,
        ):
            source_height, source_width = image.shape[:2]

            plans = []
            plan_diagnostics: list[str] = []
            detected_quad = None
            adaptive_enabled = (
                self.config.preprocess_enabled and self.config.preprocess_mode == "adaptive"
            )
            if adaptive_enabled:
                quality_before, detected_quad, plans, plan_diagnostics = build_adaptive_plan(
                    image, self.config, allow_perspective_correction=is_raster_image
                )
            else:
                quality_before = analyze_image_quality(image)
            if not adaptive_enabled and is_raster_image and self.config.correct_perspective:
                # Still report document-focused quality for baseline-only runs.
                from .perspective import detect_document_quad
                detected_quad = detect_document_quad(image)
                if detected_quad is not None:
                    quality_before = analyze_image_quality(
                        image, region_points=detected_quad.points
                    )

            page_ground_truth = (
                ground_truth_pages[page_number - 1]
                if page_number <= len(ground_truth_pages) else None
            )

            # Baseline OCR is always run first. It is the safe fallback and the
            # reference against which all proposed transforms are compared.
            baseline_lines = self.engine.recognize_page(image)
            baseline_text = "\n".join(line["text"] for line in baseline_lines)
            baseline_evaluation = evaluate_ocr_candidate(
                "original", [], baseline_lines, baseline_text=baseline_text
            )
            baseline_evaluation.metrics["image_quality_after"] = quality_before.to_dict()
            if page_ground_truth is not None:
                _attach_ground_truth_metrics(baseline_evaluation, page_ground_truth)
            evaluations: list[CandidateEvaluation] = [baseline_evaluation]
            candidate_runs: dict[str, dict[str, Any]] = {
                "original": {
                    "lines": baseline_lines,
                    "inverse_transform": np.eye(3, dtype=np.float64),
                    "image_size": [int(source_width), int(source_height)],
                    "metadata": {
                        "enabled": False,
                        "operations": [],
                        "source_size": [int(source_width), int(source_height)],
                        "processed_size": [int(source_width), int(source_height)],
                    },
                    "quality_after": quality_before.to_dict(),
                    "reasons": ["Always evaluated as the source-image baseline."],
                }
            }

            for plan in plans:
                prepared = preprocess_variant(
                    image,
                    self.config,
                    plan.operations,
                    allow_perspective_correction=is_raster_image,
                    detected_quad=detected_quad if "perspective" in plan.operations else None,
                    deskew_angle_degrees=(
                        quality_before.estimated_skew_degrees
                        if "deskew" in plan.operations else None
                    ),
                )
                # Do not keep multiple full-resolution variants resident.
                candidate_lines = self.engine.recognize_page(prepared.image)
                for line in candidate_lines:
                    if line.get("bbox") is not None:
                        line["bbox_processed"] = line["bbox"]
                        line["bbox"] = map_bbox_to_source(
                            line["bbox"], prepared.inverse_transform
                        )

                evaluation = evaluate_ocr_candidate(
                    plan.name,
                    plan.operations,
                    candidate_lines,
                    baseline_text=baseline_text,
                )
                quality_after = analyze_image_quality(prepared.image)
                evaluation.metrics["image_quality_after"] = quality_after.to_dict()
                evaluation.metrics["trigger_reasons"] = list(plan.reasons)
                if page_ground_truth is not None:
                    _attach_ground_truth_metrics(evaluation, page_ground_truth)
                evaluations.append(evaluation)
                candidate_runs[plan.name] = {
                    "lines": candidate_lines,
                    "inverse_transform": prepared.inverse_transform,
                    "image_size": [int(prepared.image.shape[1]), int(prepared.image.shape[0])],
                    "metadata": prepared.metadata,
                    "quality_after": quality_after.to_dict(),
                    "reasons": list(plan.reasons),
                }
                del prepared

            chosen = choose_candidate(
                evaluations,
                minimum_improvement=self.config.preprocessing_selection_margin,
            )
            selected_run = candidate_runs[chosen.name]
            selected_lines = selected_run["lines"]

            for line in selected_lines:
                line["normalized_text"] = normalize_arabic_for_matching(line["text"])

            selected_metadata = dict(selected_run["metadata"])
            selected_metadata.update({
                "mode": self.config.preprocess_mode if self.config.preprocess_enabled else "baseline",
                "selected_variant": chosen.name,
                "selection_method": "heuristic_v2_no_baseline_agreement",
                "ground_truth_used_for_selection": False,
                "selection_margin": self.config.preprocessing_selection_margin,
                "ground_truth_supplied_for_evaluation": page_ground_truth is not None,
                "selection_score": round(chosen.adjusted_score, 6),
                "original_quality": quality_before.to_dict(),
                "decision_diagnostics": plan_diagnostics,
                "candidates": [item.to_dict() for item in evaluations],
                "selection_score_warning": (
                    "The runtime selection score is a heuristic, not measured OCR accuracy. "
                    "Baseline text agreement is diagnostic only and does not affect selection."
                ),
                "ground_truth_policy": (
                    "Ground truth is used only for evaluation metrics and never for runtime selection."
                ),
            })

            pages.append({
                "page_number": page_number,
                "width": int(source_width),
                "height": int(source_height),
                "processed_width": int(selected_run["image_size"][0]),
                "processed_height": int(selected_run["image_size"][1]),
                "bbox_coordinate_space": "source_page_pixels",
                "preprocessing": selected_metadata,
                "text": "\n".join(line["text"] for line in selected_lines),
                "lines": selected_lines,
            })
            del image, baseline_lines, selected_lines, candidate_runs

        return {
            "engine": "PaddleOCR PP-OCRv5 Arabic",
            "device": "cpu",
            "source_file": source.name,
            "page_count": len(pages),
            "preprocessing_version": "0.3.1-decision-v2",
            "evaluation_version": "strict-and-arabic-normalized-v1",
            "ground_truth_pages_supplied": len(ground_truth_pages),
            "pages": pages,
        }
