import unittest

import numpy as np

from arabic_doc_ai.config import OCRConfig
from arabic_doc_ai.decision_engine import (
    CandidateEvaluation,
    build_adaptive_plan,
    choose_candidate,
    evaluate_ocr_candidate,
)


class AdaptivePlanningTests(unittest.TestCase):
    def test_rotated_document_proposes_deskew_candidate(self):
        import cv2

        image = np.full((500, 700, 3), 255, dtype=np.uint8)
        for y in (120, 200, 280, 360):
            cv2.line(image, (90, y), (610, y), (0, 0, 0), 3)
        matrix = cv2.getRotationMatrix2D((350, 250), -4.0, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (700, 500), borderValue=(255, 255, 255)
        )
        _, _, plans, _ = build_adaptive_plan(
            rotated, OCRConfig(correct_perspective=False, enhance_images=False, upscale_small_images=False)
        )
        self.assertIn("deskew", [plan.name for plan in plans])

    def test_small_image_proposes_upscale_candidate(self):
        image = np.full((480, 640, 3), 180, dtype=np.uint8)
        _, _, plans, _ = build_adaptive_plan(image, OCRConfig(correct_perspective=False))
        self.assertIn("upscale", [plan.name for plan in plans])

    def test_candidate_limit_is_respected(self):
        image = np.full((480, 640, 3), 180, dtype=np.uint8)
        config = OCRConfig(
            correct_perspective=False,
            max_preprocessing_candidates=1,
            enhance_dynamic_range_threshold=255,
            enhance_contrast_threshold=255,
            dark_brightness_threshold=200,
            upscale_trigger_long_side=900,
        )
        _, _, plans, _ = build_adaptive_plan(image, config)
        self.assertLessEqual(len(plans), 1)


class CandidateSelectionTests(unittest.TestCase):
    def test_original_wins_when_candidate_does_not_clear_margin(self):
        lines = [{"text": "وزارة التجارة", "confidence": 0.92, "bbox": [1, 1, 100, 30]}]
        original = evaluate_ocr_candidate(
            "original", [], lines, baseline_text="وزارة التجارة"
        )
        candidate = CandidateEvaluation(
            name="enhancement",
            operations=["enhance"],
            raw_score=original.raw_score + 0.005,
            adjusted_score=original.adjusted_score - 0.003,
            metrics={},
        )
        chosen = choose_candidate(
            [original, candidate], minimum_improvement=0.015
        )
        self.assertEqual(chosen.name, "original")
        self.assertTrue(original.selected)
        self.assertFalse(candidate.selected)

    def test_candidate_wins_when_it_clears_margin(self):
        lines = [{"text": "وزارة التجارة", "confidence": 0.92, "bbox": [1, 1, 100, 30]}]
        original = evaluate_ocr_candidate(
            "original", [], lines, baseline_text="وزارة التجارة"
        )
        candidate = CandidateEvaluation(
            name="upscale",
            operations=["upscale"],
            raw_score=original.raw_score + 0.06,
            adjusted_score=original.adjusted_score + 0.052,
            metrics={},
        )
        chosen = choose_candidate(
            [original, candidate], minimum_improvement=0.015
        )
        self.assertEqual(chosen.name, "upscale")
        self.assertTrue(candidate.selected)

    def test_baseline_agreement_is_diagnostic_not_a_score_component(self):
        lines = [{"text": "وزارة التجارة", "confidence": 0.90, "bbox": [1, 1, 100, 30]}]
        exact = evaluate_ocr_candidate(
            "perspective", ["perspective"], lines, baseline_text="وزارة التجارة"
        )
        different = evaluate_ocr_candidate(
            "perspective", ["perspective"], lines, baseline_text="نص مختلف تمامًا"
        )
        self.assertEqual(exact.raw_score, different.raw_score)
        self.assertNotEqual(
            exact.metrics["baseline_text_similarity_diagnostic"],
            different.metrics["baseline_text_similarity_diagnostic"],
        )
        self.assertNotIn("baseline_text_similarity", exact.metrics)

    def test_empty_ocr_does_not_beat_original(self):
        original_lines = [{"text": "الرقم الوطني الموحد", "confidence": 0.9, "bbox": [0, 0, 100, 20]}]
        original = evaluate_ocr_candidate(
            "original", [], original_lines, baseline_text="الرقم الوطني الموحد"
        )
        empty = evaluate_ocr_candidate(
            "enhancement", ["enhance"], [], baseline_text="الرقم الوطني الموحد"
        )
        chosen = choose_candidate([original, empty], minimum_improvement=0.01)
        self.assertEqual(chosen.name, "original")


if __name__ == "__main__":
    unittest.main()
