import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from arabic_doc_ai.config import OCRConfig
from arabic_doc_ai.pipeline import ArabicDocumentOCR


class FakeEngine:
    def __init__(self, config):
        self.calls = 0

    def recognize_page(self, image):
        self.calls += 1
        # Return deterministic baseline and slightly stronger second result.
        if self.calls == 1:
            confidence = 0.80
            text = "مستند عربي"
        else:
            confidence = 0.99
            text = "مستند عربي واضح"
        return [{"text": text, "confidence": confidence, "bbox": [10, 10, 120, 40]}]


class AdaptivePipelineTests(unittest.TestCase):
    def test_pipeline_reports_original_and_candidate_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.png"
            Image.fromarray(np.full((480, 640, 3), 180, dtype=np.uint8)).save(path)
            with patch("arabic_doc_ai.pipeline.PaddleArabicOCREngine", FakeEngine):
                result = ArabicDocumentOCR(OCRConfig(
                    correct_perspective=False,
                    enhance_images=False,
                    upscale_small_images=True,
                )).extract(path, ground_truth_text="مستند عربي واضح")

        page = result["pages"][0]
        self.assertIn("preprocessing", page)
        self.assertEqual(page["preprocessing"]["selection_method"], "heuristic_v2_no_baseline_agreement")
        self.assertIn("original", [item["name"] for item in page["preprocessing"]["candidates"]])
        self.assertIn("upscale", [item["name"] for item in page["preprocessing"]["candidates"]])
        self.assertEqual(page["bbox_coordinate_space"], "source_page_pixels")
        self.assertTrue(page["preprocessing"]["ground_truth_supplied_for_evaluation"])
        self.assertIn("cer", page["preprocessing"]["candidates"][0]["metrics"])
        self.assertIn("normalized_cer", page["preprocessing"]["candidates"][0]["metrics"])


if __name__ == "__main__":
    unittest.main()
