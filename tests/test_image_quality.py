import unittest

import cv2
import numpy as np

from arabic_doc_ai.image_quality import analyze_image_quality


class ImageQualityTests(unittest.TestCase):
    def test_quality_metrics_are_finite_and_bounded(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[30:210, 40:280] = 220
        quality = analyze_image_quality(image)
        self.assertEqual(quality.width, 320)
        self.assertEqual(quality.height, 240)
        self.assertGreaterEqual(quality.edge_density, 0.0)
        self.assertLessEqual(quality.edge_density, 1.0)
        self.assertGreaterEqual(quality.dynamic_range_p95_p5, 0.0)
        self.assertTrue(np.isfinite(quality.blur_laplacian_variance))

    def test_document_roi_ignores_dark_photo_background(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[45, 55], [575, 65], [615, 410], [75, 420]], dtype=np.int32)
        cv2.fillConvexPoly(image, polygon, (210, 220, 210))
        full = analyze_image_quality(image)
        roi = analyze_image_quality(image, region_points=polygon.astype(np.float32))
        self.assertTrue(roi.document_roi_used)
        self.assertGreater(roi.brightness_mean, full.brightness_mean)

    def test_detects_small_rotation_from_long_lines(self):
        image = np.full((500, 700, 3), 255, dtype=np.uint8)
        for y in (120, 200, 280, 360):
            cv2.line(image, (90, y), (610, y), (0, 0, 0), 3)
        matrix = cv2.getRotationMatrix2D((350, 250), -4.0, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (700, 500), borderValue=(255, 255, 255)
        )
        quality = analyze_image_quality(rotated)
        self.assertGreater(abs(quality.estimated_skew_degrees), 2.0)
        self.assertGreaterEqual(quality.skew_line_support, 2)

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            analyze_image_quality(np.zeros((100, 100), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
