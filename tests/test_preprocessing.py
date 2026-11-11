import unittest

import numpy as np

from arabic_doc_ai.config import OCRConfig
from arabic_doc_ai.perspective import detect_document_quad, order_quad_points
from arabic_doc_ai.preprocessing import map_bbox_to_source, preprocess_page, preprocess_variant


class PerspectiveDetectionTests(unittest.TestCase):
    def test_detects_document_quad_in_synthetic_photo(self):
        import cv2

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[45, 55], [575, 65], [615, 410], [75, 420]], dtype=np.int32)
        cv2.fillConvexPoly(image, polygon, (220, 230, 220))
        quad = detect_document_quad(image)
        self.assertIsNotNone(quad)
        self.assertGreater(quad.area_ratio, 0.4)
        self.assertEqual(quad.points.shape, (4, 2))

    def test_does_not_warp_image_without_safe_margin(self):
        image = np.full((480, 640, 3), 240, dtype=np.uint8)
        quad = detect_document_quad(image)
        self.assertIsNone(quad)


class PreprocessingTests(unittest.TestCase):
    def test_preprocessing_preserves_source_coordinate_mapping(self):
        import cv2

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        polygon = np.array([[45, 55], [575, 65], [615, 410], [75, 420]], dtype=np.int32)
        cv2.fillConvexPoly(image, polygon, (220, 230, 220))
        result = preprocess_page(image, OCRConfig(), allow_perspective_correction=True)
        self.assertTrue(result.metadata["perspective_corrected"])
        self.assertTrue(result.metadata["upscaled"])
        self.assertEqual(result.image.dtype, np.uint8)
        self.assertEqual(result.image.shape[2], 3)

    def test_bbox_mapping_with_simple_scale(self):
        inverse = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64)
        self.assertEqual(map_bbox_to_source([20, 40, 100, 120], inverse), [10, 20, 50, 60])

    def test_deskew_candidate_records_rotation_and_keeps_source_mapping(self):
        import cv2

        image = np.full((500, 700, 3), 255, dtype=np.uint8)
        for y in (120, 200, 280, 360):
            cv2.line(image, (90, y), (610, y), (0, 0, 0), 3)
        matrix = cv2.getRotationMatrix2D((350, 250), -4.0, 1.0)
        rotated = cv2.warpAffine(
            image, matrix, (700, 500), borderValue=(255, 255, 255)
        )
        result = preprocess_variant(
            rotated,
            OCRConfig(correct_perspective=False, upscale_small_images=False),
            ("deskew",),
            allow_perspective_correction=False,
        )
        self.assertTrue(result.metadata["deskewed"])
        self.assertGreater(abs(result.metadata["deskew_angle_degrees"]), 2.0)
        self.assertEqual(result.inverse_transform.shape, (3, 3))

    def test_upscaled_result_is_bounded(self):
        image = np.full((500, 700, 3), 245, dtype=np.uint8)
        config = OCRConfig(correct_perspective=False, preprocess_max_upscale=2.0)
        result = preprocess_page(image, config, allow_perspective_correction=False)
        self.assertLessEqual(max(result.image.shape[:2]), config.preprocess_target_long_side + 2)


if __name__ == "__main__":
    unittest.main()
